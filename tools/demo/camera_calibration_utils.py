import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import pandas as pd

def bundle_adjustment(
    kpts_3D_shared,    
    kpts_2D_dict,
    camera_matrix,
    extrinsics_init_dict,
    return_optimized_points=False,
    verbose=False 
):
    """
    Bundle adjustment: jointly optimize camera extrinsics and 3D points to minimize reprojection error.
    Returns optimized extrinsics and 3D points.
    """
    cam_names = list(kpts_2D_dict.keys())
    n_points = kpts_3D_shared.shape[0]

    # Flatten parameters: [rvec_cam0, tvec_cam0, ..., rvec_camN, tvec_camN, X0, X1, ..., XM]
    def pack_params(extrinsics_dict, points_3d):
        params = []
        for cam in cam_names:
            R = extrinsics_dict[cam][:3, :3]
            t = extrinsics_dict[cam][:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            params.extend(rvec.flatten())
            params.extend(t.flatten())
        params.extend(points_3d.flatten())
        return np.array(params)

    def unpack_params(params):
        extrinsics = {}
        idx = 0
        for cam in cam_names:
            rvec = params[idx:idx+3]
            t = params[idx+3:idx+6]
            R, _ = cv2.Rodrigues(rvec)
            ext = np.eye(4)
            ext[:3, :3] = R
            ext[:3, 3] = t
            extrinsics[cam] = ext
            idx += 6
        points_3d = params[idx:].reshape((n_points, 3))
        return extrinsics, points_3d

    # Build observation list: (cam_idx, observed_2d)
    observations = []
    for cam_idx, cam in enumerate(cam_names):
        kpts = kpts_2D_dict[cam]
        for pt_idx, kpt in enumerate(kpts):
            if np.any(kpt == -1):
                continue
            observations.append((cam_idx, pt_idx, kpt.astype(np.float32)))

    # Residual function: reprojection error for all observations
    def residuals(params):
        extrinsics, points_3d = unpack_params(params)
        res = []
        for cam_idx, pt_idx, observed_2d in observations:
            cam = cam_names[cam_idx]
            K = camera_matrix if isinstance(camera_matrix, np.ndarray) else camera_matrix[cam]
            ext = extrinsics[cam]
            X = points_3d[pt_idx]
            X_cam = ext[:3, :3] @ X + ext[:3, 3]
            x_proj = K @ X_cam
            x_proj = x_proj[:2] / x_proj[2]
            res.append((observed_2d - x_proj).flatten())
        return np.concatenate(res)

    # Initial parameter vector
    x0 = pack_params(extrinsics_init_dict, kpts_3D_shared)

    # Run least squares optimization (Levenberg-Marquardt)
    result = least_squares(
        residuals,
        x0,
        method='trf',  # Levenberg-Marquardt, good for small/medium problems
        loss='cauchy',  # Robust loss function to deal with outliers
        max_nfev=2000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
        verbose=2 if verbose else 0
    )

    # Unpack optimized parameters
    optimized_extrinsics, kpts_3D_shared_optimized = unpack_params(result.x)

    if return_optimized_points:
        return optimized_extrinsics, kpts_3D_shared_optimized, result
    
    return optimized_extrinsics, result


def refine_extrinsics_with_features(images, extrinsics, camera_matrix):
    # images: dict {cam_name: [img1, img2, ...]}
    # extrinsics: dict {cam_name: 4x4}
    # camera_matrix: 3x3

    # 1. Detect features in all images (use first frame for each camera)
    detector = cv2.SIFT_create()
    keypoints_dict = {}
    descriptors_dict = {}
    for cam_name, imgs in images.items():
        img = imgs[0]
        kps, desc = detector.detectAndCompute(img, None)
        keypoints_dict[cam_name] = kps
        descriptors_dict[cam_name] = desc
        print(f"  {cam_name}: {len(kps)} features detected.")

    # 2. Match features between camera pairs
    matcher = cv2.BFMatcher()
    matches_dict = {}
    cam_names = list(images.keys())
    for i in range(len(cam_names)):
        for j in range(i+1, len(cam_names)):
            cam1, cam2 = cam_names[i], cam_names[j]
            matches = matcher.knnMatch(descriptors_dict[cam1], descriptors_dict[cam2], k=2)
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            matches_dict[(cam1, cam2)] = good_matches
            print(f"  {cam1} <-> {cam2}: {len(matches)} raw matches, {len(good_matches)} good matches after ratio test.")

    # 3. Triangulate matched points and collect 2D-3D correspondences
    triangulated_points = []
    correspondences = {cam: [] for cam in cam_names}
    for (cam1, cam2), matches in matches_dict.items():
        kps1 = keypoints_dict[cam1]
        kps2 = keypoints_dict[cam2]
        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
        # Projection matrices
        P1 = camera_matrix @ extrinsics[cam1][:3, :]
        P2 = camera_matrix @ extrinsics[cam2][:3, :]
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        triangulated_points.append(pts3d)
        correspondences[cam1].extend(zip(pts3d, pts1))
        correspondences[cam2].extend(zip(pts3d, pts2))
        print(f"  {cam1} <-> {cam2}: {pts3d.shape[0]} points triangulated.")

    # 4. Refine extrinsics for each camera using solvePnP
    refined_extrinsics = {}
    pts3d_dict = {}
    for cam_name in cam_names:
        if len(correspondences[cam_name]) < 6:
            print(f"  {cam_name}: Not enough correspondences for PnP ({len(correspondences[cam_name])} found). Skipping.")
            refined_extrinsics[cam_name] = extrinsics[cam_name]
            continue
        pts3d = np.array([c[0] for c in correspondences[cam_name]], dtype=np.float32)
        pts2d = np.array([c[1] for c in correspondences[cam_name]], dtype=np.float32)
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
        )
        R, _ = cv2.Rodrigues(rvec)
        ext = np.eye(4)
        ext[:3, :3] = R
        ext[:3, 3] = tvec.squeeze()
        refined_extrinsics[cam_name] = ext
        pts3d_dict[cam_name] = pts3d
        print(f"  {cam_name}: Refined using {len(inliers) if inliers is not None else 0} inliers.")

    return refined_extrinsics, pts3d_dict


def validate_and_plot_calibration_scene(
    extrinsics,
    camera_matrix,
    ref_3D_kpts,
    calib_3D_kpts_dict,
    calib_2D_kpts_dict,
    calib_valid_3D_kpts_dict,
    calib_valid_2D_kpts_dict,
    img_wh=(640, 480),
    connect_indices=None
):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define common dictionaries
    colors_palette = plt.get_cmap('tab20')
    colors = {}
    camera_matrices = {}
    for i, cam_name in enumerate(calib_3D_kpts_dict.keys()):
        # Colors
        colors[cam_name] = ([colors_palette(2*i)], [colors_palette(2*i+1)]) # GT and reprojected colors
        
        # Camera matrices
        ext = extrinsics[cam_name]
        R = ext[:3, :3]
        t = ext[:3, 3]
        K = camera_matrix if isinstance(camera_matrix, np.ndarray) else camera_matrix[cam_name]
        camera_matrices[cam_name] = (R, t, K)

    # Plot world frame axes
    plot_world_frame_axes(ax)

    # Plot each camera frame
    for cam_name in extrinsics.keys():
        R = camera_matrices[cam_name][0]
        t = camera_matrices[cam_name][1]
        plot_camera_frame_axes(ax, R, t, cam_name)
    
    # Plot 3D calibration points for each camera
    for cam_name, pts_3D in calib_3D_kpts_dict.items():
        # Plot 3D points for this camera
        ax.scatter(pts_3D[:, 0], pts_3D[:, 1], pts_3D[:, 2], 
                   c=colors[cam_name][0], s=20, label=f'{cam_name}: calibration', alpha=1.0)
        
    # Plot 3D validation points for each camera
    for cam_name, pts_3D in calib_valid_3D_kpts_dict.items():
        # Plot 3D points for this camera
        ax.scatter(pts_3D[:, 0], pts_3D[:, 1], pts_3D[:, 2], 
                   c=colors[cam_name][0], s=20, label=f'{cam_name}: validation', alpha=1.0, marker='x')
        
    # Plot reference 3D points (not used for calibration)
    ax.scatter(ref_3D_kpts[:, 0], ref_3D_kpts[:, 1], ref_3D_kpts[:, 2], c='black', s=60, label='reference points')
    for i, pt in enumerate(ref_3D_kpts):
        ax.text(pt[0], pt[1], pt[2], f'{i}_ref', color='black', fontsize=8)
    
    # Connect selected indices with light blue lines (closing the loop)
    if connect_indices is not None and len(connect_indices) > 1:
        pts_3D_connected = ref_3D_kpts[connect_indices]
        # Close the loop
        pts_3D_connected = np.vstack([pts_3D_connected, pts_3D_connected[0]])
        ax.plot(pts_3D_connected[:, 0], pts_3D_connected[:, 1], pts_3D_connected[:, 2], color='deepskyblue', linewidth=2)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Camera Calibration Scene')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    ax.view_init(elev=20, azim=-30)  # Adjust view angle for better visibility
    plt.tight_layout()
    plt.draw()

    # Plot 2D calibration points for each camera and the 2D reprojection of 3D points
    for cam_name, pts_2D in calib_2D_kpts_dict.items():        
        # Reproject 3D points to 2D
        projected_2D, _, errors = project_3d_points_to_2d(
            calib_3D_kpts_dict[cam_name], # 3D points
            camera_matrices[cam_name][0], # R
            camera_matrices[cam_name][1], # t
            camera_matrices[cam_name][2], # K
            pts_2D_gt=pts_2D,
            return_error=True
        )

        print("Calibration Points")
        print_reprojection_error_report(pts_2D, projected_2D, errors, cam_name)

        # Plot 2D GT points and reprojected 2D points
        plot_2d_keypoints_with_indices(
            pts_2D, projected_2D, colors[cam_name],
            figsize=(12, 10),
            img_wh=img_wh,
            cam_name=cam_name,
            title="2D Points vs 3D Reprojected Points - Calibration"
        )

    # Plot 2D validation points for each camera and the 2D reprojection of 3D points
    for cam_name, pts_2D in calib_valid_2D_kpts_dict.items():
        # Reproject 3D points to 2D
        projected_2D, _, errors = project_3d_points_to_2d(
            calib_valid_3D_kpts_dict[cam_name],  # 3D points
            camera_matrices[cam_name][0],  # R
            camera_matrices[cam_name][1],  # t
            camera_matrices[cam_name][2],  # K
            pts_2D_gt=pts_2D,
            return_error=True
        )

        print("Validation Points")
        print_reprojection_error_report(pts_2D, projected_2D, errors, cam_name)

        # Plot 2D GT points and reprojected 2D points
        plot_2d_keypoints_with_indices(
            pts_2D, projected_2D, colors[cam_name],
            figsize=(12, 10),
            img_wh=img_wh,
            cam_name=cam_name,
            title="2D Points vs 3D Reprojected Points - Validation"
        )

    plt.show()


def plot_2d_keypoints_with_indices(
    pts_GT,           # (N, 2) array of ground truth 2D points
    pts_proj,         # (N, 2) array of reprojected 2D points
    colors,           # tuple: (color_GT, color_proj)
    title="2D Points vs 3D Reprojected Points",
    figsize=(10, 8),
    img_wh=(640, 480),
    cam_name=None
):
    """
    Plot GT and reprojected 2D keypoints with indices, OpenCV convention (origin top-left, y down).
    """
    fig = plt.figure(figsize=figsize)
    ax_temp = fig.add_subplot(111)
    color_GT, color_proj = colors

    ax_temp.scatter(pts_GT[:, 0], pts_GT[:, 1],
                    c=color_GT, s=20, label=f'{cam_name}: GT 2D' if cam_name else 'GT 2D', alpha=1.0)
    ax_temp.scatter(pts_proj[:, 0], pts_proj[:, 1],
                    c=color_proj, s=20, label=f'{cam_name}: reprojected 2D' if cam_name else 'reprojected 2D', alpha=1.0)
    
    # Add index numbers next to each 2D dot (GT)
    for idx, (x, y) in enumerate(pts_GT):
        ax_temp.text(x+4, y+4, str(idx), fontsize=11)
    # Add index numbers next to each 2D dot (reprojected)
    for idx, (x, y) in enumerate(pts_proj):
        ax_temp.text(x+4, y+4, str(idx), fontsize=11)
    
    ax_temp.set_title(title if cam_name is None else f'{cam_name}: {title}')
    ax_temp.set_xlabel('X [pixels]')
    ax_temp.set_ylabel('Y [pixels]')
    ax_temp.set_xlim([0, img_wh[0]])
    ax_temp.set_ylim([0, img_wh[1]])
    ax_temp.invert_yaxis()  # OpenCV: origin top-left, y down
    ax_temp.legend()
    ax_temp.grid(True)
    ax_temp.set_aspect('equal')
    
    plt.tight_layout()
    plt.draw()


def project_3d_points_to_2d(pts_3D, R, t, K, pts_2D_gt=None, dist_coeffs=None, return_error=False):
    """
    Projects 3D points to 2D image coordinates using OpenCV's projectPoints.
    Optionally computes reprojection error if ground truth 2D points are provided.

    Args:
        pts_3D: (N, 3) array of 3D points
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        K: (3, 3) camera intrinsic matrix
        pts_2D_gt: (N, 2) array of ground truth 2D points (optional)
        dist_coeffs: distortion coefficients (default: zeros)
        return_error: if True, also return mean and per-point reprojection error

    Returns:
        projected_2D: (N, 2) array of 2D image points
        [mean_error, errors]: if return_error and pts_2D_gt is provided
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(-1, 1)
    projected_2D, _ = cv2.projectPoints(
        pts_3D.reshape(-1, 1, 3),
        rvec,
        tvec,
        K,
        dist_coeffs
    )
    projected_2D = projected_2D.reshape(-1, 2)

    if return_error and pts_2D_gt is not None:
        errors = np.linalg.norm(pts_2D_gt - projected_2D, axis=1)
        mean_error = np.mean(errors)
        return projected_2D, mean_error, errors

    return projected_2D


def plot_world_frame_axes(ax, length=0.3, linewidth=2):
    """
    Plot the world frame axes (X=red, Y=green, Z=blue) at the origin.
    Args:
        ax: matplotlib 3D axis
        length: length of each axis arrow
        linewidth: width of the axis arrows
    """
    ax.quiver(0, 0, 0, length, 0, 0, color='r', linewidth=linewidth)
    ax.quiver(0, 0, 0, 0, length, 0, color='g', linewidth=linewidth)
    ax.quiver(0, 0, 0, 0, 0, length, color='b', linewidth=linewidth)
    ax.text(length, 0, 0, 'X', color='r')
    ax.text(0, length, 0, 'Y', color='g')
    ax.text(0, 0, length, 'Z', color='b')


def plot_camera_frame_axes(ax, R, t, cam_name=None, length=0.3, linewidth=2):
    """
    Plot the camera frame axes (X=red, Y=green, Z=blue) at the camera origin in world coordinates.
    Args:
        ax: matplotlib 3D axis
        R: (3, 3) rotation matrix (world-to-camera)
        t: (3,) translation vector (world-to-camera)
        cam_name: optional, name to display at the camera origin
        length: length of each axis arrow
        linewidth: width of the axis arrows
    """
    cam_origin = -R.T @ t
    cam_axes = R.T @ np.eye(3) * length
    ax.quiver(*cam_origin, *cam_axes[:, 0], color='r', linewidth=linewidth)
    ax.quiver(*cam_origin, *cam_axes[:, 1], color='g', linewidth=linewidth)
    ax.quiver(*cam_origin, *cam_axes[:, 2], color='b', linewidth=linewidth)
    if cam_name is not None:
        ax.text(*cam_origin, cam_name, color='k', fontsize=12)


def print_reprojection_error_report(gt_2d, proj_2d, errors, cam_name=None):
    """
    Print a formatted reprojection error report.
    Args:
        gt_2d: (N, 2) array of ground truth 2D points
        proj_2d: (N, 2) array of reprojected 2D points
        errors: (N,) array of per-point reprojection errors
    """
    # Filter out invalid points
    valid_mask = ~np.any(np.equal(gt_2d, -1), axis=1)
    gt_2d = gt_2d[valid_mask]
    proj_2d = proj_2d[valid_mask]
    errors = errors[valid_mask]
    if len(gt_2d) == 0:
        print("No valid points for reprojection error report.")
        return
    
    # Calculate mean and max errors
    if len(errors) == 0:
        print("No valid points for reprojection error report.")
        return
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"\nReprojection error report for {cam_name if cam_name else 'camera'}:")
    print(f"  Mean: {mean_error:.2f}")
    print(f"  Max:  {max_error:.2f}")
    print("Detailed point comparison:")
    print("Point | Observed (u,v) | Projected (u,v) | Error")
    print("-" * 50)
    for i, (gt, proj, err) in enumerate(zip(gt_2d, proj_2d, errors)):
        print(f"  {i:2d}  | ({gt[0]:6.1f},{gt[1]:6.1f}) | ({proj[0]:6.1f},{proj[1]:6.1f}) | {err:7.2f}")
    print("-" * 50, "\n")


def load_calibration_points_from_csv(frames_dir, csv_filename="calibration_points.csv"):
    """
    Loads calibration keypoints from a CSV file in frames_dir.
    Returns:
        - kpts_dict: dict {cam_name: np.array of shape (N, 2)} with 2D points (u, v)
        - object_3D_points_dict: dict {cam_name: np.array of shape (N, 3)} with 3D points (x, y, z)
        - full_df: the full pandas DataFrame for further analysis
    """
    csv_path = os.path.join(frames_dir, csv_filename)
    df = pd.read_csv(csv_path)

    # Group by camera
    kpts_2D_dict = {}
    kpts_3D_dict = {}
    for cam_id, group in df.groupby('camera'):
        cam_name = 'cam_' + str(cam_id)
        # 2D points (u, v)
        kpts = group[['u', 'v']].to_numpy(dtype=np.float32)
        # 3D points (x, y, z)
        pts3d = group[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        kpts_2D_dict[cam_name] = kpts
        kpts_3D_dict[cam_name] = pts3d

    return kpts_2D_dict, kpts_3D_dict


def calibrate_cameras_with_ref_kpts(
        kpts_2D_dict,
        kpts_3D_dict,
        camera_matrix,
        dist_coeffs,
        debug=False
    ):
    if debug:
        print("[Calibration] 2D keypoints")
        print_points_dict(kpts_2D_dict)
        print("[Calibration] 3D keypoints")
        print_points_dict(kpts_3D_dict)
    
    extrinsics = {}
    for cam_name, keypoints in kpts_2D_dict.items():
        image_points = np.array(keypoints, dtype=np.float32)

        # Exclude the image_points that are [-1, -1] and the corresponding 3D points
        valid_mask = ~np.any(np.equal(image_points, -1), axis=1)
        valid_image_points = image_points[valid_mask]
        valid_object_3D_points = kpts_3D_dict[cam_name][valid_mask]

        if len(valid_image_points) < 4:
            raise ValueError(f"Insufficient valid points for {cam_name}: "
                           f"need ≥4, got {len(valid_image_points)}")

        # Solve PnP for this camera
        success, rvec, tvec = cv2.solvePnP(
            valid_object_3D_points,
            valid_image_points,
            camera_matrix,
            dist_coeffs
        )
        if not success:
            raise RuntimeError(f"Failed to solvePnP for camera {cam_name}. Check input data.")
        
        # Build extrinsic matrix [R|t] that transforms world -> camera
        R, _ = cv2.Rodrigues(rvec)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = tvec.squeeze()
        extrinsics[cam_name] = extrinsic

        if debug:
            print(f"\n[Calibration] Camera: {cam_name}")
            print(f"\n[Calibration] INTRINSIC matrix:\n{camera_matrix}")
            print(f"[Calibration] EXTRINSIC matrix:\n{extrinsic} w.r.t. world frame (e.g., world -> cam)")

            # Compute Euler angles from rotation matrix
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else: # Gimbal lock
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0.0
            
            # Normalize angles to [-180, 180) degrees
            x = (np.degrees(x) + 180) % 360 - 180
            y = (np.degrees(y) + 180) % 360 - 180
            z = (np.degrees(z) + 180) % 360 - 180

            print(f"[Calibration] EXTRINSIC Euler angles (rad) (Z→Y→X) (world -> cam):\n"
                  f"  Yaw (Z): {z:.3f}, Pitch (Y): {y:.3f}, Roll (X): {x:.3f}")
            print(f"[Calibration] EXTRINSIC Rodrigues vector (world -> cam):\n"
                  f"  {rvec.squeeze()}")
            
        # Camera position in world frame (inverse transformation)
        cam_pos_world = -R.T @ tvec.squeeze()
        print(f"[Calibration] Camera position in world: {cam_pos_world}")
            
    return extrinsics


def print_points_dict(points_dict, float_fmt="{:10.3f}"):
    """
    Neatly print a dictionary of N-dimensional points per camera.
    Args:
        points_dict: dict {cam_name: np.ndarray of shape (N, D)}
        float_fmt: format string for floats
    """
    for cam_name, arr in points_dict.items():
        print(f"\nCamera: {cam_name}")
        header = " Index |   " + "".join([f"    {chr(88+i)}" for i in range(arr.shape[1])])  # X, Y, Z, ...
        print(header)
        print("-" * (9 + 12 * arr.shape[1]))
        for idx, row in enumerate(arr):
            row_str = " ".join([float_fmt.format(val) for val in row])
            print(f"  {idx:3d}  | {row_str}")
        print("-" * (9 + 12 * arr.shape[1]))