import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from pathlib import Path
import cv2

def estimate_3d_bbox(bbox_2d_list, camera_params, initial_guess=None, constraints=()):
    """
    Optimize 3D bbox parameters to minimize reprojection error across multiple views.

    Args:
        bbox_2d_list: List of (N_views, 4) arrays, each [x_min, y_min, x_max, y_max] for each camera.
        camera_params: List of dicts, each with keys 'K' (3x3), 'R' (3x3), 't' (3,), one per camera.
        initial_guess: Optional, initial guess for [x, y, z, w, h, d, yaw].
        constraints: Optional, constraints for optimizer.

    Returns:
        result: OptimizeResult from scipy.optimize.minimize
    """

    def parse_params(params):
        center = params[:3]
        dimensions = params[3:6]
        rvec = params[6:9]
        return center, dimensions, rvec

    def generate_3d_corners(center, dimensions, rvec):
        # 3D corners of a unit cube centered at origin
        l, h, w = dimensions
        x = l / 2.0
        y = h / 2.0
        z = w / 2.0
        corners = np.array([
            [-x, -y, -z], [ x, -y, -z], [ x, -y,  z], [-x, -y,  z],
            [-x,  y, -z], [ x,  y, -z], [ x,  y,  z], [-x,  y,  z]
        ])
        # Apply rotation
        R = cv2.Rodrigues(rvec)[0]  # Convert rotation vector to rotation matrix
        # Rotate and translate corners
        corners = (R @ corners.T).T + center
        return corners

    def project_points(points_3d, cam):
        # cam: dict with 'K', 'R', 't'
        P = cam['K'] @ np.hstack([cam['R'], cam['t'].reshape(3,1)])  # (3,4)
        points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (8,4)
        proj = (P @ points_3d_h.T).T  # (8,3)
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    def compute_2d_bbox(pts_2d):
        x_min, y_min = pts_2d.min(axis=0)
        x_max, y_max = pts_2d.max(axis=0)
        return np.array([x_min, y_min, x_max, y_max])

    def bbox_iou_loss(pred_bbox, gt_bbox):
        # L1 loss between bbox corners (can use IoU or GIoU loss if desired)
        return np.abs(pred_bbox - gt_bbox).sum()

    def objective(params):
        center, dimensions, rvec = parse_params(params)
        bbox_3d_corners = generate_3d_corners(center, dimensions, rvec)
        total_error = 0
        for cam_idx, bbox_2d in enumerate(bbox_2d_list):
            projected_corners = project_points(bbox_3d_corners, camera_params[cam_idx])
            pred_bbox_2d = compute_2d_bbox(projected_corners)
            total_error += bbox_iou_loss(pred_bbox_2d, bbox_2d)
        return total_error

    # Initial guess: center at origin, dimensions 1, yaw 0
    if initial_guess is None:
        initial_guess = np.array([0, 0, 5, 1, 2, 0.5, 0, 0, 0])  # [x, y, z, w, h, d, rvec(0), rvec(1), rvec(2)]

    # Add bounds: dimensions (indices 3,4,5) must be > 0
    bounds = [
        (None, None),  # x
        (None, None),  # y
        (None, None),  # z
        (1e-3, None),  # length > 0
        (1e-3, None),  # height > 0
        (1e-3, None),  # width > 0
        (None, None),  # rvec_x
        (None, None),  # rvec_y
        (None, None),  # rvec_z
    ]

    result = minimize(
        objective,
        initial_guess,
        constraints=constraints,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True}
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    # Parse optimized parameters
    center, dimensions, rvec = parse_params(result.x)
    return result, center, dimensions, rvec


def triangulate_object_bbx(cfg):
    """
    Triangulate 3D object bounding box centers from multi-view 2D bounding box trajectories
    using optimization-based 3D bbox fitting for each time step.

    Returns:
        obj_bbx_2D_dict: dict[cam_name] = (N, 4) numpy arrays of 2D bbox [x_min, y_min, x_max, y_max] per frame.
        obj_bbx_3D: (N, 7) numpy array of optimized 3D bbox params per frame: [x, y, z, w, h, d, yaw]
    """
    # Load extrinsics and intrinsics
    extrinsics = torch.load(cfg.paths.extrinsics)
    intrinsics = torch.load(cfg.paths.intrinsics)
    if isinstance(intrinsics, dict):
        K_dict = {cam: torch.from_numpy(intrinsics[cam]) if isinstance(intrinsics[cam], np.ndarray) else intrinsics[cam] for cam in extrinsics}
    else:
        K_dict = {cam: torch.from_numpy(intrinsics) if isinstance(intrinsics, np.ndarray) else intrinsics for cam in extrinsics}

    cam_names = list(K_dict.keys())

    obj_bbx_2D_dict = {}
    num_frames = None
    for cam_name in cam_names:
        bbx_path = Path(cfg.preprocess_dir) / f"{cam_name}_bboxes.csv"
        if not bbx_path.exists():
            raise FileNotFoundError(f"Bounding box file not found: {bbx_path}")

        # Compute valid frame indices based on cfg
        start_frame = cfg.idx_first_image_in_folder + cfg.start_idx_images_to_load
        end_frame = cfg.idx_first_image_in_folder + cfg.end_idx_images_to_load
        valid_frames = set(range(start_frame, end_frame))

        # Read CSV and filter rows by valid frames
        df = pd.read_csv(bbx_path)
        if 'frame' not in df.columns:
            raise ValueError(f"'frame' column not found in {bbx_path}")
        df = df[df['frame'].astype(int).isin(valid_frames)]
        df = df.set_index('frame')  # Set the index as the frame column
        obj_bbx_2D_dict[cam_name] = df

        if num_frames is None:
            num_frames = len(df)
        else:
            assert num_frames == len(df), "All cameras must have the same number of frames"

    # Prepare camera parameters for each camera
    camera_params = []
    for cam_name in cam_names:
        K = K_dict[cam_name].cpu().numpy() if isinstance(K_dict[cam_name], torch.Tensor) else K_dict[cam_name]
        T_w2c = extrinsics[cam_name]  # (4, 4)
        R = T_w2c[:3, :3]
        t = T_w2c[:3, 3]
        camera_params.append({'K': K, 'R': R, 't': t})

    # For each frame, optimize 3D bbox parameters
    obj_bbx_3D = pd.DataFrame()
    num_objects = len(obj_bbx_2D_dict[cam_names[0]].columns) // 4  # Assuming each object has 4 columns per bbox
    for j in range(num_objects):
        for i in range(num_frames):
            bbox_2d_list = []
            for cam_name in cam_names:
                df = obj_bbx_2D_dict[cam_name]
                bbox_2d = df.iloc[i, j*4:(j+1)*4].values.astype(float)
                bbox_2d = np.array([bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3]])  # [x_min, y_min, x_max, y_max]
                bbox_2d_list.append(bbox_2d)

            if i == 0:
                initial_guess = np.zeros(9)  # Initial guess for [x, y, z, w, h, d, rvec_x, rvec_y, rvec_z]
            else:
                initial_guess = obj_bbx_3D.iloc[i-1, j*9:(j+1)*9].values
                
            _, center, dimensions, rvec = estimate_3d_bbox(
                bbox_2d_list,
                camera_params,
                initial_guess=initial_guess
            )

            # Store the optimized 3D bbox parameters
            obj_bbx_3D_row = np.concatenate([center, dimensions, rvec])
            if obj_bbx_3D.empty:
                obj_bbx_3D = pd.DataFrame(columns=
                    [
                        f'obj{j+1}_center_x', f'obj{j+1}_center_y', f'obj{j+1}_center_z',
                        f'obj{j+1}_length',   f'obj{j+1}_height',   f'obj{j+1}_width',
                        f'obj{j+1}_rvec_x',   f'obj{j+1}_rvec_y',   f'obj{j+1}_rvec_z'
                    ]
                )
            obj_bbx_3D.loc[i, obj_bbx_3D.columns[j*9:(j+1)*9]] = obj_bbx_3D_row

            import pdb; pdb.set_trace()
    return obj_bbx_2D_dict, obj_bbx_3D