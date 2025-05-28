import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVOImages

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange


CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--video_name", type=str, default="demo", help="Name of the video for output files")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="Focal length of fullframe camera in mm. Leave it as None to use default values."
        "For iPhone 15p, the [0.5x, 1x, 2x, 3x] lens have typical values [13, 24, 48, 77]."
        "If the camera zoom in a lot, you can try 135, 200 or even larger values.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    # Input
    frames_dir = Path(args.frames_dir)
    assert frames_dir.exists() and frames_dir.is_dir(), f"Directory not found: {frames_dir}"

    # Find all subfolders (each is a camera view)
    cam_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    assert len(cam_dirs) > 0, f"No camera subfolders found in {frames_dir}"

    # For each camera, get sorted image paths
    frames_paths_dict = {}
    for cam_dir in cam_dirs:
        cam_frames = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
        assert len(cam_frames) > 0, f"No images found in {cam_dir}"
        frames_paths_dict[cam_dir.name] = cam_frames
        Log.info(f"[Input Dir]: {cam_dir} ({len(cam_frames)} images)")

    # Optionally: check all cameras have the same number of frames
    num_frames = [len(paths) for paths in frames_paths_dict.values()]
    assert len(set(num_frames)) == 1, "All cameras must have the same number of frames (synchronized views)"
    
    # Assume all images have the same size
    sample_img = cv2.imread(str(frames_paths_dict[cam_dirs[0].name][0]))
    height, width = sample_img.shape[:2]
    Log.info(f"[Input Dir]: {frames_dir}")
    Log.info(f"(N, W, H) = ({len(frames_paths_dict.keys())}, {width}, {height}) | {len(cam_dirs)} cameras")

    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={args.video_name}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-images to preprocess dir
    Log.info(f"[Copy Images] {frames_dir} -> {cfg.preprocess_dir}")
    for cam_name, cam_frames in frames_paths_dict.items():
        cam_out_dir = Path(cfg.preprocess_dir) / cam_name
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in tqdm(cam_frames, desc=f"Copying images for {cam_name}"):
            out_path = cam_out_dir / frame_path.name
            if not out_path.exists():
                img = cv2.imread(str(frame_path))
                cv2.imwrite(str(out_path), img)

    return cfg


import matplotlib.pyplot as plt

def show_image_matplotlib(img, title="Image", add_border=False):
    if add_border:
        # Add a 120-pixel white border to the right and bottom
        color = [255, 255, 255]  # White in BGR
        img = cv2.copyMakeBorder(
            img,
            top=0, bottom=120,
            left=0, right=120,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )
    # If image is BGR (OpenCV default), convert to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_to_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_to_show = img
    plt.imshow(img_to_show)
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_reference_kpts_from_config(cfg):
    """
    Loads reference keypoints for each camera from a YAML config file.
    The YAML file should have the following structure:
    reference_object_keypoints:
      cam_1:
        - [x1, y1]
        - [x2, y2]
        - [x3, y3]
        - [x4, y4]
      cam_2:
        - [x1, y1]
        - [x2, y2]
        - [x3, y3]
        - [x4, y4]
      ...
    Returns: dict {cam_name: np.array of shape (4, 2)}
    """
    kpts_dict = {}
    for cam_name, kpts in cfg.reference_object_keypoints.items():
        kpts_dict[cam_name] = np.array(kpts, dtype=np.int16)

    # Get the 3D reference keypoints in the world coordinate system
    object_3D_points = np.array(cfg.reference_object_keypoints_3D, dtype=np.float32)

    return kpts_dict, object_3D_points


def calibrate_cameras_with_ref_kpts(cfg, debug=False):
    """
    Calibrate all cameras with respect to the main_view using 2D keypoints from config and solvePnP.
    Returns:
        dict of {cam_name: extrinsic_matrix (4x4 [R|t])} mapping each camera to the world frame
    """
    ref_kpts_dict, object_3D_points = load_reference_kpts_from_config(cfg)
    # Reference all image keypoints to the first value (subtract the first keypoint for each camera)
    for cam_name, kpts in ref_kpts_dict.items():
        ref_kpts_dict[cam_name] = kpts - kpts[0]
    
    extrinsics = {}

    # Camera intrinsics: estimate from image size or use provided
    # We'll use the first camera's first image to get width/height
    first_cam = list(ref_kpts_dict.keys())[0]
    sample_img_path = sorted((Path(cfg.preprocess_dir) / first_cam).iterdir())[0]
    sample_img = cv2.imread(str(sample_img_path))
    height, width = sample_img.shape[:2]
    
    if cfg.f_mm is not None:
        _, _, camera_matrix = create_camera_sensor(width, height, cfg.f_mm)
        camera_matrix = camera_matrix.cpu().numpy()
    else:
        camera_matrix = estimate_K(width, height).cpu().numpy()
    # camera_matrix[0][0] = 20
    # camera_matrix[1][1] = 20
    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

    for cam_name, keypoints in ref_kpts_dict.items():
        image_points = np.array(keypoints, dtype=np.float32)
        
        # Solve PnP for this camera
        success, rvec, tvec = cv2.solvePnP(object_3D_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            raise RuntimeError(f"PnP failed for {cam_name}")
        
        R, _ = cv2.Rodrigues(rvec)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = tvec.squeeze()
        extrinsics[cam_name] = extrinsic

        if debug:
            print(f"[Calibration] {cam_name} camera INTRINSIC matrix:\n{camera_matrix}")
            print(f"[Calibration] {cam_name} camera EXTRINSIC matrix:\n{extrinsic} w.r.t. world frame")

    return extrinsics


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    frames_path = cfg.preprocess_dir
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Load images as numpy arrays for batch processing
    images = {}
    for cam_dir in sorted(Path(frames_path).iterdir()):
        if cam_dir.is_dir():
            cam_name = cam_dir.name
            cam_images = [
                cv2.imread(str(p))
                for p in tqdm(
                    sorted(cam_dir.iterdir()),
                    desc=f"Loading images for {cam_name}",
                    total=len([f for f in cam_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
                )
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
            images[cam_name] = cam_images

    # # Display the first image for each camera
    # for cam_name, cam_images in images.items():
    #     if cam_images:
    #         show_image_matplotlib(cam_images[0], title=f"First image: {cam_name}", add_border=True)

    # Calibrate the view cameras using the table corners
    extrinsics = calibrate_cameras_with_ref_kpts(cfg, debug=verbose)









TRY ADDING ALSO OTHER REFERENCE POINTS, FOR EXAMPLE THE TIPS OF THE TABLE LEGS
(BETTER 3D ESTIMATION?)









    import pdb; pdb.set_trace()

    # Get bbx tracking result
    bbx_xys_dict = {}
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        for cam_name, cam_images in images.items():
            bbx_xyxy_list = tracker.get_one_track_image_batch(cam_images, stream_mode=True)
            # Replace None with zeros for missing detections to keep tensor shape consistent
            bbx_xyxy = torch.stack([
                x if x is not None else torch.zeros(4) for x in bbx_xyxy_list
            ]).float()  # (L, 4)
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
            # Save per-camera bbx results in the camera's subfolder
            cam_out_dir = Path(cfg.preprocess_dir) / cam_name
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, cam_out_dir / Path(paths.bbx).name)
            bbx_xys_dict[cam_name] = bbx_xys
        del tracker
    else:
        for cam_name in images.keys():
            cam_out_dir = Path(cfg.preprocess_dir) / cam_name
            bbx_xys = torch.load(cam_out_dir / Path(paths.bbx).name)["bbx_xys"]
            bbx_xys_dict[cam_name] = bbx_xys
            Log.info(f"[Preprocess] bbx (xyxy, xys) from {cam_out_dir / Path(paths.bbx).name}")
    if verbose:
        for cam_name, cam_images in images.items():
            Log.info(f"[Preprocess] Drawing bounding boxes on images for {cam_name}")
            cam_out_dir = Path(cfg.preprocess_dir) / cam_name
            bbx_xyxy = torch.load(cam_out_dir / Path(paths.bbx).name)["bbx_xyxy"]
            video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, cam_images)
            save_video(video_overlay, cam_out_dir / Path(cfg.paths.bbx_xyxy_video_overlay).name)

    # Get VitPose
    vitpose_dict = {}
    for cam_name, cam_images in images.items():
        frames_path_list = [str(p) for p in sorted((Path(frames_path) / cam_name).iterdir())
                            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        cam_out_dir = Path(cfg.preprocess_dir) / cam_name
        vitpose_path = cam_out_dir / Path(paths.vitpose).name
        if not vitpose_path.exists():
            vitpose_extractor = VitPoseExtractor()
            vitpose = vitpose_extractor.extract_frames(frames_path_list, bbx_xys_dict[cam_name])
            torch.save(vitpose, vitpose_path)
            del vitpose_extractor
        else:
            vitpose = torch.load(vitpose_path)
            Log.info(f"[Preprocess] vitpose from {vitpose_path}")
        vitpose_dict[cam_name] = vitpose
        if verbose:
            Log.info(f"[Preprocess] Drawing vitpose skeletons on images for {cam_name}")
            video_overlay = draw_coco17_skeleton_batch(cam_images, vitpose, 0.5)
            save_video(video_overlay, cam_out_dir / Path(paths.vitpose_video_overlay).name)

    # Get Vit features
    vit_features_dict = {}
    for cam_name, cam_images in images.items():
        frames_path_list = [str(p) for p in sorted((Path(frames_path) / cam_name).iterdir())
                            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        cam_out_dir = Path(cfg.preprocess_dir) / cam_name
        vit_features_path = cam_out_dir / Path(paths.vit_features).name
        if not vit_features_path.exists():
            extractor = Extractor()
            vit_features = extractor.extract_frames_features(frames_path_list, bbx_xys_dict[cam_name])
            torch.save(vit_features, vit_features_path)
            del extractor
        else:
            Log.info(f"[Preprocess] vit_features from {vit_features_path}")
            vit_features = torch.load(vit_features_path)
        vit_features_dict[cam_name] = vit_features

    ####### NOT TESTED YET ########
    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        for cam_name, cam_images in images.items():
            slam_path = Path(paths.slam).with_stem(f"{Path(paths.slam).stem}_{cam_name}")
            if not slam_path.exists():
                if not cfg.use_dpvo:
                    frames_path_list = [str(p) for p in sorted((Path(frames_path) / cam_name).iterdir())
                                        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
                    simple_vo = SimpleVOImages(frames_path_list, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                    vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                    torch.save(vo_results, slam_path)
            
            ### NOT ADAPTED TO PROCESSING OF BATCHES OF IMAGES YET ###
            else:  # DPVO
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
            ##########################################################
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")
    #############################

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg, video_mode=True):
    """
    Loads data for single or multi-camera setups.
    If multi-camera, returns a dict of data per camera.
    """
    paths = cfg.paths

    # Detect multi-camera setup by checking for subfolders in preprocess_dir
    cam_dirs = sorted([d for d in Path(cfg.preprocess_dir).iterdir() if d.is_dir()])
    is_multicam = len(cam_dirs) > 0

    if is_multicam:
        data_dict = {}
        for cam_dir in cam_dirs:
            cam_name = cam_dir.name
            frames_paths = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
            length = len(frames_paths)
            sample_img = cv2.imread(str(frames_paths[0]))
            height, width = sample_img.shape[:2]

            # All per-camera files are now in their own subfolder
            bbx_path = cam_dir / Path(paths.bbx).name
            vitpose_path = cam_dir / Path(paths.vitpose).name
            vit_features_path = cam_dir / Path(paths.vit_features).name
            slam_path = cam_dir / Path(paths.slam).name

            if cfg.static_cam:
                R_w2c = torch.eye(3).repeat(length, 1, 1)
            else:
                traj = torch.load(slam_path)
                if cfg.use_dpvo:  # DPVO
                    traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
                    R_w2c = quaternion_to_matrix(traj_quat).mT
                else:  # SimpleVO
                    R_w2c = torch.from_numpy(traj[:, :3, :3])
            if cfg.f_mm is not None:
                K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
            else:
                K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

            data = {
                "length": torch.tensor(length),
                "bbx_xys": torch.load(bbx_path)["bbx_xys"],
                "kp2d": torch.load(vitpose_path),
                "K_fullimg": K_fullimg,
                "cam_angvel": compute_cam_angvel(R_w2c),
                "f_imgseq": torch.load(vit_features_path),
            }
            data_dict[cam_name] = data
        return data_dict
    else:
        # Single camera or flat folder
        if video_mode:
            length, width, height = get_video_lwh(cfg.video_path)
        else:
            frames_paths = sorted([p for p in Path(cfg.preprocess_dir).iterdir()
                                   if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
            length = len(frames_paths)
            sample_img = cv2.imread(str(frames_paths[0]))
            height, width = sample_img.shape[:2]

        if cfg.static_cam:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
        else:
            traj = torch.load(cfg.paths.slam)
            if cfg.use_dpvo:  # DPVO
                traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
                R_w2c = quaternion_to_matrix(traj_quat).mT
            else:  # SimpleVO
                R_w2c = torch.from_numpy(traj[:, :3, :3])
        if cfg.f_mm is not None:
            K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
        else:
            K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

        data = {
            "length": torch.tensor(length),
            "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
            "kp2d": torch.load(paths.vitpose),
            "K_fullimg": K_fullimg,
            "cam_angvel": compute_cam_angvel(R_w2c),
            "f_imgseq": torch.load(paths.vit_features),
        }
        return data


def create_video_from_frames(frames_dir, video_path, fps=30):
    frames_paths = sorted([p for p in Path(frames_dir).iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if not frames_paths:
        raise ValueError(f"No frames found in {frames_dir}")
    # Read first frame to get size
    sample_img = cv2.imread(str(frames_paths[0]))
    height, width = sample_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for frame_path in frames_paths:
        img = cv2.imread(str(frame_path))
        video_writer.write(img)
    video_writer.release()
    return width, height


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    if not Path(cfg.video_path).exists():
        width, height = create_video_from_frames(cfg.preprocess_dir, cfg.video_path, fps=cfg.fps)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=cfg.fps, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=cfg.fps, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg, video_mode=False)  # False for image batch processing

    # ===== HMR4D & Rendering for Multi-Camera ===== #
    if isinstance(data, dict):  # Multi-camera setup
        for cam_name, cam_data in data.items():
            # HMR4D results path for this camera
            hmr4d_result_path = Path(paths.hmr4d_results).with_stem(f"{Path(paths.hmr4d_results).stem}_{cam_name}")
            if not hmr4d_result_path.exists():
                Log.info(f"[HMR4D] Predicting for {cam_name}")
                model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
                model.load_pretrained_model(cfg.ckpt_path)
                model = model.eval().cuda()
                tic = Log.sync_time()
                pred = model.predict(cam_data, static_cam=cfg.static_cam)
                pred = detach_to_cpu(pred)
                data_time = cam_data["length"] / 30
                Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
                torch.save(pred, hmr4d_result_path)

            # ===== Render for this camera ===== #
            incam_video_path = Path(paths.incam_video).with_stem(f"{Path(paths.incam_video).stem}_{cam_name}")
            global_video_path = Path(paths.global_video).with_stem(f"{Path(paths.global_video).stem}_{cam_name}")
            incam_global_horiz_video = Path(paths.incam_global_horiz_video).with_stem(f"{Path(paths.incam_global_horiz_video).stem}_{cam_name}")

            # Update cfg for this camera's outputs
            cfg_cam = cfg.copy()
            cfg_cam.preprocess_dir = str(Path(cfg.preprocess_dir) / cam_name)
            cfg_cam.paths.hmr4d_results = str(hmr4d_result_path)
            cfg_cam.paths.incam_video = str(incam_video_path)
            cfg_cam.paths.global_video = str(global_video_path)
            cfg_cam.paths.incam_global_horiz_video = str(incam_global_horiz_video)
            cfg_cam.video_path = str(Path(cfg.video_path).with_stem(f"{Path(cfg.video_path).stem}_{cam_name}"))

            render_incam(cfg_cam)
            render_global(cfg_cam)
            if not Path(cfg_cam.paths.incam_global_horiz_video).exists():
                Log.info(f"[Merge Videos] for {cam_name}")
                merge_videos_horizontal([cfg_cam.paths.incam_video, cfg_cam.paths.global_video], cfg_cam.paths.incam_global_horiz_video)
    else:
        # Single camera or flat folder
        if not Path(paths.hmr4d_results).exists():
            Log.info("[HMR4D] Predicting")
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
            tic = Log.sync_time()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            data_time = data["length"] / 30
            Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
            torch.save(pred, paths.hmr4d_results)

        # ===== Render ===== #
        render_incam(cfg)
        render_global(cfg)
        if not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
