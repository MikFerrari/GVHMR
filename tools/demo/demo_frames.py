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

from camera_calibration_utils import *
import matplotlib.pyplot as plt


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

    # Copy calibration points to preprocess dir
    Log.info(f"[Copy Calibration Points] {frames_dir} -> {cfg.preprocess_dir}")
    copy_calibration_csv_to_preprocess(frames_dir, cfg.preprocess_dir, cfg.calibration_filename)
    copy_calibration_csv_to_preprocess(frames_dir, cfg.preprocess_dir, cfg.calibration_valid_filename)
    copy_calibration_csv_to_preprocess(frames_dir, cfg.preprocess_dir, cfg.reference_object_kpts_filename)
    copy_calibration_csv_to_preprocess(frames_dir, cfg.preprocess_dir, cfg.bundle_adjustment_filename)

    return cfg


def copy_calibration_csv_to_preprocess(frames_dir, preprocess_dir, csv_filename):
    """
    Copy a calibration CSV file from frames_dir to preprocess_dir if it exists and not already copied.
    """
    calibration_csv_path = Path(frames_dir) / csv_filename
    out_calibration_csv_path = Path(preprocess_dir) / csv_filename
    if calibration_csv_path.exists():
        if not out_calibration_csv_path.exists():
            Log.info(f"Copying calibration points from {calibration_csv_path} to {out_calibration_csv_path}")
            df = pd.read_csv(calibration_csv_path)
            df.to_csv(out_calibration_csv_path, index=False)
    else:
        Log.error(f"Calibration CSV file not found: {calibration_csv_path}. ")


@torch.no_grad()
def calibrate_cameras(cfg):
    # Check if camera extrinsics already exist
    if Path(cfg.paths.extrinsics).exists() and Path(cfg.paths.intrinsics).exists():
        Log.info(f"[Calibration] Camera extrinsics already exist at {cfg.paths.extrinsics}.")
        Log.info(f"[Calibration] Camera extrinsics already exist at {cfg.paths.intrinsics}.")
        Log.info(f"[Calibration] Skipping calibration step.")
        return
    
    Log.info(f"[Calibration] Start!")
    verbose = cfg.verbose

    # Load reference keypoints (scene peculiar points) and extract unique values across cameras
    ref_kpts_2D_dict, ref_kpts_3D_dict = load_calibration_points_from_csv(
        cfg.preprocess_dir,
        csv_filename=cfg.reference_object_kpts_filename,
    )
    ref_3D_kpts = np.vstack(list(ref_kpts_3D_dict.values()))
    _, ref_3D_kpts_idx = np.unique(ref_3D_kpts, axis=0, return_index=True)
    ref_3D_kpts = ref_3D_kpts[sorted(ref_3D_kpts_idx)]

    # Load calibration points (train and test)
    calib_kpts_2D_dict, calib_kpts_3D_dict = load_calibration_points_from_csv(
        cfg.preprocess_dir,
        csv_filename=cfg.calibration_filename,
    )
    calib_valid_kpts_2D_dict, calib_valid_kpts_3D_dict = load_calibration_points_from_csv(
        cfg.preprocess_dir,
        csv_filename=cfg.calibration_valid_filename,
    )

    if cfg.augment_calib_with_ref_kpts:
        Log.info(f"[Calibration] Augmenting calibration keypoints with reference keypoints.")
        # Concatenate reference keypoints with calibration keypoints
        for cam in calib_kpts_2D_dict:
            if cam in ref_kpts_2D_dict:
                calib_kpts_2D_dict[cam] = np.concatenate([calib_kpts_2D_dict[cam], ref_kpts_2D_dict[cam]], axis=0)
                calib_kpts_3D_dict[cam] = np.concatenate([calib_kpts_3D_dict[cam], ref_kpts_3D_dict[cam]], axis=0)
            else:
                Log.warning(f"[Calibration] Reference keypoints for camera {cam} not found. Skipping augmentation.")

    # Load calibration points for bundle adjustment
    bundleAdj_kpts_2D_dict, bundleAdj_kpts_3D_dict = load_calibration_points_from_csv(
        cfg.preprocess_dir,
        csv_filename=cfg.bundle_adjustment_filename,
    )
    bundleAdj_3D_kpts = np.vstack(list(bundleAdj_kpts_3D_dict.values()))
    _, bundleAdj_3D_kpts_idx = np.unique(bundleAdj_3D_kpts, axis=0, return_index=True)
    bundleAdj_3D_kpts = bundleAdj_3D_kpts[sorted(bundleAdj_3D_kpts_idx)]

    # We'll use the first camera's first image to get width/height
    first_cam = list(ref_kpts_2D_dict.keys())[0]
    sample_img_path = sorted((Path(cfg.preprocess_dir) / first_cam).iterdir())[0]
    sample_img = cv2.imread(str(sample_img_path))
    height, width = sample_img.shape[:2]

    # Camera intrinsics: estimate from image size or use provided
    # !!! We are assuming all cameras have the same intrinsic parameters !!!
    if cfg.f_mm is not None:
        _, _, camera_matrix = create_camera_sensor(width, height, cfg.f_mm)
        camera_matrix = camera_matrix.cpu().numpy()
    else:
        camera_matrix = estimate_K(width, height).cpu().numpy()

    dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

    # Calibrate the view cameras using calibration keypoints
    extrinsics = calibrate_cameras_with_ref_kpts(
        calib_kpts_2D_dict, 
        calib_kpts_3D_dict,
        camera_matrix,
        dist_coeffs,
        debug=verbose
    )

    if verbose:
        # Plot the initial calibration scene
        Log.info(f"[Calibration] Plotting initial calibration scene and validating calibration results.")
        validate_and_plot_calibration_scene(
            extrinsics,
            camera_matrix,
            ref_3D_kpts,
            calib_kpts_3D_dict,
            calib_kpts_2D_dict,
            ref_kpts_3D_dict, #calib_valid_kpts_3D_dict,
            ref_kpts_2D_dict, #calib_valid_kpts_2D_dict,
            connect_indices=[0,1,2,3],  # Connect the corners of the table
        )

    # Perform bundle adjustment to refine the camera extrinsics
    if cfg.perform_bundle_adjustment:
        Log.info(f"[Calibration] Improving camera extrinsics with bundle adjustment.")
        extrinsics, result = bundle_adjustment(
            bundleAdj_3D_kpts,
            bundleAdj_kpts_2D_dict,
            camera_matrix,
            extrinsics,
            return_optimized_points=False,
            verbose=verbose,
        )
        if verbose:
            Log.info(f"[Calibration] Bundle adjustment optimization result: {result}")

        if verbose:
            # Plot the optimized calibration scene
            Log.info(f"[Calibration] Plotting optimized calibration scene and validating calibration results.")
            validate_and_plot_calibration_scene(
                extrinsics,
                camera_matrix,
                ref_3D_kpts,
                calib_kpts_3D_dict,
                calib_kpts_2D_dict,
                calib_valid_kpts_3D_dict,
                calib_valid_kpts_2D_dict,
                connect_indices=[0,1,2,3],  # Connect the corners of the table
            )
    
    # Save the extrinsics and the extrinsics    
    torch.save(extrinsics, Path(cfg.paths.extrinsics))
    torch.save(camera_matrix, Path(cfg.paths.intrinsics))
    Log.info(f"[Calibration] Camera intrinsics saved to {cfg.paths.intrinsics}")
    Log.info(f"[Calibration] Camera extrinsics saved to {cfg.paths.extrinsics}")


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
    frames_path_list = {}
    start_idx = getattr(cfg, "start_idx_images_to_load", 0)
    end_idx = getattr(cfg, "end_idx_images_to_load", None)  # None means load until the end
    for cam_dir in sorted(Path(frames_path).iterdir()):
        if cam_dir.is_dir():
            cam_name = cam_dir.name
            frame_paths = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
            frame_paths = frame_paths[start_idx:end_idx]
            cam_images = [
                cv2.imread(str(p))
                for p in tqdm(frame_paths, desc=f"Loading images for {cam_name}")
            ]
            images[cam_name] = cam_images
            frames_path_list[cam_name] = frame_paths

    # Get bbx tracking result
    bbx_xys_dict = {}
    for cam_name, cam_images in images.items():
        cam_out_dir = Path(cfg.preprocess_dir) / cam_name
        bbx_path = cam_out_dir / Path(paths.bbx).name
        if not bbx_path.exists():
            tracker = Tracker()
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
            cam_out_dir = Path(cfg.preprocess_dir) / cam_name
            bbx_xys = torch.load(cam_out_dir / Path(paths.bbx).name)["bbx_xys"]
            bbx_xys_dict[cam_name] = bbx_xys
            Log.info(f"[Preprocess] bbx (xyxy, xys) from {cam_out_dir / Path(paths.bbx).name}")
        if verbose:
            Log.info(f"[Preprocess] Drawing bounding boxes on images for {cam_name}")
            cam_out_dir = Path(cfg.preprocess_dir) / cam_name
            bbx_xyxy = torch.load(cam_out_dir / Path(paths.bbx).name)["bbx_xyxy"]
            video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, cam_images)
            save_video(video_overlay, cam_out_dir / Path(cfg.paths.bbx_xyxy_video_overlay).name)

    # Get VitPose
    vitpose_dict = {}
    for cam_name, cam_images in images.items():
        cam_out_dir = Path(cfg.preprocess_dir) / cam_name
        vitpose_path = cam_out_dir / Path(paths.vitpose).name
        if not vitpose_path.exists():
            vitpose_extractor = VitPoseExtractor()
            vitpose = vitpose_extractor.extract_frames(frames_path_list[cam_name], bbx_xys_dict[cam_name])
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
        cam_out_dir = Path(cfg.preprocess_dir) / cam_name
        vit_features_path = cam_out_dir / Path(paths.vit_features).name
        if not vit_features_path.exists():
            extractor = Extractor()
            vit_features = extractor.extract_frames_features(frames_path_list[cam_name], bbx_xys_dict[cam_name])
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
    start_idx = getattr(cfg, "start_idx_images_to_load", 0)
    end_idx = getattr(cfg, "end_idx_images_to_load", None)  # None means load until the end

    # Detect multi-camera setup by checking for subfolders in preprocess_dir
    cam_dirs = sorted([d for d in Path(cfg.preprocess_dir).iterdir() if d.is_dir()])
    is_multicam = len(cam_dirs) > 0

    if is_multicam:
        data_dict = {}
        for cam_dir in cam_dirs:
            cam_name = cam_dir.name
            frames_paths = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
            frames_paths = frames_paths[start_idx:end_idx]
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

            # Load the cameras extrinsics and use them to reference all cameras to the world frame
            extrinsics = torch.load(cfg.paths.extrinsics)
            T_w2c = torch.from_numpy(extrinsics[cam_name]).float() if cam_name in extrinsics else None

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
                "cam_extrinsics": T_w2c,
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


def create_video_from_frames(frames_dir, video_path, start_idx, end_idx, fps=30):
    frames_paths = sorted([p for p in Path(frames_dir).iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    frames_paths = frames_paths[start_idx:end_idx]
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
    if not Path(video_path).exists():
        Log.info(f"[Render Incam] Creating video from frames in {cfg.preprocess_dir}")
        start_idx = getattr(cfg, "start_idx_images_to_load", 0)
        end_idx = getattr(cfg, "end_idx_images_to_load", None)  # None means load until the end
        width, height = create_video_from_frames(
            cfg.preprocess_dir,
            video_path,
            start_idx,
            end_idx,
            fps=cfg.fps
        )
    else:
        Log.info(f"[Render Incam] Using existing video {video_path}")
        _, width, height = get_video_lwh(video_path)
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

        # bbx
        bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

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


def render_global_multicam(cfg, cam_name):
    """
    Render the predicted vertices in the world (global) frame for a given camera,
    using the inverse of the extrinsics to transform from camera-global to world-global.
    Visualizes the coordinate axes at the world origin.
    """
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

    # Get the predicted SMPLX parameters in the CAMERA-GLOBAL frame
    pred_incam = pred["smpl_params_incam"]

    # root_transl_incam = pred_incam["transl"]        # (batch_size, 3)  ([x,y,z] in camera coordinates)
    # root_orient_incam = pred_incam["global_orient"] # (batch_size, 3)  (Rodrigues vector for rotation)
    # body_pose_incam = pred_incam["body_pose"]       # (batch_size, 63) (Rodrigues vectors for 21 body joints)
    # betas_incam = pred_incam["betas"]               # (batch_size, 10) (shape parameters)

    # smpl
    smplx_out = smplx(**to_cuda(pred_incam))
    verts_incam = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])  # (L, V, 3)
    joints_incam = einsum(J_regressor, verts_incam, "j v, l v i -> l j i")  # (L, J, 3)
    joints_incam = smplx_out.joints  # (L, J, 3) - this is the same as above
    assert np.allclose(joints_incam.cpu().numpy(), joints_incam.cpu().numpy()), "Joints mismatch!"

    # --- Transform pred_verts and pred_joints from camera-global to WORLD-GLOBAL frame using extrinsics ---
    extrinsics = torch.load(cfg.paths.extrinsics)
    if cam_name not in extrinsics:
        raise ValueError(f"Camera {cam_name} not found in extrinsics. Available cameras: {list(extrinsics.keys())}")
    T_w2c = extrinsics[cam_name]  # (4, 4) numpy array
    T_c2w = np.linalg.inv(T_w2c)  # (4, 4) numpy array

    # Ensure T_c2w is a torch tensor
    if not torch.is_tensor(T_c2w):
        T_c2w = torch.from_numpy(T_c2w).to(verts_incam.device).float()

    # Convert verts and joints to torch tensors for transformation
    pred_verts_torch = verts_incam.detach()  # (L, V, 3)
    pred_joints_torch = joints_incam.detach()  # (L, J, 3)

    # Add a homogeneous coordinate for transformation
    verts_incam_h = torch.cat(
        [
            pred_verts_torch,
            torch.ones(pred_verts_torch.shape[0], pred_verts_torch.shape[1], 1,
                       device=pred_verts_torch.device, dtype=pred_verts_torch.dtype)
        ],
        dim=-1
    )  # (L, V, 4)
    joints_incam_h = torch.cat(
        [
            pred_joints_torch,
            torch.ones(pred_joints_torch.shape[0], pred_joints_torch.shape[1], 1,
                       device=pred_joints_torch.device, dtype=pred_joints_torch.dtype)
        ],
        dim=-1
    )  # (L, J, 4)

    # Transform vertices and joints to world-global frame
    verts_world_h = np.einsum('ij, l v j -> l v i', T_c2w.cpu(), verts_incam_h.cpu())  # (L, V, 4)
    verts_world = verts_world_h[:, :, :3]  # (L, V, 3)
    joints_world_h = np.einsum('ij, l p j -> l p i', T_c2w.cpu(), joints_incam_h.cpu())  # (L, J, 4)
    joints_world = joints_world_h[:, :, :3]  # (L, J, 3)

    # Convert back to torch tensors
    verts_glob = torch.from_numpy(verts_world).to(verts_incam.device).float()  # (L, V, 3)
    joints_glob = torch.from_numpy(joints_world).to(joints_incam.device).float()  # (L, J, 3)
    # ------------------------------------------------------------------------------------------------------

    if cfg.verbose:
        Log.info(f"[Render Global] Camera {cam_name} - Predicted vertices in world-global frame: {verts_glob.shape}")
        Log.info(f"[Render Global] Camera {cam_name} - Predicted joints in world-global frame: {joints_glob.shape}")

        # --- Plot joints_glob in 3D using matplotlib ---
        def plot_vertices_3d(joints, frame_idx=0, title="3D Joints"):  # joints: (L, J, 3)
            js = joints[frame_idx].cpu().numpy() if hasattr(joints, 'cpu') else joints[frame_idx]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(js[:, 0], js[:, 1], js[:, 2], c='r', marker='o')
            # for i, (x, y, z) in enumerate(js):
            #     ax.text(x, y, z, str(i), fontsize=8)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
            plt.draw()

        global_plot_idx = 100  # Change to desired frame index
        plot_vertices_3d(verts_incam, frame_idx=global_plot_idx, title=f"[Cam] 3D Joints (frame {global_plot_idx})")
        plot_vertices_3d(verts_glob, frame_idx=global_plot_idx, title=f"[World] 3D Joints (frame {global_plot_idx})")
        plt.show()

    # -- rendering code -- #
    video_path = cfg.video_path
    _, width, height = get_video_lwh(video_path)
    K = torch.load(cfg.paths.intrinsics)
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)

    # choose camera view to render
    R = torch.from_numpy(T_w2c[:3, :3]).float().cuda()  # (3, 3)
    T = torch.from_numpy(T_w2c[:3, 3]).float().cuda()  # (3,)
    renderer.create_camera(R, T)  # Set the camera pose

    # Use as img the current frame from the video
    writer = get_writer(global_video_path, fps=cfg.fps, crf=CRF)
    video_reader = get_video_reader(video_path)

    for i, img in tqdm(enumerate(video_reader), desc=f"Rendering Global Multicam"):
        # Render the mesh for the current frame
        img = renderer.render_mesh(verts_glob[i], background=img, colors=[0.8, 0.6, 0.2])

        # Draw the table rectangle
        img = render_table_rectangle(K, T_w2c, img)

        # Draw world coordinate axes at the origin (length=0.5m)
        img = draw_axes_on_image(img, K, T_w2c, length=0.5)

        writer.write_frame(img)

    writer.close()


def render_table_rectangle(K, T_w2c, img, color=(0, 255, 255), thickness=2):
    """
    Draws the table rectangle defined by the first 4 points in cfg.reference_object_keypoints_3D
    onto the given image using the provided camera intrinsics and extrinsics.
    """
    # Ensure K is a numpy array
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy().astype(np.float64)

    # Load reference keypoints (scene peculiar points) and extract unique values across cameras
    _, ref_kpts_3D_dict = load_calibration_points_from_csv(
        cfg.preprocess_dir,
        csv_filename=cfg.reference_object_kpts_filename,
    )
    ref_3D_kpts = np.vstack(list(ref_kpts_3D_dict.values()))
    _, ref_3D_kpts_idx = np.unique(ref_3D_kpts, axis=0, return_index=True)
    ref_3D_kpts = ref_3D_kpts[sorted(ref_3D_kpts_idx)]

    # Get the first 4 table corners (Nx3)
    pts_w = np.array(ref_3D_kpts[:4], dtype=np.float32) # (4, 3)
    
    # Extract rotation matrix (3x3) and translation vector (3,)
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]

    # Convert rotation matrix to Rodrigues vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    # Project to image
    pts_img, _ = cv2.projectPoints(pts_w, rvec, tvec, K, distCoeffs=np.zeros(4))
    pts_img = pts_img.squeeze().astype(int)

    # Draw rectangle (connect corners)
    for i in range(4):
        pt1 = tuple(pts_img[i])
        pt2 = tuple(pts_img[(i + 1) % 4])
        img = cv2.line(img, pt1, pt2, color, thickness)

    return img


def draw_axes_on_image(img, K, T_w2c, length=0.5):
    """
    Draws XYZ axes at the world origin on the image.
    img: (H, W, 3) numpy array
    K: (3, 3) camera intrinsics
    T_w2c: (4, 4) world-to-camera extrinsic (identity for world origin)
    length: axis length in meters
    """
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy().astype(np.float64)

    # World origin and axes endpoints in world frame
    origin = np.array([0, 0, 0])
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])
    pts_w = np.stack([origin, x_axis, y_axis, z_axis], axis=0).T  # (4, 3)
    
    # Extract rotation matrix (3x3) and translation vector (3,)
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]

    # Convert rotation matrix to Rodrigues vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Project to image
    pts_img, _ = cv2.projectPoints(pts_w, rvec, tvec, K, distCoeffs=np.zeros(4))
    pts_img = pts_img.squeeze().astype(int)
    
    # Draw axes
    img = cv2.line(img, tuple(pts_img[0]), tuple(pts_img[1]), (255, 0, 0), 3)  # X - Red
    img = cv2.line(img, tuple(pts_img[0]), tuple(pts_img[2]), (0, 255, 0), 3)  # Y - Green
    img = cv2.line(img, tuple(pts_img[0]), tuple(pts_img[3]), (0, 0, 255), 3)  # Z - Blue
    
    return img


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Camera calibration ===== #
    calibrate_cameras(cfg)

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
                data_time = cam_data["length"] / cfg.fps # 30
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
            render_global_multicam(cfg_cam, cam_name)
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
