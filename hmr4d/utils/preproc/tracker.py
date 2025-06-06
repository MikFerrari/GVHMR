from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt") # yolov8<n/s/m/l/x>.pt")

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history
    
    def track_image_batch(self, images, stream_mode=False):
        """
        Run YOLO detection on a batch of RGB images (list of numpy arrays or torch tensors).
        Returns a list of lists of dicts: [[{"id": id, "bbx_xyxy": bbx_xyxy}, ...], ...] for each image.
        """
        cfg = {
            "device": "cuda",
            "conf": 0.5,
            "classes": 0,  # human
            "verbose": False,
            "stream": stream_mode, # False for batch processing, True for frame-by-frame
        }
        results = self.yolo.predict(images, **cfg)
        batch_result_frames = []
        for result in tqdm(results, total=len(images), desc="YoloV8 BBox Tracking"):
            result_frame = []
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes
                if boxes.xyxy is not None and len(boxes.xyxy) > 0:
                    bbx_xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
                    track_ids = list(range(len(bbx_xyxy)))
                    result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            batch_result_frames.append(result_frame)
        return batch_result_frames  # one list per image

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track
    
    def get_one_track_image_batch(self, images, stream_mode=False):
        """
        Run YOLO detection on a batch of RGB images and return the largest detected human bounding box for each image.
        Returns a list of (4,) torch tensors with the bounding box coordinates (xyxy), or None if no detection for that image.
        """
        track_histories = self.track_image_batch(images, stream_mode=stream_mode)  # List of [ {id, bbx_xyxy}, ... ] per image
        results = []
        for frame in track_histories:
            if not frame:
                results.append(None)  # No detection for this image
                continue
            bbx_xyxys = np.array([det["bbx_xyxy"] for det in frame])  # (N, 4)
            areas = (bbx_xyxys[:, 2] - bbx_xyxys[:, 0]) * (bbx_xyxys[:, 3] - bbx_xyxys[:, 1])
            idx = np.argmax(areas)
            bbx_xyxy_one_track = torch.tensor(bbx_xyxys[idx])  # (4,)
            results.append(bbx_xyxy_one_track)
        return results  # List of torch tensors or None
