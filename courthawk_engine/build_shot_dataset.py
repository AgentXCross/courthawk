"""
This script is used to generate the dataset for shot classification.

CourtHawk is run on an input video to determine the frames where a shot
was hit. This file then crops out the player on the close side of the court in the frames:
[hitting frame - 2, hitting frame + 2] and saves the crop to
data/pose_estimation_data/images.

We then need to hand write the labels in shot_classification_metadata.csv.

Run with python3 -m courthawk_engine.build_shot_dataset
"""

import csv
from pathlib import Path

from .core import (
    Video,
    Point,
    BoundingBox,
    CourtSide,
)
from .court_keypoint_detector import CourtKeypointDetector
from .trackers import PlayerTracker, BallTracker

import cv2
import numpy as np
import torch


ENGINE_DIR = Path(__file__).resolve().parent


def build_dataset(input_video: Video) -> None:
    """
    Builds the shot classification dataset from an input video.

    Assumes the close-side player in input_video is right-handed. Only the
    close-side player's shots are cropped and saved. Shots hit by the
    far-side player are skipped entirely.

    Outputs the shot frames to output_path and numbers it to be the smallest possible
    based on the current images in output_path.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    output_path: Path = ENGINE_DIR / "data" / "pose_estimation_data" / "images"
    output_path.mkdir(parents = True, exist_ok = True)

    metadata_path: Path = ENGINE_DIR / "data" / "pose_estimation_data" / "shot_classification_metadata.csv"

    court_keypoints_model_path: Path = ENGINE_DIR / "models" / "keypoints_model.pt"
    player_tracker_model_path: Path = ENGINE_DIR / "models" / "yolov8x_player_tracker.pt"
    ball_tracker_model_path: Path = ENGINE_DIR / "models" / "yolo5_ball_detector.pt"

    player_tracker: PlayerTracker = PlayerTracker(model_path = player_tracker_model_path, device = device)
    ball_tracker: BallTracker = BallTracker(model_path = ball_tracker_model_path, device = device)
    court_keypoint_detector: CourtKeypointDetector = CourtKeypointDetector(model_path = court_keypoints_model_path)

    player_bbox_detections: list[dict[int, BoundingBox]] = player_tracker.detect_frames(input_video.frames)
    ball_bbox_detections: list[BoundingBox | None] = ball_tracker.detect_frames(input_video.frames)
    ball_bbox_detections: list[BoundingBox] = ball_tracker.interpolate_ball_positions(ball_bbox_detections)

    n_frames = input_video.num_frames
    sample_frames = [
        input_video.frames[i]
        for i in [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
    ]
    court_keypoints: list[Point] = court_keypoint_detector.predict_average(sample_frames)

    player_bbox_detections, player_ids = player_tracker.choose_and_filter_players(court_keypoints, player_bbox_detections)

    player_sides = player_tracker.determine_court_sides(court_keypoints, player_bbox_detections, player_ids)
    close_side_player_id = next(pid for pid, side in player_sides.items() if side == CourtSide.CLOSE)

    ball_shot_frames: list[int] = ball_tracker.get_ball_shot_frames(ball_bbox_detections, player_bbox_detections)

    # Continue numbering from whatever's already in output_path, rather than starting over at 1
    existing_numbers = [int(p.stem) for p in output_path.glob("*.png") if p.stem.isdigit()]
    next_number = max(existing_numbers, default = 0) + 1

    with open(metadata_path, "a", newline = "") as metadata_file:
        metadata_writer = csv.writer(metadata_file)

        for frame_num in ball_shot_frames:
            ball_center = ball_bbox_detections[frame_num].center
            frame_players = player_bbox_detections[frame_num]

            # "who hit the ball" logic is whichever player's bbox center is closest to the ball on the shot frame
            hitting_player_id = min(
                frame_players.keys(),
                key = lambda pid: (
                    (frame_players[pid].center.x - ball_center.x) ** 2 +
                    (frame_players[pid].center.y - ball_center.y) ** 2
                )
            )

            if hitting_player_id != close_side_player_id:
                continue

            for f in range(max(0, frame_num - 2), min(n_frames, frame_num + 3)):
                player_bbox = player_bbox_detections[f].get(close_side_player_id)
                if player_bbox is None:
                    continue

                x1, y1, x2, y2 = int(player_bbox.tl.x), int(player_bbox.tl.y), int(player_bbox.br.x), int(player_bbox.br.y)
                crop: np.ndarray = input_video.frames[f][max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                image_name = f"{next_number:04d}.png"
                cv2.imwrite(str(output_path / image_name), crop)
                metadata_writer.writerow([f"images/{image_name}", ""])
                next_number += 1

    print(f"Saved crops up to {output_path / f'{next_number - 1:04d}.png'}")


if __name__ == "__main__":
    input_video_path = ENGINE_DIR / "data" / "input_videos" / "sinner_deminaur_3.mp4"
    build_dataset(Video.read_video(input_video_path))
