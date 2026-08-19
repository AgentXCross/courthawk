"""
This script is primarily for testing. It runs an image through the entire 
pipeline and produces the output video. This file should not be accessed
by the frontend or backend folders.

Run with python3 -m courthawk_engine.main
"""

from pathlib import Path

from .core import (
    Video,
    Point,
    BoundingBox,
    CourtSide,
    Handedness
)
from .court_keypoint_detector import CourtKeypointDetector
from .minicourt import MiniCourt
from .pose_estimation import PoseEstimator, ShotClassifier, ShotType
from .renderer import Renderer
from .trackers import PlayerTracker, BallTracker

import cv2
import numpy as np
import torch

# Anchored on this file's own location
ENGINE_DIR = Path(__file__).resolve().parent


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    input_video_path: Path = ENGINE_DIR / "data" / "input_videos" / "sinner_zverev.mp4"
    court_keypoints_model_path: Path = ENGINE_DIR / "models" / "keypoints_model.pt"
    player_tracker_model_path: Path = ENGINE_DIR / "models" / "yolov8x_player_tracker.pt"
    ball_tracker_model_path: Path = ENGINE_DIR / "models" / "yolo5_ball_detector.pt"
    pose_estimator_model_path: Path = ENGINE_DIR / "models" / "pose_landmarker.task"
    shot_classifier_model_path: Path = ENGINE_DIR / "models" / "shot_classifier_random_forest.pkl"
    output_video_path: Path = ENGINE_DIR / "data" / "output_videos" / "sinner_zverev_output.mp4"

    PLAYER_SPEED_STRIDE = 5  # frames between player speed samples

    COURT_SIDE_HANDEDNESS: dict[CourtSide, Handedness] = {
        CourtSide.CLOSE: Handedness.RIGHT,
        CourtSide.FAR: Handedness.RIGHT
    }

    video = Video.read_video(input_video_path) 

    player_tracker = PlayerTracker(model_path = player_tracker_model_path, device = device)
    ball_tracker = BallTracker(model_path = ball_tracker_model_path, device = device)

    player_bbox_detections: list[dict[int, BoundingBox]] = player_tracker.detect_frames(
        video.frames,
        read_from_stub = False,
        stub_path = None
    )
    ball_bbox_detections: list[BoundingBox | None] = ball_tracker.detect_frames(
        video.frames,
        read_from_stub = False,
        stub_path = None
    )
    
    ball_bbox_detections: list[BoundingBox] = ball_tracker.interpolate_ball_positions(ball_bbox_detections)

    court_line_detector = CourtKeypointDetector(court_keypoints_model_path)

    sample_frames: list[np.ndarray] = [
        video.frames[i] 
        for i in 
        [0, video.num_frames // 3, 2 * video.num_frames // 3, video.num_frames - 1]
    ]
    court_keypoints: list[Point] = court_line_detector.predict_average(sample_frames)

    player_bbox_detections: list[dict[CourtSide, BoundingBox]] = player_tracker.choose_and_filter_players(court_keypoints, player_bbox_detections)
    player_bbox_detections = player_tracker.interpolate_player_positions(player_bbox_detections)

    # Mirror the crop before pose estimation when far side XOR left handed
    should_mirror: dict[CourtSide, bool] = {
        side: (side == CourtSide.FAR) ^ (COURT_SIDE_HANDEDNESS[side] == Handedness.LEFT)
        for side in CourtSide
    }
    print(f"Should mirror: {should_mirror}")

    mini_court = MiniCourt(video.frames[0])

    ball_shot_frames: list[int] = ball_tracker.get_ball_shot_frames(ball_bbox_detections, player_bbox_detections)

    pose_estimator = PoseEstimator(model_path = pose_estimator_model_path)
    shot_classifier = ShotClassifier(model_path = shot_classifier_model_path)
    
    shot_types, hitting_player_ids = pose_estimator.classify_shots(video.frames, ball_shot_frames, player_bbox_detections, ball_bbox_detections, shot_classifier, should_mirror)

    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bbox_to_mini_court_coords(player_bbox_detections, ball_bbox_detections, court_keypoints)
   
    player_shots_data: list[dict[str, int | float]] = mini_court.get_shot_stats(ball_shot_frames, player_mini_court_detections, ball_mini_court_detections, video.fps)

    player_speed_stats: list[dict[str, int | float]] = mini_court.get_player_speed_stats(player_mini_court_detections, video.fps, PLAYER_SPEED_STRIDE)

    # Draw Outputs
    renderer = Renderer()
    output_video_frames: list[np.ndarray] = renderer.render(
        video.frames,
        court_keypoints,
        player_bbox_detections,
        ball_bbox_detections,
        player_mini_court_detections,
        ball_mini_court_detections,
        ball_shot_frames,
        shot_types,
        hitting_player_ids,
        list(CourtSide),
        mini_court,
        player_shots_data,
        player_speed_stats
    )

    for i, frame in enumerate(output_video_frames): # draw frame number
        cv2.putText(frame, f"Frame #{i + 1}", (50, 170), cv2.FONT_HERSHEY_TRIPLEX, 2, renderer.color_1, 5)
        
    output_video = Video(
        frames = output_video_frames,
        fps = video.fps,
        num_frames = len(output_video_frames)
    )
    output_video.save_video(output_video_path)


if __name__ == "__main__":
    main()