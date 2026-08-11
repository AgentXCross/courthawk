"""
Entry point to the CourtHawk engine.

This module exposes the public API used by the backend. The backend passes
an input video to analyze_point(), which executes the full pipeline and returns 
the analysis results.
"""

import cv2
import numpy as np

import uuid
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from trackers import PlayerTracker, BallTracker
from court_keypoint_detector import CourtKeypointDetector
from minicourt import MiniCourt
from pose_estimation import PoseEstimator, ShotClassifier, ShotType
from renderer import Renderer
from core import (
    Video,
    Point,
    BoundingBox,
    constants,
    euclidean_distance
)


MODELS_DIR = Path("models")
OUTPUT_DIR = Path("data/output_videos")
PLAYER_SPEED_STRIDE = 20  # frames between player speed samples


@dataclass
class Shot:
    frame: int
    player_id: int
    shot_type: ShotType
    ball_speed_kmh: float


@dataclass
class PlayerSpeedSample:
    frame: int
    speeds_kmh: dict[int, float] # player_id -> speed at this frame


@dataclass
class PointAnalysis:
    """
    PointAnalysis stores all the data that the frontend will display.
    The frontend will show the annotated video in the center with only the bboxes and court keypoints.
    The minicourt will be placed on the left side of the screen and the statistics on the right side.
    """
    annotated_video_path: Path # Video with only bboxes and court keypoints overlayed
    fps: float

    real_court_keypoints: list[Point]
    player_ids: list[int]
    player_court_foot_positions: list[dict[int, Point]]
    ball_court_positions: list[Point]

    shots: list[Shot]
    player_speeds: list[PlayerSpeedSample]


# Cached models

@cache
def _player_tracker() -> PlayerTracker:
    return PlayerTracker(model_path = MODELS_DIR / "yolov8x_player_tracker.pt")

@cache
def _ball_tracker() -> BallTracker:
    return BallTracker(model_path = MODELS_DIR / "yolo5_ball_detector.pt")

@cache
def _court_keypoint_detector() -> CourtKeypointDetector:
    return CourtKeypointDetector(model_path = MODELS_DIR / "keypoints_model.pt")

@cache
def _pose_estimator() -> PoseEstimator:
    return PoseEstimator(model_path = MODELS_DIR / "pose_landmarker.task")

@cache
def _shot_classifier() -> ShotClassifier:
    return ShotClassifier(model_path = MODELS_DIR / "shot_classifier.pkl")

@cache
def _renderer() -> Renderer:
    return Renderer()


# Real-court (meters) geometry, replaces MiniCourt

def _real_court_keypoints() -> list[Point]:
    """
    The 14 court keypoints in real-world meters, (0, 0) is the top left corner.
    """
    width = constants.DOUBLE_LINE_WIDTH
    length = constants.BASELINE_TO_NET * 2
    alley = constants.DOUBLE_ALLEY_WIDTH
    service = constants.BASELINE_TO_SERVICE

    return [
        Point(0, 0),
        Point(width, 0),
        Point(0, length),
        Point(width, length),
        Point(alley, 0),
        Point(alley, length),
        Point(width - alley, 0),
        Point(width - alley, length),
        Point(alley, service),
        Point(width - alley, service),
        Point(alley, length - service),
        Point(width - alley, length - service),
        Point(width / 2, service),
        Point(width / 2, length - service)
    ]


def _get_homography(source_keypoints: list[Point], destination_keypoints: list[Point]) -> np.ndarray:
    source = np.array([[kp.x, kp.y] for kp in source_keypoints], dtype = np.float32)
    destination = np.array([[kp.x, kp.y] for kp in destination_keypoints], dtype = np.float32)
    H, _ = cv2.findHomography(source, destination)
    return H


def _transform_point(point: Point, H: np.ndarray) -> Point:
    np_point = np.array([[[point.x, point.y]]], dtype = np.float32)
    transformed = cv2.perspectiveTransform(np_point, H)
    return Point(transformed[0][0][0], transformed[0][0][1])


# Main entry point
    
def analyze_point(video: Video) -> PointAnalysis:
    player_tracker = _player_tracker()
    ball_tracker = _ball_tracker()

    player_bbox_detections: list[dict[int, BoundingBox]] = player_tracker.detect_frames(video.frames)
    ball_bbox_detections: list[BoundingBox] = ball_tracker.detect_frames(video.frames)

    ball_bbox_detections = ball_tracker.interpolate_ball_positions(ball_bbox_detections)

    n_frames = video.num_frames
    sample_frames = [
        video.frames[i] 
        for i in 
        [0, n_frames // 3, 2 * n_frames // 3, n_frames - 1]
    ]
    detected_court_keypoints = _court_keypoint_detector().predict_average(sample_frames)

    player_bbox_detections, player_ids = player_tracker.choose_and_filter_players(
        detected_court_keypoints, 
        player_bbox_detections
    )

    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_bbox_detections, player_bbox_detections)

    shot_types, hitting_player_ids = _pose_estimator().classify_shots(
        video.frames, 
        ball_shot_frames, 
        player_bbox_detections, 
        ball_bbox_detections, 
        _shot_classifier()
    )

    # Homography: Video-pixel court keypoints -> Real-life court in meters
    real_court_keypoints = _real_court_keypoints()
    H = _get_homography(detected_court_keypoints, real_court_keypoints)

    player_court_foot_positions: list[dict[int, Point]] = []
    ball_court_positions: list[Point] = []

    for frame_num, player_bboxes_this_frame in enumerate(player_bbox_detections):
        player_court_foot_positions.append({
            player_id: _transform_point(bbox.foot, H)
            for player_id, bbox in player_bboxes_this_frame.items()
        })

        ball_court_positions.append(
            _transform_point(ball_bbox_detections[frame_num].center, H)
        )

    # Shot stats: Distances are already in meters
    shots: list[Shot] = []
    for i in range(len(ball_shot_frames)):
        start_frame = ball_shot_frames[i]

        if i < len(ball_shot_frames) - 1:
            end_frame = min(start_frame + 20, ball_shot_frames[i + 1])
        else:
            end_frame = min(start_frame + 20, len(video.frames) - 1)

        time_seconds = (end_frame - start_frame) / video.fps

        distance_meters = euclidean_distance(
            ball_court_positions[start_frame],
            ball_court_positions[end_frame]
        )
        ball_speed_kmh = (distance_meters / time_seconds) * 3.6

        shots.append(Shot(
            frame = start_frame,
            player_id = hitting_player_ids[i],
            shot_type = shot_types[i],
            ball_speed_kmh = ball_speed_kmh
        ))

    # Player speed stats
    player_speeds: list[PlayerSpeedSample] = []
    for frame_num in range(PLAYER_SPEED_STRIDE, len(player_court_foot_positions), PLAYER_SPEED_STRIDE):
        speeds_kmh: dict[int, float] = {}
        time_seconds = PLAYER_SPEED_STRIDE / video.fps

        for player_id in player_ids:
            prev = player_court_foot_positions[frame_num - PLAYER_SPEED_STRIDE].get(player_id)
            curr = player_court_foot_positions[frame_num].get(player_id)

            if prev is not None and curr is not None:
                distance_meters = euclidean_distance(prev, curr)
                speeds_kmh[player_id] = (distance_meters / time_seconds) * 3.6
            else:
                speeds_kmh[player_id] = 0.0

        player_speeds.append(PlayerSpeedSample(frame_num, speeds_kmh))

    # Annotated video: bboxes + court keypoints only
    renderer = _renderer()
    annotated_frames = renderer.draw_player_bboxes(video.frames, player_bbox_detections)
    annotated_frames = renderer.draw_ball_bboxes(annotated_frames, ball_bbox_detections)
    annotated_frames = renderer.draw_keypoints_on_video(annotated_frames, detected_court_keypoints)

    OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    annotated_video_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.avi"

    output_video = Video(
        frames = annotated_frames,
        fps = video.fps,
        num_frames = len(annotated_frames)
    )
    output_video.save_video(annotated_video_path)

    return PointAnalysis(
        annotated_video_path = annotated_video_path,
        fps = video.fps,
        real_court_keypoints = real_court_keypoints,
        player_ids = player_ids,
        player_court_foot_positions = player_court_foot_positions,
        ball_court_positions = ball_court_positions,
        shots = shots,
        player_speeds = player_speeds
    )
