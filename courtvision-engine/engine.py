"""
Entry point to the CourtVision engine.

This module exposes the public API used by the backend. The backend passes
an input video to analyze_point(), which executes the full pipeline and returns 
the analysis results.
"""

from trackers import PlayerTracker, BallTracker
from court_keypoint_detector import CourtKeypointDetector
from minicourt import MiniCourt
from pose_estimation import PoseEstimator, ShotClassifier
from core import (
    Video,
    Point,
)

import cv2

from dataclasses import dataclass


@dataclass
class PointAnalysis:
    """PointAnalysis stores all the data that the frontend will display."""
    # Court
    court_keypoints: list[Point]


def analyze_point(video: Video) -> PointAnalysis:
    ...