"""
PoseEstimator class and feature extraction functions from pose estimation.
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import cv2
import numpy as np
from PIL import Image

import random
from collections import Counter
from pathlib import Path

from ..core import (
    Point,
    BoundingBox,
    Line
)
from .shot_classifier import ShotClassifier, ShotType


def _dot_product(u: Point, v: Point) -> float:
    """Returns the dot product of 2 vectors."""
    return u.x * v.x + u.y * v.y


def _norm(u: Point) -> float:
    """Returns the norm of a vector. Vectors are represented using Point."""
    return np.sqrt(u.x * u.x + u.y * u.y)


def _angle(a: Point, b: Point, c: Point) -> float:
    """Returns the angle in degrees at point b in the triangle a-b-c."""
    ba = a - b
    bc = c - b

    dot_product = _dot_product(ba, bc)
    ba_norm = _norm(ba)
    bc_norm = _norm(bc)

    cos_theta = np.clip(dot_product / (ba_norm * bc_norm), -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_theta)))


def _vec_angle(v1: Point, v2: Point):
    """Angle in degrees between two vectors."""
    dot_product = _dot_product(v1, v2)
    v1_norm = _norm(v1)
    v2_norm = _norm(v2)

    cos_theta = np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_theta)))


def _line_angle(p1: Point, p2: Point) -> float:
    """Angle in degrees of the line p1 to p2 from horizontal."""
    vector = p2 - p1

    angle = np.arctan2(vector.y, vector.x) # Because tan theta = y / x

    return float(np.degrees(angle))


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Boost local contrast via CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (2, 2))

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class PoseEstimator:
    """
    PoseEstimator class stores relevant MediaPipe indices and pose estimation model.

    get_keypoints() returns a feature vector for inference.
    classify_shots() classifies the shots.
    """
    # Relevant MediaPipe landmark indices
    L_SHOULDER, R_SHOULDER = mp_vision.PoseLandmark.LEFT_SHOULDER, mp_vision.PoseLandmark.RIGHT_SHOULDER
    L_ELBOW, R_ELBOW = mp_vision.PoseLandmark.LEFT_ELBOW, mp_vision.PoseLandmark.RIGHT_ELBOW
    L_WRIST, R_WRIST = mp_vision.PoseLandmark.LEFT_WRIST, mp_vision.PoseLandmark.RIGHT_WRIST
    L_HIP, R_HIP = mp_vision.PoseLandmark.LEFT_HIP, mp_vision.PoseLandmark.RIGHT_HIP
    L_KNEE, R_KNEE = mp_vision.PoseLandmark.LEFT_KNEE, mp_vision.PoseLandmark.RIGHT_KNEE


    def __init__(self, model_path: Path):
        base_options = mp_python.BaseOptions(model_asset_path = str(model_path))
        options = mp_vision.PoseLandmarkerOptions(base_options = base_options)
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)


    def _to_rgb(self, frame: np.ndarray | str) -> np.ndarray:
        """Converts a BGR image or image file to an RGB image with CLAHE applied on the L channel in LAB space."""
        if isinstance(frame, str):
            img = np.array(Image.open(frame).convert('RGB'))
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return _apply_clahe(img)
    

    def get_keypoints(self, frame: np.ndarray | str) -> np.ndarray | None:
        """
        Runs MediaPipe pose estimator on a frame and returns a feature vector for shot
        classification. 
        
        Returns None if no pose is detected.

        See the README.md for more details on the features.
        """
        img_rgb: np.ndarray = self._to_rgb(frame)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks: # No pose detected
            return None

        landmarks = result.pose_landmarks[0]
        height, width = img_rgb.shape[:2] 

        keypoints: list[Point] = [
            Point(landmark.x * width, landmark.y * height) 
            for landmark in landmarks
        ]

        # Normalize: center on mid-hip, scale by torso height
        mid_hip: Point = (keypoints[self.L_HIP] + keypoints[self.R_HIP]) / 2

        keypoints = [point - mid_hip for point in keypoints]

        mid_shoulder: Point = (keypoints[self.L_SHOULDER] + keypoints[self.R_SHOULDER]) / 2

        torso_height = _norm(mid_shoulder)

        if torso_height > 0:
            keypoints = [p / torso_height for p in keypoints]
        else:
            print(f"Warning torso height is non positive.\n")

        # Get the relevant Points
        left_shoulder: Point = keypoints[self.L_SHOULDER]
        right_shoulder: Point = keypoints[self.R_SHOULDER]
        left_elbow: Point = keypoints[self.L_ELBOW]
        right_elbow: Point = keypoints[self.R_ELBOW]
        left_wrist: Point = keypoints[self.L_WRIST]
        right_wrist: Point = keypoints[self.R_WRIST]
        left_hip: Point = keypoints[self.L_HIP]
        right_hip: Point = keypoints[self.R_HIP]
        left_knee: Point = keypoints[self.L_KNEE]
        right_knee: Point = keypoints[self.R_KNEE]

        mid_shoulder = (left_shoulder + right_shoulder) / 2 # This new mid_shoulder is centered around the mid-hip and scaled

        # 1. Max wrist height relative to mid-shoulder
        # Expect the serve to be a greater value than forehands and backhands
        max_wrist_height = max(-(left_wrist.y - mid_shoulder.y), -(right_wrist.y - mid_shoulder.y))

        # 2 & 3. Left and right elbow angles
        # Expect serve to be a greater angle than groundstrokes
        l_elbow_angle: float = _angle(left_shoulder, left_elbow, left_wrist)
        r_elbow_angle: float = _angle(right_shoulder, right_elbow, right_wrist)

        # 4. Shoulder tilt relative to the horizontal
        # Expect the serve to be a larger angle compared to groundstrokes
        shoulder_angle = _line_angle(left_shoulder, right_shoulder)

        # 5. Torso lean (angle of mid-hip to mid-shoulder from vertical)
        torso_lean = np.degrees(np.arctan2(mid_shoulder.x, -mid_shoulder.y))

        # 6 & 7. Left and right wrist x positions
        # Expect right wrist to be positive for forehand, negative for backhand, and near 0 for serves.
        # Left wrist may vary but should follow right write for groundstrokes, and near 0 for serves.
        l_wrist_x = left_wrist.x
        r_wrist_x = right_wrist.x

        # 8 & 9: Left and right wrist y positions
        # Expect both to be high for serve, lower for groundstrokes
        l_wrist_y = -left_wrist.y
        r_wrist_y = -right_wrist.y

        # 10 & 11: Left and right arm extension
        l_extension: float = _norm(left_wrist - left_shoulder)
        r_extension: float = _norm(right_wrist - right_shoulder)

        # 12 & 13: Left and right elbow x positions
        l_elbow_x = left_elbow.x
        r_elbow_x = right_elbow.x

        # 14: Absolute wrist separation
        wrist_x_separation = abs(right_wrist.x - left_wrist.x)


        return np.array([
            max_wrist_height,   # 1
            l_elbow_angle,      # 2
            r_elbow_angle,      # 3
            shoulder_angle,     # 4
            torso_lean,         # 5
            l_wrist_x,          # 6
            r_wrist_x,          # 7
            l_wrist_y,          # 8
            r_wrist_y,          # 9
            l_extension,        # 10
            r_extension,        # 11
            l_elbow_x,          # 12
            r_elbow_x,          # 13
            wrist_x_separation, # 14
        ], dtype = np.float32)


    def classify_shots(
            self, 
            video_frames: list[np.ndarray], 
            ball_shot_frames: list[int], 
            player_bbox_detections: list[dict[int, BoundingBox]], 
            ball_bbox_detections: list[BoundingBox], 
            classifier: ShotClassifier,
            should_mirror: dict[int, bool]
        ) -> tuple[list[ShotType], list[int]]:
        """
        For each ball hit frame, samples up to 7 frames (frame - 3 to frame + 3), crops the hitting
        player, runs pose estimation on each crop, and majority-votes the shot type.

        If far side XOR lefty evaluates to true for a player's id, the crop for that player before running it
        through pose estimation is reversed. 

        Returns ShotType.UNKNOWN only if no pose is detected in any of the 7 frames.
        Ties are broken randomly.
        """
        assert len(player_bbox_detections) == len(ball_bbox_detections) and len(video_frames) == len(player_bbox_detections)
        
        num_frames = len(video_frames)
        shot_types: list[ShotType] = []
        hitting_player_ids: list[int] = []

        for frame_num in ball_shot_frames:
            ball_box = ball_bbox_detections[frame_num]
            ball_center = ball_box.center

            frame_players = player_bbox_detections[frame_num]

            hitting_player_id = min(
                frame_players.keys(),
                key = lambda pid: (
                    (frame_players[pid].center.x - ball_center.x) ** 2 +
                    (frame_players[pid].center.y - ball_center.y) ** 2
                )
            )
            hitting_player_ids.append(hitting_player_id)

            votes: list[ShotType | None] = []
            for f in range(max(0, frame_num - 3), min(num_frames, frame_num + 4)):
                player_bbox = player_bbox_detections[f].get(hitting_player_id)

                if player_bbox is None:
                    continue

                x1, y1, x2, y2 = int(player_bbox.tl.x), int(player_bbox.tl.y), int(player_bbox.br.x), int(player_bbox.br.y)
                crop: np.ndarray = video_frames[f][max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                if should_mirror.get(hitting_player_id, False):
                    crop = cv2.flip(crop, 1)

                keypoints = self.get_keypoints(crop)
                if keypoints is not None:
                    votes.append(classifier.predict(keypoints))

            if not votes:
                shot_types.append(ShotType.UNKNOWN)
            else:
                count = Counter(votes)
                max_count = max(count.values())
                candidates = [s for s, c in count.items() if c == max_count]
                shot_types.append(random.choice(candidates)) # Only actually random when there is a tie

        return shot_types, hitting_player_ids
