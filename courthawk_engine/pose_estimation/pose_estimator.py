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
        base_options = mp_python.BaseOptions(model_asset_path = model_path)
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

        left_extension: float = _norm(left_wrist - left_shoulder)
        right_extension: float = _norm(right_wrist - right_shoulder)

        l_elbow_angle: float = _angle(left_shoulder, left_elbow, left_wrist)
        r_elbow_angle: float = _angle(right_shoulder, right_elbow, right_wrist)

        # 1. Max wrist height relative to mid-shoulder
        max_wrist_height = max(-(left_wrist.y - mid_shoulder.y), -(right_wrist.y - mid_shoulder.y))

        # 2. Vertical wrist separation
        vert_wrist_sep = abs(left_wrist.y - right_wrist.y)

        # 3. Horizontal wrist separation
        horiz_wrist_sep = abs(left_wrist.x - right_wrist.x)

        # 4 & 5. More/less extended arm elbow angles
        if left_extension >= right_extension:
            more_extended_elbow, less_extended_elbow = l_elbow_angle, r_elbow_angle
        else:
            more_extended_elbow, less_extended_elbow = r_elbow_angle, l_elbow_angle

        # 6. Arm extension asymmetry
        ext_asymmetry = abs(left_extension - right_extension)

        # 7. Shoulder rotation angle
        shoulder_angle = _line_angle(left_shoulder, right_shoulder)

        # 8. Hip rotation angle
        hip_angle = _line_angle(left_hip, right_hip)

        # 9. Hip-shoulder twist
        twist = shoulder_angle - hip_angle

        # 10. Torso lean (angle of spine from vertical, mid_hip is at origin)
        torso_lean = np.degrees(np.arctan2(mid_shoulder.x, -mid_shoulder.y))

        # 11. Wrist x-coordinate product
        wrist_product = float(left_wrist.x) * float(right_wrist.x)

        # 12. Min cross-body shoulder-to-wrist distance
        cross_dist = min(_norm(left_wrist - right_shoulder), _norm(right_wrist - left_shoulder))

        # 13. Angle between forearm vectors
        forearm_angle = _vec_angle(left_wrist - left_elbow, right_wrist - right_elbow)

        # 14. Elbow angle difference
        elbow_angle_diff = abs(l_elbow_angle - r_elbow_angle)

        # 15. Average wrist height minus average elbow height
        wrist_vs_elbow = -((left_wrist.y + right_wrist.y) / 2 - (left_elbow.y + right_elbow.y) / 2)

        # 16. Legs crossed: 1 if knee x-ordering is opposite to hip x-ordering
        legs_crossed = float((left_knee.x - right_knee.x) * (left_hip.x - right_hip.x) < 0)

        # 17. Right wrist offset
        right_wrist_offset = (right_wrist.x - mid_shoulder.x) / _norm(right_shoulder - left_shoulder)

        # 18. Left wrist offset
        left_wrist_offset = (left_wrist.x - mid_shoulder.x) / _norm(right_shoulder - left_shoulder)

        return np.array([
            max_wrist_height,       # 1
            vert_wrist_sep,         # 2
            horiz_wrist_sep,        # 3
            more_extended_elbow,    # 4
            less_extended_elbow,    # 5
            ext_asymmetry,          # 6
            shoulder_angle,         # 7
            hip_angle,              # 8
            twist,                  # 9
            torso_lean,             # 10
            wrist_product,          # 11
            cross_dist,             # 12
            forearm_angle,          # 13
            elbow_angle_diff,       # 14
            wrist_vs_elbow,         # 15
            legs_crossed,           # 16
            right_wrist_offset,     # 17
            left_wrist_offset,      # 18
        ], dtype = np.float32)


    def classify_shots(
            self, 
            video_frames: list[np.ndarray], 
            ball_shot_frames: list[int], 
            player_bbox_detections: list[dict[int, BoundingBox]], 
            ball_bbox_detections: list[BoundingBox], 
            classifier: ShotClassifier
        ) -> tuple[list[ShotType], list[int]]:
        """
        For each ball hit frame, samples up to 7 frames (frame - 3 to frame + 3), crops the hitting
        player, runs pose estimation on each crop, and majority-votes the shot type.

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
