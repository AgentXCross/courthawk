import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import cv2
import numpy as np
from PIL import Image

MODEL_PATH = 'models/pose_landmarker.task'

def _angle(a, b, c):
    """Angle in degrees at point b in the triangle a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def _vec_angle(v1, v2):
    """Angle in degrees between two vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def _line_angle(p1, p2):
    """Angle in degrees of the line p1 to p2 from horizontal."""
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def _apply_clahe(img_rgb):
    """Boost local contrast via CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (2, 2))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

class PoseEstimator:
    # MediaPipe landmark indices
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26

    def __init__(self, model_path = MODEL_PATH):
        base_options = mp_python.BaseOptions(model_asset_path = model_path)
        options = mp_vision.PoseLandmarkerOptions(base_options = base_options)
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def _to_rgb(self, frame):
        """Accepts a BGR numpy array, RGB numpy array, or file path."""
        if isinstance(frame, str):
            img = np.array(Image.open(frame).convert('RGB'))
        elif frame.shape[2] == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            img = frame
        return _apply_clahe(img)

    def get_keypoints(self, frame):
        """
        Runs MediaPipe Pose on a frame and returns a 16-feature vector for shot
        classification. Returns None if no pose is detected.
        See features.txt for full feature descriptions and normalization details.
        """
        img_rgb = self._to_rgb(frame)
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = img_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        h, w = img_rgb.shape[:2]
        kps = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        # Normalize: center on mid-hip, scale by torso height
        mid_hip = (kps[self.L_HIP] + kps[self.R_HIP]) / 2
        mid_shoulder = (kps[self.L_SHOULDER] + kps[self.R_SHOULDER]) / 2
        kps -= mid_hip
        mid_shoulder -= mid_hip
        torso_height = np.linalg.norm(mid_shoulder)
        if torso_height > 0:
            kps /= torso_height

        ls = kps[self.L_SHOULDER]
        rs = kps[self.R_SHOULDER]
        le = kps[self.L_ELBOW]
        re = kps[self.R_ELBOW]
        lw = kps[self.L_WRIST]
        rw = kps[self.R_WRIST]
        lh = kps[self.L_HIP]
        rh = kps[self.R_HIP]
        lk = kps[self.L_KNEE]
        rk = kps[self.R_KNEE]
        mid_s = (ls + rs) / 2

        l_ext = np.linalg.norm(lw - ls)
        r_ext = np.linalg.norm(rw - rs)

        l_elbow_angle = _angle(ls, le, lw)
        r_elbow_angle = _angle(rs, re, rw)

        # 1. Max wrist height relative to mid-shoulder
        max_wrist_height = max(-(lw[1] - mid_s[1]), -(rw[1] - mid_s[1]))

        # 2. Vertical wrist separation
        vert_wrist_sep = abs(lw[1] - rw[1])

        # 3. Horizontal wrist separation
        horiz_wrist_sep = abs(lw[0] - rw[0])

        # 4 & 5. More/less extended arm elbow angles
        if l_ext >= r_ext:
            more_ext_elbow, less_ext_elbow = l_elbow_angle, r_elbow_angle
        else:
            more_ext_elbow, less_ext_elbow = r_elbow_angle, l_elbow_angle

        # 6. Arm extension asymmetry
        ext_asymmetry = abs(l_ext - r_ext)

        # 7. Shoulder rotation angle
        shoulder_angle = _line_angle(ls, rs)

        # 8. Hip rotation angle
        hip_angle = _line_angle(lh, rh)

        # 9. Hip-shoulder twist
        twist = shoulder_angle - hip_angle

        # 10. Torso lean (angle of spine from vertical, mid_hip is at origin)
        torso_lean = np.degrees(np.arctan2(mid_s[0], -mid_s[1]))

        # 11. Wrist x-coordinate product
        wrist_product = float(lw[0]) * float(rw[0])

        # 12. Min cross-body shoulder-to-wrist distance
        cross_dist = min(np.linalg.norm(lw - rs), np.linalg.norm(rw - ls))

        # 13. Angle between forearm vectors
        forearm_angle = _vec_angle(lw - le, rw - re)

        # 14. Elbow angle difference
        elbow_angle_diff = abs(l_elbow_angle - r_elbow_angle)

        # 15. Average wrist height minus average elbow height
        wrist_vs_elbow = -((lw[1] + rw[1]) / 2 - (le[1] + re[1]) / 2)

        # 16. Legs crossed: 1 if knee x-ordering is opposite to hip x-ordering
        legs_crossed = float((lk[0] - rk[0]) * (lh[0] - rh[0]) < 0)

        return np.array([
            max_wrist_height,   # 1
            vert_wrist_sep,     # 2
            horiz_wrist_sep,    # 3
            more_ext_elbow,     # 4
            less_ext_elbow,     # 5
            ext_asymmetry,      # 6
            shoulder_angle,     # 7
            hip_angle,          # 8
            twist,              # 9
            torso_lean,         # 10
            wrist_product,      # 11
            cross_dist,         # 12
            forearm_angle,      # 13
            elbow_angle_diff,   # 14
            wrist_vs_elbow,     # 15
            legs_crossed,       # 16
        ], dtype = np.float32)

    def classify_shots(self, video_frames, ball_shot_frames, player_detections, ball_detections, classifier):
        """
        For each ball hit frame, samples up to 7 frames (frame - 3 to frame + 3), crops the hitting
        player, runs pose estimation on each crop, and majority-votes the shot type.
        Returns 'unknown' only if no pose is detected in any of the 7 frames.
        Ties are broken randomly.
        """
        import random
        from collections import Counter

        n = len(video_frames)
        shot_types = []
        hitting_player_ids = []

        for frame_num in ball_shot_frames:
            ball_box = ball_detections[frame_num]
            ball_cx = (ball_box[0] + ball_box[2]) / 2
            ball_cy = (ball_box[1] + ball_box[3]) / 2

            frame_players = player_detections[frame_num]
            hitting_player_id = min(
                frame_players.keys(),
                key = lambda pid: (
                    ((frame_players[pid][0] + frame_players[pid][2]) / 2 - ball_cx) ** 2 +
                    ((frame_players[pid][1] + frame_players[pid][3]) / 2 - ball_cy) ** 2
                )
            )
            hitting_player_ids.append(hitting_player_id)

            votes = []
            for f in range(max(0, frame_num - 3), min(n, frame_num + 4)):
                player_box = player_detections[f].get(hitting_player_id)
                if player_box is None:
                    continue
                x1, y1, x2, y2 = int(player_box[0]), int(player_box[1]), int(player_box[2]), int(player_box[3])
                crop = video_frames[f][max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                keypoints = self.get_keypoints(crop)
                if keypoints is not None:
                    votes.append(classifier.predict(keypoints))

            if not votes:
                shot_types.append('unknown')
            else:
                count = Counter(votes)
                max_count = max(count.values())
                candidates = [s for s, c in count.items() if c == max_count]
                shot_types.append(random.choice(candidates))

        return shot_types, hitting_player_ids
