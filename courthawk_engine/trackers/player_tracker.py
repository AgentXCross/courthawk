"""
PlayerTracker class.

Tracks the players using a YOLO model. 
Filters the identified people for the 2 players. 
Players are stored as BoundingBox objects.
"""

from ultralytics import YOLO

import pickle
from pathlib import Path

import cv2
import numpy as np

from ..core import (
    BoundingBox,
    Point,
    euclidean_distance
)


class PlayerTracker:
    """
    Tracks players over the frames of the video.

    detect_frames() should be run first. It runs the YOLO model on all frames 
    and only tracks the "person" class. 

    choose_and_filter_players() should be run after. It requires the court keypoints.
    """
    def __init__(self, model_path: Path):
        self.model: YOLO = YOLO(model_path)


    def choose_and_filter_players(
            self, 
            court_keypoints: list[Point], 
            player_detections: list[dict[int, BoundingBox]] # one dict per frame mapping track ID to their BoundingBox
        ) -> tuple[list[dict[int, BoundingBox]], list[int]]:
        """
        Chooses the players on the first frame. Calls choose_players.

        Filters and returns the player_detections dict, only keeping the indices of the chosen players.
        Also returns the indices of the chosen players.
        """
        player_detections_first_frame: dict[int, BoundingBox] = player_detections[0]
        chosen_players: list[int] = self.choose_players(court_keypoints, player_detections_first_frame)

        filtered_player_detections: list[dict[int, BoundingBox]] = []

        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections, chosen_players


    def choose_players(
            self,
            court_keypoints: list[Point],
            player_dict: dict[int, BoundingBox],
            alpha: float = 0.2,
            beta: float = 0.8
        ):
        """
        Filters for the 2 players. Each person is scored as:

            alpha * (sum of distances to the nearest 3 court keypoints)
            + beta * (y-distance to the nearest baseline)

        Lower score is better. Returns the indices of the 2 selected players.
        """
        # court_keypoints[0:2] are the near baseline corners, court_keypoints[2:4] are the
        # far baseline corners (see the keypoint ordering in engine.py's _real_court_keypoints())
        near_baseline_y = (court_keypoints[0].y + court_keypoints[1].y) / 2
        far_baseline_y = (court_keypoints[2].y + court_keypoints[3].y) / 2

        scores: list[tuple[int, float]] = []
        for track_id, bbox in player_dict.items():
            player_foot = bbox.foot

            # Calculate distance from each person to all the court keypoints
            dists = []
            for keypoint in court_keypoints:
                dist = euclidean_distance(player_foot, keypoint)
                dists.append(dist)

            # Sort and then take the sum of the smallest 3 values
            dists.sort()
            sum_dist_of_min_3 = float(sum(dists[:3]))

            baseline_y_dist = min(abs(player_foot.y - near_baseline_y), abs(player_foot.y - far_baseline_y))

            score = alpha * sum_dist_of_min_3 + beta * baseline_y_dist
            scores.append((track_id, score))

        # Sort and then take the smallest 2 values who we choose as the player
        scores.sort(key = lambda x: x[1])
        chosen_players = [scores[0][0], scores[1][0]]
        return chosen_players


    def detect_frames(
            self, 
            frames: list[np.ndarray],
            read_from_stub: bool = False, 
            stub_path: Path | None = None
        ) -> list[dict[int, BoundingBox]]:
        """
        Runs detect_frame() on all the frames. Returns a list of dicts that map
        a tracking index to a its BoundingBox. 
        """
        player_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            assert len(player_detections) == len(frames)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        assert len(player_detections) == len(frames)
        return player_detections

    
    def detect_frame(self, frame: np.ndarray) -> dict[int, BoundingBox]:
        """
        Runs YOLO model to detect person position on a single frame.
        Returns as a dict mapping track index to BoundingBox.
        """
        results = self.model.track(frame, persist = True, classes = [0])[0] # class 0 is "person"

        player_dict: dict[int, BoundingBox] = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            x1, y1, x2, y2 = box.xyxy.tolist()[0]

            bbox = BoundingBox(
                Point(x1, y1), Point(x2, y2)
            )
            player_dict[track_id] = bbox

        return player_dict

