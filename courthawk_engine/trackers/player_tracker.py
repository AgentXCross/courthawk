"""
PlayerTracker class.

Tracks the players using a YOLO model. 
Filters the identified people for the 2 players. 
Determines which side the players are on.
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
    CourtSide,
    euclidean_distance
)


def _is_close_side(point: Point, court_keypoints: list[Point]) -> bool:
    """
    Returns True if point is on the close (camera-side) half of the court.

    Uses the midpoint of the two center service line keypoints (indices 11, 12) as the
    y-cutoff.
    """
    y_cutoff = (court_keypoints[11].y + court_keypoints[12].y) / 2
    return point.y > y_cutoff


class PlayerTracker:
    """
    Tracks players over the frames of the video.

    determine_court_sides() determines which side a player is on, either close or far.

    detect_frames() should be run first. It runs the YOLO model on all frames 
    and only tracks the "person" class. 

    choose_and_filter_players() should be run after. It requires the court keypoints.
    """
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model: YOLO = YOLO(model_path)
        self.device: str = device


    def determine_court_sides(
            self,
            court_keypoints: list[Point],
            player_bbox_detections: list[dict[int, BoundingBox]],
            player_ids: list[int]
    ) -> dict[int, CourtSide]:
        """
        Determines which side of the court each player is on.
        This is only calculated once on the first frame.
        Returns a dict mapping player_id to which side of the court they are on.

        Takes pixel-space court_keypoints and player_bbox_detections (the same data
        choose_players() uses, before any homography to real-world meters), so this same
        function works whether it's called from engine.py or main.py, since both already
        compute pixel-space keypoints and player detections independently.
        """
        assert(len(player_ids) == 2)
        assert(len(court_keypoints) == 14)
        assert(len(player_bbox_detections) > 0)

        player_id_to_side: dict[int, CourtSide] = {}

        first_frame_bboxes: dict[int, BoundingBox] = player_bbox_detections[0]

        player_1_foot_position: Point = first_frame_bboxes[player_ids[0]].foot
        player_2_foot_position: Point = first_frame_bboxes[player_ids[1]].foot

        if _is_close_side(player_1_foot_position, court_keypoints):
            player_id_to_side[player_ids[0]] = CourtSide.CLOSE
        else:
            player_id_to_side[player_ids[0]] = CourtSide.FAR

        if _is_close_side(player_2_foot_position, court_keypoints):
            if player_id_to_side[player_ids[0]] == CourtSide.CLOSE:
                print(f"Players were determined to be on the same side. Overriding one player so they are on opposite sides.")
                player_id_to_side[player_ids[1]] = CourtSide.FAR
            else:
                player_id_to_side[player_ids[1]] = CourtSide.CLOSE
        else:
            if player_id_to_side[player_ids[0]] == CourtSide.FAR:
                print(f"Players were determined to be on the same side. Overriding one player so they are on opposite sides.")
                player_id_to_side[player_ids[1]] = CourtSide.CLOSE
            else:
                player_id_to_side[player_ids[1]] = CourtSide.FAR

        return player_id_to_side


    def choose_and_filter_players(
            self, 
            court_keypoints: list[Point], 
            player_bbox_detections: list[dict[int, BoundingBox]] # one dict per frame mapping track ID to their BoundingBox
        ) -> tuple[list[dict[int, BoundingBox]], list[int]]:
        """
        Chooses the players using the first frame with at least 2 confirmed detections
        (the tracker may not have confirmed any track IDs yet on frame 0 itself). Calls
        choose_players.

        Filters and returns the player_detections dict, only keeping the indices of the chosen players.
        Also returns the indices of the chosen players.
        """
        selection_frame: dict[int, BoundingBox] = next(
            (player_dict for player_dict in player_bbox_detections if len(player_dict) >= 2),
            player_bbox_detections[0]
        )
        chosen_players: list[int] = self.choose_players(court_keypoints, selection_frame)

        filtered_player_detections: list[dict[int, BoundingBox]] = []

        for player_dict in player_bbox_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections, chosen_players


    def choose_players(
            self,
            court_keypoints: list[Point],
            player_dict: dict[int, BoundingBox],
            alpha: float = 0.2,
            beta: float = 0.8,
            gamma: float = 0.5
        ):
        """
        Filters for the 2 players. Each person is scored as:

            gamma * (alpha * (sum of eucliden distances to the nearest 3 court keypoints)
            + beta * (euclidean distance to the nearest baseline center hash mark))

        gamma only applies to candidates on the close side of the court (gamma = 1 for
        far-side candidates). Camera perspective magnifies pixel distances near the
        camera, so a close-side player's own score would otherwise be inflated relative
        to a far-side player standing the equivalent real-world distance from their
        nearest keypoints.

        Lower score is better. Returns the indices of the 2 selected players.
        """
        # court_keypoints[0] and court_keypoints[1] are the far baseline corners.
        # court_keypoints[2] and court_keypoints[3] are the close baseline corners.
        far_hash_mark: Point = (court_keypoints[0] + court_keypoints[1]) / 2
        near_hash_mark: Point = (court_keypoints[2] + court_keypoints[3]) / 2

        scores: list[tuple[int, float]] = []
        for track_id, bbox in player_dict.items():
            player_foot = bbox.foot

            # Calculate distance from each person to all the court keypoints
            dists = []
            for keypoint in court_keypoints:
                dist = euclidean_distance(player_foot, keypoint)
                dists.append(dist)

            # Sort and then take the sum of the smallest 3 values
            assert len(dists) == 14
            dists.sort()
            sum_dist_of_min_3 = float(sum(dists[:3]))

            hash_dist = min(euclidean_distance(player_foot, far_hash_mark), euclidean_distance(player_foot, near_hash_mark))

            score = alpha * sum_dist_of_min_3 + beta * hash_dist
            if _is_close_side(player_foot, court_keypoints):
                score *= gamma

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
        results = self.model.track(
            frame, 
            persist = True, # persist = True keeps the tracker state alive
            classes = [0], # class 0 is "person"
            device = self.device,
        )[0] 

        player_dict: dict[int, BoundingBox] = {}

        for box in results.boxes:
            if box.id is None:
                # Tracker hasn't confirmed/assigned an ID to this detection yet this frame
                continue

            track_id = int(box.id.tolist()[0])
            x1, y1, x2, y2 = box.xyxy.tolist()[0]

            bbox = BoundingBox(
                Point(x1, y1), Point(x2, y2)
            )
            player_dict[track_id] = bbox

        return player_dict

