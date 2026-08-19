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
import pandas as pd

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

    detect_frames() should be run first. It runs the YOLO model on all frames
    and only tracks the "person" class.

    choose_and_filter_players() should be run after. It requires the court keypoints.
    It also determines which side of the court each player is on and returns their
    bounding boxes keyed by CourtSide.
    """
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model: YOLO = YOLO(model_path)
        self.device: str = device


    def _determine_track_id_sides(
            self,
            court_keypoints: list[Point],
            selection_frame: dict[int, BoundingBox],
            chosen_players: list[int]
    ) -> dict[int, CourtSide]:
        """
        Determines which side of the court each of the 2 chosen track IDs is on, using
        their positions on selection_frame.
        Returns a dict mapping track_id to CourtSide.
        """
        assert(len(chosen_players) == 2)
        assert(len(court_keypoints) == 14)

        player_id_to_side: dict[int, CourtSide] = {}

        player_1_foot_position: Point = selection_frame[chosen_players[0]].foot
        player_2_foot_position: Point = selection_frame[chosen_players[1]].foot

        if _is_close_side(player_1_foot_position, court_keypoints):
            player_id_to_side[chosen_players[0]] = CourtSide.CLOSE
        else:
            player_id_to_side[chosen_players[0]] = CourtSide.FAR

        if _is_close_side(player_2_foot_position, court_keypoints):
            if player_id_to_side[chosen_players[0]] == CourtSide.CLOSE:
                print(f"Players were determined to be on the same side. Overriding one player so they are on opposite sides.")
                player_id_to_side[chosen_players[1]] = CourtSide.FAR
            else:
                player_id_to_side[chosen_players[1]] = CourtSide.CLOSE
        else:
            if player_id_to_side[chosen_players[0]] == CourtSide.FAR:
                print(f"Players were determined to be on the same side. Overriding one player so they are on opposite sides.")
                player_id_to_side[chosen_players[1]] = CourtSide.CLOSE
            else:
                player_id_to_side[chosen_players[1]] = CourtSide.FAR

        return player_id_to_side


    def choose_and_filter_players(
            self,
            court_keypoints: list[Point],
            player_bbox_detections: list[dict[int, BoundingBox]] # one dict per frame mapping track ID to their BoundingBox
        ) -> list[dict[CourtSide, BoundingBox]]:
        """
        Chooses the 2 players using the first frame with at least 2 confirmed detections,
        determines which side of the court each is on, and returns their bounding boxes re-keyed by
        CourtSide instead of by raw track ID.
        
        CourtSide.CLOSE and CourtSide.FAR are the player identifiers used everywhere downstream of this point.
        """
        selection_frame: dict[int, BoundingBox] = next(
            (player_dict for player_dict in player_bbox_detections if len(player_dict) >= 2),
            player_bbox_detections[0]
        )
        chosen_players: list[int] = self.choose_players(court_keypoints, selection_frame)
        track_id_to_side = self._determine_track_id_sides(court_keypoints, selection_frame, chosen_players)

        filtered_player_detections: list[dict[CourtSide, BoundingBox]] = []

        for player_dict in player_bbox_detections:
            filtered_player_dict = {
                track_id_to_side[track_id]: bbox
                for track_id, bbox in player_dict.items()
                if track_id in track_id_to_side
            }
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections


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


    def interpolate_player_positions(
            self,
            player_bbox_detections: list[dict[CourtSide, BoundingBox]]
    ) -> list[dict[CourtSide, BoundingBox]]:
        """
        Forward/backward fills and interpolates player bounding boxes for frames where the player
        wasn't detected by the tracker. 
        """
        num_frames: int = len(player_bbox_detections)
        interpolated: list[dict[CourtSide, BoundingBox]] = [{} for _ in range(num_frames)]

        for side in CourtSide:
            positions_as_list: list[tuple[float, float, float, float]] = [
                (frame[side].tl.x, frame[side].tl.y, frame[side].br.x, frame[side].br.y)
                if side in frame
                else (None, None, None, None)
                for frame in player_bbox_detections
            ]

            df = pd.DataFrame(positions_as_list, columns = ["x1", "y1", "x2", "y2"])
            df = df.interpolate().bfill().ffill()

            for frame_num, (x1, y1, x2, y2) in enumerate(df.to_numpy().tolist()):
                interpolated[frame_num][side] = BoundingBox(
                    Point(x1, y1), Point(x2, y2)
                )
                
        return interpolated


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

