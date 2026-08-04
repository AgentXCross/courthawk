"""
BallTracker class.

Tracks the ball path using a YOLO model.
Also handles determining when a ball is hit by a player.
The ball is primarily stored as a BoundingBox object.
"""


from ultralytics import YOLO

import pickle
from pathlib import Path

import cv2
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from core import (
    BoundingBox,
    euclidean_distance
)


class BallTracker:
    def __init__(self, model_path: Path):
        self.model = YOLO(model_path)


    def detect_frames(
            self, 
            frames: np.ndarray, 
            read_from_stub: bool = False, 
            stub_path: Path = None
        ) -> list[BoundingBox | None]:
        """
        Runs detect_frames on all frames. Returns as a list of BoundingBox
        or None for frames where no ball was detected. 
        """
        ball_detections: list[BoundingBox | None] = []

        # Load stubs
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_bbox: BoundingBox | None = self.detect_frame(frame)
            ball_detections.append(ball_bbox)
        
        if stub_path is not None: # Save the ball_detections if we have a path
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    
    def detect_frame(self, frame: np.ndarray) -> BoundingBox | None:
        """
        Runs YOLO model to detect ball position on a single frame.
        Returns as a BoundingBox.
        """
        results = self.model.predict(frame, conf = 0.10)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            return BoundingBox(
                Point(x1, y1), Point(x2, y2)
            )
            
        return None


    def interpolate_ball_positions(
            self, 
            ball_positions: list[BoundingBox | None]
        ) -> list[BoundingBox]:
        """
        Interpolate all ball positions by converting the list of BoundingBox or None
        from detect_frames into a pd.DataFrame and then filling in the missing values
        with a linear assumption.

        Retuns a list of BoundingBox with no None values.
        """
        ball_positions_as_list: list[tuple[float, float, float, float]] = 
            [(bbox.tl.x, bbox.tl.y, bbox.br.x, bbox.br.y) 
             if bbox is not None
             else (None, None, None, None)
             for bbox in ball_positions
             ]
        assert len(ball_positions) == len(ball_positions_as_list)

        df_ball_positions = pd.DataFrame(ball_positions_as_list, columns = ["x1", "y1", "x2", "y2"])

        # Interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # Ensure the first frame has a detection
        df_ball_positions = df_ball_positions.ffill() # Ensure the last frame has a detection

        new_ball_positions_as_list = df_ball_positions.to_numpy().tolist()

        new_ball_positions: list[BoundingBox] = [
            BoundingBox(
                Point(x1, y1), Point(x2, y2)
            )
            for x1, y1, x2, y2 in new_ball_positions_as_list
        ]

        return new_ball_positions

    
    def get_ball_shot_frames(
            self, 
            ball_positions: list[BoundingBox], 
            player_detections: list[dict[int, BoundingBox]]
        ) -> list[int]:
        """
        Determines the frame numbers where the ball is hit by a player.

        This is calculated by graphing out the relationship between time and
        distance to the closest player. We consider the ball hit when we hit the local
        minimums in the graph with a certain buffer.
        """
        assert len(ball_positions) == len(player_detections)

        ball_positions_as_list: list[tuple[float, float, float, float]] = [
            (bbox.tl.x, bbox.tl.y, bbox.br.x, bbox.br.y)
            for bbox in ball_positions
        ]

        # Build ball positions DataFrame
        df_ball_positions = pd.DataFrame(ball_positions_as_list, columns = ["x1", "y1", "x2", "y2"])
        df_ball_positions = (
            df_ball_positions
            .interpolate()
            .bfill()
            .ffill()
        )

        df_ball_positions["mid_x"] = (df_ball_positions["x1"] + df_ball_positions["x2"]) / 2
        df_ball_positions["mid_y"] = (df_ball_positions["y1"] + df_ball_positions["y2"]) / 2

        # Build player position DataFrame, one row per frame, one column set per player
        player_ids = sorted({ # Grab player_ids, this ensures we don't miss any players
            player_id
            for frame_detections in player_detections
            for player_id in frame_detections
        })
        assert len(player_ids) == 2

        player_position_rows = []
        for frame in player_detections: # A frame is a dict[int, BoundingBox]
            row = {}
            for player_id, bbox in frame.items():
                center_point: Point = bbox.center
                row[f"p{player_id}_mid_x"] = center_point.x
                row[f"p{player_id}_mid_y"] = center_point.y
            player_position_rows.append(row)

        df_players = (
            pd.DataFrame(player_position_rows)
            .interpolate()
            .bfill()
            .ffill()
        )

        # Find the distance to the closest player in each frame
        closest_distances: list[float] = []

        for frame_index in range(len(ball_positions)):
            ball = Point(
                x = df_ball_positions.at(frame_index, "mid_x"),
                y = df_ball_positions.at(frame_index, "mid_y")
            )

            distances_this_frame: list[float] = [] # list[float, float]

            for player_id in player_ids:
                player = Point(
                    x = df_players.at(frame_index, f"p{player_id}_mid_x"),
                    y = df_players.at(frame_index, f"p{player_id}_mid_y")
                )

                distances_this_frame.append(
                    euclidean_distance(ball, player)
                )

            closest_distances.append(min(distances_this_frame))

        assert len(closest_distances) == len(ball_positions)

        # Find local minima
        peaks, _ = find_peaks(
            -np.array(closest_distances),
            prominence = 50, # only count valleys where the ball gets at least 50px closer than surrounding frames
            distance = 15 # min frame numbers between different peaks
        )

        return peaks.tolist()


    # Will remove this function later, put in Renderer
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, bbox in zip(video_frames, ball_detections):
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    "Ball", (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.9,
                    (1, 255, 214),
                    2
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (1, 255, 214), 2)
            output_video_frames.append(frame)
        return output_video_frames

