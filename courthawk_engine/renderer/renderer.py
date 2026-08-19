"""
Render Class.

Draws keypoints, minicourt, player boxes, ball box, and stats onto the output frames.
"""

import numpy as np
import cv2

from ..core import (
    Point,
    BoundingBox,
    CourtSide
)
from ..pose_estimation import ShotType
from ..minicourt import MiniCourt


class Renderer:
    """
    Draws keypoints, minicourt, player boxes, ball box, and stats onto the output frames.
    The outside will only call the render method that will call all other methods.

    This class is only useful for the testing stage since on the app the stats will not be 
    overlayed onto the video.
    """
    def __init__(
            self, 
            font = cv2.FONT_HERSHEY_TRIPLEX,
            color_1: tuple = (1, 255, 214), # Neon yellow
            color_2: tuple = (140, 0, 236), # Hot pink
        ):
        self.font = font
        self.color_1 = color_1
        self.color_2 = color_2


    def render(
            self,
            video_frames: list[np.ndarray],
            court_keypoints: list[Point],
            player_bbox_detections: list[dict[CourtSide, BoundingBox]],
            ball_bbox_detections: list[BoundingBox],
            player_point_detections_mini: list[dict[CourtSide, Point]], # homography already applied
            ball_point_detections_mini: list[Point], # homography already applied
            ball_shot_frames: list[int],
            shot_types: list[ShotType],
            hitting_player_ids : list[CourtSide],
            player_ids: list[CourtSide],
            mini_court: MiniCourt,
            player_shots_data: list[dict[str, int | float]],
            player_speeds_data: list[dict[str, int | float]]

        ) -> list[np.ndarray]:
        """Calls all other methods in this class to render all ouputs onto the video."""

        video_frames = self.draw_keypoints_on_video(video_frames, court_keypoints)
        video_frames = self.draw_player_bboxes(video_frames, player_bbox_detections)
        video_frames = self.draw_shot_types(video_frames, player_bbox_detections, ball_shot_frames, shot_types, hitting_player_ids)
        video_frames = self.draw_ball_bboxes(video_frames, ball_bbox_detections)
        video_frames = self.draw_minicourt(video_frames, mini_court)
        video_frames = self.draw_players_on_minicourt(video_frames, player_point_detections_mini)
        video_frames = self.draw_ball_on_minicourt(video_frames, ball_point_detections_mini)
        video_frames = self.draw_stats(video_frames, player_shots_data, player_speeds_data, player_ids)

        return video_frames


    def _draw_keypoints_on_image(self, image: np.ndarray, keypoints: list[Point]) -> np.ndarray:
        for i in range(len(keypoints)):
            x = int(keypoints[i].x)
            y = int(keypoints[i].y)

            cv2.putText(
                image, 
                str(i + 1),
                (x, y - 10),
                self.font,
                1,
                self.color_1,
                2
            )
            cv2.circle(image, (x, y), 10, self.color_1, -1)

        return image


    def draw_keypoints_on_video(self, video_frames: list[np.ndarray], keypoints: list[Point]) -> list[np.ndarray]:
        """
        Draws the court keypoints on the video. Returns the new video frames.
        """
        assert len(keypoints) == 14

        output_frames: list[np.ndarray] = []

        for frame in video_frames:
            output_frames.append(self._draw_keypoints_on_image(frame, keypoints))

        return output_frames


    def draw_player_bboxes(
            self,
            video_frames: list[np.ndarray],
            player_bbox_detections: list[dict[CourtSide, BoundingBox]]
        ) -> list[np.ndarray]:
        """Draws player bounding boxes on the output frames."""
        assert len(video_frames) == len(player_bbox_detections)

        output_video_frames: list[np.ndarray] = []

        for frame, player_bbox_dict in zip(video_frames, player_bbox_detections):
            for player_id, bbox in player_bbox_dict.items():
                cv2.putText(
                    frame,
                    f"Player: {player_id.value.capitalize()}",
                    (int(bbox.tl.x), int(bbox.tl.y - 10)),
                    self.font,
                    0.9,
                    self.color_2,
                    2
                )
                cv2.rectangle(
                    frame,
                    (int(bbox.tl.x), int(bbox.tl.y)),
                    (int(bbox.br.x), int(bbox.br.y)),
                    self.color_2,
                    2
                )

            output_video_frames.append(frame)

        return output_video_frames


    def draw_shot_types(
            self,
            video_frames: list[np.ndarray],
            player_bbox_detections: list[dict[CourtSide, BoundingBox]],
            ball_shot_frames: list[int],
            shot_types: list[ShotType],
            hitting_player_ids : list[CourtSide]
        ) -> list[np.ndarray]:
        """
        Draws shot types when a player hits a shot.
        Displays the shot type for a + or - 5 frame window.
        """
        assert len(video_frames) == len(player_bbox_detections)
        assert len(ball_shot_frames) == len(shot_types) and len(ball_shot_frames) == len(hitting_player_ids)

        # active maps a frame number to a list of which player and shot type
        active: dict[int, list[tuple[CourtSide, ShotType]]] = {}

        for shot_frame, shot_type, player_id in zip(ball_shot_frames, shot_types, hitting_player_ids):
            for frame in range(
                max(0, shot_frame - 5),
                min(len(video_frames), shot_frame + 6)
            ):
                if active.get(frame, None) is None:
                    active[frame] = [(player_id, shot_type)]
                else:
                    active[frame].append((player_id, shot_type))

        for frame_num, (frame, player_dict) in enumerate(zip(video_frames, player_bbox_detections)):
            if frame_num in active:
                for player_id, shot_type in active[frame_num]:
                    bbox = player_dict[player_id]
                    cv2.putText(
                        frame,
                        shot_type.value,
                        (int(bbox.tl.x), int(bbox.br.y + 25)),
                        self.font,
                        0.9,
                        self.color_2,
                        2
                    )

        return video_frames
        

    def draw_ball_bboxes(self, video_frames: list[np.ndarray], ball_bbox_detections: list[BoundingBox]) -> list[np.ndarray]:
        """Draws the ball bounding boxes."""
        assert len(video_frames) == len(ball_bbox_detections)

        output_video_frames: list[np.ndarray] = []

        for frame, bbox in zip(video_frames, ball_bbox_detections):
            if bbox:
                cv2.putText(
                    frame,
                    "Ball",
                    (int(bbox.tl.x), int(bbox.tl.y - 10)),
                    self.font,
                    0.9,
                    self.color_1,
                    2
                )
                cv2.rectangle(
                    frame,
                    (int(bbox.tl.x), int(bbox.tl.y)),
                    (int(bbox.br.x), int(bbox.br.y)),
                    self.color_1,
                    2
                )

                output_video_frames.append(frame)

        return output_video_frames


    def _draw_minicourt_background(self, frame: np.ndarray, minicourt: MiniCourt) -> np.ndarray:
        """Draws the background of the mini-court on a single frame."""
        overlay = frame.copy()

        cv2.rectangle(
            overlay, 
            (minicourt.background_tl.x, minicourt.background_tl.y),
            (minicourt.background_br.x, minicourt.background_br.y),
            (0, 0, 0),
            cv2.FILLED
        )

        return cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)


    def _draw_court_on_minicourt(self, frame: np.ndarray, minicourt: MiniCourt) -> np.ndarray:
        """Draws the lines, net, and court keypoints on the minicourt."""
        # Lines
        for line in minicourt.drawing_lines:
            start_point: Point = line.end_1
            end_point: Point = line.end_2

            cv2.line(
                frame, 
                (int(start_point.x), int(start_point.y)),
                (int(end_point.x), int(end_point.y)),
                (255, 255, 255),
                3
            )

        # Met
        net_start_point = Point(
            int(minicourt.drawing_keypoints[0].x),
            (minicourt.drawing_keypoints[0].y + minicourt.drawing_keypoints[2].y) // 2
        )

        net_end_point = Point(
            int(minicourt.drawing_keypoints[1].x),
            (minicourt.drawing_keypoints[0].y + minicourt.drawing_keypoints[2].y) // 2
        )

        cv2.line(
            frame,
            (int(net_start_point.x), int(net_start_point.y)),
            (int(net_end_point.x), int(net_end_point.y)),
            (255, 255, 255),
            3
        )

        # Keypoints
        for keypoint in minicourt.drawing_keypoints:
            x = int(keypoint.x)
            y = int(keypoint.y)

            cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)

        return frame


    def draw_minicourt(self, video_frames: list[np.ndarray], minicourt: MiniCourt):
        """Draws the minicourt background, the lines, net, and keypoints."""
        output_frames = []

        for frame in video_frames:
            self._draw_minicourt_background(frame, minicourt)
            self._draw_court_on_minicourt(frame, minicourt)

            output_frames.append(frame)

        return output_frames


    def draw_players_on_minicourt(
            self,
            video_frames: list[np.ndarray],
            player_mini_court_positions: list[dict[CourtSide, Point]],
        ):
        """Draws the players on the minicourt as dots."""
        assert len(video_frames) == len(player_mini_court_positions)

        for frame_num, frame in enumerate(video_frames):
            for _, point_position in player_mini_court_positions[frame_num].items():
                cv2.circle(frame, (int(point_position.x), int(point_position.y)), 10, self.color_2, -1)

        return video_frames

    
    def draw_ball_on_minicourt(
            self,
            video_frames: list[np.ndarray],
            ball_mini_court_positions: list[Point]
    ):
        """Draws the ball on the minicourt as a dot."""
        assert len(video_frames) == len(ball_mini_court_positions)

        for frame, point_position in zip(video_frames, ball_mini_court_positions):
            cv2.circle(frame, (int(point_position.x), int(point_position.y)), 10, self.color_1, -1)

        return video_frames


    def draw_stats(
        self,
        video_frames: list[np.ndarray],
        player_shots_data: list[dict[str, int | float]], # keys are frame, player_who_hit, ball_speed_kmh
        player_speeds_data: list[dict[str, int | float]], # keys are frame, player_{side.value}_speed_kmh
        player_ids: list[CourtSide]
    ) -> list[np.ndarray]:
        """Displays player and ball speed stats on the output video."""
        p1_id, p2_id = CourtSide.CLOSE, CourtSide.FAR
        shot_idx = 0
        speed_idx = 0
        p1_shot_speed = p2_shot_speed = None
        p1_speed = p2_speed = None

        font_scale = 0.6
        thickness = 1
        x, y = 30, 185 # places it right under the Frame #
        label_w = 260
        val_w = 130
        row_h = 35
        n_rows = 3
        box_w = label_w + val_w * 2
        box_h = row_h * n_rows + 15

        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            # track last shot speed per player
            while shot_idx < len(player_shots_data) and player_shots_data[shot_idx]['frame'] <= frame_num:
                s = player_shots_data[shot_idx]
                if s['player_who_hit'] == p1_id:
                    p1_shot_speed = s['ball_speed_kmh']
                else:
                    p2_shot_speed = s['ball_speed_kmh']
                shot_idx += 1

            # track last per-stride speed per player
            while speed_idx < len(player_speeds_data) and player_speeds_data[speed_idx]['frame'] <= frame_num:
                s = player_speeds_data[speed_idx]
                p1_speed = s[f'player_{p1_id.value}_speed_kmh']
                p2_speed = s[f'player_{p2_id.value}_speed_kmh']
                speed_idx += 1

            def display_val(val): return f"{val:.1f} km/h" if val is not None else "N/A"

            rows = [
                ("", p1_id.value.capitalize(), p2_id.value.capitalize()),
                ("Last Shot Speed", display_val(p1_shot_speed), display_val(p2_shot_speed)),
                ("Player Speed", display_val(p1_speed), display_val(p2_speed)),
            ]

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)

            for r, (label, v1, v2) in enumerate(rows):
                row_y = y + 10 + r * row_h
                cv2.putText(frame, label, (x + 5, row_y + 20), self.font, font_scale, (255, 255, 255), thickness)
                cv2.putText(frame, v1, (x + 5 + label_w, row_y + 20), self.font, font_scale, (255, 255, 255), thickness)
                cv2.putText(frame, v2, (x + 5 + label_w + val_w, row_y + 20), self.font, font_scale, (255, 255, 255), thickness)

            output_frames.append(frame)

        return output_frames