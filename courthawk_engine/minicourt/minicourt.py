"""
MiniCourt class.
"""

import cv2
from cv2.typing import MatLike
import numpy as np

from ..core import (
    convert_pixel_distance_to_meters,
    convert_meters_to_pixel_distance,
    euclidean_distance,
    constants,
    Point,
    Line,
    BoundingBox
)


class MiniCourt:
    """
    Determine the coordinates of where to place the mini-court.
    Calculates the coordinate transforms that convert a point of the real court to 
    the mini-court.
    Computes shot and player speed stats.
    """
    def __init__(self, frame: np.ndarray):
        self.rectangle_width: int = 380 # Width of the mini-court
        self.rectangle_height: int = 720 # Length of the mini-court
        self.buffer: int = 10 # Between the entire mini-court and the edge of the video
        self.padding_court: int = 60 # On the sides of the court, vertical padding is determined in set_mini_court_position

        # The following attributes are set by the functions at the end of the constructor
        self.background_tl: Point | None = None
        self.background_br: Point | None = None

        self.court_tl: Point | None = None
        self.court_br: Point | None = None

        self.drawing_keypoints: list[Point] | None = None

        self.drawing_lines: list[Line] | None = None

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_keypoints()
        self.set_court_lines()


    def convert_meters_to_pixels(self, meters: float) -> float:
        """Scales real life tennis court distances in meters to the mini-courts own pixel space."""
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.mini_court_width()
        )


    def convert_pixels_to_meters(self, pixels: float) -> float:
        """Converts pixel distance on the mini-court to real life meters."""
        return convert_pixel_distance_to_meters(
            pixels,
            constants.DOUBLE_LINE_WIDTH,
            self.mini_court_width()
        )


    def set_canvas_background_box_position(self, frame: np.ndarray) -> None:
        """
        Sets the location for the endpoints of entire mini-court section.
        Sets the background_tr and background_br attributes.
        """
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Anchor to top-right corner, clamp to frame bounds
        end_x = frame_width - self.buffer
        start_x = end_x - self.rectangle_width

        start_y = self.buffer
        end_y = min(start_y + self.rectangle_height, frame_height - self.buffer)

        assert start_x >= 0 and end_x >= 0 and start_y >= 0 and end_y >= 0

        self.background_tl = Point(start_x, start_y)
        self.background_br = Point(end_x, end_y)


    def set_mini_court_position(self):
        """
        Sets the locations of the endpoints of the actual court in the mini-court area.
        Sets court_tl and court_br attributes.
        """
        court_start_x = self.background_tl.x + self.padding_court
        court_end_x = self.background_br.x - self.padding_court

        assert court_end_x >= court_start_x

        # Vertically center the court within the background rectangle
        court_height = self.convert_meters_to_pixels(constants.BASELINE_TO_NET * 2)
        available_height = self.background_br.y - self.background_tl.y
        assert court_height <= available_height

        v_padding = (available_height - court_height) / 2

        court_start_y = self.background_tl.y + v_padding
        court_end_y = court_start_y + court_height

        self.court_tl = Point(court_start_x, court_start_y)
        self.court_br = Point(court_end_x, court_end_y)


    def set_court_keypoints(self) -> None:
        """Sets the court keypoints as a list of Points, stored as an attribute."""
        drawing_keypoints: list[None | Point] = [None] * 14

        # tl ----- (br.x, tl.y)
        # |                   | 
        # |                   | 
        # |                   | 
        # |                   | 
        # |                   | 
        # |                   | 
        # |                   | 
        # (tl.x, br.y) ----- br

        # point 0: Doubles top left
        drawing_keypoints[0] = Point(self.court_tl.x, self.court_tl.y)
        # point 1: Doubles top right
        drawing_keypoints[1] = Point(self.court_br.x, self.court_tl.y)
        # point 2: Doubles bottom left
        drawing_keypoints[2] = Point(self.court_tl.x, self.court_br.y)
        # point 3: Doubles bottom right
        drawing_keypoints[3] = Point(self.court_br.x, self.court_br.y)
        # point 4: Singles top left
        drawing_keypoints[4] = Point(
            self.court_tl.x + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_tl.y
        )
        # point 5: Singles bottom left
        drawing_keypoints[5] = Point(
            self.court_tl.x + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_br.y
        )
        # point 6: Singles top right
        drawing_keypoints[6] = Point(
            self.court_br.x - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_tl.y
        )
        # point 7: Singles bottom right
        drawing_keypoints[7] = Point(
            self.court_br.x - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_br.y
        )
        # point 8: Mini-Court top left
        drawing_keypoints[8] = Point(
            self.court_tl.x + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_tl.y + self.convert_meters_to_pixels(constants.BASELINE_TO_SERVICE)
        )
        # point 9: Mini-Court top right
        drawing_keypoints[9] = Point(
            self.court_br.x - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_tl.y + self.convert_meters_to_pixels(constants.BASELINE_TO_SERVICE)
        )
        # point 10: Mini-Court bottom left
        drawing_keypoints[10] = Point(
            self.court_tl.x + self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_br.y - self.convert_meters_to_pixels(constants.BASELINE_TO_SERVICE)
        )
        # point 11: Mini-Court bottom right
        drawing_keypoints[11] = Point(
            self.court_br.x - self.convert_meters_to_pixels(constants.DOUBLE_ALLEY_WIDTH),
            self.court_br.y - self.convert_meters_to_pixels(constants.BASELINE_TO_SERVICE)
        )
        # point 12: Mini-Court top center
        drawing_keypoints[12] = Point(
            (self.court_tl.x + self.court_br.x) / 2,
            self.court_tl.y + self.convert_meters_to_pixels(constants.BASELINE_TO_SERVICE)
        )
        # point 13: Mini-Court bottom center
        drawing_keypoints[13] = Point(
            (self.court_tl.x + self.court_br.x) / 2,
            self.court_br.y - self.convert_meters_to_pixels(constants.BASELINE_TO_SERVICE)
        )

        assert all(isinstance(keypoint, Point) for keypoint in drawing_keypoints)
        self.drawing_keypoints = drawing_keypoints

    
    def set_court_lines(self):
        """Sets the lines between the keypoints on the drawing, stores as an attribute."""
        lines_between_keypoints = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (2, 3),
            (12, 13),
        ]

        drawing_lines = []
        for (endpoint_1, endpoint_2) in lines_between_keypoints:
            drawing_lines.append(Line(
                self.drawing_keypoints[endpoint_1],
                self.drawing_keypoints[endpoint_2]
            ))

        assert all(isinstance(line, Line) for line in drawing_lines)
        self.drawing_lines = drawing_lines

    
    def mini_court_width(self) -> int:
        """Returns the court width."""
        return self.rectangle_width - 2 * self.padding_court


    def get_homography(self, court_keypoints: list[Point]) -> MatLike:
        """
        Calculates the homography (3 x 3 matrix) that maps points from the original video frame
        to the mini-court display.

        court_keypoints is the real keypoints on the original video.
        """
        src = np.array([[court_keypoints[i].x , court_keypoints[i].y] for i in range(14)], dtype = np.float32)
        dst = np.array([[self.drawing_keypoints[i].x, self.drawing_keypoints[i].y] for i in range(14)], dtype = np.float32)

        H, _ = cv2.findHomography(src, dst)
        return H


    def transform_point(self, original_point: Point, H: MatLike) -> Point:
        """Uses the homography to transform the original_point."""
        np_original_point = np.array([[[original_point.x, original_point.y]]], dtype = np.float32)
        transformed = cv2.perspectiveTransform(np_original_point, H)
        return Point(transformed[0][0][0], transformed[0][0][1])


    def convert_bbox_to_mini_court_coords(
            self,
            player_boxes: list[dict[int, BoundingBox]],
            ball_boxes: list[BoundingBox],
            original_court_keypoints: list[Point]
    ) -> tuple[list[dict[int, Point]], list[Point]]:
        """
        Converts all player bounding boxes and ball bounding boxes from the original
        court to the mini-court display.

        Returns the player boxes and ball boxes transformed. 
        """
        H = self.get_homography(original_court_keypoints)
        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box: BoundingBox = ball_boxes[frame_num]
            ball_position: Point = ball_box.center

            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position: Point = bbox.foot
                output_player_bbox_dict[player_id] = self.transform_point(foot_position, H)

            output_ball_boxes.append(self.transform_point(ball_position, H))
            output_player_boxes.append(output_player_bbox_dict)

        return output_player_boxes, output_ball_boxes


    def get_shot_stats(
            self, 
            ball_shot_frames: list[int],
            player_mini_court_detections: list[dict[int, Point]], 
            ball_mini_court_detections: list[Point], 
            fps: float
        ) -> list[dict[str, int | float]]:
        """
        Gets shot stats. 

        Determines which player hit the ball based on which player is closest to the ball at
        the time it's hit. Calculates the speed of the ball based on the distance the ball travelled
        between the frame it was hit to 20 frames later.
        """
        shot_stats: list[dict] = []
        for i in range(len(ball_shot_frames) - 1):
            start_frame = ball_shot_frames[i]
            end_frame = min(start_frame + 20, ball_shot_frames[i + 1])

            time_seconds = (end_frame - start_frame) / fps

            ball_speed_mps = self.convert_pixels_to_meters(
                euclidean_distance(ball_mini_court_detections[start_frame], ball_mini_court_detections[end_frame])
            ) / time_seconds

            ball_speed_kmh = ball_speed_mps * 3.6

            player_who_hit = min(
                player_mini_court_detections[start_frame].keys(),
                key = lambda player_id: euclidean_distance(
                    ball_mini_court_detections[start_frame],
                    player_mini_court_detections[start_frame][player_id]
                )
            )

            shot_stats.append({
                'frame': start_frame,
                'player_who_hit': player_who_hit,
                'ball_speed_kmh': ball_speed_kmh,
            })

        return shot_stats

    
    def get_player_speed_stats(
            self, 
            player_mini_court_detections: list[dict[int, Point]], 
            fps: float,
            stride: int
        ) -> list[dict[str, int | float]]:
        """
        Given a list of dicts of player detections, determines the players avg speeds for every
        stride of frames.

        Returns a list of dicts of each players speed at every stride frames.
        """
        assert len(player_mini_court_detections) > 0

        speed_stats: list[dict[str, int | float]] = []
        player_ids = list(player_mini_court_detections[0].keys())
        time_seconds = stride / fps

        for frame_num in range(stride, len(player_mini_court_detections), stride):
            stat = {'frame': frame_num}

            for pid in player_ids:
                if pid in player_mini_court_detections[frame_num] and pid in player_mini_court_detections[frame_num - stride]:
                    dist = euclidean_distance(
                        player_mini_court_detections[frame_num - stride][pid],
                        player_mini_court_detections[frame_num][pid]
                    )
                    stat[f'player_{pid}_speed_kmh'] = (self.convert_pixels_to_meters(dist) / time_seconds) * 3.6
                else:
                    stat[f'player_{pid}_speed_kmh'] = 0.0
            speed_stats.append(stat)

        return speed_stats
    