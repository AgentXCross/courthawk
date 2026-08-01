import cv2
from utils import (
    convert_pixel_distance_to_meters, 
    convert_meters_to_pixel_distance,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_bbox,
    measure_distance
)
import court_dimension_constants
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.rectangle_width = 380
        self.rectangle_height = 720
        self.buffer = 10
        self.padding_court = 60

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_keypoints()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(
            meters,
            court_dimension_constants.DOUBLE_LINE_WIDTH,
            self.court_width
        )

    def convert_pixels_to_meters(self, pixels):
        return convert_pixel_distance_to_meters(
            pixels,
            court_dimension_constants.DOUBLE_LINE_WIDTH,
            self.court_width
        )

    def set_court_keypoints(self):
        drawing_key_points = [0] * 28
        # point 0: Doubles top left
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1: Doubles top right
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2: Doubles bottom left
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = int(self.court_start_y) + self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_NET * 2)
        # point 3: Doubles bottom right
        drawing_key_points[6] = drawing_key_points[0] + self.court_width
        drawing_key_points[7] = drawing_key_points[5]
        # point 4: Singles top left
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[9] = drawing_key_points[1]
        # point 5: Singles bottom left
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[11] = drawing_key_points[5]
        # point 6: Singles top right
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[13] = drawing_key_points[3]
        # point 7: Singles bottom right
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(court_dimension_constants.DOUBLE_ALLEY_WIDTH)
        drawing_key_points[15] = drawing_key_points[7]
        # point 8: Mini-Court top left
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_SERVICE)
        # point 9: Mini-Court top right
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(court_dimension_constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # point 10: Mini-Court bottom left
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_SERVICE)
        # point 11: Mini-Court bottom right
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(court_dimension_constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # point 12: Mini-Court top center
        drawing_key_points[24] = (drawing_key_points[16] + drawing_key_points[18]) // 2
        drawing_key_points[25] = drawing_key_points[17]
        # point 13: Mini-Court bottom center
        drawing_key_points[26] = (drawing_key_points[20] + drawing_key_points[22]) // 2
        drawing_key_points[27] = drawing_key_points[21]
        self.drawing_key_points = drawing_key_points
    
    def set_court_lines(self):
        self.lines = [
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

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_width = self.court_end_x - self.court_start_x

        # Vertically center the court within the background rectangle
        court_height = self.convert_meters_to_pixels(court_dimension_constants.BASELINE_TO_NET * 2)
        available_height = self.end_y - self.start_y
        v_padding = (available_height - court_height) / 2
        self.court_start_y = self.start_y + v_padding
        self.court_end_y = self.court_start_y + court_height

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        # Anchor to top-right corner, clamp to frame bounds
        self.end_x = frame.shape[1] - self.buffer
        self.start_x = self.end_x - self.rectangle_width
        self.start_y = self.buffer
        self.end_y = min(self.start_y + self.rectangle_height, frame.shape[0] - self.buffer)

    def draw_court(self, frame):
        # Draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 3)

        # Draw Net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 255, 255), 3)

        # Draw Keypoints on Mini-Court
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
        return frame

    def draw_background_rectangle(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 0), cv2.FILLED)
        return cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_width
    
    def get_court_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coords(
            self, 
            object_position, 
            closest_keypoint, 
            closest_keypoint_index, 
            player_height_in_pixels,
            player_height_in_meters
    ):
        dist_from_keypoint_x_pixels, dist_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_keypoint)
        
        # Convert pixel distance to meters
        dist_from_keypoint_x_meters = convert_pixel_distance_to_meters(
            dist_from_keypoint_x_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )
        dist_from_keypoint_y_meters = convert_pixel_distance_to_meters(
            dist_from_keypoint_y_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )

        # Convert to mini-court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(dist_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(dist_from_keypoint_y_meters)

        closest_mini_court_keypoint = (self.drawing_key_points[closest_keypoint_index * 2], self.drawing_key_points[closest_keypoint_index * 2 + 1])

        mini_court_player_position = (
            closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_court_keypoint[1] + mini_court_y_distance_pixels
        )
        return mini_court_player_position
    
    def get_homography(self, court_keypoints):
        src = np.array([[court_keypoints[i * 2], court_keypoints[i * 2 + 1]] for i in range(14)], dtype=np.float32)
        dst = np.array([[self.drawing_key_points[i * 2], self.drawing_key_points[i * 2 + 1]] for i in range(14)], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def transform_point(self, point, H):
        pt = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)
        return (transformed[0][0][0], transformed[0][0][1])

    def convert_bbox_to_mini_court_coords(
            self,
            player_boxes,
            ball_boxes,
            original_court_keypoints
    ):
        H = self.get_homography(original_court_keypoints)
        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num]
            ball_position = get_center_bbox(ball_box)

            output_player_bbox_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                output_player_bbox_dict[player_id] = self.transform_point(foot_position, H)

            output_ball_boxes.append(self.transform_point(ball_position, H))
            output_player_boxes.append(output_player_bbox_dict)

        return output_player_boxes, output_ball_boxes

    def draw_players_on_mini_court(self, frames, player_positions, color):
        for frame_num, frame in enumerate(frames):
            for _, position in player_positions[frame_num].items():
                cv2.circle(frame, (int(position[0]), int(position[1])), 10, color, -1)
        return frames

    def draw_ball_on_mini_court(self, frames, ball_positions, color):
        for frame_num, frame in enumerate(frames):
            position = ball_positions[frame_num]
            cv2.circle(frame, (int(position[0]), int(position[1])), 10, color, -1)
        return frames
    
    def get_shot_stats(self, ball_shot_frames, player_mini_court_detections, ball_mini_court_detections, fps):
        shot_stats = []
        for i in range(len(ball_shot_frames) - 1):
            start_frame = ball_shot_frames[i]
            end_frame = min(start_frame + 20, ball_shot_frames[i + 1])
            time_seconds = (end_frame - start_frame) / fps

            ball_speed_kmh = self.convert_pixels_to_meters(
                measure_distance(ball_mini_court_detections[start_frame], ball_mini_court_detections[end_frame])
            ) / time_seconds * 3.6

            player_who_hit = min(
                player_mini_court_detections[start_frame].keys(),
                key = lambda pid: measure_distance(
                    ball_mini_court_detections[start_frame],
                    player_mini_court_detections[start_frame][pid]
                )
            )
            shot_stats.append({
                'frame': start_frame,
                'player_who_hit': player_who_hit,
                'ball_speed_kmh': ball_speed_kmh,
            })
        return shot_stats
    
    def get_player_speed_stats(self, player_mini_court_detections, fps, stride):
        speed_stats = []
        player_ids = list(player_mini_court_detections[0].keys())
        time_seconds = stride / fps

        for frame_num in range(stride, len(player_mini_court_detections), stride):
            stat = {'frame': frame_num}
            for pid in player_ids:
                if pid in player_mini_court_detections[frame_num] and pid in player_mini_court_detections[frame_num - stride]:
                    dist = measure_distance(
                        player_mini_court_detections[frame_num - stride][pid],
                        player_mini_court_detections[frame_num][pid]
                    )
                    stat[f'player_{pid}_speed_kmh'] = (self.convert_pixels_to_meters(dist) / time_seconds) * 3.6
                else:
                    stat[f'player_{pid}_speed_kmh'] = 0.0
            speed_stats.append(stat)
        return speed_stats
    
    def draw_stats(self, frames, player_shots_data, player_speed_stats, player_ids):
        p1_id, p2_id = list(player_ids)[0], list(player_ids)[1]
        shot_idx  = 0
        speed_idx = 0
        p1_shot_speed = p2_shot_speed = None
        p1_speed = p2_speed = None
        p1_all_speeds = []
        p2_all_speeds = []

        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.6
        thickness = 1
        x, y = 30, 185
        label_w = 260
        val_w = 130
        row_h = 35
        n_rows = 4
        box_w = label_w + val_w * 2
        box_h = row_h * n_rows + 15

        output_frames = []
        for frame_num, frame in enumerate(frames):
            # track last shot speed per player
            while shot_idx < len(player_shots_data) and player_shots_data[shot_idx]['frame'] <= frame_num:
                s = player_shots_data[shot_idx]
                if s['player_who_hit'] == p1_id:
                    p1_shot_speed = s['ball_speed_kmh']
                else:
                    p2_shot_speed = s['ball_speed_kmh']
                shot_idx += 1

            # accumulate for running average
            while speed_idx < len(player_speed_stats) and player_speed_stats[speed_idx]['frame'] <= frame_num:
                s = player_speed_stats[speed_idx]
                p1_speed = s[f'player_{p1_id}_speed_kmh']
                p2_speed = s[f'player_{p2_id}_speed_kmh']
                p1_all_speeds.append(p1_speed)
                p2_all_speeds.append(p2_speed)
                speed_idx += 1

            def display_val(val): return f"{val:.1f} km/h" if val is not None else "N/A"

            p1_avg = sum(p1_all_speeds) / len(p1_all_speeds) if p1_all_speeds else None
            p2_avg = sum(p2_all_speeds) / len(p2_all_speeds) if p2_all_speeds else None

            rows = [
                ("", f"Player {p1_id}", f"Player {p2_id}"),
                ("Last Shot Speed", display_val(p1_shot_speed), display_val(p2_shot_speed)),
                ("Player Current Speed", display_val(p1_speed), display_val(p2_speed)),
                ("Avg Player Speed", display_val(p1_avg), display_val(p2_avg)),
            ]

            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), cv2.FILLED)
            frame = cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)

            for r, (label, v1, v2) in enumerate(rows):
                row_y = y + 10 + r * row_h
                cv2.putText(frame, label, (x + 5, row_y + 20), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(frame, v1, (x + 5 + label_w, row_y + 20), font, font_scale, (255, 255, 255), thickness)
                cv2.putText(frame, v2, (x + 5 + label_w + val_w, row_y + 20), font, font_scale, (255, 255, 255), thickness)

            output_frames.append(frame)
        return output_frames