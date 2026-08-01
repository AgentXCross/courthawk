from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # Interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill() # Ensure the first frame has a detection

        return df_ball_positions.to_numpy().tolist()
    
    def get_ball_shot_frames(self, ball_positions, player_detections):
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        df_ball_positions['mid_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2

        # Build player position DataFrame, one row per frame, one column set per player
        player_ids = list(player_detections[0].keys())
        rows = []
        for frame in player_detections:
            row = {}
            for pid, bbox in frame.items():
                row[f'p{pid}_cx'] = (bbox[0] + bbox[2]) / 2
                row[f'p{pid}_cy'] = (bbox[1] + bbox[3]) / 2
            rows.append(row)
        df_players = pd.DataFrame(rows).interpolate().bfill()

        bx = df_ball_positions['mid_x'].values
        by = df_ball_positions['mid_y'].values

        # Distance from ball to each player per frame
        dists = []
        for pid in player_ids:
            dx = bx - df_players[f'p{pid}_cx'].values
            dy = by - df_players[f'p{pid}_cy'].values
            dists.append((dx ** 2 + dy ** 2) ** 0.5)

        # Closest player distance per frame
        min_dist = pd.Series(np.minimum(*dists))

        # Find local minima
        peaks, _ = find_peaks(-min_dist, prominence = 50, distance = 15)
        return list(peaks)

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        ball_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections
    
    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.10)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
            break
            
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, bbox in zip(video_frames, ball_detections):
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.putText(frame,
                            "Ball", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9,
                            (1, 255, 214),
                            2
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (1, 255, 214), 2)
            output_video_frames.append(frame)
        return output_video_frames

