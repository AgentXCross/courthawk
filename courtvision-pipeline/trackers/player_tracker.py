from ultralytics import YOLO
import cv2
import pickle
from utils import measure_distance, get_center_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_bbox(bbox)

            # Calculate distance from each person to all the court keypoints
            dists = []
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                dist = measure_distance(player_center, court_keypoint)
                dists.append(dist)
            
            # Sort and then take the average of the smallest 3 values
            dists.sort()
            avg_dist = sum(dists[:3]) / 3

            distances.append((track_id, avg_dist))
        # Sort and then take the smallest 2 values who are the players
        distances.sort(key = lambda x: x[1])
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        player_detections = []
        
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections
    
    def detect_frame(self, frame):
        results = self.model.track(frame, persist = True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_class_id = box.cls.tolist()[0]
            object_class_name = id_name_dict[object_class_id]
            if object_class_name == "person":
                player_dict[track_id] = result
            
        return player_dict

    def draw_shot_types(self, video_frames, player_detections, ball_shot_frames, shot_types, hitting_player_ids):
        active = {}  # frame_num -> (player_id, shot_type)
        for shot_frame, shot_type, player_id in zip(ball_shot_frames, shot_types, hitting_player_ids):
            for f in range(max(0, shot_frame - 5), min(len(video_frames), shot_frame + 6)):
                active[f] = (player_id, shot_type)

        for frame_num, (frame, player_dict) in enumerate(zip(video_frames, player_detections)):
            if frame_num in active:
                player_id, shot_type = active[frame_num]
                bbox = player_dict.get(player_id)
                if bbox is not None:
                    cv2.putText(frame,
                                shot_type, (int(bbox[0]), int(bbox[3] + 25)),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                0.9,
                                (140, 0, 236),
                                2
                    )
        return video_frames

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, 
                            f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9,
                            (140, 0, 236),
                            2
                )
                cv2.rectangle(
                    frame, 
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (140, 0, 236),
                    2
                )
            output_video_frames.append(frame)
        return output_video_frames

