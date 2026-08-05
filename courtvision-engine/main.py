from trackers import PlayerTracker, BallTracker
from court_keypoint_detector import CourtKeypointDetector
from minicourt import MiniCourt
from pose_estimation import PoseEstimator, ShotClassifier

from core import Video

import cv2

def main():
    # Read Video
    input_video_path = "input-videos/sinner_zverev.mp4"
    video = Video.read_video(input_video_path)

    # Detect and Track: Players and Ball
    player_tracker = PlayerTracker(model_path = "models/yolov8x.pt")
    ball_tracker = BallTracker(model_path = "models/yolo5_last.pt")
    player_detections = player_tracker.detect_frames(
        video.frames,
        read_from_stub = False,
        stub_path = "tracker_stubs/player_detections_sinner_zverev.pkl"
    )
    ball_detections = ball_tracker.detect_frames(
        video.frames,
        read_from_stub = False,
        stub_path = "tracker_stubs/ball_detections_sinner_zverev.pkl"
    )
    ## ball_detections is a list of lists where each element is [x1, y1, x2, y2]
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Detect Court Keypoints
    court_model_path = "models/keypoints_model.pt"
    court_line_detector = CourtLineDetector(court_model_path)
    ## court_keypoints is a list of 28 values
    ## Every 2 values corresponds to one keypoint on the court
    n = len(video.frames)
    ## Predict keypoints on 4 separate frames and average
    sample_frames = [video.frames[i] for i in [0, n // 3, 2 * n // 3, n - 1]]
    court_keypoints = court_line_detector.predict_average(sample_frames)

    # Choose the 2 Players
    ## player_detections is a list of dictionaries
    ## player_ids is a tuple with 2 elements representing the ids of the 2 players
    ## Each dictionairy corresponds to a frame, player_id's are the keys
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    player_ids = player_detections[0].keys()

    # Mini-Court
    mini_court = MiniCourt(video.frames[0])

    # Detect Ball Shots
    ## ball_shot_frames is a list of frames where the ball was shot
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections, player_detections)

    # Classify Shot Types
    pose_estimator = PoseEstimator(model_path = "models/pose_landmarker.task")
    shot_classifier = ShotClassifier(model_path = "models/shot_classifier.pkl")
    ## shot_types is a list of ["forehand", "backhand", "serve", "unknown"] corresponding to each shot from the ball_shot_frames
    ## hitting_player_ids is a list of length len(shot_types) and len(ball_shot_frames) containg the ID of which player hit each shot
    shot_types, hitting_player_ids = pose_estimator.classify_shots(video.frames, ball_shot_frames, player_detections, ball_detections, shot_classifier)

    # Convert Positions to Mini-Court Positions
    ## player_mini_court_detections is a list of dict. Keys are player_ids and values are coordinates on the mini-court
    ## ball_mini_court_detections is a list of tuples. 
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bbox_to_mini_court_coords(player_detections, ball_detections, court_keypoints)

    # Shot Stats
    ## player_shots_data is a list of dicts with keys: frame, player_who_hit, ball_speed_kmh
    player_shots_data = mini_court.get_shot_stats(ball_shot_frames, player_mini_court_detections, ball_mini_court_detections, video.fps)

    # Player Speed Stats
    ## player_speed_stats is a list of dicts with keys: frame, player_1_speed_kmh, player_2_speed_kmh
    player_speed_stats = mini_court.get_player_speed_stats(player_mini_court_detections, video.fps, 20)

    # Draw Outputs
    render_functions = []
    output_video_frames = player_tracker.draw_bboxes(video.frames, player_detections)
    output_video_frames = player_tracker.draw_shot_types(output_video_frames, player_detections, ball_shot_frames, shot_types, hitting_player_ids)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_players_on_mini_court(output_video_frames, player_mini_court_detections, (140, 0, 236))
    output_video_frames = mini_court.draw_ball_on_mini_court(output_video_frames, ball_mini_court_detections, (1, 255, 214))
    output_video_frames = mini_court.draw_stats(output_video_frames, player_shots_data, player_speed_stats, player_ids)

    # Draw Frame Number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame #{i + 1}", (50, 170), cv2.FONT_HERSHEY_TRIPLEX, 2, (1, 255, 214), 5)
        
    # Save Outputs
    output_video = Video()
    output_video.save_video("output-videos/sinner_zverev_output.avi")

if __name__ == "__main__":
    main()