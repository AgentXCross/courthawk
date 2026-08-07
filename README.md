# CourtHawk: Tennis Player and Ball Tracking

#### Currently restructuring the codebase to be more readable and maintainable. Also building a front and backend.

![Demo](courthawk_engine/data/output_videos/sinner_zverev_output.gif)

CourtHawk is a computer vision system for analyzing tennis footage from the standard TV angle, tracking players and the ball, detecting and classifying shot types, and overlaying stats on a mini-court diagram.

## Project Overview

Given a match video (currently only serves individual points) from the standard angle, the pipeline:
1. Detects and tracks both players and the ball across every frame
2. Identifies court keypoints and maps the court to a mini-court view
3. Detects frames where the ball is struck
4. Classifies each shot as a forehand, backhand, or serve using pose estimation
5. Computes ball speed and player movement speed
6. Renders all of the above back onto the original video

## ML Models and their Purpose

| Model | Purpose |
|---|---|
| YOLOv8x | Detects and tracks players frame-by-frame. |
| YOLOv5 (fine-tuned) | Detects the tennis ball (fine-tuned on tennis ball dataset). Results are not perfect due to the high speed of the ball. Considering replacing with a heat-map model. |
| ResNet-18 (find-tuned) | Predicts 14 court keypoints used to build the homography for the mini-court. |
| MediaPipe Pose Landmarker | Estimates 33-point body pose on cropped player images. Results are not perfect due to the low quality of the video and high speed of player movements. Considering enchancing cropped images using AI image enchancing models. |
| XGBoost Classifier | Classifies pose features into forehand / backhand / serve using outputs from MediaPipe Pose Landmarker. |

## Shot Classification

For each detected ball strike, the pipeline samples up to 7 frames (+ and - 3 around the hit frame), crops the hitting player out of each, and runs MediaPipe pose estimation on the crop. 16 geometric features are extracted from the landmarks (wrist heights, elbow angles, shoulder/hip rotation, torso lean, etc.) and fed to an XGBoost model. The final shot type is decided by majority vote across the 7 frames or labeled as “unknown” if pose landmarks cannot be detected by MediaPipe in any of the 7 frames.

See `pose_estimation/features.txt` for the full feature list.

## File Structure

```
CourtHawk/
├── courthawk_engine/
│   ├── core/                        # Video I/O, Data Structures, Unit Conversions, Constants
│   ├── court_keypoint_detector/     # Predicts and refines court keypoints
│   ├── trackers/                    # YOLOv8 player tracking, YOLOv5 ball tracking
│   ├── minicourt/                   # Mini-court overlay, homography, shot/speed stats
│   ├── pose_estimation/             # MediaPipe pose extraction and XGBoost shot classification
│   ├── renderer/                    # Draws tracking/stat overlays back onto the video
│   ├── models/                      # Trained model weights (gitignored)
│   ├── data/                        # Input/output videos and training data (gitignored, except input/output videos)
│   ├── stubs/                       # Cached detection results (pickle) to skip re-inference for testing
│   ├── development/                 # Notebooks used to build and prototype the pipeline
│   ├── engine.py                    # Public API entry point used by the backend
│   └── main.py                      # Standalone script to run the full pipeline for testing
│
├── backend/                         # API layer (in progress)
└── frontend/                        # UI layer (in progress)
```
