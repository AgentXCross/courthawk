# CourtHawk

Computer vision system for tennis that detects and tracks players and ball, estimates player poses, classifies shots, detects court geometry, and projects movements onto a bird's-eye-view using homography.

<p align="center">
  <img src="courthawk_engine/data/output_videos/sinner_zverev_output.gif" width="70%">
</p>

## How It Works

### Input

As input, CourtHawk expects a video from the standard TV angle of 1 tennis point starting from the moment the server hits the ball and ending as soon as the point finishes (Currently working on a method to accept an entire tennis match and determining how to split up points). Both tennis players should be visible from the very first frame. The court geometry is assumed to be constant during the duration of the video because the system only runs inference on the court keypoint model using a few arbitrary frames. 

### Player Tracker

Player tracking is done by a pretrained `YOLOv8x` model. The model is directed to only track the "person" class (index 0). The ball boys and chair umpire are likely to also be tracked by the model, so to select the actual players, we calculate the sum of the Euclidean Distances of every detected "person" to their nearest 3 court keypoints (to be computed later), and select the 2 persons with the minimum values.

The player detections are returned as bounding boxes for every frame. A bounding box is defined using 2 points: The top-left point and the bottom-right point.

There is a chance the model fails to detect the players during certain frames. To ensure this doesn't crash any computations, missing detections are linearly interpolated between the nearest known detections. Any remaining missing detections at the beginning or end of the video are backward-filled or forward-filled, respectively.

### Ball Tracker

Ball tracking is done by a fine-tuned `YOLOv5` model. The model is fine-tuned with a public dataset from RoboFlow consisting of 428 training set images, 100 validation set images, and 50 test set images, all containing corresponding bounding boxes for the tennis balls. The dataset can be accessed at: [RoboFlow Ball Detection Dataset](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6). The model is restriced to track only 1 tennis ball. 

Since the ball moves at a very high speed and is frequently blocked by players, missing detections are also linearly interpolated between the nearest known detections. Any remaining missing detections at the beginning or end of the video are backward-filled or forward-filled.

Currently working replacing YOLO for ball tracking and instead adapting a model that produces a heatmap of the ball's location over the entire image (model would be analogous to a semantic segmentation model).

### Court Keypoints

Court keypoint detection is done using a fine-tuned `ResNet-18` model. The model is fine-tuned with a public dataset consisting of 8841 images, separated into 75% training and 25% validation. The dataset can be accessed at: [Tennis Court Detector GitHub](https://github.com/yastrebksv/TennisCourtDetector). The model is trained to predict 28 independent values, corresponding to the 14 keypoints on a court. Thus, this task is treated as regression.

For each predicted keypoint, a crop is made around it, centered about the prediction. From the crop, we threshold the image to find bright pixels and apply the Probabilistic Hough Transform to detect line segments. Duplicated line segments within a distance threshold on both ends are merged together. If only 2 line segments remain, we compute their intersection, and if it exists, we replace the prediction with the intersections. If any of the conditions fail, we keep the original prediction.

The court keypoint locations stay constant during the duration of the video. So, we compute the court keypoints for a few random frames and average their result as the final keypoints.

A better method to consider is to instead use a model that predicts 14 heatmaps, one for each keypoint. This way, we are not predicting 28 independent values. Something important to consider is to not have only the ball pixels have value 1 and all other pixels have value 0. The Gaussian heatmap value at pixel $(x, y)$, with the true keypoint at $(x_k, y_k)$ is given by 

$$H(x, y) = \exp(- \frac{(x - x_k)^2 + (y - y_k)^2}{2 \sigma^2})$$

where $\sigma$ determines how spread out the distribution is. The distribution depends on the value of $\sigma$, but an example distribution is 

$$
\begin{bmatrix}
0.00 & 0.01 & 0.04 & 0.01 & 0.00 \\
0.01 & 0.14 & 0.37 & 0.14 & 0.01 \\
0.04 & 0.37 & 1.00 & 0.37 & 0.04 \\
0.01 & 0.14 & 0.37 & 0.14 & 0.01 \\
0.00 & 0.01 & 0.04 & 0.01 & 0.00
\end{bmatrix}
$$

### Mini-Court

The minicourt provides a bird-eye's view of the point. The player and ball positions are approxiamated since it is impossible to determine the height of the tennis ball given only a single angle of the point. 

We first extract the foot position of the players and the center position of the ball on the actual court. To translate the positions onto minicourt, we calculate the homography. The homography is a matrix $H \in \mathbb{R}^{3 \times 3}$ that maps points from the original image plane to corresponding points on the mini-court plane. The homography requires at least 4 points to calculate as it has 8 degrees of freedom.

$$
\begin{bmatrix}
x_{mini} \\
y_{mini} \\
1
\end{bmatrix}
\sim
\begin{bmatrix}
x' \\
y' \\
z' 
\end{bmatrix}
= 
\begin{bmatrix}
h_1 & h_2 & h_3 \\
h_4 & h_5 & h_6 \\
h_7 & h_8 & h_9
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$

### Shot Detection

To detect when a shot occurs, we start by graphing the minimum distance between the ball and either player over time. When a player hits the ball, this distance should briefly approach a minimum before increasing as the ball travels toward their opponent. Therefore, local minima in the distance curve are treated as potential shots.

To reduce false detections caused by noise in the ball and player tracking, we restrict the minimum time between consecutive shots and the prominence of each local minimum. This filters out small fluctuations that are unlikely to be actual shots.

<p align="center">
  <img src="courthawk_engine/data/assets/ball_hit.png" width="70%">
</p>

### Shot Classification

When we detected the shots using the algorithm above, we also determined which player hit the ball. Since we have the bounding boxes of the player across all the frames, we can crop out the player in the frames near the shot frame (specifically we crop out the player in 7 frames $[\text{shot frame} - 3, \text{shot frame} + 3]$). 

For each cropped frame, we run MediaPipe's Pose Estimation model. MediaPipe detects 33 body landmarks, including the shoulders, elbows, wrists, hips, knees, and ankles. Each landmark contains an estimated $(x, y)$ position. 

Rather than use the landmark positions directly, we extract a smaller set of features relevant to tennis that describe the players pose. The positions are centered at the players mid-shoulder position and then normalized by their torso length so that the representation is less sensitive to the player's size or distance from camera.

1. Max wrist height: Higher of two wrists relative to the mid-shoulder
2. Vertical wrist separation: Absolute difference between wrist y-positions
3. Horizontal wrist separation: Absolute difference between wrist x-positions
4. More extended arm elbow angle: Elbow angle (wrist -> elbow -> shoulder) of the arm with greater shoulder to wrist distance
5. Less extended arm elbow anlge: Elbow angle of the other arm
6. Arm extension asymmetry: Absolute difference in shoulder to wrist distance between the 2 arms
7. Shoulder rotation angle: Angle of the shoulder line relative to the horizontal
8. Hip rotation angle: Angle of the hip line relative to the horizontal
9. Hip-shoulder twist: Shoulder rotation angle - hip rotation angle
10. Torso lean: Angle of the mid-shoulder to mid-hip relative to the vertical
11. Wrist x-coordinate produce: right wrist x-position * left wrist x-position
12. Mininum cross body shoulder-to-wrist distance: min(dist(left wrist, right shoulder), dist(right wrist, left shoulder))
13. Angle between forearm vectors: Angle between (left elbow to left wrist) and (right elbow to right wrist)
14. Elbow angle difference: Absolute difference between left elbow and right elbow angle
15. Wrist average height - elbow average height
16. Legs cross: Binary feature
17. Right wrist offset: (Right wrist x-position - mid-shoulder x-position) / shoulder width
18. Left wrist offset: (Left wrist x-position - mid-shoudler x-position) / shoulder width

The extracted feature vector is used to train a `Softmax/Multinomial Logistic Regression` model to classify shots for 3 classes: forehand, backhand, and serve. The training dataset was extracted from videos by hand. The system applies the model for all 7 frames and decides by a majority vote.

Due to the small size of players, the cropped box contains a very low quality image, making it difficult for MediaPipe to determine the 33 landmarks. We should consider using methods like Real-ESRGAN to upscale the images before inference. Additionally, we should consider training 2 separate models for the players on the 2 sides of the court. 

### Speed Statistics

For each player, we keep track of 2 statistics:
- Last Shot Speed
- Player Current Speed

The last shot speed is calculated by using when the player last hit the ball as a start frame and a few frames later as the end frame. We can calculated the speed of the shot using $s_{avg} = \frac{\Delta d}{\Delta t}$ from start frame to end frame. The player current speed is calculated every few frames using the same formula.

### Output

Running `courthawk_engine/main.py` will produce a video with all of the above overlayed on the orginal video. When running the system using the deployed app (coming soon), the results will be displayed next to the video.

## File Structure

```
CourtHawk/
├── courthawk_engine/
│   ├── core/                        # Video I/O, Data Structures, Unit Conversions, Constants
│   ├── court_keypoint_detector/     # Predicts and refines court keypoints
│   ├── trackers/                    # YOLOv8 player tracking, YOLOv5 ball tracking
│   ├── minicourt/                   # Mini-court overlay, homography, shot/speed stats
│   ├── pose_estimation/             # MediaPipe pose extraction and shot classification
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
