"""
CourtKeypointDetector class.

Determine the 14 court keypoints in a tennis court.
"""

from dataclasses import dataclass

import torch
import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import numpy as np
from pathlib import Path

from core import (
    Point,
    Line,
    euclidean_distance,
)


def _detect_lines(crop: np.ndarray) -> list[Line]:
    """
    Converts the crop to greyscale and applies a binary threshold to isolate bright
    court lines. 
    Runs the Probabilistic Hough Transform to detect line segments.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength = 10, maxLineGap = 30)

    if lines is None:
        return []
    
    lines = np.squeeze(lines)
    if lines.ndim == 1:
        lines = [lines]

    return [
        Line(
            end_1 = Point(x1, y1),
            end_2 = Point(x2, y2)
        )
        for x1, y1, x2, y2 in lines
    ]


def _merge_lines(
        lines: list[Line],
        merge_distance: int = 30 # Pixel tolerance for lines that are considered the same
    ) -> list[Line]:
    """
    Merge line segments whose ends are merge_distance pixels close together.
    Duplicate segments are replaced with a single segment that has endpoints equal to
    the average of the duplicate segments.
    """
    if merge_distance < 0:
        raise ValueError("merge_distance must be non-negative")
    
    merged_lines: list[Line] = []
    used = [False] * len(lines) # Boolean mask of the lines we have already merged

    for index, line in enumerate(lines):
        if used[index]:
            continue

        end_1_x_sum = line.end_1.x
        end_1_y_sum = line.end_1.y
        end_2_x_sum = line.end_2.x
        end_2_y_sum = line.end_2.y
        merged_count = 1

        used[index] = True

        for other_index in range(index + 1, len(lines)):
            if used[other_index]:
                continue

            other_line = lines[other_index]

            current_merged_line = Line(
                end_1 = Point(end_1_x_sum / merged_count, end_1_y_sum / merged_count),
                end_2 = Point(end_2_x_sum / merged_count, end_2_y_sum / merged_count)
            )

            # Check both endpoint orientations since we can't control which end is which
            same_orientation = (
                euclidean_distance(current_merged_line.end_1, other_line.end_1) < merge_distance
                and euclidean_distance(current_merged_line.end_2, other_line.end_2) < merge_distance
            )

            reverse_orientation = (
                euclidean_distance(current_merged_line.end_1, other_line.end_2) < merge_distance 
                and euclidean_distance(current_merged_line.end_2, other_line.end_1) < merge_distance
            )

            if same_orientation:
                end_1_x_sum += other_line.end_1.x
                end_1_y_sum += other_line.end_1.y
                end_2_x_sum += other_line.end_2.x
                end_2_y_sum += other_line.end_2.y
            elif reverse_orientation:
                end_1_x_sum += other_line.end_2.x
                end_1_y_sum += other_line.end_2.y
                end_2_x_sum += other_line.end_1.x
                end_2_y_sum += other_line.end_1.y
            else:
                continue

            merged_count += 1
            used[other_index] = True

        merged_lines.append(
            Line(
                Point(end_1_x_sum / merged_count, end_1_y_sum / merged_count),
                Point(end_2_x_sum / merged_count, end_2_y_sum / merged_count)
            )
        )

    assert(len(merged_lines) <= len(lines))

    return merged_lines


def _point_of_intersection(line1: Line, line2: Line) -> Point | None:
    """
    Returns the (x, y) Point intersection of the two Line segments.
    Returns None if the lines are parallel or coincident.

    The intersection point of 2 lines defined by endpoints (x1, y1, x2, y2) and (x3, y3, x4, y4)
    is given by:

    Parametric forms of the lines are
        (x, y) = (x1, y1) + u(x2 - x1, y2 - y1)
        (x, y) = (x3, y3) + v(x4 - x3, y4 - y3)

        For line 1: x = x1 + u(x2 - x1), y = y1 + u(y2 - y1)
        For line 2: x = x3 + v(x4 - x3), y = y3 + v(y4 - y3)

    Thus the intersection is given solving:
        x1 + u(x2 - x1) = x3 + v(x4 - x3)
        y1 + u(y2 - y1) = y3 + v(y4 - y3)
    
    Let A = x2 - x1, B = x4 - x3, C = x3 - x1
    Let D = y2 - y1, E = y4 - y3, F = y3 - y1

    Au - Bv = C
    Du - Ev = F

    Multiply the first equation by E: AEu - BEv = CE
    Multiply the second equation by B: BDu - BEv = BF 

    Subtracting the two equations
    AEu - BDu = CE - BF
    u = (CE - BF) / (AE - BD)

    Then x = x1 + Au, y = y1 + Du.
    """
    x1, y1, x2, y2 = line1.end_1.x, line1.end_1.y, line1.end_2.x, line1.end_2.y
    x3, y3, x4, y4 = line2.end_1.x, line2.end_1.y, line2.end_2.x, line2.end_2.y

    A = x2 - x1
    B = x4 - x3
    C = x3 - x1
    D = y2 - y1
    E = y4 - y3
    F = y3 - y1

    denom = A * E - B * D
    if abs(denom) < 1e-6:
        return None

    u = (C * E - B * F) / denom
    
    return Point(x1 + A * u, y1 + D * u)


class CourtKeypointDetector:
    """
    CourtKeypointDetector detects the 14 keypoints of a tennis court.
    A ResNet model first predicts the 28 values corresponding to the 14 coordinates.
    Then we refine the keypoints by detecting the white lines around its predictions.

    Keypoint Indices (Each keypoint is (x, y) where (0, 0) is the top left corner):
        0: Doubles top left
        1: Doubles top right
        2: Doubles bottom left
        3: Doubles bottom right
        4: Singles top left
        5: Singles bottom left
        6: Singles top right
        7: Singles bottom right
        8: Mini-court top left
        9: Mini-court top right
        10: Mini-court bottom left
        11: Mini-court bottom right
        12: Mini-court top center
        13: Mini-court bottom center
    """
    def __init__(self, model_path: Path):
        self.model = models.resnet18(weights = None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2) # Edit the last layer to predict 28 values
        state_dict = torch.load(model_path, map_location = 'cpu')
        self.model.load_state_dict(state_dict)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])
        ])


    def predict(self, image: np.ndarray) -> list[Point]:
        """Predicts the keypoints. Returns as a list of Points."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)
        # unsqueeze allows the image to go from shape (3, 224, 224) to (1, 3, 224, 224)
        # since the model expects the batch size dimension

        with torch.no_grad():
            outputs = self.model(image_tensor) # shape: (1, 28)

        keypoints = outputs.squeeze().cpu().numpy() # array of 28
        original_h, original_w = image.shape[:2]

        keypoints[::2] *= original_w / 224.0 # scale the x-coords
        keypoints[1::2] *= original_h / 224.0 # scale the y-coords

        keypoints = [ # list[Point]
            Point(keypoints[i], keypoints[i + 1])
            for i in range(0, len(keypoints), 2)
        ]

        keypoints = self.refine_keypoints(image, keypoints)

        return keypoints


    def predict_average(self, frames: list[np.ndarray]) -> list[Point]:
        """
        Calls predict on each frame and returns the element-wise average of all
        keypoint predictions as a list of Points.

        predict_average is the function called by the user. It calls predict which in turn
        calls refine_keypoints.
        """
        assert len(frames) > 0
        all_keypoints: list[list[Point]] = [self.predict(frame) for frame in frames]
        num_frames = len(frames)
        num_keypoints = len(all_keypoints[0])

        assert(len(all_keypoints[0]) == 14)
        keypoint_sums = [Point(0, 0)] * len(all_keypoints[0])

        for i in range(num_frames):
            for j in range(num_keypoints):
                keypoint_sums[j] = keypoint_sums[j] + all_keypoints[i][j]

        return [
            point / num_frames for point in keypoint_sums
        ]

            
    def refine_keypoints(
            self, 
            image: np.ndarray, 
            keypoints: list[Point],
            crop_size: int = 50
    ) -> list[Point]:
        """
        Crops an crop_size * 2 x crop_size * 2 window around each initial keypoint.
        Detects lines in that crop and replaces the keypoint with the intersection of those lines if exactly
        2 distinct lines are found and their intersection falls within the crop.

        Keypoints with no clean intersection are left unchanged.

        Produces output to indicate if a replacement is made or not.
        """
        assert(len(keypoints) == 14)
        
        refined_keypoints: list[Point] = keypoints.copy()
        img_height, img_width = image.shape[:2] # numpy arrays are row, col, channel

        for i in range(len(keypoints)):
            refined = False
            x_center = int(keypoints[i].x)
            y_center = int(keypoints[i].y)

            x_min = max(0, x_center - crop_size)
            x_max = min(img_width, x_center + crop_size)
            y_min = max(0, y_center - crop_size)
            y_max = min(img_height, y_center + crop_size)

            cropped = image[y_min:y_max, x_min:x_max]
            if cropped.size == 0: # Nothing was cropped out
                print(f"Keypoint #{i}: Nothing cropped out around prediction.")
                continue

            lines: list[Line] = _detect_lines(cropped)
            if len(lines) > 1: # We require at least 2 lines to find a POI
                lines = _merge_lines(lines, 30)
                if len(lines) == 2: # Require exactly 2 lines after merging to determine an intersection
                    poi = _point_of_intersection(lines[0], lines[1])
                    if poi is not None: # Can be None if lines are parallel or coincident
                        if 0 < poi.x < cropped.shape[1] and 0 < poi.y < cropped.shape[0]:
                            refined_keypoint = Point(x_min + poi.x, y_min + poi.y)
                            refined = True

            if refined:
                refined_keypoints[i] = refined_keypoint
                print(f"Keypoint #{i}: Successfully refined.")
            else:
                print(f"Keypoint #{i}: Not refined. Original prediction holds.")

        print("\n")

        assert(len(refined_keypoints) == 14)
        return refined_keypoints
