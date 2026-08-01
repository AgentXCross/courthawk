import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from scipy.spatial import distance as scipy_distance


def _detect_lines(image):
    """
    Converts the crop to greyscale, applies a binary threshold to isolate bright
    court lines, then runs the Probabilistic Hough Transform to find line segments.
    Returns a list of segments, each as [x1, y1, x2, y2].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength = 10, maxLineGap = 30)
    if lines is None:
        return []
    lines = np.squeeze(lines)
    if lines.ndim == 1:
        lines = [lines]
    return list(lines)


def _merge_lines(lines):
    """
    Collapses near-duplicate segments (both endpoint pairs within 20px of each other)
    into a single averaged segment. Prevents the same physical court line from being
    counted twice when computing the intersection.
    """
    lines = sorted(lines, key = lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []
    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1:]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    d1 = scipy_distance.euclidean((x1, y1), (x3, y3))
                    d2 = scipy_distance.euclidean((x2, y2), (x4, y4))
                    if d1 < 20 and d2 < 20:
                        line = np.array([int((x1+x3)/2), int((y1+y3)/2), int((x2+x4)/2), int((y2+y4)/2)], dtype=np.int32)
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines

def _line_intersection(line1, line2):
    """
    Returns the (x, y) intersection of the infinite extensions of two segments
    using the parametric line formula. Returns None if the lines are parallel.
    """
    x1, y1, x2, y2 = map(float, line1)
    x3, y3, x4, y4 = map(float, line2)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet18(weights = None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
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

    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]

        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        keypoints = self.refine_keypoints(image, keypoints)

        return keypoints

    def predict_average(self, frames):
        """
        Runs predict on each frame and returns the element-wise average of all
        keypoint arrays.
        """
        all_keypoints = np.array([self.predict(frame) for frame in frames])
        return all_keypoints.mean(axis = 0)

    def refine_keypoints(self, image, keypoints, crop_size = 50):
        """
        Crops an 100x100 window around each initial keypoint, detects lines in that
        crop, and replaces the keypoint with the intersection of those lines if exactly
        2 distinct lines are found and their intersection falls within the crop.
        Keypoints with no clean intersection are left unchanged.
        """
        refined = keypoints.copy()
        img_h, img_w = image.shape[:2]

        for i in range(0, len(keypoints), 2):
            x_ct = int(keypoints[i])
            y_ct = int(keypoints[i + 1])

            x_min = max(x_ct - crop_size, 0)
            x_max = min(x_ct + crop_size, img_w)
            y_min = max(y_ct - crop_size, 0)
            y_max = min(y_ct + crop_size, img_h)

            crop = image[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            lines = _detect_lines(crop)
            if len(lines) > 1:
                lines = _merge_lines(lines)
                if len(lines) == 2:
                    pt = _line_intersection(lines[0], lines[1])
                    if pt is not None:
                        nx, ny = int(pt[0]), int(pt[1])
                        if 0 < nx < crop.shape[1] and 0 < ny < crop.shape[0]:
                            refined[i] = x_min + nx
                            refined[i + 1] = y_min + ny

        return refined

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(
                image,
                str(i // 2), (x, y - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (1, 255, 214),
                2
            )
            cv2.circle(image, (x, y), 10, (1, 255, 214), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)
        return output_frames
