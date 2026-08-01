"""Bounding Box and Point dataclass definitions and utility functions.""" 

from dataclasses import dataclass


@dataclass(frozen = True)
class Point:
    x: float
    y: float


@dataclass(frozen = True)
class BoundingBox:
    # top left
    x1: float 
    y1: float

    # bottom right
    x2: float
    y2: float 

    @property
    def center(self) -> Point:
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        return Point(center_x, center_y)

    @property
    def foot(self) -> Point:
        center_x = (self.x1 + self.x2) / 2
        bottom_y = max(self.y1, self.y2)
        return Point(center_x, bottom_y)

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y2)


def measure_distance(p1: Point, p2: Point) -> float:
    """Euclidean distance."""
    a = p2.x - p1.x
    b = p2.y - p1.y
    c = a * a + b * b
    return c ** 0.5


def closest_keypoint_index(point: Point, keypoints: list[Point], keypoint_indices: list[int]) -> int:
    """Returns index of the closest keypoint to point."""
    closest_distance = float('inf')
    keypoint_index = 0

    for index in keypoint_indices:
        keypoint = keypoints[index]
        distance = measure_distance(point, keypoint)

        if distance < closest_distance:
            closest_distance = distance
            keypoint_index = index

    return keypoint_index


def measure_offset(from_point: Point, to_point: Point) -> Point:
    """Returns offset (dx, dy)."""
    return Point(
        x = to_point.x - from_point.x,
        y = to_point.y - from_point.y,
    )