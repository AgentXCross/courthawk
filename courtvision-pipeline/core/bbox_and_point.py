"""Bounding Box and Point dataclass definitions and utility functions.""" 

from dataclasses import dataclass


@dataclass(frozen = True)
class Point:
    x: float
    y: float


@dataclass(frozen = True)
class BoundingBox:
    tl: Point # top left: (x1, y1)
    br: Point # bottom right: (x2, y2)

    @property
    def center(self) -> Point:
        center_x = (self.tl.x + self.br.x) / 2
        center_y = (self.tl.y + self.br.y) / 2
        return Point(center_x, center_y)

    @property
    def foot(self) -> Point:
        center_x = (self.tl.x + self.br.x) / 2
        bottom_y = max(self.tl.y, self.br.y)
        return Point(center_x, bottom_y)

    @property
    def height(self) -> float:
        return abs(self.tl.y - self.br.y)


def euclidean_distance(p1: Point, p2: Point) -> float:
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
        distance = euclidean_distance(point, keypoint)

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