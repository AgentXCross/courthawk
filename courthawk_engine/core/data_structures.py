"""
Data structures and their utility functions.
- Handedness
- CourtSide
- Point
- Line
- BoundingBox
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Handedness(Enum):
    LEFT = "left"
    RIGHT = "right"


class CourtSide(Enum):
    CLOSE = "close"
    FAR = "far"


@dataclass(frozen = True)
class Point:
    """A Point is defined by its x and y-coordinates."""
    x: float
    y: float


    def __add__(self, other: Point) -> Point:
        return Point(
            self.x + other.x,
            self.y + other.y
        )

    def __sub__(self, other: Point) -> Point:
        return Point(
            self.x - other.x,
            self.y - other.y
        )

    def __mul__(self, scalar: float) -> Point:
        return Point(
            x = self.x * scalar,
            y = self.y * scalar,
        )

    def __truediv__(self, scalar: float) -> Point:
        return Point(
            x = self.x / scalar,
            y = self.y / scalar,
        )


@dataclass(frozen = True)
class Line:
    """A Line is defined by 2 Points representing its endpoints."""
    # One end of the line
    end_1: Point

    # Other end of the line
    end_2: Point


@dataclass(frozen = True)
class BoundingBox:
    """A BoundingBox is defined by 2 Points representing its top left and bottom right coordinates."""
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