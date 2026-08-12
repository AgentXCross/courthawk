"""
Video utilities and data structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Video:
    "Represents a video loaded into memory."

    frames: list[np.ndarray]
    fps: float
    num_frames: int

    @classmethod # This method can be called without an instance of the class
    def read_video(cls, video_path: Path) -> Video:
        """Loads a video."""
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
    
        frames: list[np.ndarray] = []
    
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frames.append(frame)

        capture.release()
    
        return cls(
            frames = frames,
            fps = fps,
            num_frames = len(frames)
        )

    def save_video(self, output_path: Path) -> None:
        """Save the video to disk as MP4 (H.264), so it's playable in a browser <video> tag."""
        if not self.frames:
            raise ValueError("Cannot save an empty video.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            self.fps,
            (
                self.frames[0].shape[1],
                self.frames[0].shape[0],
            ),
        )

        for frame in self.frames:
            writer.write(frame)

        writer.release()

        return
    