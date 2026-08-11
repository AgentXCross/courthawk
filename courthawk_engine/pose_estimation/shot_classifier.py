"""Runs the shot classification model."""

import pickle
import numpy as np
from pathlib import Path

from enum import Enum


class ShotType(Enum):
    SERVE = "serve"
    FOREHAND = "forehand"
    BACKHAND = "backhand"


_CLASSES = (ShotType.BACKHAND, ShotType.FOREHAND, ShotType.SERVE)


class ShotClassifier:
    """Wraps the trained shot classifier."""
    def __init__(self, model_path: Path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)


    def predict(self, features: np.ndarray) -> ShotType:
        """Predicts the shot type from a 16-feature normalized pose feature vector."""
        class_index = self.model.predict(np.array(features).reshape(1, -1))[0] # Required since model accepts and returns in batches
        return _CLASSES[class_index]
