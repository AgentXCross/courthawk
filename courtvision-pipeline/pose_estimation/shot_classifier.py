import pickle
import numpy as np

CLASSES = ['backhand', 'forehand', 'serve']

class ShotClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, keypoints):
        """
        Predicts the shot type from a 66-feature normalized keypoint vector.
        Returns a string: 'forehand', 'backhand', or 'serve'.
        """
        return CLASSES[self.model.predict(np.array(keypoints).reshape(1, -1))[0]]
