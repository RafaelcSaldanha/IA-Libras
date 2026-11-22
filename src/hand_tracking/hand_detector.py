
import cv2
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(self, model_path=None, num_hands=2, confidence=0.5):

        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent
                / "models"
                / "hand_landmarker.task"
            )

        model_path = Path(model_path)

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence
        )

        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        return self.detector.detect(mp_image)

    def close(self):
        self.detector.close()
