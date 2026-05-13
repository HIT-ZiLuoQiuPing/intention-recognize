"""MediaPipe wrapper for extracting right-arm and right-hand keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import COORD_DIM, KEYPOINT_COUNT


POSE_RIGHT_ARM_INDICES = (12, 14, 16)


@dataclass
class KeypointFrame:
    coords: np.ndarray
    missing_points: int
    results: object


class MediaPipeKeypointExtractor:
    """Extract 24 points: right shoulder, elbow, wrist, and 21 right-hand points."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pose_visibility_threshold: float = 0.3,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for camera collection and real-time inference. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.mp = mp
        self.pose_visibility_threshold = pose_visibility_threshold
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.drawer = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

    def close(self) -> None:
        self.holistic.close()

    def __enter__(self) -> "MediaPipeKeypointExtractor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def process_bgr(self, frame_bgr: np.ndarray) -> KeypointFrame:
        import cv2

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        coords = np.full((KEYPOINT_COUNT, COORD_DIM), np.nan, dtype=np.float32)

        missing_points = 0
        if results.pose_landmarks is not None:
            for local_idx, pose_idx in enumerate(POSE_RIGHT_ARM_INDICES):
                landmark = results.pose_landmarks.landmark[pose_idx]
                if getattr(landmark, "visibility", 1.0) >= self.pose_visibility_threshold:
                    coords[local_idx] = [landmark.x, landmark.y, landmark.z]
                else:
                    missing_points += 1
        else:
            missing_points += len(POSE_RIGHT_ARM_INDICES)

        hand_offset = len(POSE_RIGHT_ARM_INDICES)
        if results.right_hand_landmarks is not None:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                coords[hand_offset + idx] = [landmark.x, landmark.y, landmark.z]
        else:
            missing_points += 21

        return KeypointFrame(coords=coords, missing_points=missing_points, results=results)

    def draw(self, frame_bgr: np.ndarray, results: object) -> np.ndarray:
        annotated = frame_bgr.copy()
        if getattr(results, "pose_landmarks", None) is not None:
            self.drawer.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp.solutions.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
            )
        if getattr(results, "right_hand_landmarks", None) is not None:
            self.drawer.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                self.mp.solutions.holistic.HAND_CONNECTIONS,
                self.styles.get_default_hand_landmarks_style(),
                self.styles.get_default_hand_connections_style(),
            )
        return annotated
