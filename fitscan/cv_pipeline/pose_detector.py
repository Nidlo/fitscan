"""MediaPipe pose detection. Implements IPoseDetector."""

from __future__ import annotations

import numpy as np
from loguru import logger

from fitscan.domain.exceptions import PoseDetectionError
from fitscan.domain.models import BodyPose, Keypoint

# MediaPipe landmark names used in this project
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    9: "mouth_left",
    10: "mouth_right",
    7: "left_ear",
    8: "right_ear",
    31: "left_foot_index",
    32: "right_foot_index",
}


class PoseDetector:
    """Detects 33 body landmarks using MediaPipe Pose."""

    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

    def __init__(self) -> None:
        self._pose = None

    def _init_pose(self) -> None:
        if self._pose is not None:
            return
        try:
            import mediapipe as mp

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            )
        except ImportError:
            logger.warning("MediaPipe not available — using stub pose detector.")
            self._pose = None

    def detect(self, image: np.ndarray) -> BodyPose:
        self._init_pose()
        if self._pose is None:
            raise PoseDetectionError("MediaPipe not installed.")

        results = self._pose.process(image)
        if not results.pose_landmarks:
            raise PoseDetectionError(
                "No human pose detected. Ensure the full body is visible."
            )
        h, w = image.shape[:2]
        keypoints: dict[str, Keypoint] = {}
        for idx, name in LANDMARK_NAMES.items():
            lm = results.pose_landmarks.landmark[idx]
            keypoints[name] = Keypoint(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
        return BodyPose(keypoints=keypoints, image_width=w, image_height=h)

    def close(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def find_occluded_landmarks(
        pose: BodyPose,
        threshold: float = 0.3,
    ) -> list[str]:
        """Return the names of landmarks whose MediaPipe `visibility` falls
        below `threshold`. The S5 layered-AI flow feeds these to
        `ClaudeValidator.verify_landmarks` for vision-based verification —
        clothing folds, low-light arms, hair-over-shoulder are exactly the
        cases where MediaPipe's confidence drops below the
        cv-pipeline-trustable bar but a human (or a vision model) can still
        place the joint by silhouette.

        The default threshold matches MediaPipe's
        `MIN_DETECTION_CONFIDENCE` so anything the model itself isn't
        confident enough to use as a tracking anchor is candidate for
        re-verification.
        """
        return sorted(
            name for name, kp in pose.keypoints.items() if kp.visibility < threshold
        )
