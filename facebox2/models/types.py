from typing import TypedDict, List, Any

class FaceDetectionResult(TypedDict, total=False):
    box: List[int]         # [x, y, width, height]
    landmarks: List[List[int]]  # List of landmarks (e.g. [left_eye, right_eye, nose, left_mouth, right_mouth])
    confidence: float      # Detection confidence
