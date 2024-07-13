import numpy as np
import cv2
from typing import List
from .base import FaceRecognitionModel
from .types import FaceDetectionResult
from .ghostfacenetv1_face_detector import YoloV5FaceDetector

class GhostFaceNetV1Model(FaceRecognitionModel):
    def __init__(self):
        super().__init__()
        self.app = None
    
    def load_model(self, model_path: str = None) -> None:
        self.face_detector = YoloV5FaceDetector()
        pass

    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        self.validate_image(image)
        face_detections: List[FaceDetectionResult] = []
        bbs, pps, ccs = self.face_detector.detect_in_image(image, score_threshold=0.3)

        for bb, pp, cc in zip(bbs, pps, ccs):
            face_detections.append(FaceDetectionResult(
                box=bb,
                landmarks=pp,
                confidence=cc,
            ))
        face_detections = sorted(face_detections, key=lambda x: x['confidence'], reverse=True)
        return face_detections
    
    def extract_faces(self, image: np.ndarray, face_detections: List[FaceDetectionResult]) -> List[np.ndarray]:
        landmarks = np.array([face['landmarks'] for face in face_detections])
        return self.face_detector.face_align_landmarks(image, landmarks)
    
    def get_embeddings(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        from deepface import DeepFace
        embeddings = []
        for image in face_images:
            embedding = DeepFace.represent(image, model_name="GhostFaceNet", detector_backend="skip")
            embeddings.append(np.array(embedding[0]["embedding"]))
        return embeddings
