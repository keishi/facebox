import numpy as np
import cv2
from typing import List
from .base import FaceRecognitionModel
from .types import FaceDetectionResult

class InsightFaceModel(FaceRecognitionModel):
    def __init__(self):
        super().__init__()
        self.app = None
    
    def load_model(self, model_path: str = None) -> None:
        if self.app is None:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis("buffalo_l", allowed_modules=['detection', 'recognition'], providers=["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.recognition_model = self.app.models["recognition"]
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        self.validate_image(image)
        self.load_model()
        faces = self.app.get(image)
        face_detections: List[FaceDetectionResult] = []
        for face in faces:
            face_detections.append(FaceDetectionResult(
                box=face.bbox,
                landmarks=[
                    face.kps[0], # left eye
                    face.kps[1], # right eye
                    face.kps[2], # nose
                    face.kps[3], # left mouth
                    face.kps[4], # right mouth
                ],
                confidence=face.det_score
            ))
        return face_detections
    
    def extract_faces(self, image: np.ndarray, face_detections: List[FaceDetectionResult]) -> List[np.ndarray]:
        self.validate_image(image)
        self.validate_detections(face_detections)
        aligned_faces = []
        for detection in face_detections:
            aligned_face = self.align_face(image, detection['landmarks'])
            aligned_faces.append(aligned_face)
        return aligned_faces
    
    def get_embeddings(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        self.validate_face_images(faces)
        self.load_model()
        embeddings = []
        embeddings = self.recognition_model.get_feat(faces)
        return [x for x in embeddings]

    def align_face(self, image: np.ndarray, landmarks: List[np.ndarray], image_size: int = 112) -> np.ndarray:
        """
        Align a single face in an image based on landmarks.
        
        Args:
            image (Image.Image): The input PIL Image.
            landmarks (List[np.ndarray]): The landmarks of the face (5 points: left eye, right eye, nose, left mouth, right mouth).
            
        Returns:
            Image.Image: The aligned face image.
        """
        from skimage.transform import SimilarityTransform, warp
        src = np.array(landmarks, dtype='float32')
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        assert src.shape == (5, 2)

        if image_size % 112 == 0:
            ratio = float(image_size) / 112.0
            diff_x = 0
        else:
            ratio = float(image_size) / 128.0
            diff_x = 8.0 * ratio

        tform = SimilarityTransform()
        tform.estimate(src, dst)

        M = tform.params[0:2, :]

        warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)
        return warped
