import numpy as np
import cv2
from typing import List
from .base import FaceRecognitionModel
from .types import FaceDetectionResult
from retinaface import RetinaFace
from deepface import DeepFace

class Facenet512Model(FaceRecognitionModel):
    def __init__(self):
        super().__init__()
        self.app = None
    
    def load_model(self, model_path: str = None) -> None:
        pass

    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        from retinaface import RetinaFace
        self.validate_image(image)
        face_detections: List[FaceDetectionResult] = []
        faces = RetinaFace.detect_faces(image)
        if isinstance(faces, tuple):
            print('tuple', faces)
            return []
        faces = faces.values()
        for face in faces:
            face_detections.append(FaceDetectionResult(
                box=face['facial_area'],
                landmarks=[
                    face['landmarks']['right_eye'],
                    face['landmarks']['left_eye'],
                    face['landmarks']['nose'],
                    face['landmarks']['mouth_right'],
                    face['landmarks']['mouth_left'],
                ],
                confidence=face['score'],
            ))
        return face_detections
    
    def extract_faces(self, image: np.ndarray, face_detections: List[FaceDetectionResult]) -> List[np.ndarray]:
        from retinaface.commons import postprocess

        target_size = (160, 160)
        self.validate_image(image)
        self.validate_detections(face_detections)
        aligned_faces = []
        for detection in face_detections:
            facial_area = detection['box']
            face_image = image[facial_area[1] : facial_area[3], facial_area[0] : facial_area[2]]
            left_eye = detection['landmarks'][1]
            right_eye = detection['landmarks'][0]
            nose = detection['landmarks'][2]
            aligned_face = postprocess.alignment_procedure(face_image, right_eye, left_eye, nose)
            # resize and padding
            factor_0 = target_size[0] / aligned_face.shape[0]
            factor_1 = target_size[1] / aligned_face.shape[1]
            factor = min(factor_0, factor_1)
            dsize = (int(aligned_face.shape[1] * factor), int(aligned_face.shape[0] * factor))
            aligned_face = cv2.resize(aligned_face, dsize)

            diff_0 = target_size[0] - aligned_face.shape[0]
            diff_1 = target_size[1] - aligned_face.shape[1]
            aligned_face = np.pad(
                aligned_face,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )
            aligned_faces.append(aligned_face)
        return aligned_faces
    
    def get_embeddings(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        from deepface import DeepFace
        embeddings = []
        for image in face_images:
            embedding = DeepFace.represent(image, model_name="Facenet512", detector_backend="skip")
            embeddings.append(np.array(embedding[0]["embedding"]))
        return embeddings
