import numpy as np
import cv2
from typing import List
from .base import FaceRecognitionModel
from .types import FaceDetectionResult
import torch
from ..utils.image_utils import cv2pil

class EvolveModel(FaceRecognitionModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def load_model(self, model_path: str = None) -> None:
        if self.model is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            from .evolve_backbone.model_irse import IR_50
            self.model = IR_50((112, 112))
            #'backbone_ir50_asia.pth'
            # 'backbone_ir50_ms1m_epoch120.pth'
            self.model.load_state_dict(torch.load('/Users/keishi/Downloads/backbone_ir50_asia.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        from .evolve_align.detector import detect_faces

        self.validate_image(image)
        pil_image = cv2pil(image)
        try:
            bbox_list, landmarks_list = detect_faces(pil_image)
        except Exception as e:
            print(e)
            return []
        faces = []
        for bbox, landmarks in zip(bbox_list, landmarks_list):
            faces.append(FaceDetectionResult(
                box=bbox,
                landmarks=landmarks,
                confidence=1.0
            ))
        return faces
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
        from .evolve_align.align_trans import get_reference_facial_points, warp_and_crop_face

        self.validate_image(image)
        self.validate_detections(face_detections)
        aligned_faces = []
        
        crop_size = 112 # specify size of aligned faces, align and crop with padding
        scale = crop_size / 112.
        reference = get_reference_facial_points(default_square = True) * scale
        for detection in face_detections:
            landmarks = detection['landmarks']
            facial5points = [[landmarks[j], landmarks[j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(image, facial5points, reference, crop_size=(crop_size, crop_size))
            aligned_faces.append(warped_face)
        return aligned_faces
    
    def get_embeddings(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        self.validate_face_images(faces)
        self.load_model()
        embeddings = []
        for image in faces:
            # load numpy to tensor
            ccropped = image
            ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
            ccropped = np.reshape(ccropped, [1, 3, 112, 112])
            ccropped = np.array(ccropped, dtype = np.float32)
            ccropped = (ccropped - 127.5) / 128.0
            ccropped = torch.from_numpy(ccropped)

            features = self.model(ccropped.to(self.device)).cpu()
            features = features.detach().numpy()
            for feature in features:
                embeddings.append(feature)
        return embeddings
