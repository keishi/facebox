from typing import List, Dict, Any
import numpy as np
from .types import FaceDetectionResult

class FaceRecognitionModel:
    def __init__(self):
        self.model = None
    
    def load_model(self, model_path: str = None) -> None:
        """
        Load the face recognition model. Implement lazy loading to speed up initialization.
        
        Args:
            model_path (str): Path to the model. Defaults to None.
        """
        raise NotImplementedError("load_model method not implemented")
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetectionResult]:
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image in BGR format.
        
        Returns:
            List[FaceDetectionResult]: A list of detected faces with bounding boxes and landmarks.
        """
        raise NotImplementedError("detect_faces method not implemented")
    
    def extract_faces(self, image: np.ndarray, detections: List[FaceDetectionResult]) -> List[np.ndarray]:
        """
        Extract and align faces from the image based on detections.
        
        Args:
            image (np.ndarray): Input image in BGR format.
            detections (List[FaceDetectionResult]): List of detected faces with bounding boxes and landmarks.
        
        Returns:
            List[np.ndarray]: A list of aligned face images.
        """
        raise NotImplementedError("extract_faces method not implemented")
    
    def get_embeddings(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get face embeddings for the aligned face images.
        
        Args:
            faces (List[np.ndarray]): List of aligned face images.
        
        Returns:
            List[np.ndarray]: A list of embeddings for each aligned face image.
        """
        raise NotImplementedError("get_embeddings method not implemented")
    
    def validate_image(self, image: np.ndarray) -> None:
        """
        Validate that the input is a valid OpenCV image (NumPy array).
        
        Args:
            image (np.ndarray): Input image in BGR format.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a NumPy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-dimensional array with 3 channels (BGR format).")
    
    def validate_detections(self, detections: List[FaceDetectionResult]) -> None:
        """
        Validate that detections are in the correct format.
        
        Args:
            detections (List[FaceDetectionResult]): List of detected faces with bounding boxes and landmarks.
        """
        if not isinstance(detections, list):
            raise ValueError("Detections must be a list of dictionaries.")
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError("Each detection must be a dictionary.")
            if 'box' not in detection or 'landmarks' not in detection or 'confidence' not in detection:
                raise ValueError("Each detection must contain 'box', 'landmarks', and 'confidence' keys.")
    
    def validate_face_images(self, faces: List[np.ndarray]) -> None:
        """
        Validate that faces are in the correct format.
        
        Args:
            faces (List[np.ndarray]): List of aligned face images.
        """
        if not isinstance(faces, list):
            raise ValueError("Faces must be a list of NumPy arrays.")
        for face in faces:
            self.validate_image(face)
