from typing import List, Dict, Any
import numpy as np

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
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image in BGR format.
        
        Returns:
            List[Dict[str, Any]]: A list of detected faces with bounding boxes and landmarks.
        """
        raise NotImplementedError("detect_faces method not implemented")
    
    def extract_faces(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Extract and align faces from the image based on detections.
        
        Args:
            image (np.ndarray): Input image in BGR format.
            detections (List[Dict[str, Any]]): List of detected faces with bounding boxes and landmarks.
        
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
