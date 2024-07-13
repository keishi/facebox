import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any
import cv2
from skimage.transform import SimilarityTransform, warp
from ..models.types import FaceDetectionResult

def pil2cv(image):
    """PIL型 -> OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def cv2pil(image):
    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def draw_detection_results(image: Image.Image, detections: List[FaceDetectionResult]) -> Image.Image:
    """
    Draw the detection results onto the image.
    
    Args:
        image (Image.Image): The input PIL Image.
        detections (List[FaceDetectionResult]): List of detection results with bounding boxes.
        
    Returns:
        Image.Image: The image with detection results drawn on it.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    for detection in detections:
        # Draw bounding box
        box = detection['box']
        box_scale = max(box[2] - box[0], box[3] - box[1]) / 120

        draw.rectangle(box, outline='red', width=int(2 * box_scale))

        if 'landmarks' in detection:
            # Draw landmarks
            for point in detection['landmarks']:
                draw.ellipse((int(point[0] - 2 * box_scale), int(point[1] - 2 * box_scale), int(point[0] + 2 * box_scale), int(point[1] + 2 * box_scale)), fill='purple', outline='purple')
    
    return image
