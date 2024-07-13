# import unittest
# from PIL import Image
# import numpy as np
# from facebox2.models.facenet512_model import Facenet512Model
# from facebox2.utils.image_utils import draw_detection_results, pil2cv, cv2pil
# import cv2

# class TestFacenet512Model(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = Facenet512Model()
#         cls.model.load_model()

#     def test_get_embeddings(self):
#         image = Image.open('tests/dataset/test1.jpg')
#         image = pil2cv(image)
#         result = self.model.detect_faces(image)

#         draw_detection_results(cv2pil(image), result).save('/tmp/test2_detection.jpg')

#         extracted_faces = self.model.extract_faces(image, result)
#         for i, face in enumerate(extracted_faces):
#             cv2.imwrite(f'/tmp/test1_aligned_face_{i}.jpg', face)
#         print(isinstance(extracted_faces, list))

#         embeddings = self.model.get_embeddings(extracted_faces)
#         self.assertIsInstance(embeddings, list)
#         if len(embeddings) > 0:
#             self.assertIsInstance(embeddings[0], np.ndarray)
#             self.assertEqual(len(embeddings[0].shape), 1)  # Embeddings should be 1D arrays

# if __name__ == '__main__':
#     unittest.main()
