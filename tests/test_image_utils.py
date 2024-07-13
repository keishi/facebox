# import unittest
# from PIL import Image
# import numpy as np
# from facebox.utils.image_utils import pil_to_image

# class TestImageUtils(unittest.TestCase):
#     def test_pil_to_image(self):
#         pil_image = Image.new('RGB', (100, 100), color = 'red')
#         np_image = pil_to_image(pil_image)
#         self.assertIsInstance(np_image, np.ndarray)
#         self.assertEqual(np_image.shape, (100, 100, 3))
#         self.assertEqual(np_image[0, 0, 0], 255)  # Check red channel

# if __name__ == '__main__':
#     unittest.main()
