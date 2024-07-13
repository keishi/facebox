import os
import sys

sys.path.append(os.path.abspath('./facebox'))

from facebox2.models.insightface_model import InsightFaceModel
from facebox2.utils.image_utils import pil2cv, cv2pil
from tqdm import tqdm
from PIL import Image
import typing
import time
import numpy as np


def batchify(generator: typing.Iterator, size: int):
    batch = []
    for x in generator:
        batch.append(x)
        if len(batch) == size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def unbatchify(generator: typing.Iterator):
    for batch in generator:
        for x in batch:
            yield x

images_dir = './small-facial3'

if not os.path.exists(images_dir):
    # download and extract https://www.foo.com/small-facial3.zip
    os.system(f"wget https://www.foo.com/small-facial3.zip")
    os.system(f"unzip small-facial3.zip")
    assert os.path.exists(images_dir)

image_files = []
for root, dirs, files in os.walk(images_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_files.append(os.path.join(root, file))
image_files.sort()

image_files = image_files[:1024]
print(f"Number of images: {len(image_files)}")

face_model = InsightFaceModel()
face_model.load_model()

def extract_faces(face_model, image_files) -> typing.List[np.ndarray]:
    for image_file in image_files:
        image = Image.open(image_file)
        image = pil2cv(image)
        face_detections = face_model.detect_faces(image)
        faces = face_model.extract_faces(image, face_detections)
        for face in faces:
            yield face

def run(face_model, image_files, batch_size):
    start_time = time.time()
    embeddings = []
    if batch_size is None:
        for image_file in image_files:
            image = Image.open(image_file)
            image = pil2cv(image)
            # face_detections = face_model.detect_faces(image)
            # faces = face_model.extract_faces(image, face_detections)
            faces = [image]
            for face in faces:
                embeddings.extend(face_model.get_embeddings([face]))
    else:
        aligned_faces = extract_faces(face_model, image_files)
        for batch in batchify(aligned_faces, batch_size):
            embeddings.extend(face_model.get_embeddings(batch))
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds (batch size: {batch_size})")

for batch_size in [None, 1, 16, 32, 64, 128, 256]:
    run(face_model, image_files, batch_size)
