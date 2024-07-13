import os
import sys

sys.path.append(os.path.abspath('/Users/keishi/facebox'))

from PIL import Image
import numpy as np
from facebox2.utils.image_utils import pil2cv, cv2pil
import typing
from facebox2.models.insightface_model import InsightFaceModel
import pickle
from tqdm import tqdm

def load_images(images):
    for image in images:
        img = Image.open(image)
        img.resize((112, 112))
        img = pil2cv(img)
        yield img

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

dataset_dir = "/Users/keishi/facial3/dataset"

print("Listing up images")
images = []
labels = []
for name in sorted(os.listdir(dataset_dir)):
    subdir = os.path.join(dataset_dir, name)
    if not os.path.isdir(subdir):
        continue
    for basename in os.listdir(subdir):
        ext = os.path.splitext(basename)[1]
        if ext.lower() == ".jpg" or ext.lower() == ".webp" or ext.lower() == ".png":
            images.append(os.path.join(dataset_dir, name, basename))
            labels.append(name)
print(f'Found {len(images)} images')

embedding_cache = {}
if os.path.exists("embedding_cache.pkl"):
    with open("embedding_cache.pkl", "rb") as f:
        embedding_cache = pickle.load(f)

unknown_images = []
for i in range(len(images)):
    if images[i] not in embedding_cache:
        unknown_images.append(images[i])

print(f"Found {len(unknown_images)} unknown images")

if len(unknown_images) > 0:
    print("Loaading model")
    model = InsightFaceModel()
    model.load_model()

    print("Calculating embeddings")
    face_images = load_images(unknown_images)
    batches = batchify(face_images, 32)
    embedding_batches = map(model.get_embeddings, batches)
    progress = tqdm(total=len(unknown_images))
    for i, x in enumerate(unbatchify(embedding_batches)):
        embedding_cache[unknown_images[i]] = x / np.linalg.norm(x)
        progress.update(1)
    progress.close()

    with open("embedding_cache.pkl", "wb") as f:
        pickle.dump(embedding_cache, f)

embeddings = [embedding_cache[x] for x in images]

qualities = [0] * len(images)
threshold = 0.73
rng = np.random.default_rng(0)
for i in tqdm(range(len(images))):
    num_correct = 0
    num_total = 0
    if rng.random() < 0.9:
        continue
    for j in range(len(images)):
        if i == j:
            continue
        if rng.random() < 0.95:
            continue
        cos_distance = 1 - np.dot(embeddings[i], embeddings[j])
        is_same_prediction = cos_distance < threshold
        is_same_truth = labels[i] == labels[j]
        if is_same_prediction == is_same_truth:
            num_correct += 1
        num_total += 1
    accuracy = num_correct / num_total
    qualities[i] = accuracy
        
with open("quality_estimator_dataset.pkl", "wb") as f:
    pickle.dump((images, qualities), f)
