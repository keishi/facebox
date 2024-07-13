import os
import sys

sys.path.append(os.path.abspath('.'))

from facebox2.models.ghostfacenetv1_model import GhostFaceNetV1Model
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from facebox2.models.insightface_model import InsightFaceModel
from facebox2.models.facenet512_model import Facenet512Model
from facebox2.models.evolve_model import EvolveModel
from facebox2.utils.image_utils import pil2cv, cv2pil
from sklearn.datasets import fetch_lfw_pairs
from deepface import DeepFace
from tqdm import tqdm
import cv2

import random

# parse pairsDevTrain.txt
# first line is the number of pairs
# each row is tab separated
# same person pairs have 3 columns
# person_id, image_id1, image_id2
# different person pairs have 4 columns
# person_id1, image_id1, person_id2, image_id2
# image file is at /Users/keishi/scikit_learn_data/lfw_home/lfw_funneled/{person_id}/{person_id}_{image_id:04}.jpg
# return list of tuples (image_path1, image_path2, is_same)
def parse_pairs(pairs_filename):
    with open(pairs_filename, 'r') as f:
        num_pairs = int(f.readline().strip())
        pairs = []
        for line in f:
            row = line.strip().split('\t')
            if len(row) == 3:
                person_id, image_id1, image_id2 = row
                person_id1 = person_id2 = person_id
                is_same = True
            else:
                person_id1, image_id1, person_id2, image_id2 = row
                is_same = False
            image_path1 = f'/Users/keishi/scikit_learn_data/lfw_home/lfw_funneled/{person_id1}/{person_id1}_{image_id1.zfill(4)}.jpg'
            image_path2 = f'/Users/keishi/scikit_learn_data/lfw_home/lfw_funneled/{person_id2}/{person_id2}_{image_id2.zfill(4)}.jpg'
            pairs.append((image_path1, image_path2, is_same))
    return pairs

def download_lfw_pairs(subset='test'):
    lfw_pairs = fetch_lfw_pairs(subset=subset, color=True, resize=2)

def compute_distances2(model, image_pairs, align_faces=False, scale_image=1):
    print('compute_distances2')
    distances = []
    labels = []

    for img_a_path, img_b_path, is_same in tqdm(image_pairs):
        # img_a = cv2.imread(img_a_path)
        # img_b = cv2.imread(img_b_path)
        img_a = Image.open(img_a_path)
        img_b = Image.open(img_b_path)
        img_a = img_a.convert('RGB')
        img_b = img_b.convert('RGB')
        # img_a.save('/tmp/image_a.jpg')
        # img_b.save('/tmp/image_b.jpg')
        img_a = pil2cv(img_a)
        img_b = pil2cv(img_b)
        #print('DeepFace.represent', img_a.shape, img_a.dtype, img_a.min(), img_a.max())
        face_a = DeepFace.represent(img_a, model_name="Facenet512", detector_backend="retinaface")[0]
        face_b = DeepFace.represent(img_b, model_name="Facenet512", detector_backend="retinaface")[0]
        embedding_a = face_a['embedding']
        embedding_b = face_b['embedding']
        if embedding_a is None or embedding_b is None:
            continue
        embedding_a = np.array(embedding_a)
        embedding_b = np.array(embedding_b)
        distance = np.linalg.norm(embedding_a - embedding_b)
        print(distance)
        distances.append(distance)
        labels.append(is_same)
    
    return np.array(distances), np.array(labels)


def compute_distances(model, image_pairs, align_faces=False, scale_image=1):
    norms = []
    distances = []
    labels = []
    
    for img_a_path, img_b_path, is_same in tqdm(image_pairs):
        if isinstance(img_a_path, Image.Image):
            image_a = img_a_path
        else:
            image_a = Image.open(img_a_path)
        if isinstance(img_b_path, Image.Image):
            image_b = img_b_path
        else:
            image_b = Image.open(img_b_path)

        # remove
        image_a.resize((112, 112))
        image_b.resize((112, 112))
        # remove

        if scale_image != 1:
            image_a = image_a.resize((image_a.width // scale_image, image_a.height // scale_image))
            image_b = image_b.resize((image_b.width // scale_image, image_b.height // scale_image))

        if image_a.mode != 'RGB':
            image_a = image_a.convert('RGB')
        if image_b.mode != 'RGB':
            image_b = image_b.convert('RGB')

        image_a = pil2cv(image_a)
        image_b = pil2cv(image_b)

        if align_faces:
            faces_a = model.detect_faces(image_a)
            faces_b = model.detect_faces(image_b)
            if len(faces_a) == 0 or len(faces_b) == 0:
                continue
            aligned_face_a = model.extract_faces(image_a, faces_a)[0]
            aligned_face_b = model.extract_faces(image_b, faces_b)[0]
            image_a = aligned_face_a
            image_b = aligned_face_b

        embedding_a, embedding_b = model.get_embeddings([image_a, image_b])
        
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        norms.append((norm_a, norm_b))
        
        #normalize
        embedding_a = embedding_a / np.linalg.norm(embedding_a)
        embedding_b = embedding_b / np.linalg.norm(embedding_b)
        
        # cosine distance
        distance = 1 - np.dot(embedding_a, embedding_b)

        distances.append(distance)
        labels.append(is_same)

    return np.array(norms), np.array(distances), np.array(labels)

def plot_roc_curve(distances, labels):
    fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=0)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc, thresholds


def calculate_accuracy(distances, labels, thresholds):
    accuracies = []
    
    for threshold in thresholds:
        predictions = distances < threshold
        accuracy = accuracy_score(labels, predictions)
        accuracies.append(accuracy)
    
    return accuracies

def analyze_correlation(norms, distances, labels, threshold=0.5):
    norms_a, norms_b = zip(*norms)
    
    plt.figure()
    plt.hist(norms_a, bins=50, alpha=0.5, label='Norms of Embedding A')
    plt.hist(norms_b, bins=50, alpha=0.5, label='Norms of Embedding B')
    plt.legend(loc='upper right')
    plt.title('Distribution of Embedding Norms')
    plt.show()

    norm_mean_a = np.mean(norms_a)
    norm_mean_b = np.mean(norms_b)
    
    print(f'Mean Norm of Embedding A: {norm_mean_a}')
    print(f'Mean Norm of Embedding B: {norm_mean_b}')
    
    correct_predictions = np.array([d < threshold for d in distances]) == labels  # Example threshold for cosine distance
    correct_norms = np.array(norms)[correct_predictions]
    incorrect_norms = np.array(norms)[~correct_predictions]
    
    plt.figure()
    plt.hist(correct_norms.flatten(), bins=50, alpha=0.5, label='Correct Predictions')
    plt.hist(incorrect_norms.flatten(), bins=50, alpha=0.5, label='Incorrect Predictions')
    plt.legend(loc='upper right')
    plt.title('Norms for Correct and Incorrect Predictions')
    plt.show()


def main():
    if False:
        download_lfw_pairs()
        dataset = parse_pairs('/Users/keishi/scikit_learn_data/lfw_home/pairsDevTest.txt')
        align_faces = True
        scale_image = 2
    else:
        num_pairs = 200
        dataset_dir = "/Users/keishi/facial3/dataset"
        align_faces = False
        scale_image = 1
        # dataset_dir = "/Users/keishi/facial3/evaluate/dataset"
        # align_faces = True
        # scale_image = 1
        # dataset_dir = "/Users/keishi/facial3/datasets/lfw-deepfunneled"
        # align_faces = True
        # scale_image = 1

        print("Loading dataset...")
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

        print("Generating dataset...")
        rng = random.Random(0)
        dataset = []
        unique_labels = sorted(list(set(labels)))
        label_to_images = {}
        for label, image in zip(labels, images):
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].append(image)
        # Generate 10 positive pairs
        positive_pairs = []
        while len(positive_pairs) < num_pairs // 2:
            name = rng.choice(unique_labels)
            if len(label_to_images[name]) < 2:
                continue
            img_a, img_b = rng.sample(label_to_images[name], 2)
            positive_pairs.append((img_a, img_b, True))
        dataset.extend(positive_pairs)

        # Generate 10 negative pairs
        negative_pairs = []
        while len(negative_pairs) < num_pairs // 2:
            try:
                name_a, name_b = rng.sample(unique_labels, 2)
            except:
                print(unique_labels)
                raise
            try:
                img_a = rng.choice(label_to_images[name_a])
                img_b = rng.choice(label_to_images[name_b])
            except:
                print(name_a, name_b)
                print(label_to_images[name_a])
                print(label_to_images[name_b])
                raise
            img_a = rng.choice(label_to_images[name_a])
            img_b = rng.choice(label_to_images[name_b])
            negative_pairs.append((img_a, img_b, False))
        dataset.extend(negative_pairs)

    print("Computing distances...")
    #model = Facenet512Model()
    model = InsightFaceModel()
    #model = EvolveModel()
    #model = GhostFaceNetV1Model()
    model.load_model()
    norms, distances, labels = compute_distances(model, dataset, align_faces=align_faces, scale_image=scale_image)
    #distances, labels = compute_distances2(model, dataset, align_faces=align_faces, scale_image=scale_image)

    analyze_correlation(norms, distances, labels, 25.57)

    print("Plotting ROC curve...")
    roc_auc, thresholds = plot_roc_curve(distances, labels)
    print(f'ROC AUC: {roc_auc:.2f}')

    accuracies = calculate_accuracy(distances, labels, thresholds)
    best_accuracy = max(accuracies)
    best_threshold = thresholds[np.argmax(accuracies)]
    print(f'Best Accuracy: {best_accuracy:.2f} at threshold {best_threshold:.2f}')


if __name__ == "__main__":
    main()
