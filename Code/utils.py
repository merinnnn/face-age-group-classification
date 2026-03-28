import os, random, time, joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from collections import Counter

from skimage.feature import hog, local_binary_pattern
from skimage import img_as_ubyte

from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# CONSTANTS
RANDOM_SEED = 42
IMAGE_SIZE  = (128, 128)    # default for classical models   
CNN_SIZE    = (244, 244)    # VGG/ResNet
AGE_LABELS  = {0:"Child", 1:"Young", 2:"Middle-Aged", 3:"Senior"}
NUM_CLASSES = 4

# Set all seeds
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def load_dataset(data_dir, label_file):
    with open(label_file, 'r') as f:
        rows = [l.strip().split() for l in f if l.strip()]
    paths, labels = [], []
    for path, label in rows:
        paths.append(os.path.join(data_dir, path))
        labels.append(int(label))
    return paths, labels

# Preprocess image for classical models
def load_image(path, size=IMAGE_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    img = cv2.resize(img, size)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


