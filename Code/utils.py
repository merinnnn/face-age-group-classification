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
SEED = 42
IMAGE_SIZE  = (128, 128)    # default for classical models   
CNN_SIZE    = (244, 244)    # VGG/ResNet
AGE_LABELS  = {0:"Child", 1:"Young", 2:"Middle-Aged", 3:"Senior"}
NUM_CLASSES = 4

# Set all seeds
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def load_dataset(data_dir, label_file):
    """Load dataset paths and labels from a label file."""
    with open(label_file, 'r') as f:
        rows = [l.strip().split() for l in f if l.strip()]
    paths, labels = [], []
    for path, label in rows:
        path = os.path.basename(path)   # strip any subfolder prefix
        p = os.path.join(data_dir, path)
        if os.path.exists(p):
            paths.append(p)
            labels.append(int(label))
    return paths, labels

# Preprocess image for classical models
def load_image(path, size=IMAGE_SIZE, grayscale=False):
    """Load and resize image."""
    if grayscale:
        # Read directly as grayscale to guarantee 2D array for HOG
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        return cv2.resize(img, size)
    else:
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Cannot read: {path}")
        img = cv2.resize(img, size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Feature extractors
def extract_hog(path, size=IMAGE_SIZE):
    """HOG feature descriptor."""
    img = load_image(path, size)
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   feature_vector=True,
                   channel_axis=-1)
    return features

def extract_lbp(path, size=IMAGE_SIZE, P=24, R=3):
    """LBP feature descriptor."""
    img = load_image(path, size=size, grayscale=True)
    lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist

def extract_hog_lbp(path, size=IMAGE_SIZE):
    """Combined HOG + LBP features."""
    hog_features = extract_hog(path, size)
    lbp_features = extract_lbp(path, size)
    return np.concatenate((hog_features, lbp_features))

def batch_extract(paths, extractor, label=""):
    """Extract features for a list of paths with progress printing."""
    out = []
    n = len(paths)
    for i, p in enumerate(paths):
        if i % 1000 == 0:
            print(f"  {label}: {i}/{n}")
        out.append(extractor(p))
    return np.array(out)

# SVM training
def train_linear_SVM(X_train, y_train):
    """Train a linear SVM classifier."""
    classifier = svm.SVC(kernel='linear', class_weight='balanced')
    classifier.fit(X_train, y_train)
    return classifier

def train_rbf_SVM(X_train, y_train, C=10):
    """Train an RBF-kernel SVM classifier."""
    classifier = svm.SVC(kernel='rbf', C=C, gamma='scale',
                         class_weight='balanced', random_state=SEED)
    classifier.fit(X_train, y_train)
    return classifier

# MLP training
def train_MLP(X_train, y_train, hidden_layer_sizes=(512, 256), max_iter=100):
    """Train a Multi-layer Perceptron classifier."""
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        early_stopping=True,
        random_state=SEED
    )
    classifier.fit(X_train, y_train)
    return classifier

# Evaluation
def evaluate(name, y_true, y_pred, train_time=None, model_path=None):
    """Print accuracy and full classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"Model: {name}")
    if train_time:
        print(f"Training Time: {train_time:.1f} seconds")
    if model_path and os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model Size: {size:.2f} MB")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(metrics.classification_report(
        y_true, y_pred,
        target_names=[AGE_LABELS[i] for i in range(4)]
    ))
    return acc

def plot_confusion_matrix(name, y_true, y_pred, ax=None):
    """Plot a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[AGE_LABELS[i] for i in range(4)]
    )
    disp.plot(ax=ax or plt.gca(), colorbar=False, xticks_rotation=30)
    if ax:
        ax.set_title(name)

def qualitative_grid(test_paths, test_labels, predictions_dict, n=8):
    """Show n random test images with GT + all model predictions."""
    idxs = random.sample(range(len(test_paths)), n)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4.5))
    for ax, idx in zip(axes.flatten(), idxs):
        img = load_image(test_paths[idx])
        ax.imshow(img)
        ax.axis('off')
        gt = AGE_LABELS[test_labels[idx]]
        lines = [f"GT: {gt}"]
        for mname, preds in predictions_dict.items():
            mark = "✓" if preds[idx] == test_labels[idx] else "✗"
            lines.append(f"{mark} {mname}: {AGE_LABELS[preds[idx]]}")
        ax.set_title("\n".join(lines), fontsize=7, loc='left')
    plt.tight_layout()
    plt.show()