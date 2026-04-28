import numpy as np
from skimage import img_as_ubyte
from skimage.feature import haar_like_feature, haar_like_feature_coord, hog
from skimage.feature import local_binary_pattern
from skimage.transform import integral_image
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# Colab notebooks should do:
#   ROOT = "/content/drive/MyDrive/CW_Folder_UG"
#   sys.path.insert(0, f"{ROOT}/Code")
#   from components import *
from utils import SEED, flatten_images, preprocess_classical, preprocess_many


# FEATURE EXTRACTORS

# HOG
def _extract_hog_feature(
    path,
    image_size=(128, 128),
    contrast=None,
    crop_face=False,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
):
    """Extract one HOG descriptor from one image.

    Reference: HOG is introduced in Lecture05, 
    and implemented with skimage.feature.hog in Lab_05_Solved.ipynb.
    """
    image = preprocess_classical(
        path,
        image_size=image_size,
        contrast=contrast,
        crop_face=crop_face,
    )
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
    )

def extract_hog_features(
    paths,
    image_size=(128, 128),
    contrast=None,
    crop_face=False,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    max_images=None,
):
    """Create a HOG feature matrix from image paths.

    Reference: batch version of the HOG descriptor shown in Lab_05_Solved.ipynb.
    """
    selected_paths = paths if max_images is None else paths[:max_images]
    features = [
        _extract_hog_feature(
            path,
            image_size=image_size,
            contrast=contrast,
            crop_face=crop_face,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
        )
        for path in selected_paths
    ]
    return np.asarray(features)

# LBP
def _extract_lbp_feature(
    path,
    image_size=(128, 128),
    contrast=None,
    crop_face=False,
    points=24,
    radius=3,
    method="uniform",
):
    """Extract one LBP histogram from one image.

    Reference: LBP is listed in Lecture05 as a texture descriptor useful for classification.
    """
    image = preprocess_classical(
        path,
        image_size=image_size,
        contrast=contrast,
        crop_face=crop_face,
    )
    image_u8 = img_as_ubyte(np.clip(image, 0, 1))
    lbp = local_binary_pattern(image_u8, P=points, R=radius, method=method)

    if method == "uniform":
        n_bins = points + 2
        hist_range = (0, n_bins)
    else:
        n_bins = int(lbp.max() + 1)
        hist_range = (0, n_bins)

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=hist_range, density=True)
    return hist.astype(np.float32)

def extract_lbp_features(
    paths,
    image_size=(128, 128),
    contrast=None,
    crop_face=False,
    points=24,
    radius=3,
    method="uniform",
    max_images=None,
):
    """Create an LBP feature matrix from image paths.

    Reference: batch implementation of the LBP descriptor listed in Lecture05.
    """
    selected_paths = paths if max_images is None else paths[:max_images]
    features = [
        _extract_lbp_feature(
            path,
            image_size=image_size,
            contrast=contrast,
            crop_face=crop_face,
            points=points,
            radius=radius,
            method=method,
        )
        for path in selected_paths
    ]
    return np.asarray(features)

# HOG + LBP
def extract_hog_lbp_features(
    paths,
    image_size=(128, 128),
    contrast=None,
    crop_face=False,
    hog_params=None,
    lbp_params=None,
    max_images=None,
):
    """Create concatenated HOG + LBP hand-crafted features.

    Reference: HOG and LBP are both listed as hand-crafted feature descriptors
    in Lecture05; concatenating descriptors follows the Lecture05 pipeline of
    forming one feature descriptor/vector before classification.
    """
    hog_params = {} if hog_params is None else dict(hog_params)
    lbp_params = {} if lbp_params is None else dict(lbp_params)

    X_hog = extract_hog_features(
        paths,
        image_size=image_size,
        contrast=contrast,
        crop_face=crop_face,
        max_images=max_images,
        **hog_params,
    )
    X_lbp = extract_lbp_features(
        paths,
        image_size=image_size,
        contrast=contrast,
        crop_face=crop_face,
        max_images=max_images,
        **lbp_params,
    )
    return np.concatenate([X_hog, X_lbp], axis=1)

# CLASSIFIERS

# LINEAR CLASSIFIER
def create_linear_classifier(C=1.0, class_weight="balanced", max_iter=5000, **kwargs):
    """Create a linear classifier.

    Reference: linear classifiers are introduced in Lecture05. 
    This uses sklearn.svm.LinearSVC.
    """
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=kwargs.pop("random_state", SEED),
        **kwargs,
    )

# SVM CLASSIFIERS
def create_svm_classifier(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    degree=3,
    class_weight="balanced",
    probability=False,
    **kwargs,
):
    """Create an SVM classifier with configurable kernel.

    Reference: SVMs, polynomial kernels, RBF/Gaussian kernels, and multi-class
    SVMs are covered in Lecture05; SVC training is used in Lab_05_Solved.ipynb
    and Lab_06_Solved.ipynb.
    """
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        class_weight=class_weight,
        probability=probability,
        random_state=kwargs.pop("random_state", SEED),
        **kwargs,
    )

# LINEAR SVM CLASSIFIER
def create_linear_svm(C=1.0, class_weight="balanced", **kwargs):
    """Create a linear SVM classifier.

    Reference: directly follows train_SVM_solved.py, where
    svm.SVC(kernel='linear') defines the linear SVM component.
    """
    return create_svm_classifier(
        kernel="linear",
        C=C,
        class_weight=class_weight,
        **kwargs,
    )

# POLYNOMIAL SVM CLASSIFIER
def create_polynomial_svm(
    C=1.0,
    degree=3,
    gamma="scale",
    class_weight="balanced",
    **kwargs,
):
    """Create a polynomial-kernel SVM classifier.

    Reference: polynomial SVMs are discussed in Lecture05 and used as a BoVW
    hyper-parameter change in Lab_06_Solved.ipynb.
    """
    return create_svm_classifier(
        kernel="poly",
        C=C,
        gamma=gamma,
        degree=degree,
        class_weight=class_weight,
        **kwargs,
    )

# RBF SVM CLASSIFIER
def create_rbf_svm(C=1.0, gamma="scale", class_weight="balanced", **kwargs):
    """Create an RBF-kernel SVM classifier.

    Reference: the RBF/Gaussian kernel is covered in Lecture05, and RBF SVM is
    used for BoVW classification in Lab_06_Solved.ipynb.
    """
    return create_svm_classifier(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        **kwargs,
    )

# KNN CLASSIFIER
def create_knn_classifier(n_neighbors=5, weights="uniform", metric="minkowski", **kwargs):
    """Create a K-nearest neighbours classifier.

    Reference: KNN appears in the MNIST comparison tables in Lecture05,
    Lecture06, and Lecture07.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        **kwargs,
    )

# MLP CLASSIFIER
def create_mlp_classifier(
    hidden_layer_sizes=(50,),
    activation="relu",
    solver="sgd",
    alpha=1e-4,
    learning_rate_init=0.1,
    max_iter=100,
    early_stopping=False,
    **kwargs,
):
    """Create a multi-layer perceptron classifier.

    Reference: MLPs are covered in Lecture06 and implemented with
    sklearn.neural_network.MLPClassifier in Lab_06_Solved.ipynb.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=kwargs.pop("random_state", SEED),
        **kwargs,
    )

# RANDOM FOREST CLASSIFIER
def create_random_forest_classifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    **kwargs,
):
    """Create a random forest classifier.

    Reference: random forests are listed in Lecture05 as decision-tree based
    classifiers for image classification.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=kwargs.pop("random_state", SEED),
        **kwargs,
    )
