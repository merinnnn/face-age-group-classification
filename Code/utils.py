import os
import random
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from skimage import color, exposure, img_as_float, img_as_ubyte, transform
from skimage.draw import rectangle_perimeter
from skimage.io import imread

# import tensorflow as tf
# from mtcnn import MTCNN

SEED =104
random.seed(SEED)
np.random.seed(SEED)

AGE_LABELS = {
    0: "Child",
    1: "Young",
    2: "Middle-Aged",
    3: "Senior",
}

_MTCNN_CACHE = {}
_MTCNN_DETECT_MAX_SIDE = 1000

def get_project_paths(root="/content/drive/MyDrive/CW_Folder_UG"):
    """Paths for the Colab folders."""
    root = Path(root)
    data_dir = root / "CW_Dataset"
    personal_dir = root / "Personal_Dataset"
    models_dir = root / "Models"
    models_dir.mkdir(exist_ok=True)
    return {
        "ROOT": root,
        "TRAIN_IMGS": data_dir / "train",
        "TRAIN_LBL": data_dir / "train" / "train_labels.txt",
        "TEST_IMGS": data_dir / "test",
        "TEST_LBL": data_dir / "test" / "test_labels.txt",
        "PERSONAL_IMGS": personal_dir,
        "PERSONAL_LBL": personal_dir / "personal_labels.txt",
        "MODELS_DIR": models_dir,
    }

def load_labels(image_dir, label_file):
    """Load name and label from the .txt file"""
    image_dir = Path(image_dir)
    paths, labels = [], []
    with open(label_file, "r") as file:
        for line in file:
            filename, label = line.strip().split()
            paths.append(image_dir / filename)
            labels.append(int(label))
    return paths, labels

def load_dataset(root="/content/drive/MyDrive/CW_Folder_UG"):
    """Load dataset from the given path."""
    paths = get_project_paths(root)
    train_paths, train_labels = load_labels(paths["TRAIN_IMGS"], paths["TRAIN_LBL"])
    test_paths, test_labels = load_labels(paths["TEST_IMGS"], paths["TEST_LBL"])
    
    print("Train:", len(train_paths), "images")
    print("Test:", len(test_paths), "images")
    return train_paths, train_labels, test_paths, test_labels

def load_personal_dataset(root="/content/drive/MyDrive/CW_Folder_UG"):
    """Load the personal/in-the-wild dataset."""
    paths = get_project_paths(root)
    personal_paths, personal_labels = load_labels(paths["PERSONAL_IMGS"], paths["PERSONAL_LBL"])
    print("Personal:", len(personal_paths), "images")
    return personal_paths, personal_labels

def make_train_val_split(train_paths, train_labels, test_size=0.15, random_state=SEED):
    """Split dataset into train/val."""
    return train_test_split(train_paths, train_labels, test_size=test_size, random_state=random_state)


# PREPROCESSING

def read_rgb(path):
    """Load RGB image as float in [0,1]. Reference: Lab_02_Solved.ipynb (1: Negative four ways)."""
    return img_as_float(imread(path))

def resize_image(image, size=(128, 128)):
    """Resize image to the given size."""
    return transform.resize(image, size, anti_aliasing=True)

def adjust_contrast(image, method=None, gamma=1.0):
    """Contrast adjustment. Reference: Lab_02_Solved.ipynb (3: Contrast adjustments)."""
    if method is None:
        return image
    elif method == "stretch":
        p2, p98 = np.percentile(image, (2, 98))
        return exposure.rescale_intensity(image, in_range=(p2, p98))
    elif method =="equalize":
        return exposure.equalize_hist(image)
    elif method == "gamma":
        return exposure.adjust_gamma(image, gamma)
    else:
        raise ValueError(f"Unknown method: {method}")
 
def _get_mtcnn():
    """Load MTCNN model, with caching."""
    if 'mtcnn' not in _MTCNN_CACHE:
        from mtcnn import MTCNN
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        
        _MTCNN_CACHE["mtcnn"] = MTCNN()
    return _MTCNN_CACHE["mtcnn"]

def _detect_faces_mtcnn(image):
    """
    Detect faces using MTCNN.
    Downsample to _MTCNN_DETECT_MAX_SIDE so MTCNN sees faces at consistent scale.

    Returns list of (confidence, (x, y, w, h)).
    """
    mtcnn = _get_mtcnn()
    
    h, w = image.shape[:2]
    long_side = max(h, w)

    if long_side > _MTCNN_DETECT_MAX_SIDE:
        scale = _MTCNN_DETECT_MAX_SIDE / long_side
        detect_h = round(h * scale)
        detect_w = round(w * scale)
        small_image = transform.resize(image, (detect_h, detect_w), anti_aliasing=True, preserve_range=True)
        sx = w / detect_w
        sy = h / detect_h
    else:
        small_image = image
        sx = sy = 1.0

    img_u8 = img_as_ubyte(np.clip(small_image, 0, 1))
    faces = mtcnn.detect_faces(img_u8)

    results = []
    for face in faces:
        if face["confidence"] < 0.9:
            continue
        x, y, w, h = face["box"]
        x = max(0, int(x * sx))
        y = max(0, int(y * sy))
        w = max(1, int(w * sx))
        h = max(1, int(h * sy))
        results.append((face["confidence"], (x, y, w, h)))
    return results

def crop_largest_face(image, margin=0.0):
    """Crop the largest detected face from the image. Returns original if no faces detected."""
    faces = _detect_faces_mtcnn(image)
    if not faces:
        return image

    _, (x, y, w, h) = max(faces, key=lambda item: item[0])
    cx, cy = x + w // 2, y + h // 2
    half = int(max(w, h) * (0.5 + margin))
    y0 = max(0, cy - half)
    y1 = min(image.shape[0], cy + half)
    x0 = max(0, cx - half)
    x1 = min(image.shape[1], cx + half)
    return image[y0:y1, x0:x1]

def get_largest_face_box(image):
    """Return (x, y, w, h) of the highest-confidence detected face, or None."""
    faces = _detect_faces_mtcnn(image)
    if not faces:
        return None
    _, box = max(faces, key=lambda item: item[0])
    return box

def get_all_face_boxes(image):
    """Return list of (x, y, w, h) for all detected faces, sorted by confidence descending."""
    faces = _detect_faces_mtcnn(image)
    return [box for _, box in sorted(faces, key=lambda f: f[0], reverse=True)]

def draw_box(image, thickness=3):
    """Draw a box around the detected face for visualization."""
    faces = _detect_faces_mtcnn(image)
    if not faces:
        return image.copy()
    
    _, (x, y, w, h) = max(faces, key=lambda item: item[0])
    out = np.clip(image.copy(), 0, 1)
    for t in range(thickness):
        rr, cc = rectangle_perimeter(
            (y + t, x + t),
            (y + h - t, x + w - t),
            shape=out.shape[:2],
        )
        out[rr, cc] = [1.0, 0.0, 0.0]
    return out

def match_dataset_resolution(img, max_side=200):
    """Downsample an oversized image so its longest edge is at most max_side px."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_shape = (round(h * scale), round(w * scale))
    return transform.resize(img, new_shape, anti_aliasing=True, preserve_range=True)

def preprocess_pipeline(path, image_size=(128, 128), contrast=None, crop_face=False, match_resolution=False, resolution_max_side=200):
    """RGB load -> optional face crop -> optional resolution match -> grayscale -> optional resize -> optional contrast adjustment."""
    image = read_rgb(path)
    if crop_face:
        image = crop_largest_face(image)
    if match_resolution:
        image = match_dataset_resolution(image, max_side=resolution_max_side)
    image = color.rgb2gray(image)
    if image_size is not None:
        image = resize_image(image, image_size)
    image = adjust_contrast(image, method=contrast)
    return image.astype(np.float32)
    

class HOGTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer: list of image paths to HOG feature matrix."""
    def __init__(self, image_size=(128, 128), crop_face=True,
                 orientations=9, pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2), block_norm='L2-Hys'):
        self.image_size = image_size
        self.crop_face = crop_face
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from skimage.feature import hog
        features = []
        for p in X:
            img = preprocess_pipeline(p, image_size=self.image_size,
                                      crop_face=self.crop_face,
                                      match_resolution=self.crop_face)
            features.append(hog(img, orientations=self.orientations,
                                 pixels_per_cell=self.pixels_per_cell,
                                 cells_per_block=self.cells_per_block,
                                 block_norm=self.block_norm,
                                 feature_vector=True))
        return np.array(features)


class SIFTBoVWTransformer(BaseEstimator, TransformerMixin):
    """sklearn transformer: list of image paths to BoVW histogram matrix."""
    def __init__(self, n_codewords=200, crop_face=True):
        self.n_codewords = n_codewords
        self.crop_face = crop_face

    def fit(self, X, y=None):
        import cv2
        from sklearn.cluster import MiniBatchKMeans
        sift = cv2.SIFT_create()
        des_list = []
        for p in X:
            img_rgb = read_rgb(str(p))
            if self.crop_face:
                img_rgb = crop_largest_face(img_rgb)
                img_rgb = match_dataset_resolution(img_rgb)
            img = img_as_ubyte(color.rgb2gray(img_rgb))
            _, des = sift.detectAndCompute(img, None)
            if des is not None:
                des_list.append(des)
        des_array = np.vstack(des_list)
        batch_size = des_array.shape[0] // 4
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_codewords, batch_size=batch_size, n_init='auto').fit(des_array)
        return self

    def transform(self, X):
        import cv2
        sift = cv2.SIFT_create()
        hist_list = []
        for p in X:
            img_rgb = read_rgb(str(p))
            if self.crop_face:
                img_rgb = crop_largest_face(img_rgb)
                img_rgb = match_dataset_resolution(img_rgb)
            img = img_as_ubyte(color.rgb2gray(img_rgb))
            _, des = sift.detectAndCompute(img, None)
            hist = np.zeros(self.n_codewords)
            if des is not None:
                for j in self.kmeans_.predict(des):
                    hist[j] += 1 / len(des)
            hist_list.append(hist)
        return np.vstack(hist_list)