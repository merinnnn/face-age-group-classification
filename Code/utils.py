from collections import Counter
from pathlib import Path
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import color, exposure, img_as_float, img_as_ubyte, transform
from skimage.io import imread

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

AGE_LABELS = {
    0: "Child",
    1: "Young",
    2: "Middle-Aged",
    3: "Senior",
}

# ENVIRONMENT SETUP

def find_project_root(start=Path.cwd()):
    """Find the coursework folder whether the notebook is run locally or in Colab."""
    start = Path(start).resolve()
    candidates = [start, *start.parents]
    candidates.append(Path("/content/drive/MyDrive/CW_Folder_UG"))

    for candidate in candidates:
        if (candidate / "CW_Dataset").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find CW_Dataset. Check your notebook working directory or Drive path."
    )

def get_project_paths(root=None):
    """Return the standard coursework paths used by the model notebooks."""
    root = Path(root) if root is not None else find_project_root()
    data_dir = root / "CW_Dataset"
    code2_dir = root / "Code 2"
    models_dir = root / "Models"
    output_dir = code2_dir / "outputs"

    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    return {
        "ROOT": root,
        "DATA_DIR": data_dir,
        "TRAIN_IMGS": data_dir / "train",
        "TRAIN_LBL": data_dir / "train" / "train_labels.txt",
        "TEST_IMGS": data_dir / "test",
        "TEST_LBL": data_dir / "test" / "test_labels.txt",
        "MODELS_DIR": models_dir,
        "OUTPUT_DIR": output_dir,
    }

# DATA LOADING

def load_labels(image_dir, label_file):
    """Load image paths and integer labels from a coursework label file."""
    image_dir = Path(image_dir)
    paths = []
    labels = []

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            filename, label = line.strip().split()
            paths.append(image_dir / filename)
            labels.append(int(label))

    return paths, labels

def load_coursework_dataset(root=None, verbose=True):
    """Load train/test paths and labels for the coursework dataset."""
    paths = get_project_paths(root)
    train_paths, train_labels = load_labels(paths["TRAIN_IMGS"], paths["TRAIN_LBL"])
    test_paths, test_labels = load_labels(paths["TEST_IMGS"], paths["TEST_LBL"])

    if verbose:
        print("Train samples:", len(train_paths))
        print("Test samples:", len(test_paths))
        print("Train class distribution:", Counter(train_labels))
        print("Test class distribution:", Counter(test_labels))

    return train_paths, train_labels, test_paths, test_labels

# DATA SPLITTING

def make_validation_split(train_paths, train_labels, test_size=0.1, seed=SEED):
    """Create a stratified train/validation split from the provided training set."""
    return train_test_split(
        train_paths,
        train_labels,
        test_size=test_size,
        stratify=train_labels,
        random_state=seed,
    )

# IMAGE PREPROCESSING

def read_rgb(path):
    """Load an image as RGB float in [0, 1].

    Reference: repurposed from Lab_02 code cell 3 (imread)
    and cell 12 (colour/dtype handling with skimage).
    """
    img = imread(path)

    if img.ndim == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img_as_float(img)


def resize_image(img, image_size):
    """Resize while keeping intensities in [0, 1].

    Reference: repurposed from Lab_05 code cell 20.
    """
    resized = transform.resize(
        img,
        image_size,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(resized, 0, 1)

def adjust_contrast(img, method=None, gamma=1.0):
    """Apply optional contrast preprocessing.

    Reference: repurposed from Lab_02 code cell 19.
    """
    if method is None:
        return img

    is_rgb = img.ndim == 3
    work = img.copy()

    if is_rgb:
        hsv = color.rgb2hsv(work)
        channel = hsv[:, :, 2]
    else:
        channel = work

    if method == "stretch":
        p2, p98 = np.percentile(channel, (2, 98))
        channel = exposure.rescale_intensity(channel, in_range=(p2, p98))
    elif method == "equalize":
        channel = exposure.equalize_hist(channel)
    elif method == "gamma":
        channel = exposure.adjust_gamma(channel, gamma=gamma)
    else:
        raise ValueError("method must be None, 'stretch', 'equalize', or 'gamma'")

    if is_rgb:
        hsv[:, :, 2] = channel
        return np.clip(color.hsv2rgb(hsv), 0, 1)

    return np.clip(channel, 0, 1)

def crop_largest_face(img, margin=0.20):
    """Crop the largest detected face; return the original image if no face is found.

    Reference: repurposed from Lab_05 code cell 17.
    """
    gray = color.rgb2gray(img)
    gray_u8 = img_as_ubyte(gray)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray_u8, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return img

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = int(max(w, h) * margin)
    y0 = max(0, y - pad)
    y1 = min(img.shape[0], y + h + pad)
    x0 = max(0, x - pad)
    x1 = min(img.shape[1], x + w + pad)

    return img[y0:y1, x0:x1]

def preprocess_classical(path, image_size=(128, 128), contrast=None, crop_face=False):
    """Preprocess one image for HOG/SIFT/SVM/MLP-style classical models.

    Reference: new combination of tutorial pieces: image loading from Lab_02
    code cell 3, grayscale conversion from Lab_02 code cell 21, resizing adapted
    from Lab_05 code cell 20, and optional contrast from Lab_02 code cell 19.
    """
    img = read_rgb(path)

    if crop_face:
        img = crop_largest_face(img)

    gray = color.rgb2gray(img)
    gray = resize_image(gray, image_size)
    gray = adjust_contrast(gray, method=contrast)

    return gray.astype(np.float32)

def preprocess_cnn(
    path, image_size=(224, 224), contrast=None, crop_face=False, imagenet_norm=False
):
    """Preprocess one image for a CNN if you are not using torchvision transforms.

    Reference: new NumPy/skimage version inspired by Lab_08 code cell 7, where
    CNN inputs are resized/cropped, converted to tensors, and normalized. This
    function keeps the output as a NumPy array for visual testing and later reuse.
    """
    img = read_rgb(path)

    if crop_face:
        img = crop_largest_face(img)

    img = resize_image(img, image_size)
    img = adjust_contrast(img, method=contrast)
    img = img.astype(np.float32)

    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    return img

def preprocess_many(paths, preprocess_fn, max_images=None, **kwargs):
    """Apply a preprocessing function to many image paths and stack the results."""
    selected_paths = paths if max_images is None else paths[:max_images]
    images = [preprocess_fn(path, **kwargs) for path in selected_paths]
    return np.stack(images)

def flatten_images(images):
    """Flatten image arrays for scikit-learn classifiers."""
    return images.reshape(images.shape[0], -1)
