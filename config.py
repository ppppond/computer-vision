import os

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
IMG_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
