import os
import sys
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get(
        "EEG_IMAGE_DECODE_REPO",
        "/data/gaoy/projects/EEG_Image_decode",
    )
).expanduser()

LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    os.environ.get(
        "EEG_IMAGE_DECODE_DATA",
        "/data/gaoy/projects/datasets/EEG_Image_decode",
    )
).expanduser()

EEG_DATA_PATH = DATA_ROOT / "Preprocessed_data_250Hz"
TRAIN_IMAGE_DIR = DATA_ROOT / "images_set" / "training_images"
TEST_IMAGE_DIR = DATA_ROOT / "images_set" / "test_images"
CLIP_TRAIN_FEATURES = DATA_ROOT / "ViT-H-14_features_train.pt"
CLIP_TEST_FEATURES = DATA_ROOT / "ViT-H-14_features_test.pt"
EEG_EMB_DIR = DATA_ROOT / "emb_eeg"
FINETUNE_CKPT_DIR = DATA_ROOT / "fintune_ckpts"
GENERATED_IMAGE_DIR = DATA_ROOT / "generated_imgs"

GENERATED_IMAGE_TENSOR_DIR = DATA_ROOT / "generated_imgs_tensor"
TEST_IMAGE_TENSOR_DIR = DATA_ROOT / "images_set" / "test_images_tensor"
LOW_LEVEL_VAE_IMAGE_DIR = DATA_ROOT / "vae_imgs" / "epoch_170"
RECONSTRUCTED_IMAGE_DIR = DATA_ROOT / "reconstructed_imgs"

SUBJECT = "sub-08"
ENCODER_TYPE = "ATM_S"

RELEASED_EEG_FEATURES = EEG_EMB_DIR / f"{ENCODER_TYPE}_eeg_features_{SUBJECT}.pt"
RELEASED_EEG_TRAIN_FEATURES = EEG_EMB_DIR / f"{ENCODER_TYPE}_eeg_features_{SUBJECT}_train.pt"
RELEASED_EEG_TEST_FEATURES = EEG_EMB_DIR / f"{ENCODER_TYPE}_eeg_features_{SUBJECT}_test.pt"
DIFFUSION_PRIOR_CKPT = FINETUNE_CKPT_DIR / SUBJECT / "diffusion_prior.pt"
ATMS_CHECKPOINT = Path(
    os.environ.get(
        "EEG_IMAGE_DECODE_ATMS_CKPT",
        REPO_ROOT / "models" / "contrast" / "ATMS" / "02-01_00-39" / SUBJECT / "40.pth",
    )
).expanduser()


def ensure_repo_on_path():
    for root in (REPO_ROOT, LOCAL_REPO_ROOT):
        for path in (root, root / "Generation"):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
