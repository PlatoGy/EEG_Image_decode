import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def concept_from_folder(folder):
    return folder[folder.index("_") + 1:] if "_" in folder else folder


def collect_image_paths(image_dir):
    folders = [
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    ]
    folders.sort()

    texts = []
    images = []
    for folder in folders:
        description = concept_from_folder(folder)
        texts.append(f"This picture is {description}")

        folder_path = Path(image_dir) / folder
        image_names = [
            name for name in os.listdir(folder_path)
            if name.lower().endswith(IMAGE_EXTS)
        ]
        image_names.sort()
        images.extend(str(folder_path / name) for name in image_names)

    return folders, texts, images


@torch.no_grad()
def encode_texts(model, texts, device, batch_size):
    import clip

    features = []
    dtype = next(model.parameters()).dtype
    for start in tqdm(range(0, len(texts), batch_size), desc="text features"):
        batch_texts = texts[start:start + batch_size]
        text_inputs = torch.cat([clip.tokenize(t) for t in batch_texts]).to(device)
        text_features = model.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1).detach()
        features.append(text_features.float().cpu())
        del text_inputs, text_features
    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size):
    features = []
    dtype = next(model.parameters()).dtype
    for start in tqdm(range(0, len(image_paths), batch_size), desc="image features"):
        batch_paths = image_paths[start:start + batch_size]
        image_inputs = torch.stack([
            preprocess(Image.open(path).convert("RGB"))
            for path in batch_paths
        ]).to(device=device, dtype=dtype)

        image_features = model.encode_image(image_inputs)
        features.append(image_features.float().cpu())
        del image_inputs, image_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(features, dim=0)


def encode_split(args, model, preprocess, device, split, image_dir, output_path):
    folders, texts, images = collect_image_paths(image_dir)
    print(split, "concepts:", len(folders), "images:", len(images))
    print(split, "image_dir:", image_dir)
    print(split, "output:", output_path)

    text_features = encode_texts(model, texts, device, args.text_batch_size)
    img_features = encode_images(model, preprocess, images, device, args.batch_size)

    norms = img_features.norm(dim=1)
    print(
        split,
        "raw image feature norm:",
        f"mean={norms.mean().item():.6f}",
        f"std={norms.std().item():.6f}",
        f"min={norms.min().item():.6f}",
        f"max={norms.max().item():.6f}",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_features": text_features,
            "img_features": img_features,
            "folders": folders,
            "image_paths": images,
        },
        output_path,
    )
    print("saved:", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract raw ViT-H-14 image features for generation prior training.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")
    parser.add_argument("--model-type", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    parser.add_argument("--train-image-dir", default=None)
    parser.add_argument("--test-image-dir", default=None)
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--test-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    precision = args.precision if device.type == "cuda" else "fp32"

    import open_clip

    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        args.model_type,
        pretrained=args.pretrained,
        precision=precision,
        device=device,
    )
    model.eval().requires_grad_(False)

    train_image_dir = args.train_image_dir or str(Path(args.data_root) / "images_set" / "training_images")
    test_image_dir = args.test_image_dir or str(Path(args.data_root) / "images_set" / "test_images")
    train_output = args.train_output or str(Path(args.data_root) / "ViT-H-14_features_train_raw.pt")
    test_output = args.test_output or str(Path(args.data_root) / "ViT-H-14_features_test_raw.pt")

    print("device:", device)
    print("precision:", precision)
    print("batch_size:", args.batch_size)

    if args.split in ("train", "both"):
        encode_split(args, model, preprocess_train, device, "train", train_image_dir, train_output)
    if args.split in ("test", "both"):
        encode_split(args, model, preprocess_train, device, "test", test_image_dir, test_output)


if __name__ == "__main__":
    main()
