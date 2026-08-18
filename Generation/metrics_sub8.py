import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from path_config import DATA_ROOT, GENERATED_IMAGE_DIR, TEST_IMAGE_DIR


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def torch_checkpoint_dir():
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")).expanduser()
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def concept_name(folder):
    try:
        return folder[folder.index("_") + 1 :]
    except ValueError:
        return folder


def image_files(path):
    return sorted(
        [item for item in Path(path).iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS]
    )


def load_image(path, size=512):
    image = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_ground_truth():
    folders = sorted([d for d in os.listdir(TEST_IMAGE_DIR) if (TEST_IMAGE_DIR / d).is_dir()])
    names = []
    tensors = []
    for folder in folders:
        files = image_files(TEST_IMAGE_DIR / folder)
        if not files:
            raise FileNotFoundError(f"No test image found in {TEST_IMAGE_DIR / folder}")
        names.append(concept_name(folder))
        tensors.append(load_image(files[0]))
    return names, torch.stack(tensors)


def load_reconstructions(names, subject, repeat):
    root = GENERATED_IMAGE_DIR / subject
    tensors = []
    missing = []
    for name in names:
        path = root / name / f"{repeat}.png"
        if not path.exists():
            missing.append(path)
            continue
        tensors.append(load_image(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} generated images for repeat {repeat}; first missing: {missing[0]}")
    return torch.stack(tensors)


def resize_batch(images, size):
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def normalize_batch(images, mean, std):
    mean = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def pixcorr(recons, gt):
    recons = resize_batch(recons, 425).flatten(1).cpu().numpy()
    gt = resize_batch(gt, 425).flatten(1).cpu().numpy()
    values = [np.corrcoef(gt[i], recons[i])[0, 1] for i in range(len(gt))]
    return float(np.mean(values))


def ssim_metric(recons, gt):
    recons = resize_batch(recons, 425).clamp(0, 1)
    gt = resize_batch(gt, 425).clamp(0, 1)
    try:
        from skimage.color import rgb2gray
        from skimage.metrics import structural_similarity as skimage_ssim

        gt_gray = rgb2gray(gt.permute(0, 2, 3, 1).cpu().numpy())
        recon_gray = rgb2gray(recons.permute(0, 2, 3, 1).cpu().numpy())
        values = [
            skimage_ssim(
                rec,
                im,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
            for im, rec in zip(gt_gray, recon_gray)
        ]
        return float(np.mean(values))
    except ImportError:
        from pytorch_msssim import ssim

        print("scikit-image is not installed; falling back to pytorch-msssim for SSIM.")
        return float(ssim(recons, gt, data_range=1.0, size_average=True).item())


@torch.no_grad()
def extract_features(images, model, feature_layer, preprocess, device, batch_size):
    feats = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start : start + batch_size].to(device))
        out = model(batch)
        if feature_layer is not None:
            out = out[feature_layer]
        feats.append(out.float().flatten(1).cpu())
    return torch.cat(feats, dim=0).numpy()


def two_way_identification(recons, gt, model, feature_layer, preprocess, device, batch_size):
    pred_features = extract_features(recons, model, feature_layer, preprocess, device, batch_size)
    gt_features = extract_features(gt, model, feature_layer, preprocess, device, batch_size)
    r = np.corrcoef(gt_features, pred_features)
    r = r[: len(gt), len(gt) :]
    congruents = np.diag(r)
    success = r < congruents
    return float(np.mean(np.sum(success, axis=0)) / (len(gt) - 1))


def build_deep_metrics(device):
    from torchvision.models import (
        AlexNet_Weights,
        EfficientNet_B1_Weights,
        Inception_V3_Weights,
        alexnet,
        efficientnet_b1,
        inception_v3,
    )
    import clip

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(
        alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]
    ).to(device)
    alex_model.eval().requires_grad_(False)
    alex_preprocess = lambda x: normalize_batch(resize_batch(x, 256), imagenet_mean, imagenet_std)

    inception_weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(
        inception_v3(weights=inception_weights), return_nodes=["avgpool"]
    ).to(device)
    inception_model.eval().requires_grad_(False)
    inception_preprocess = lambda x: normalize_batch(resize_batch(x, 342), imagenet_mean, imagenet_std)

    clip_model, _ = clip.load("ViT-L/14", device=device, download_root=str(torch_checkpoint_dir()))
    clip_model.eval().requires_grad_(False)
    clip_preprocess = lambda x: normalize_batch(
        resize_batch(x, 224),
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    )

    eff_weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(efficientnet_b1(weights=eff_weights), return_nodes=["avgpool"]).to(device)
    eff_model.eval().requires_grad_(False)
    eff_preprocess = lambda x: normalize_batch(resize_batch(x, 255), imagenet_mean, imagenet_std)

    return {
        "AlexNet(2)": (alex_model, "features.4", alex_preprocess, "two_way"),
        "AlexNet(5)": (alex_model, "features.11", alex_preprocess, "two_way"),
        "InceptionV3": (inception_model, "avgpool", inception_preprocess, "two_way"),
        "CLIP": (clip_model.encode_image, None, clip_preprocess, "two_way"),
        "EffNet-B": (eff_model, "avgpool", eff_preprocess, "distance"),
    }


def effnet_distance(recons, gt, model, feature_layer, preprocess, device, batch_size):
    pred_features = extract_features(recons, model, feature_layer, preprocess, device, batch_size)
    gt_features = extract_features(gt, model, feature_layer, preprocess, device, batch_size)
    return float(np.array([sp.spatial.distance.correlation(gt_features[i], pred_features[i]) for i in range(len(gt))]).mean())


def run_repeat(recons, gt, device, batch_size, deep_metrics):
    recons = recons.to(device)
    gt = gt.to(device)
    row = {
        "PixCorr": pixcorr(recons, gt),
        "SSIM": ssim_metric(recons, gt),
    }
    for name, (model, feature_layer, preprocess, kind) in deep_metrics.items():
        if kind == "two_way":
            row[name] = two_way_identification(recons, gt, model, feature_layer, preprocess, device, batch_size)
        elif kind == "distance":
            row[name] = effnet_distance(recons, gt, model, feature_layer, preprocess, device, batch_size)
    return row


def main():
    parser = argparse.ArgumentParser(description="Compute Subject-08 reconstruction metrics.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--basic-only", action="store_true", help="Only compute PixCorr and SSIM.")
    parser.add_argument("--output", default=str(DATA_ROOT / "metrics_sub8.csv"))
    args = parser.parse_args()

    device = torch.device(args.device)
    names, gt = load_ground_truth()
    print(f"Loaded {len(names)} ground-truth test images from {TEST_IMAGE_DIR}")

    deep_metrics = {} if args.basic_only else build_deep_metrics(device)
    rows = []
    for repeat in tqdm(range(args.repeats), desc="repeats"):
        recons = load_reconstructions(names, args.subject, repeat)
        result = run_repeat(recons, gt, device, args.batch_size, deep_metrics)
        result["repeat"] = repeat
        rows.append(result)
        print(pd.DataFrame([result]).to_string(index=False))

    df = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    summary = df.drop(columns=["repeat"]).agg(["mean", "std"]).T.reset_index(names="Metric")
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved per-repeat metrics: {output}")
    print(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
