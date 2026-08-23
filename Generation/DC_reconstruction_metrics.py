import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import scipy as sp
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def image_files(folder):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    ])


def concept_from_folder(folder):
    return folder[folder.index("_") + 1:] if "_" in folder else folder


def load_image_tensor(path, image_size):
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def collect_pairs(generated_dir, ground_truth_dir, repeats, image_size, start_index, num_concepts):
    gt_folders = [
        d for d in os.listdir(ground_truth_dir)
        if os.path.isdir(os.path.join(ground_truth_dir, d))
    ]
    gt_folders.sort()

    pairs = []
    selected = gt_folders[start_index:start_index + num_concepts if num_concepts is not None else None]
    for folder in selected:
        concept = concept_from_folder(folder)
        gt_folder = Path(ground_truth_dir) / folder
        gt_images = image_files(gt_folder)
        if not gt_images:
            raise FileNotFoundError(f"No ground-truth image in {gt_folder}")
        gt_path = gt_folder / gt_images[0]

        recon_folder = Path(generated_dir) / concept
        if not recon_folder.exists():
            raise FileNotFoundError(f"Missing generated folder for concept {concept}: {recon_folder}")
        recon_images = image_files(recon_folder)
        if repeats is not None:
            recon_images = recon_images[:repeats]
        if not recon_images:
            raise FileNotFoundError(f"No generated images in {recon_folder}")

        for image_name in recon_images:
            pairs.append((concept, gt_path, recon_folder / image_name))

    concepts = [p[0] for p in pairs]
    all_images = torch.stack([load_image_tensor(p[1], image_size) for p in pairs], dim=0)
    all_brain_recons = torch.stack([load_image_tensor(p[2], image_size) for p in pairs], dim=0)
    return concepts, all_images, all_brain_recons


@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, device="cuda"):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    return np.mean(success_cnt) / (len(all_images) - 1)


def pixcorr_metric(all_brain_recons, all_images):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()

    scores = []
    for i in tqdm(range(len(all_images)), desc="PixCorr"):
        scores.append(np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1])
    return float(np.mean(scores))


def ssim_metric(all_brain_recons, all_images):
    try:
        from skimage.color import rgb2gray
        from skimage.metrics import structural_similarity as skimage_ssim
    except ImportError as exc:
        raise ImportError("scikit-image is required for SSIM in this script.") from exc

    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu())

    scores = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images), desc="SSIM"):
        scores.append(
            skimage_ssim(
                rec,
                im,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )
    return float(np.mean(scores))


def alexnet_metrics(all_brain_recons, all_images, device):
    from torchvision.models import AlexNet_Weights, alexnet

    weights = AlexNet_Weights.IMAGENET1K_V1
    model = create_feature_extractor(
        alexnet(weights=weights),
        return_nodes=["features.4", "features.11"],
    ).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    alexnet2 = two_way_identification(all_brain_recons.to(device).float(), all_images, model, preprocess, "features.4", device)
    alexnet5 = two_way_identification(all_brain_recons.to(device).float(), all_images, model, preprocess, "features.11", device)
    return float(alexnet2), float(alexnet5)


def inception_metric(all_brain_recons, all_images, device):
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(inception_v3(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return float(two_way_identification(all_brain_recons, all_images, model, preprocess, "avgpool", device))


def clip_metric(all_brain_recons, all_images, device):
    import clip

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    return float(two_way_identification(all_brain_recons, all_images, clip_model.encode_image, preprocess, None, device))


def efficientnet_metric(all_brain_recons, all_images, device):
    from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gt = model(preprocess(all_images).to(device))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fake = model(preprocess(all_brain_recons).to(device))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return float(np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean())


def swav_metric(all_brain_recons, all_images, device):
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    model = create_feature_extractor(model, return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gt = model(preprocess(all_images).to(device))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fake = model(preprocess(all_brain_recons).to(device))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return float(np.array([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]).mean())


def parse_args():
    parser = argparse.ArgumentParser(description="Run reconstruction metrics for generated EEG images.")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--ground-truth-dir", default="/data/gaoy/projects/datasets/EEG_Image_decode/images_set/test_images")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-concepts", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pixcorr", "ssim", "alexnet", "inception", "clip", "efficientnet", "swav"],
        choices=["pixcorr", "ssim", "alexnet", "inception", "clip", "efficientnet", "swav"],
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.generated_dir).parents[1] / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts, all_images, all_brain_recons = collect_pairs(
        args.generated_dir,
        args.ground_truth_dir,
        args.repeats,
        args.image_size,
        args.start_index,
        args.num_concepts,
    )
    all_images = all_images.to(device)
    all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0, 1)

    print("pairs:", len(concepts))
    print("ground truth:", tuple(all_images.shape))
    print("recon:", tuple(all_brain_recons.shape))

    results = {}
    if "pixcorr" in args.metrics:
        results["PixCorr"] = pixcorr_metric(all_brain_recons, all_images)
        print("PixCorr:", results["PixCorr"])
    if "ssim" in args.metrics:
        results["SSIM"] = ssim_metric(all_brain_recons, all_images)
        print("SSIM:", results["SSIM"])
    if "alexnet" in args.metrics:
        results["AlexNet(2)"], results["AlexNet(5)"] = alexnet_metrics(all_brain_recons, all_images, device)
        print("AlexNet(2):", results["AlexNet(2)"])
        print("AlexNet(5):", results["AlexNet(5)"])
    if "inception" in args.metrics:
        results["InceptionV3"] = inception_metric(all_brain_recons, all_images, device)
        print("InceptionV3:", results["InceptionV3"])
    if "clip" in args.metrics:
        results["CLIP"] = clip_metric(all_brain_recons, all_images, device)
        print("CLIP:", results["CLIP"])
    if "efficientnet" in args.metrics:
        results["EffNet-B"] = efficientnet_metric(all_brain_recons, all_images, device)
        print("EffNet-B:", results["EffNet-B"])
    if "swav" in args.metrics:
        results["SwAV"] = swav_metric(all_brain_recons, all_images, device)
        print("SwAV:", results["SwAV"])

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            writer.writerow([key, value])
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("metrics_dir:", output_dir)


if __name__ == "__main__":
    main()
