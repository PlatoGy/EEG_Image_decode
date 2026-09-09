import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import scipy as sp
import torch
from PIL import Image
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


def collect_images(generated_dir, ground_truth_dir, repeats, image_size, start_index, num_concepts):
    gt_folders = [
        d for d in os.listdir(ground_truth_dir)
        if os.path.isdir(os.path.join(ground_truth_dir, d))
    ]
    gt_folders.sort()
    selected = gt_folders[start_index:start_index + num_concepts if num_concepts is not None else None]

    concepts = []
    gt_paths = []
    generated_paths = []
    generated_labels = []

    for class_index, folder in enumerate(selected):
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
            raise FileNotFoundError(f"No generated image in {recon_folder}")

        concepts.append(concept)
        gt_paths.append(gt_path)
        for image_name in recon_images:
            generated_paths.append(recon_folder / image_name)
            generated_labels.append(class_index)

    gt_images = torch.stack([load_image_tensor(path, image_size) for path in gt_paths], dim=0)
    generated_images = torch.stack([load_image_tensor(path, image_size) for path in generated_paths], dim=0)
    generated_labels = torch.tensor(generated_labels, dtype=torch.long)
    return {
        "concepts": concepts,
        "gt_paths": gt_paths,
        "generated_paths": generated_paths,
        "gt_images": gt_images,
        "generated_images": generated_images,
        "generated_labels": generated_labels,
    }


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def extract_features(images, model, preprocess, feature_layer, device, batch_size, desc):
    features = []
    for start in tqdm(range(0, len(images), batch_size), desc=desc):
        end = min(start + batch_size, len(images))
        batch = torch.stack([preprocess(image) for image in images[start:end]], dim=0).to(device)
        outputs = model(batch)
        if feature_layer is not None:
            outputs = outputs[feature_layer]
        features.append(outputs.float().flatten(1).cpu())
        del batch, outputs
    return torch.cat(features, dim=0)


@torch.no_grad()
def extract_inception_logits(images, model, preprocess, device, batch_size, desc):
    logits = []
    for start in tqdm(range(0, len(images), batch_size), desc=desc):
        end = min(start + batch_size, len(images))
        batch = torch.stack([preprocess(image) for image in images[start:end]], dim=0).to(device)
        outputs = model(batch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits.append(outputs.float().cpu())
        del batch, outputs
    return torch.cat(logits, dim=0)


def inception_preprocess(image_size=299):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_inception_feature_model(device):
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(inception_v3(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    return model


def load_inception_classifier(device):
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights).to(device)
    model.eval().requires_grad_(False)
    return model


def fid_from_features(real_features, fake_features, eps=1e-6):
    real = real_features.double().numpy()
    fake = fake_features.double().numpy()
    mu_real = np.mean(real, axis=0)
    mu_fake = np.mean(fake, axis=0)
    sigma_real = np.cov(real, rowvar=False)
    sigma_fake = np.cov(fake, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = sp.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sp.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


def polynomial_mmd2_unbiased(x, y, degree=3, gamma=None, coef0=1.0):
    x = x.double()
    y = y.double()
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    k_xx = (gamma * x.mm(x.t()) + coef0).pow(degree)
    k_yy = (gamma * y.mm(y.t()) + coef0).pow(degree)
    k_xy = (gamma * x.mm(y.t()) + coef0).pow(degree)

    m = x.shape[0]
    n = y.shape[0]
    sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def kid_from_features(real_features, fake_features, subset_size=100, num_subsets=100, seed=42):
    total = min(real_features.shape[0], fake_features.shape[0])
    subset_size = min(int(subset_size), total)
    if subset_size < 2:
        raise ValueError("KID subset size must be at least 2.")

    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(int(num_subsets)):
        real_idx = torch.randperm(real_features.shape[0], generator=generator)[:subset_size]
        fake_idx = torch.randperm(fake_features.shape[0], generator=generator)[:subset_size]
        values.append(polynomial_mmd2_unbiased(real_features[real_idx], fake_features[fake_idx]).item())
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def inception_score_from_logits(logits, splits=10, eps=1e-16):
    probs = torch.softmax(logits.float(), dim=1).numpy()
    splits = min(int(splits), len(probs))
    if splits < 1:
        raise ValueError("inception score splits must be at least 1.")
    scores = []
    for part in np.array_split(probs, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    scores = np.asarray(scores, dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=1) if len(scores) > 1 else 0.0)


def fid_kid_is_metrics(data, device, args):
    from torchvision.models import Inception_V3_Weights

    preprocess = inception_preprocess(args.inception_image_size)
    feature_model = load_inception_feature_model(device)
    gt_images = data["gt_images"].float()
    generated_images = data["generated_images"].float()

    real_features = extract_features(
        gt_images, feature_model, preprocess, "avgpool", device, args.feature_batch_size, "Inception real features"
    )
    fake_features = extract_features(
        generated_images, feature_model, preprocess, "avgpool", device, args.feature_batch_size, "Inception generated features"
    )
    del feature_model
    cleanup_cuda()

    if args.fid_real_mode == "repeat":
        labels = data["generated_labels"]
        real_for_distribution = real_features[labels]
    else:
        real_for_distribution = real_features

    results = {}
    if "fid" in args.metrics:
        results["FID"] = fid_from_features(real_for_distribution, fake_features)
    if "kid" in args.metrics:
        kid_mean, kid_std = kid_from_features(
            real_for_distribution,
            fake_features,
            subset_size=args.kid_subset_size,
            num_subsets=args.kid_num_subsets,
            seed=args.seed,
        )
        results["KID"] = kid_mean
        results["KID_std"] = kid_std
    if "is" in args.metrics:
        classifier = load_inception_classifier(device)
        logits = extract_inception_logits(
            generated_images,
            classifier,
            preprocess,
            device,
            args.feature_batch_size,
            "Inception generated logits",
        )
        del classifier
        cleanup_cuda()
        is_mean, is_std = inception_score_from_logits(logits, splits=args.inception_score_splits)
        results["IS"] = is_mean
        results["IS_std"] = is_std

    # Keep torchvision from warning that the imported weights are unused in some execution paths.
    _ = Inception_V3_Weights.DEFAULT
    return results


def l2_normalize(features):
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def clip_image_features(images, device, batch_size, model_name):
    import clip

    model, _ = clip.load(model_name, device=device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    features = extract_features(images, model.encode_image, preprocess, None, device, batch_size, f"{model_name} features")
    del model
    cleanup_cuda()
    return l2_normalize(features.float())


@torch.no_grad()
def inception_retrieval_features(images, device, batch_size, image_size):
    preprocess = inception_preprocess(image_size)
    model = load_inception_feature_model(device)
    features = extract_features(images, model, preprocess, "avgpool", device, batch_size, "Inception retrieval features")
    del model
    cleanup_cuda()
    return l2_normalize(features.float())


@torch.no_grad()
def efficientnet_retrieval_features(images, device, batch_size):
    from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.Resize((255, 255), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = extract_features(images, model, preprocess, "avgpool", device, batch_size, "EfficientNet retrieval features")
    del model
    cleanup_cuda()
    return l2_normalize(features.float())


def retrieval_features(images, device, args):
    if args.retrieval_backbone == "clip":
        return clip_image_features(images, device, args.feature_batch_size, args.clip_model)
    if args.retrieval_backbone == "inception":
        return inception_retrieval_features(images, device, args.feature_batch_size, args.inception_image_size)
    if args.retrieval_backbone == "efficientnet":
        return efficientnet_retrieval_features(images, device, args.feature_batch_size)
    raise ValueError(f"Unknown retrieval backbone: {args.retrieval_backbone}")


def save_retrieval_rows(output_dir, rows):
    path = output_dir / "retrieval_200way.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generated_index",
            "concept",
            "generated_path",
            "target_rank",
            "target_score",
            "top1_concept",
            "top1_score",
            "top5_concepts",
            "top5_scores",
        ])
        writer.writerows(rows)
    print("saved retrieval rows:", path)


def topk_accuracy_metrics(data, device, args, output_dir):
    gt_features = retrieval_features(data["gt_images"].float(), device, args)
    generated_features = retrieval_features(data["generated_images"].float(), device, args)
    labels = data["generated_labels"]
    concepts = data["concepts"]

    scores = generated_features @ gt_features.t()
    max_k = min(5, scores.shape[1])
    top_scores, top_indices = torch.topk(scores, k=max_k, dim=1)
    top1_correct = (top_indices[:, 0].cpu() == labels).float()
    top5_correct = (top_indices.cpu() == labels[:, None]).any(dim=1).float()
    target_scores = scores[torch.arange(scores.shape[0]), labels].cpu()
    target_ranks = (scores.cpu() > target_scores[:, None]).sum(dim=1) + 1

    rows = []
    for idx in range(scores.shape[0]):
        top5_ids = top_indices[idx].cpu().tolist()
        top5_vals = top_scores[idx].cpu().tolist()
        rows.append([
            idx,
            concepts[int(labels[idx].item())],
            str(data["generated_paths"][idx]),
            int(target_ranks[idx].item()),
            float(target_scores[idx].item()),
            concepts[int(top5_ids[0])],
            float(top5_vals[0]),
            ";".join(concepts[int(class_id)] for class_id in top5_ids),
            ";".join(f"{float(value):.8f}" for value in top5_vals),
        ])
    save_retrieval_rows(output_dir, rows)

    return {
        "200way_top1_acc": float(top1_correct.mean().item()),
        "200way_top5_acc": float(top5_correct.mean().item()),
        "200way_mean_rank": float(target_ranks.float().mean().item()),
        "200way_median_rank": float(target_ranks.float().median().item()),
    }


def save_results(output_dir, args, results):
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in results.items():
            writer.writerow([key, value])
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run extra reconstruction metrics: FID, KID, IS, and 200-way top-k accuracy.")
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--ground-truth-dir", default="/data/gaoy/projects/datasets/EEG_Image_decode/images_set/test_images")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-concepts", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fid", "kid", "is", "topk"],
        choices=["fid", "kid", "is", "topk"],
    )

    parser.add_argument("--fid-real-mode", choices=["unique", "repeat"], default="unique")
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--kid-num-subsets", type=int, default=100)
    parser.add_argument("--inception-score-splits", type=int, default=10)
    parser.add_argument("--inception-image-size", type=int, default=299)

    parser.add_argument("--retrieval-backbone", choices=["clip", "inception", "efficientnet"], default="clip")
    parser.add_argument("--clip-model", default="ViT-L/14")

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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.generated_dir).parents[1] / "extra_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = collect_images(
        args.generated_dir,
        args.ground_truth_dir,
        args.repeats,
        args.image_size,
        args.start_index,
        args.num_concepts,
    )
    print("concepts:", len(data["concepts"]))
    print("generated images:", len(data["generated_paths"]))
    print("ground truth images:", len(data["gt_paths"]))
    print("generated tensor:", tuple(data["generated_images"].shape))
    print("ground truth tensor:", tuple(data["gt_images"].shape))
    print("metrics:", args.metrics)
    print("retrieval backbone:", args.retrieval_backbone)

    results = {}
    if any(metric in args.metrics for metric in ("fid", "kid", "is")):
        results.update(fid_kid_is_metrics(data, device, args))
        for key, value in results.items():
            print(f"{key}: {value}")

    if "topk" in args.metrics:
        topk_results = topk_accuracy_metrics(data, device, args, output_dir)
        results.update(topk_results)
        for key, value in topk_results.items():
            print(f"{key}: {value}")

    save_results(output_dir, args, results)
    print("metrics_dir:", output_dir)


if __name__ == "__main__":
    main()
