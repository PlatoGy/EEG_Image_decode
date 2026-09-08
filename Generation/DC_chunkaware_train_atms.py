import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from DC_featboundary_model import (
    FEATBOUNDARY_BOUNDARIES,
    FEATBOUNDARY_CHUNKS,
    HardBoundaryMasker,
    constrained_top3_boundaries,
    interpolate_scores_to_raw_time,
    temporal_change_score,
)
from DC_train_atms import ATMS, build_feature_datasets, subject_id_from_name


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def load_atms(checkpoint_path, num_subjects, device):
    model = ATMS(63, 250, num_subjects=num_subjects)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.to(device)


def configure_train_scope(model, train_scope):
    for param in model.parameters():
        param.requires_grad_(False)

    if train_scope == "proj":
        for param in model.proj_eeg.parameters():
            param.requires_grad_(True)
    elif train_scope == "enc_proj":
        for param in model.enc_eeg.parameters():
            param.requires_grad_(True)
        for param in model.proj_eeg.parameters():
            param.requires_grad_(True)
    elif train_scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f"Unknown train scope: {train_scope}")

    model.logit_scale.requires_grad_(True)


def count_trainable(module):
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    total = sum(param.numel() for param in module.parameters())
    return trainable, total


class FrozenFeatureBoundaryExtractor(nn.Module):
    def __init__(self, checkpoint_path, num_subjects=2, device="cuda", min_chunk_length=20, smoothing_kernel=5):
        super().__init__()
        self.atms = freeze_module(load_atms(checkpoint_path, num_subjects, device))
        self.masker = HardBoundaryMasker().to(device)
        self.min_chunk_length = int(min_chunk_length)
        self.smoothing_kernel = int(smoothing_kernel)
        self.device = torch.device(device)

    @torch.no_grad()
    def forward(self, eeg, subject_ids):
        eeg = eeg.to(self.device)
        subject_ids = subject_ids.to(self.device)

        encoded = self.atms.encoder(eeg, None, subject_ids)
        temporal_feature = self.atms.enc_eeg[0](encoded)
        eeg_embedding = self.atms.enc_eeg[1](temporal_feature)
        z_global = self.atms.proj_eeg(eeg_embedding)

        feature_scores = temporal_change_score(temporal_feature, self.smoothing_kernel)
        raw_scores = interpolate_scores_to_raw_time(feature_scores, sequence_length=eeg.shape[-1])
        boundaries, lengths, score_peaks = constrained_top3_boundaries(
            raw_scores,
            min_len=self.min_chunk_length,
            sequence_length=eeg.shape[-1],
        )
        chunked, masks = self.masker(eeg, boundaries)
        return {
            "chunked": chunked.float(),
            "z_global": z_global.float(),
            "boundaries": boundaries.float(),
            "lengths": lengths.float(),
            "score_peaks": score_peaks.float(),
            "masks": masks.float(),
        }


def chunk_targets(img_features, labels, z_global):
    return {
        "img": img_features.repeat_interleave(4, dim=0),
        "labels": labels.repeat_interleave(4),
        "global": z_global.repeat_interleave(4, dim=0),
    }


def chunk_atms_forward(model, chunked, subject_ids):
    batch_size, num_chunks, channels, time = chunked.shape
    flat_chunked = chunked.reshape(batch_size * num_chunks, channels, time)
    flat_subject_ids = subject_ids.repeat_interleave(num_chunks)
    features = model(flat_chunked, flat_subject_ids).float()
    return features.reshape(batch_size, num_chunks, -1), features


def train_one_epoch(model, boundary_extractor, dataloader, optimizer, device, subject_id, img_features_all, args):
    model.train()
    if args.train_scope != "all":
        model.encoder.eval()
    img_features_all = img_features_all[::10].to(device).float()
    mse_loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_img_mse = 0.0
    total_clip = 0.0
    total_distill = 0.0
    correct = 0
    total = 0
    boundary_values = []
    length_values = []

    for batch_idx, (eeg_data, labels, text_features, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long, device=device)

        boundary_cache = boundary_extractor(eeg_data, subject_ids)
        chunked = boundary_cache["chunked"]
        targets = chunk_targets(img_features, labels, boundary_cache["z_global"])
        _, chunk_features = chunk_atms_forward(model, chunked, subject_ids)

        img_mse = mse_loss_fn(chunk_features, targets["img"])
        clip_loss = model.loss_func(chunk_features, targets["img"], model.logit_scale)
        distill_loss = mse_loss_fn(chunk_features, targets["global"].detach())
        loss = (
            args.mse_weight * img_mse * 10
            + args.clip_weight * clip_loss * 10
            + args.distill_weight * distill_loss * 10
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_img_mse += img_mse.item()
        total_clip += clip_loss.item()
        total_distill += distill_loss.item()

        logits = model.logit_scale * chunk_features.detach() @ img_features_all.T
        predicted = torch.argmax(logits, dim=1)
        correct += (predicted == targets["labels"]).sum().item()
        total += targets["labels"].numel()
        boundary_values.append(boundary_cache["boundaries"].cpu())
        length_values.append(boundary_cache["lengths"].cpu())

    boundaries = torch.cat(boundary_values, dim=0)
    lengths = torch.cat(length_values, dim=0)
    num_batches = batch_idx + 1
    return {
        "loss": total_loss / num_batches,
        "img_mse": total_img_mse / num_batches,
        "clip_loss": total_clip / num_batches,
        "distill_loss": total_distill / num_batches,
        "train_top1": correct / total,
        "boundary_mean": boundaries.mean(dim=0),
        "length_mean": lengths.mean(dim=0),
    }


@torch.no_grad()
def retrieval_metrics(embeds, img_features):
    paired_cosine = F.cosine_similarity(embeds.float(), img_features.float(), dim=1).mean()
    retrieval = embeds.float() @ img_features.float().T
    top1 = (retrieval.argmax(dim=1) == torch.arange(embeds.shape[0], device=embeds.device)).float().mean()
    return float(paired_cosine.item()), float(top1.item())


@torch.no_grad()
def evaluate_chunk_atms(model, boundary_extractor, dataloader, device, subject_id, img_features_all):
    model.eval()
    img_features_all = img_features_all.to(device).float()
    chunk_features = []
    img_targets = []
    boundary_values = []
    length_values = []
    score_peak_values = []

    for eeg_data, labels, text_features, img_features in dataloader:
        eeg_data = eeg_data.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long, device=device)
        boundary_cache = boundary_extractor(eeg_data, subject_ids)
        features_by_chunk, _ = chunk_atms_forward(model, boundary_cache["chunked"], subject_ids)
        chunk_features.append(features_by_chunk.cpu())
        img_targets.append(img_features.cpu())
        boundary_values.append(boundary_cache["boundaries"].cpu())
        length_values.append(boundary_cache["lengths"].cpu())
        score_peak_values.append(boundary_cache["score_peaks"].cpu())

    chunk_features = torch.cat(chunk_features, dim=0).to(device)
    img_targets = torch.cat(img_targets, dim=0).to(device)
    boundaries = torch.cat(boundary_values, dim=0)
    lengths = torch.cat(length_values, dim=0)
    score_peaks = torch.cat(score_peak_values, dim=0)

    metrics = {}
    for chunk_id in range(4):
        cosine, top1 = retrieval_metrics(chunk_features[:, chunk_id, :], img_targets)
        metrics[f"z{chunk_id + 1}_cosine"] = cosine
        metrics[f"z{chunk_id + 1}_top1"] = top1
    mean_features = chunk_features.mean(dim=1)
    mean_cosine, mean_top1 = retrieval_metrics(mean_features, img_targets)
    metrics["z_chunk_mean_cosine"] = mean_cosine
    metrics["z_chunk_mean_top1"] = mean_top1
    metrics["boundary_mean"] = boundaries.float().mean(dim=0)
    metrics["boundary_std"] = boundaries.float().std(dim=0, unbiased=False)
    metrics["length_mean"] = lengths.float().mean(dim=0)
    metrics["length_std"] = lengths.float().std(dim=0, unbiased=False)
    metrics["score_peak_mean"] = score_peaks.float().mean(dim=0)
    return metrics


def tensor_values(tensor):
    return [float(x) for x in tensor.reshape(-1).tolist()]


def save_checkpoint(model, path, args, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict_path = path.with_name(f"{path.stem}_state_dict.pth")
    torch.save(
        {
            "mode": "chunkaware_atms",
            "epoch": epoch,
            "model": model.state_dict(),
            "chunkaware_config": {
                "base_atms_ckpt": args.base_atms_ckpt,
                "boundary_atms_ckpt": args.boundary_atms_ckpt or args.base_atms_ckpt,
                "train_scope": args.train_scope,
                "min_chunk_length": args.min_chunk_length,
                "smoothing_kernel": args.smoothing_kernel,
                "mse_weight": args.mse_weight,
                "clip_weight": args.clip_weight,
                "distill_weight": args.distill_weight,
                "fixed_reference_chunks": FEATBOUNDARY_CHUNKS,
                "fixed_reference_boundaries": FEATBOUNDARY_BOUNDARIES,
            },
        },
        path,
    )
    torch.save(model.state_dict(), state_dict_path)


def write_result_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(args, run_dir, device):
    subject_id = subject_id_from_name(args.subject)
    train_dataset, test_dataset, text_train, img_train, text_test, img_test = build_feature_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    boundary_ckpt = args.boundary_atms_ckpt or args.base_atms_ckpt
    boundary_extractor = FrozenFeatureBoundaryExtractor(
        boundary_ckpt,
        num_subjects=args.num_subjects,
        device=device,
        min_chunk_length=args.min_chunk_length,
        smoothing_kernel=args.smoothing_kernel,
    )

    model = load_atms(args.base_atms_ckpt, args.num_subjects, device)
    configure_train_scope(model, args.train_scope)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    boundary_trainable, boundary_total = count_trainable(boundary_extractor)
    model_trainable, model_total = count_trainable(model)
    print("boundary extractor checkpoint:", boundary_ckpt)
    print("chunk-aware ATMS init checkpoint:", args.base_atms_ckpt)
    print("boundary extractor trainable/total:", boundary_trainable, boundary_total)
    print("chunk-aware ATMS trainable/total:", model_trainable, model_total)
    print("train scope:", args.train_scope)
    print("loss weights:", {"mse": args.mse_weight, "clip": args.clip_weight, "distill": args.distill_weight})
    print("run_dir:", run_dir)

    rows = []
    best_mean_top1 = -1.0
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model,
            boundary_extractor,
            train_loader,
            optimizer,
            device,
            subject_id,
            img_train,
            args,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_img_mse": train_stats["img_mse"],
            "train_clip_loss": train_stats["clip_loss"],
            "train_distill_loss": train_stats["distill_loss"],
            "train_top1": train_stats["train_top1"],
            "lr": optimizer.param_groups[0]["lr"],
            "logit_scale": float(model.logit_scale.detach().cpu().item()),
        }

        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            eval_stats = evaluate_chunk_atms(model, boundary_extractor, test_loader, device, subject_id, img_test)
            for key, value in eval_stats.items():
                if torch.is_tensor(value):
                    continue
                row[f"test_{key}"] = value
            row["test_b1_mean"], row["test_b2_mean"], row["test_b3_mean"] = tensor_values(eval_stats["boundary_mean"])
            row["test_len1_mean"], row["test_len2_mean"], row["test_len3_mean"], row["test_len4_mean"] = tensor_values(eval_stats["length_mean"])

            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train Loss: {train_stats['loss']:.4f}, Train Top1: {train_stats['train_top1']:.4f}, "
                f"Mean Chunk Cosine: {eval_stats['z_chunk_mean_cosine']:.4f}, "
                f"Mean Chunk Top1: {eval_stats['z_chunk_mean_top1']:.4f}, "
                f"Z1/Z2/Z3/Z4 Top1: "
                f"{eval_stats['z1_top1']:.4f}/"
                f"{eval_stats['z2_top1']:.4f}/"
                f"{eval_stats['z3_top1']:.4f}/"
                f"{eval_stats['z4_top1']:.4f}"
            )

            if eval_stats["z_chunk_mean_top1"] > best_mean_top1:
                best_mean_top1 = eval_stats["z_chunk_mean_top1"]
                save_checkpoint(model, run_dir / "best_chunk_mean_top1.pth", args, epoch + 1)
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} - "
                f"Train Loss: {train_stats['loss']:.4f}, Train Top1: {train_stats['train_top1']:.4f}"
            )

        rows.append(row)
        write_result_csv(run_dir / "results.csv", rows)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(model, run_dir / "checkpoints" / f"{epoch + 1}.pth", args, epoch + 1)
            save_checkpoint(model, run_dir / "latest.pth", args, epoch + 1)

    save_checkpoint(model, run_dir / "final.pth", args, args.epochs)
    print("saved final:", run_dir / "final.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train chunk-aware ATMS for feature-boundary masked EEG chunks.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--base-atms-ckpt", required=True)
    parser.add_argument("--boundary-atms-ckpt", default=None)
    parser.add_argument("--vit-train-features", default=None)
    parser.add_argument("--vit-test-features", default=None)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-scope", choices=["proj", "enc_proj", "all"], default="enc_proj")

    parser.add_argument("--mse-weight", type=float, default=0.9)
    parser.add_argument("--clip-weight", type=float, default=0.1)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--min-chunk-length", type=int, default=20)
    parser.add_argument("--smoothing-kernel", type=int, default=5)

    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--output-root", default="/data/gaoy/projects/datasets/EEG_Image_decode/runs/chunkaware_atms")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    args.vit_train_features = args.vit_train_features or str(Path(args.data_root) / "ViT-H-14_features_train.pt")
    args.vit_test_features = args.vit_test_features or str(Path(args.data_root) / "ViT-H-14_features_test.pt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.subject / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train(args, run_dir, device)


if __name__ == "__main__":
    main()
