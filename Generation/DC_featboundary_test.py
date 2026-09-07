import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from DC_featboundary_model import (
    FEATBOUNDARY_BOUNDARIES,
    FeatureBoundaryChunkATMS,
)
from DC_train_atms import load_eeg_split, subject_id_from_name


class EEGTensorDataset(Dataset):
    def __init__(self, eeg):
        self.eeg = eeg

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_values(tensor):
    return [float(x) for x in tensor.reshape(-1).tolist()]


@torch.no_grad()
def collect_boundaries(args, device):
    eeg, _ = load_eeg_split(args.data_root, args.subject, False, (args.time_start, args.time_end))
    eeg = eeg[: args.num_samples].float()
    loader = DataLoader(EEGTensorDataset(eeg), batch_size=args.batch_size, shuffle=False, num_workers=0)
    subject_id = subject_id_from_name(args.subject)
    conditioner = FeatureBoundaryChunkATMS(
        args.atms_ckpt,
        num_subjects=args.num_subjects,
        device=device,
        min_chunk_length=args.min_chunk_length,
        smoothing_kernel=args.smoothing_kernel,
    )
    conditioner.eval()

    boundaries = []
    lengths = []
    score_peaks = []
    raw_scores = []
    feature_scores = []
    for eeg_batch in loader:
        eeg_batch = eeg_batch.to(device)
        subject_ids = torch.full((eeg_batch.size(0),), subject_id, dtype=torch.long, device=device)
        cache = conditioner(eeg_batch, subject_ids)
        boundaries.append(cache["boundaries"].cpu())
        lengths.append(cache["lengths"].cpu())
        score_peaks.append(cache["score_peaks"].cpu())
        raw_scores.append(cache["raw_change_score"].cpu())
        feature_scores.append(cache["feature_change_score"].cpu())

    return {
        "boundaries": torch.cat(boundaries, dim=0),
        "lengths": torch.cat(lengths, dim=0),
        "score_peaks": torch.cat(score_peaks, dim=0),
        "raw_scores": torch.cat(raw_scores, dim=0),
        "feature_scores": torch.cat(feature_scores, dim=0),
    }


def save_boundary_csv(results, output_dir):
    path = output_dir / "boundary_results.csv"
    boundaries = results["boundaries"]
    lengths = results["lengths"]
    score_peaks = results["score_peaks"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "b1",
            "b2",
            "b3",
            "L1",
            "L2",
            "L3",
            "L4",
            "score_b1",
            "score_b2",
            "score_b3",
        ])
        for idx in range(boundaries.shape[0]):
            writer.writerow([
                idx,
                *[int(x) for x in boundaries[idx].tolist()],
                *[int(x) for x in lengths[idx].tolist()],
                *[float(x) for x in score_peaks[idx].tolist()],
            ])
    print("saved boundary csv:", path)


def save_score_plots(results, output_dir, num_plots, seed):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib unavailable, skip plots:", exc)
        return

    plot_dir = output_dir / "score_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    boundaries = results["boundaries"]
    raw_scores = results["raw_scores"]
    rng = random.Random(seed)
    sample_indices = list(range(raw_scores.shape[0]))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[: min(num_plots, len(sample_indices))]

    for idx in sample_indices:
        score = raw_scores[idx].numpy()
        b1, b2, b3 = [int(x) for x in boundaries[idx].tolist()]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(score.shape[0]), score, linewidth=1.6)
        for boundary, color in zip((b1, b2, b3), ("tab:red", "tab:orange", "tab:green")):
            ax.axvline(boundary, color=color, linestyle="--", linewidth=1.2)
        ax.set_title(f"sample {idx}: b=({b1}, {b2}, {b3})")
        ax.set_xlabel("raw temporal position")
        ax.set_ylabel("change score")
        ax.set_xlim(0, score.shape[0] - 1)
        fig.tight_layout()
        path = plot_dir / f"sample_{idx:03d}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
    print("saved score plots:", plot_dir)


def print_summary(results, min_chunk_length):
    boundaries = results["boundaries"].float()
    lengths = results["lengths"].float()
    score_peaks = results["score_peaks"].float()
    boundary_mean = tensor_values(boundaries.mean(dim=0))
    boundary_std = tensor_values(boundaries.std(dim=0, unbiased=False))
    length_mean = tensor_values(lengths.mean(dim=0))
    length_std = tensor_values(lengths.std(dim=0, unbiased=False))
    score_peak_mean = tensor_values(score_peaks.mean(dim=0))
    unique_boundaries = torch.unique(boundaries.long(), dim=0).shape[0]
    collapsed_tail = torch.all(lengths[:, 1:] <= float(min_chunk_length), dim=1).float().mean().item()
    all_same = unique_boundaries == 1

    print("fixed reference boundaries:", FEATBOUNDARY_BOUNDARIES)
    print("b mean:", boundary_mean)
    print("b std:", boundary_std)
    print("length mean:", length_mean)
    print("length std:", length_std)
    print("score peak mean:", score_peak_mean)
    print("unique boundary triplets:", unique_boundaries)
    print("tail chunks all at min_len ratio:", collapsed_tail)
    print("all samples same boundaries:", all_same)


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity test feature-change EEG boundaries.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--atms-ckpt", required=True)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-chunk-length", type=int, default=20)
    parser.add_argument("--smoothing-kernel", type=int, default=5)
    parser.add_argument("--plot-samples", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or Path(args.data_root) / "runs" / "featboundary_test" / args.subject)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = collect_boundaries(args, device)
    print_summary(results, args.min_chunk_length)
    save_boundary_csv(results, output_dir)
    save_score_plots(results, output_dir, args.plot_samples, args.seed)
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
