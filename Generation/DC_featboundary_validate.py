import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from DC_featboundary_model import FeatureBoundaryChunkATMS, make_featboundary_router_modules
from DC_train_atms import load_eeg_split, subject_id_from_name
from DC_train_diffusion_prior import load_img_features


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


@torch.no_grad()
def retrieval_metrics(embeds, img_features):
    embeds = embeds.float()
    img_features = img_features.float()
    paired_cosine = F.cosine_similarity(embeds, img_features, dim=1).mean()
    retrieval = embeds @ img_features.T
    top1 = (retrieval.argmax(dim=1) == torch.arange(embeds.shape[0], device=embeds.device)).float().mean()
    return float(paired_cosine.item()), float(top1.item())


def tensor_to_list(tensor):
    return [float(x) for x in tensor.reshape(-1).tolist()]


def load_featboundary_models(args, device):
    state = torch.load(args.featboundary_prior_ckpt, map_location=device)
    if "diffusion_prior" not in state:
        raise ValueError("featboundary checkpoint must contain diffusion_prior, router, and gamma.")

    feat_config = state.get("featboundary_config", {})
    min_chunk_length = feat_config.get("min_chunk_length", args.min_chunk_length)
    smoothing_kernel = feat_config.get("smoothing_kernel", args.smoothing_kernel)

    conditioner = FeatureBoundaryChunkATMS(
        args.atms_ckpt,
        num_subjects=args.num_subjects,
        device=device,
        min_chunk_length=min_chunk_length,
        smoothing_kernel=smoothing_kernel,
    )
    pipe = make_featboundary_router_modules(device, cond_dim=1024, dropout=args.dropout, init_gamma=args.init_gamma)
    pipe.diffusion_prior.load_state_dict(state["diffusion_prior"])
    pipe.router_condition.router.load_state_dict(state["router"])
    pipe.router_condition.gamma.data.copy_(state["gamma"].to(device))
    conditioner.eval()
    pipe.diffusion_prior.eval()
    pipe.router_condition.eval()

    print("loaded feature-boundary checkpoint:", args.featboundary_prior_ckpt)
    print("gamma:", float(pipe.router_condition.gamma.detach().cpu().item()))
    print("featboundary_config:", feat_config)
    return conditioner, pipe, feat_config


@torch.no_grad()
def extract_condition_cache(args, device, conditioner):
    eeg, _ = load_eeg_split(args.data_root, args.subject, False, (args.time_start, args.time_end))
    eeg = eeg[: args.num_samples].float()
    loader = DataLoader(EEGTensorDataset(eeg), batch_size=args.atms_batch_size, shuffle=False, num_workers=0)
    subject_id = subject_id_from_name(args.subject)

    z_global = []
    z_chunks = []
    boundaries = []
    lengths = []
    score_peaks = []
    for eeg_batch in loader:
        eeg_batch = eeg_batch.to(device)
        subject_ids = torch.full((eeg_batch.size(0),), subject_id, dtype=torch.long, device=device)
        cache = conditioner(eeg_batch, subject_ids)
        z_global.append(cache["z_global"].float().cpu())
        z_chunks.append(cache["z_chunks"].float().cpu())
        boundaries.append(cache["boundaries"].float().cpu())
        lengths.append(cache["lengths"].float().cpu())
        score_peaks.append(cache["score_peaks"].float().cpu())

    return {
        "z_global": torch.cat(z_global, dim=0),
        "z_chunks": torch.cat(z_chunks, dim=0),
        "boundaries": torch.cat(boundaries, dim=0),
        "lengths": torch.cat(lengths, dim=0),
        "score_peaks": torch.cat(score_peaks, dim=0),
    }


def condition_for_mode(pipe, condition_cache, timesteps, mode, fixed_gamma):
    z_global = condition_cache["z_global"].to(pipe.device)
    z_chunks = condition_cache["z_chunks"].to(pipe.device)
    batch_size = z_global.shape[0]
    if timesteps.ndim == 0 or timesteps.numel() == 1:
        timesteps = torch.full((batch_size,), float(timesteps.reshape(-1)[0].item()), device=pipe.device)
    weights = pipe.router_condition.router(timesteps.to(pipe.device)).to(z_chunks.dtype)
    z_chunk = torch.sum(z_chunks * weights[:, :, None], dim=1)

    if mode == "full":
        gamma = pipe.router_condition.gamma.to(z_global.dtype)
        return z_global + gamma * z_chunk
    if mode == "global_only":
        return z_global
    if mode == "chunk_only":
        return z_chunk
    if mode == "chunk_scaled":
        gamma = pipe.router_condition.gamma.to(z_global.dtype)
        return gamma * z_chunk
    if mode == "fixed_pos_gamma":
        return z_global + float(fixed_gamma) * z_chunk
    raise ValueError(f"Unknown condition mode: {mode}")


@torch.no_grad()
def generate_prior_outputs(pipe, condition_cache, args, mode, seed):
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps

    self_device_cache = {
        "z_global": condition_cache["z_global"].to(pipe.device),
        "z_chunks": condition_cache["z_chunks"].to(pipe.device),
    }
    timesteps, _ = retrieve_timesteps(pipe.scheduler, args.prior_steps, pipe.device, None)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    batch_size = self_device_cache["z_global"].shape[0]
    h_t = torch.randn(batch_size, pipe.diffusion_prior.embed_dim, generator=generator, device=pipe.device)

    for timestep in tqdm(timesteps, desc=f"prior {mode}", leave=False):
        t = torch.ones(h_t.shape[0], dtype=torch.float, device=pipe.device) * timestep
        if mode == "uncond" or args.guidance_scale == 0:
            noise_pred = pipe.diffusion_prior(h_t, t, None)
        else:
            c_t = condition_for_mode(pipe, self_device_cache, t, mode, args.fixed_gamma)
            noise_pred_cond = pipe.diffusion_prior(h_t, t, c_t)
            noise_pred_uncond = pipe.diffusion_prior(h_t, t, None)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        h_t = pipe.scheduler.step(noise_pred, int(timestep.item()), h_t, generator=generator).prev_sample
    return h_t.float().cpu()


@torch.no_grad()
def generate_modes(pipe, condition_cache, args, modes):
    outputs = {mode: [] for mode in modes}
    num_samples = condition_cache["z_global"].shape[0]
    for start in range(0, num_samples, args.prior_batch_size):
        end = min(start + args.prior_batch_size, num_samples)
        batch_cache = {
            "z_global": condition_cache["z_global"][start:end],
            "z_chunks": condition_cache["z_chunks"][start:end],
        }
        for mode in modes:
            outputs[mode].append(generate_prior_outputs(pipe, batch_cache, args, mode, seed=args.seed + start))
    return {mode: torch.cat(parts, dim=0) for mode, parts in outputs.items()}


def save_embedding_summary(output_dir, condition_cache, img_features):
    rows = []
    z_global = condition_cache["z_global"].float()
    z_chunks = condition_cache["z_chunks"].float()
    metrics = {
        "z_global": z_global,
        "z_chunk_mean": z_chunks.mean(dim=1),
        "z1": z_chunks[:, 0, :],
        "z2": z_chunks[:, 1, :],
        "z3": z_chunks[:, 2, :],
        "z4": z_chunks[:, 3, :],
    }
    for name, embeds in metrics.items():
        cosine, top1 = retrieval_metrics(embeds, img_features)
        rows.append([
            name,
            cosine,
            top1,
            float(embeds.norm(dim=1).mean().item()),
            float(embeds.norm(dim=1).std(unbiased=False).item()),
            "",
            "",
        ])

    path = output_dir / "chunk_embedding_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["embedding", "paired_cosine", "top1", "norm_mean", "norm_std", "cos_to_full_prior", "l2_to_full_prior"])
        writer.writerows(rows)
    print("saved chunk embedding summary:", path)


def save_prior_summary(output_dir, outputs, img_features):
    full = outputs.get("full")
    rows = []
    for mode, embeds in outputs.items():
        cosine, top1 = retrieval_metrics(embeds, img_features)
        if full is not None and mode != "full":
            cos_to_full = F.cosine_similarity(embeds.float(), full.float(), dim=1).mean().item()
            l2_to_full = (embeds.float() - full.float()).norm(dim=1).mean().item()
        else:
            cos_to_full = 1.0
            l2_to_full = 0.0
        rows.append([
            mode,
            cosine,
            top1,
            float(embeds.norm(dim=1).mean().item()),
            float(embeds.norm(dim=1).std(unbiased=False).item()),
            float(cos_to_full),
            float(l2_to_full),
        ])

    path = output_dir / "prior_mode_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "paired_cosine", "top1", "norm_mean", "norm_std", "cos_to_full_prior", "l2_to_full_prior"])
        writer.writerows(rows)
    print("saved prior mode summary:", path)


def save_sample_summary(output_dir, condition_cache, img_features, outputs):
    z_global = condition_cache["z_global"].float()
    z_chunks = condition_cache["z_chunks"].float()
    boundaries = condition_cache["boundaries"].long()
    lengths = condition_cache["lengths"].long()
    score_peaks = condition_cache["score_peaks"].float()

    path = output_dir / "sample_validation.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
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
            "z_global_img_cos",
            "z1_img_cos",
            "z2_img_cos",
            "z3_img_cos",
            "z4_img_cos",
        ]
        for mode in outputs:
            header.append(f"{mode}_img_cos")
        writer.writerow(header)

        for idx in range(boundaries.shape[0]):
            row = [
                idx,
                *[int(x) for x in boundaries[idx].tolist()],
                *[int(x) for x in lengths[idx].tolist()],
                *[float(x) for x in score_peaks[idx].tolist()],
                float(F.cosine_similarity(z_global[idx:idx + 1], img_features[idx:idx + 1], dim=1).item()),
                *[
                    float(F.cosine_similarity(z_chunks[idx:idx + 1, chunk_id, :], img_features[idx:idx + 1], dim=1).item())
                    for chunk_id in range(4)
                ],
            ]
            for mode, embeds in outputs.items():
                row.append(float(F.cosine_similarity(embeds[idx:idx + 1].float(), img_features[idx:idx + 1], dim=1).item()))
            writer.writerow(row)
    print("saved sample validation:", path)


def save_outputs(output_dir, condition_cache, outputs):
    torch.save(condition_cache, output_dir / "condition_cache.pt")
    torch.save(outputs, output_dir / "prior_mode_outputs.pt")
    print("saved condition cache and prior outputs")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate whether feature-boundary chunk information changes the prior output.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--atms-ckpt", required=True)
    parser.add_argument("--featboundary-prior-ckpt", required=True)
    parser.add_argument("--vit-test-features", default=None)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-gamma", type=float, default=0.1)
    parser.add_argument("--min-chunk-length", type=int, default=20)
    parser.add_argument("--smoothing-kernel", type=int, default=5)

    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--atms-batch-size", type=int, default=256)
    parser.add_argument("--prior-batch-size", type=int, default=64)
    parser.add_argument("--prior-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fixed-gamma", type=float, default=0.1)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "global_only", "fixed_pos_gamma", "chunk_only", "chunk_scaled", "uncond"],
        choices=["full", "global_only", "fixed_pos_gamma", "chunk_only", "chunk_scaled", "uncond"],
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-outputs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    args.vit_test_features = args.vit_test_features or str(Path(args.data_root) / "ViT-H-14_features_test_raw.pt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(
        args.output_dir
        or Path(args.data_root) / "runs" / "featboundary_validate" / args.subject / Path(args.featboundary_prior_ckpt).stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    conditioner, pipe, _ = load_featboundary_models(args, device)
    condition_cache = extract_condition_cache(args, device, conditioner)
    img_features = load_img_features(args.vit_test_features)[: condition_cache["z_global"].shape[0]].float()

    print("boundary mean:", tensor_to_list(condition_cache["boundaries"].mean(dim=0)))
    print("boundary std:", tensor_to_list(condition_cache["boundaries"].std(dim=0, unbiased=False)))
    print("length mean:", tensor_to_list(condition_cache["lengths"].mean(dim=0)))
    print("length std:", tensor_to_list(condition_cache["lengths"].std(dim=0, unbiased=False)))

    save_embedding_summary(output_dir, condition_cache, img_features)
    outputs = generate_modes(pipe, condition_cache, args, args.modes)
    save_prior_summary(output_dir, outputs, img_features)
    save_sample_summary(output_dir, condition_cache, img_features, outputs)
    if args.save_outputs:
        save_outputs(output_dir, condition_cache, outputs)

    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
