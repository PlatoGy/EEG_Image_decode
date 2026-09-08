import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from DC_featboundary_model import (
    FEATBOUNDARY_BOUNDARIES,
    FEATBOUNDARY_CHUNKS,
    FeatureBoundaryChunkATMS,
    make_featboundary_router_modules,
    make_optimizer_and_scheduler,
)
from DC_featboundary_train import (
    RawEEGToImageEmbeddingDataset,
    count_trainable,
    repeat_train_image_features,
    retrieval_metrics,
    seed_everything,
    tensor_values,
    validate_inputs,
)
from DC_train_atms import load_eeg_split, subject_id_from_name
from DC_train_diffusion_prior import load_img_features


def set_gamma_mode(pipe, args):
    pipe.router_condition.gamma.data.fill_(float(args.init_gamma))
    if args.gamma_mode == "fixed":
        pipe.router_condition.gamma.requires_grad_(False)
    elif args.gamma_mode == "learned":
        pipe.router_condition.gamma.requires_grad_(True)
    else:
        raise ValueError(f"Unknown gamma mode: {args.gamma_mode}")


def save_checkpoint(pipe, path, mode, args, epoch=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": mode,
            "epoch": epoch,
            "diffusion_prior": pipe.diffusion_prior.state_dict(),
            "router": pipe.router_condition.router.state_dict(),
            "gamma": pipe.router_condition.gamma.detach().cpu(),
            "featboundary_config": {
                "init_chunks": FEATBOUNDARY_CHUNKS,
                "init_boundaries": FEATBOUNDARY_BOUNDARIES,
                "min_chunk_length": args.min_chunk_length,
                "smoothing_kernel": args.smoothing_kernel,
                "router_entropy_reg_weight": args.router_entropy_reg_weight,
                "freeze_router_epochs": args.freeze_router_epochs,
                "gamma_mode": args.gamma_mode,
                "gamma_value": args.init_gamma,
            },
        },
        path,
    )


def load_resume_checkpoint(pipe, checkpoint_path, device, args):
    state = torch.load(checkpoint_path, map_location=device)
    if "diffusion_prior" not in state:
        raise ValueError("Feature-boundary resume checkpoint must contain diffusion_prior, router, and gamma.")
    pipe.diffusion_prior.load_state_dict(state["diffusion_prior"])
    pipe.router_condition.router.load_state_dict(state["router"])
    if args.gamma_mode == "learned":
        pipe.router_condition.gamma.data.copy_(state["gamma"].to(device))
    else:
        pipe.router_condition.gamma.data.fill_(float(args.init_gamma))
    print("loaded feature-boundary checkpoint:", checkpoint_path)


def train_featboundary_gamma(args, run_dir, device):
    if not args.atms_ckpt:
        raise ValueError("--atms-ckpt is required.")

    subject_id = subject_id_from_name(args.subject)
    train_eeg, _ = load_eeg_split(args.data_root, args.subject, True, (args.time_start, args.time_end))
    raw_img_train = load_img_features(args.vit_train_features)
    target_img_train = repeat_train_image_features(raw_img_train)
    validate_inputs(train_eeg, raw_img_train, target_img_train)

    dataset = RawEEGToImageEmbeddingDataset(train_eeg, target_img_train, subject_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    conditioner = FeatureBoundaryChunkATMS(
        args.atms_ckpt,
        num_subjects=args.num_subjects,
        device=device,
        min_chunk_length=args.min_chunk_length,
        smoothing_kernel=args.smoothing_kernel,
    )
    pipe = make_featboundary_router_modules(
        device,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        init_gamma=args.init_gamma,
    )
    set_gamma_mode(pipe, args)

    if args.resume_prior_ckpt:
        load_resume_checkpoint(pipe, args.resume_prior_ckpt, device, args)
        set_gamma_mode(pipe, args)

    global_trainable, global_total = count_trainable(conditioner.global_atms)
    chunk_trainable, chunk_total = count_trainable(conditioner.chunk_atms)
    prior_trainable, prior_total = count_trainable(pipe.diffusion_prior)
    router_trainable, router_total = count_trainable(pipe.router_condition)
    gamma_trainable = int(pipe.router_condition.gamma.requires_grad)
    print("frozen global/feature ATMS trainable/total:", global_trainable, global_total)
    print("frozen shared chunk ATMS trainable/total:", chunk_trainable, chunk_total)
    print("trainable diffusion prior trainable/total:", prior_trainable, prior_total)
    print("trainable router_condition trainable/total:", router_trainable, router_total)
    print("gamma mode/value/trainable:", args.gamma_mode, float(pipe.router_condition.gamma.detach().cpu().item()), gamma_trainable)
    print("fixed reference chunks:", FEATBOUNDARY_CHUNKS)
    print("feature-boundary min chunk length:", args.min_chunk_length)
    print("feature-boundary smoothing kernel:", args.smoothing_kernel)
    print("router entropy reg weight:", args.router_entropy_reg_weight)
    print("freeze router epochs:", args.freeze_router_epochs)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(pipe, dataloader, args.epochs, args.lr)
    loss_csv = run_dir / "loss.csv"
    eval_csv = run_dir / "prior_eval.csv"
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch_index",
            "epoch",
            "loss",
            "diffusion_loss",
            "router_entropy_penalty",
            "lr",
            "gamma",
            "gamma_mode",
            "gamma_trainable",
            "router_frozen",
            "b1_mean",
            "b2_mean",
            "b3_mean",
            "b1_std",
            "b2_std",
            "b3_std",
            "len1_mean",
            "len2_mean",
            "len3_mean",
            "len4_mean",
            "len1_std",
            "len2_std",
            "len3_std",
            "len4_std",
            "score_b1_mean",
            "score_b2_mean",
            "score_b3_mean",
            "w1_mean",
            "w2_mean",
            "w3_mean",
            "w4_mean",
        ])
    if args.eval_every > 0:
        with open(eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch_index", "epoch", "split", "paired_cosine", "top1", "num_samples"])

    test_eeg = None
    test_img = None
    if args.eval_every > 0:
        test_eeg, _ = load_eeg_split(args.data_root, args.subject, False, (args.time_start, args.time_end))
        test_img = load_img_features(args.vit_test_features)

    for epoch_idx in range(args.epochs):
        router_frozen = epoch_idx < args.freeze_router_epochs
        stats = pipe.train_epoch(
            dataloader,
            conditioner,
            optimizer,
            lr_scheduler,
            router_entropy_reg_weight=args.router_entropy_reg_weight,
            freeze_router=router_frozen,
        )
        lr = optimizer.param_groups[0]["lr"]
        gamma = float(pipe.router_condition.gamma.detach().cpu().item())
        boundary_mean = tensor_values(stats["boundary_mean"])
        boundary_std = tensor_values(stats["boundary_std"])
        length_mean = tensor_values(stats["length_mean"])
        length_std = tensor_values(stats["length_std"])
        score_peak_mean = tensor_values(stats["score_peak_mean"])
        router_weight_mean = tensor_values(stats["router_weight_mean"])

        print(
            f"epoch: {epoch_idx}, loss: {stats['loss']}, diffusion_loss: {stats['diffusion_loss']}, "
            f"router_entropy_penalty: {stats['router_entropy_penalty']}, gamma: {gamma}, "
            f"gamma_mode: {args.gamma_mode}, router_frozen: {router_frozen}, "
            f"b_mean: {boundary_mean}, b_std: {boundary_std}, "
            f"len_mean: {length_mean}, len_std: {length_std}, "
            f"score_peak_mean: {score_peak_mean}, router_w_mean: {router_weight_mean}"
        )
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch_idx,
                epoch_idx + 1,
                stats["loss"],
                stats["diffusion_loss"],
                stats["router_entropy_penalty"],
                lr,
                gamma,
                args.gamma_mode,
                int(pipe.router_condition.gamma.requires_grad),
                int(router_frozen),
                *boundary_mean,
                *boundary_std,
                *length_mean,
                *length_std,
                *score_peak_mean,
                *router_weight_mean,
            ])

        should_save = (epoch_idx + 1) % args.save_every == 0 or epoch_idx + 1 == args.epochs
        if should_save:
            save_checkpoint(pipe, run_dir / "checkpoints" / f"epoch_{epoch_idx + 1:03d}.pth", args.mode, args, epoch_idx + 1)
            save_checkpoint(pipe, run_dir / "latest.pth", args.mode, args, epoch_idx + 1)

        if args.eval_every > 0 and ((epoch_idx + 1) % args.eval_every == 0 or epoch_idx + 1 == args.epochs):
            cosine, top1 = retrieval_metrics(pipe, conditioner, test_eeg, test_img, subject_id, device, args)
            print(f"eval epoch: {epoch_idx}, split: test, paired_cosine: {cosine}, top1: {top1}")
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch_idx, epoch_idx + 1, "test", cosine, top1, args.eval_num_samples])

    save_checkpoint(pipe, run_dir / args.save_name, args.mode, args, args.epochs)
    print("saved feature-boundary gamma-test diffusion prior:", run_dir / args.save_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Feature-Boundary Dynamic 4-Chunk + Router with controllable gamma.")
    parser.add_argument("--mode", choices=["featboundary_gamma"], default="featboundary_gamma")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--atms-ckpt", default=None)
    parser.add_argument("--vit-train-features", default=None)
    parser.add_argument("--vit-test-features", default=None)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond-dim", type=int, default=1024)
    parser.add_argument("--init-gamma", type=float, default=0.1)
    parser.add_argument("--gamma-mode", choices=["learned", "fixed"], default="fixed")
    parser.add_argument("--resume-prior-ckpt", default=None)
    parser.add_argument("--save-name", default="diffusion_prior.pt")
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument("--min-chunk-length", type=int, default=20)
    parser.add_argument("--smoothing-kernel", type=int, default=5)
    parser.add_argument("--router-entropy-reg-weight", type=float, default=0.01)
    parser.add_argument("--freeze-router-epochs", type=int, default=20)

    parser.add_argument("--output-root", default="/data/gaoy/projects/datasets/EEG_Image_decode/runs/diffusion_prior")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-num-samples", type=int, default=200)
    parser.add_argument("--eval-prior-steps", type=int, default=50)
    parser.add_argument("--eval-guidance-scale", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    args.vit_train_features = args.vit_train_features or str(Path(args.data_root) / "ViT-H-14_features_train_raw.pt")
    args.vit_test_features = args.vit_test_features or str(Path(args.data_root) / "ViT-H-14_features_test_raw.pt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.subject / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("mode:", args.mode)
    print("run_dir:", run_dir)
    train_featboundary_gamma(args, run_dir, device)
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
