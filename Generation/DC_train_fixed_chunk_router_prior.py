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
from torch.utils.data import DataLoader, Dataset


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
from DC_chunk_router import (
    FrozenFixedChunkATMS,
    make_fixed_chunk_router_modules,
    make_optimizer_and_scheduler,
)
from DC_train_atms import FeatureDataset, load_clip_features, load_eeg_split, subject_id_from_name
from DC_train_diffusion_prior import (
    load_img_features,
    load_train_eeg_embeddings_from_atms,
    prior_retrieval_metrics,
    save_state_dict,
    train_with_checkpoints,
    validate_training_inputs,
)


class RawEEGToImageEmbeddingDataset(Dataset):
    def __init__(self, eeg, h_embeddings, subject_id):
        self.eeg = eeg
        self.h_embeddings = h_embeddings
        self.subject_id = int(subject_id)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],
            "h_embedding": self.h_embeddings[idx],
            "subject_id": torch.tensor(self.subject_id, dtype=torch.long),
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repeat_train_image_features(raw_img_train):
    return raw_img_train.view(1654, 10, 1, 1024).repeat(1, 1, 4, 1).view(-1, 1024).float()


def count_trainable(module):
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    total = sum(param.numel() for param in module.parameters())
    return trainable, total


def validate_fixed_inputs(train_eeg, raw_img_train, repeated_img_train):
    expected_eeg_shape = (1654 * 10 * 4, 63, 250)
    expected_raw_img_shape = (1654 * 10, 1024)
    expected_repeated_shape = (1654 * 10 * 4, 1024)
    if tuple(train_eeg.shape) != expected_eeg_shape:
        raise ValueError(f"Expected train EEG {expected_eeg_shape}, got {tuple(train_eeg.shape)}")
    if tuple(raw_img_train.shape) != expected_raw_img_shape:
        raise ValueError(f"Expected raw image features {expected_raw_img_shape}, got {tuple(raw_img_train.shape)}")
    if tuple(repeated_img_train.shape) != expected_repeated_shape:
        raise ValueError(f"Expected repeated image features {expected_repeated_shape}, got {tuple(repeated_img_train.shape)}")
    expected_repeated = raw_img_train.repeat_interleave(4, dim=0)
    max_repeat_diff = (repeated_img_train - expected_repeated).abs().max().item()
    if max_repeat_diff != 0.0:
        raise ValueError(f"Image target repeat is not sample-wise repeat_interleave(4): {max_repeat_diff}")
    norms = raw_img_train.norm(dim=1)
    if torch.allclose(norms, torch.ones_like(norms), rtol=1e-3, atol=1e-3):
        raise ValueError("Image targets look L2-normalized; use raw/un-normalized ViT-H features.")
    print("fixed check: train EEG", tuple(train_eeg.shape))
    print("fixed check: raw image targets", tuple(raw_img_train.shape))
    print("fixed check: repeated image targets", tuple(repeated_img_train.shape))
    print("fixed check: repeat max_abs_diff", max_repeat_diff)
    print(
        "fixed check: raw target norm",
        f"mean={norms.mean().item():.6f}",
        f"std={norms.std().item():.6f}",
    )


@torch.no_grad()
def fixed_prior_retrieval_metrics(pipe, conditioner, eeg, img_features, subject_id, device, args):
    num_samples = min(args.eval_num_samples, eeg.shape[0], img_features.shape[0])
    eeg = eeg[:num_samples].float().to(device)
    img_features = img_features[:num_samples].float().to(device)
    subject_ids = torch.full((num_samples,), subject_id, dtype=torch.long, device=device)
    condition_cache = conditioner(eeg, subject_ids)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    outputs = pipe.generate(
        condition_cache,
        num_inference_steps=args.eval_prior_steps,
        guidance_scale=args.eval_guidance_scale,
        generator=generator,
    )
    paired_cosine = torch.nn.functional.cosine_similarity(outputs.float(), img_features, dim=1).mean()
    retrieval = outputs.float() @ img_features.T
    top1 = (retrieval.argmax(dim=1) == torch.arange(num_samples, device=device)).float().mean()
    return float(paired_cosine.item()), float(top1.item())


def save_fixed_checkpoint(pipe, path, mode, epoch=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": mode,
            "epoch": epoch,
            "diffusion_prior": pipe.diffusion_prior.state_dict(),
            "router": pipe.router_condition.router.state_dict(),
            "gamma": pipe.router_condition.gamma.detach().cpu(),
        },
        path,
    )


def train_fixed_chunk_router(args, run_dir, device):
    if not args.atms_ckpt:
        raise ValueError("--atms-ckpt is required for fixed_chunk_router mode.")

    subject_id = subject_id_from_name(args.subject)
    train_eeg, _ = load_eeg_split(args.data_root, args.subject, True, (args.time_start, args.time_end))
    raw_img_train = load_img_features(args.vit_train_features)
    target_img_train = repeat_train_image_features(raw_img_train)
    validate_fixed_inputs(train_eeg, raw_img_train, target_img_train)

    dataset = RawEEGToImageEmbeddingDataset(train_eeg, target_img_train, subject_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    conditioner = FrozenFixedChunkATMS(args.atms_ckpt, num_subjects=args.num_subjects, device=device)
    pipe = make_fixed_chunk_router_modules(device, cond_dim=args.cond_dim, dropout=args.dropout, init_gamma=args.init_gamma)
    global_trainable, global_total = count_trainable(conditioner.global_atms)
    chunk_trainable, chunk_total = count_trainable(conditioner.chunk_atms)
    prior_trainable, prior_total = count_trainable(pipe.diffusion_prior)
    router_trainable, router_total = count_trainable(pipe.router_condition)
    print("frozen global ATMS trainable/total:", global_trainable, global_total)
    print("frozen chunk ATMS trainable/total:", chunk_trainable, chunk_total)
    print("trainable diffusion prior trainable/total:", prior_trainable, prior_total)
    print("trainable router+gamma trainable/total:", router_trainable, router_total)
    if args.resume_prior_ckpt:
        state = torch.load(args.resume_prior_ckpt, map_location=device)
        if "diffusion_prior" in state:
            pipe.diffusion_prior.load_state_dict(state["diffusion_prior"])
            pipe.router_condition.router.load_state_dict(state["router"])
            pipe.router_condition.gamma.data.copy_(state["gamma"].to(device))
        else:
            pipe.diffusion_prior.load_state_dict(state)
        print("loaded prior checkpoint:", args.resume_prior_ckpt)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(pipe, dataloader, args.epochs, args.lr)
    loss_csv = run_dir / "loss.csv"
    eval_csv = run_dir / "prior_eval.csv"
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch_index", "epoch", "loss", "lr", "gamma"])
    if args.eval_every > 0:
        with open(eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch_index", "epoch", "split", "paired_cosine", "top1", "num_samples"])

    test_eeg = None
    test_img = None
    if args.eval_every > 0:
        test_eeg, _ = load_eeg_split(args.data_root, args.subject, False, (args.time_start, args.time_end))
        test_img = load_img_features(args.vit_test_features)

    for epoch_idx in range(args.epochs):
        loss = pipe.train_epoch(dataloader, conditioner, optimizer, lr_scheduler)
        lr = optimizer.param_groups[0]["lr"]
        gamma = float(pipe.router_condition.gamma.detach().cpu().item())
        print(f"epoch: {epoch_idx}, loss: {loss}, gamma: {gamma}")
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch_idx, epoch_idx + 1, loss, lr, gamma])

        should_save = (epoch_idx + 1) % args.save_every == 0 or epoch_idx + 1 == args.epochs
        if should_save:
            save_fixed_checkpoint(pipe, run_dir / "checkpoints" / f"epoch_{epoch_idx + 1:03d}.pth", args.mode, epoch_idx + 1)
            save_fixed_checkpoint(pipe, run_dir / "latest.pth", args.mode, epoch_idx + 1)

        if args.eval_every > 0 and ((epoch_idx + 1) % args.eval_every == 0 or epoch_idx + 1 == args.epochs):
            cosine, top1 = fixed_prior_retrieval_metrics(pipe, conditioner, test_eeg, test_img, subject_id, device, args)
            print(f"eval epoch: {epoch_idx}, split: test, paired_cosine: {cosine}, top1: {top1}")
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch_idx, epoch_idx + 1, "test", cosine, top1, args.eval_num_samples])

    save_fixed_checkpoint(pipe, run_dir / args.save_name, args.mode, args.epochs)
    print("saved diffusion prior:", run_dir / args.save_name)


def train_original(args, run_dir, device):
    raw_img_train = load_img_features(args.vit_train_features)
    target_img_train = repeat_train_image_features(raw_img_train)

    if args.eeg_train_embeds:
        eeg_features_train = torch.load(args.eeg_train_embeds, map_location="cpu").float()
        print("loaded EEG train embeddings:", args.eeg_train_embeds, tuple(eeg_features_train.shape))
    elif args.atms_ckpt:
        eeg_features_train = load_train_eeg_embeddings_from_atms(args, device)
        torch.save(eeg_features_train, run_dir / f"ATM_S_eeg_features_{args.subject}_train.pt")
        print("extracted EEG train embeddings:", tuple(eeg_features_train.shape))
    else:
        raise ValueError("original mode requires --eeg-train-embeds or --atms-ckpt.")

    validate_training_inputs(eeg_features_train, raw_img_train, target_img_train, run_dir)
    dataset = EmbeddingDataset(c_embeddings=eeg_features_train, h_embeddings=target_img_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    diffusion_prior = DiffusionPriorUNet(cond_dim=args.cond_dim, dropout=args.dropout)
    pipe = Pipe(diffusion_prior, device=device)
    if args.resume_prior_ckpt:
        pipe.diffusion_prior.load_state_dict(torch.load(args.resume_prior_ckpt, map_location=device))
        print("loaded prior checkpoint:", args.resume_prior_ckpt)

    eval_data = None
    if args.eeg_test_embeds:
        eval_data = {
            "test_eeg": torch.load(args.eeg_test_embeds, map_location="cpu").float(),
            "test_img": load_img_features(args.vit_test_features),
        }
    train_with_checkpoints(pipe, dataloader, args, run_dir, eval_data)
    save_state_dict(pipe, run_dir / args.save_name)
    print("saved diffusion prior:", run_dir / args.save_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Train original or fixed raw-EEG chunk router diffusion prior.")
    parser.add_argument("--mode", choices=["original", "fixed_chunk_router"], default="fixed_chunk_router")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--atms-ckpt", default=None)
    parser.add_argument("--eeg-train-embeds", default=None)
    parser.add_argument("--eeg-test-embeds", default=None)
    parser.add_argument("--vit-train-features", default=None)
    parser.add_argument("--vit-test-features", default=None)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--atms-batch-size", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond-dim", type=int, default=1024)
    parser.add_argument("--init-gamma", type=float, default=0.1)
    parser.add_argument("--resume-prior-ckpt", default=None)
    parser.add_argument("--save-name", default="diffusion_prior.pt")
    parser.add_argument("--save-every", type=int, default=10)

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
    if args.mode == "original":
        train_original(args, run_dir, device)
    else:
        train_fixed_chunk_router(args, run_dir, device)
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
