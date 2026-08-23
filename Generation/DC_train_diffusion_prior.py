import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
from DC_train_atms import ATMS, FeatureDataset, extract_embeddings, load_clip_features, load_eeg_split, subject_id_from_name


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train_eeg_embeddings_from_atms(args, device):
    train_eeg, train_labels = load_eeg_split(
        args.data_root,
        args.subject,
        True,
        (args.time_start, args.time_end),
    )
    text_train, img_train = load_clip_features(args.vit_train_features)
    train_text_per_sample = text_train[train_labels]
    train_img_per_sample = img_train.view(1654, 10, 1024).repeat_interleave(4, dim=1).view(-1, 1024)
    dataset = FeatureDataset(train_eeg, train_labels, train_text_per_sample, train_img_per_sample)

    model = ATMS(63, 250, num_subjects=args.num_subjects)
    model.load_state_dict(torch.load(args.atms_ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()

    subject_id = subject_id_from_name(args.subject)
    embeddings = extract_embeddings(model, dataset, args.atms_batch_size, device, subject_id)
    return embeddings


def make_target_img_embeddings(vit_train_features):
    _, emb_img_train = load_clip_features(vit_train_features)
    emb_img_train_4 = emb_img_train.view(1654, 10, 1, 1024).repeat(1, 1, 4, 1).view(-1, 1024)
    return emb_img_train_4.float()


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion prior from EEG embeddings to ViT-H image embeddings.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--eeg-train-embeds", default=None)
    parser.add_argument("--atms-ckpt", default=None)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--atms-batch-size", type=int, default=1024)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)

    parser.add_argument("--vit-train-features", default=None)
    parser.add_argument("--output-root", default="/data/gaoy/projects/datasets/EEG_Image_decode/runs/diffusion_prior")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond-dim", type=int, default=1024)
    parser.add_argument("--resume-prior-ckpt", default=None)
    parser.add_argument("--save-name", default="diffusion_prior.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eeg_train_embeds is None and args.atms_ckpt is None:
        raise ValueError("Provide --eeg-train-embeds or --atms-ckpt.")

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    args.vit_train_features = args.vit_train_features or str(Path(args.data_root) / "ViT-H-14_features_train.pt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    timestamp = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.subject / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.eeg_train_embeds is not None:
        eeg_features_train = torch.load(args.eeg_train_embeds, map_location="cpu").float()
        print("loaded EEG train embeddings:", args.eeg_train_embeds, tuple(eeg_features_train.shape))
    else:
        eeg_features_train = load_train_eeg_embeddings_from_atms(args, device)
        torch.save(eeg_features_train, run_dir / f"ATM_S_eeg_features_{args.subject}_train.pt")
        print("extracted EEG train embeddings:", tuple(eeg_features_train.shape))

    emb_img_train_4 = make_target_img_embeddings(args.vit_train_features)
    if eeg_features_train.shape != emb_img_train_4.shape:
        raise ValueError(
            f"EEG/image embedding shape mismatch: {tuple(eeg_features_train.shape)} vs {tuple(emb_img_train_4.shape)}"
        )
    torch.save(emb_img_train_4, run_dir / "ViT-H-14_img_features_train_repeat4.pt")

    dataset = EmbeddingDataset(c_embeddings=eeg_features_train, h_embeddings=emb_img_train_4)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    diffusion_prior = DiffusionPriorUNet(cond_dim=args.cond_dim, dropout=args.dropout)
    print("diffusion prior parameters:", sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
    pipe = Pipe(diffusion_prior, device=device)
    if args.resume_prior_ckpt:
        pipe.diffusion_prior.load_state_dict(torch.load(args.resume_prior_ckpt, map_location=device))
        print("loaded prior checkpoint:", args.resume_prior_ckpt)

    pipe.train(dataloader, num_epochs=args.epochs, learning_rate=args.lr)

    save_path = run_dir / args.save_name
    torch.save(pipe.diffusion_prior.state_dict(), save_path)
    print("saved diffusion prior:", save_path)
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
