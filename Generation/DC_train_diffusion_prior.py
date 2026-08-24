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
from torch import nn
import torch.optim as optim
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


def load_img_features(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        obj = obj["img_features"]
    return obj.float()


def validate_training_inputs(eeg_features_train, raw_img_train, repeated_img_train, run_dir):
    expected_eeg_shape = (1654 * 10 * 4, 1024)
    expected_raw_img_shape = (1654 * 10, 1024)
    errors = []

    if tuple(eeg_features_train.shape) != expected_eeg_shape:
        errors.append(f"Expected train EEG embeddings {expected_eeg_shape}, got {tuple(eeg_features_train.shape)}")
    if tuple(raw_img_train.shape) != expected_raw_img_shape:
        errors.append(f"Expected raw CLIP image targets {expected_raw_img_shape}, got {tuple(raw_img_train.shape)}")
    if tuple(repeated_img_train.shape) != expected_eeg_shape:
        errors.append(f"Expected repeated CLIP targets {expected_eeg_shape}, got {tuple(repeated_img_train.shape)}")

    raw_norms = raw_img_train.norm(dim=1)
    repeated_norms = repeated_img_train.norm(dim=1)
    unit_like = torch.allclose(raw_norms, torch.ones_like(raw_norms), rtol=1e-3, atol=1e-3)
    if unit_like:
        errors.append(
            "CLIP image targets look L2-normalized; expected raw/un-normalized features. "
            f"norm mean={raw_norms.mean().item():.6f}, std={raw_norms.std().item():.6f}"
        )

    expected_repeated = raw_img_train.repeat_interleave(4, dim=0)
    max_repeat_diff = (repeated_img_train - expected_repeated).abs().max().item()
    if max_repeat_diff != 0.0:
        errors.append(
            "16540 -> 66160 target repeat is not sample-wise repeat_interleave(4). "
            f"max_abs_diff={max_repeat_diff}"
        )

    print("strict check: train EEG embeddings", tuple(eeg_features_train.shape))
    print("strict check: raw image CLIP targets", tuple(raw_img_train.shape))
    print("strict check: repeated image CLIP targets", tuple(repeated_img_train.shape))
    print(
        "raw target norm stats:",
        f"mean={raw_norms.mean().item():.6f}",
        f"std={raw_norms.std().item():.6f}",
        f"min={raw_norms.min().item():.6f}",
        f"max={raw_norms.max().item():.6f}",
    )
    print(
        "repeat alignment:",
        "sample i -> image_idx i//4 -> concept image_idx//10 -> image_in_concept image_idx%10 -> rep i%4",
    )
    print("repeat max_abs_diff:", max_repeat_diff)

    pairing_path = run_dir / "pairing_check.csv"
    with open(pairing_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index",
            "image_index",
            "concept_index",
            "image_in_concept",
            "repetition_index",
            "target_equal_raw_image",
            "eeg_norm",
            "target_norm",
        ])
        for sample_index in range(20):
            image_index = sample_index // 4
            concept_index = image_index // 10
            image_in_concept = image_index % 10
            repetition_index = sample_index % 4
            target_equal = torch.equal(repeated_img_train[sample_index], raw_img_train[image_index])
            writer.writerow([
                sample_index,
                image_index,
                concept_index,
                image_in_concept,
                repetition_index,
                bool(target_equal),
                float(eeg_features_train[sample_index].norm().item()),
                float(repeated_norms[sample_index].item()),
            ])
            print(
                "PAIR",
                f"sample={sample_index:03d}",
                f"image_idx={image_index:04d}",
                f"concept={concept_index:04d}",
                f"image_in_concept={image_in_concept}",
                f"rep={repetition_index}",
                f"target_equal_raw={bool(target_equal)}",
            )
    print("saved pairing check:", pairing_path)
    if errors:
        raise ValueError("Strict training input check failed:\n" + "\n".join(f"- {error}" for error in errors))


@torch.no_grad()
def prior_retrieval_metrics(pipe, eeg_features, img_features, device, num_samples, seed, prior_steps, guidance_scale):
    n = min(num_samples, eeg_features.shape[0], img_features.shape[0])
    eeg_features = eeg_features[:n].float().to(device)
    img_features = img_features[:n].float().to(device)
    outputs = []

    for idx in range(n):
        generator = torch.Generator(device=device).manual_seed(seed + idx)
        h = pipe.generate(
            c_embeds=eeg_features[idx:idx + 1],
            num_inference_steps=prior_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        outputs.append(h.float().cpu())

    outputs = torch.cat(outputs, dim=0).to(device)
    paired_cosine = torch.nn.functional.cosine_similarity(outputs, img_features, dim=1).mean()
    retrieval = outputs @ img_features.T
    top1 = (retrieval.argmax(dim=1) == torch.arange(n, device=device)).float().mean()
    return float(paired_cosine.item()), float(top1.item())


def save_state_dict(pipe, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pipe.diffusion_prior.state_dict(), path)


def train_with_checkpoints(pipe, dataloader, args, run_dir, eval_data):
    pipe.diffusion_prior.train()
    device = pipe.device
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(pipe.diffusion_prior.parameters(), lr=args.lr)

    from diffusers.optimization import get_cosine_schedule_with_warmup

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * args.epochs),
    )
    num_train_timesteps = pipe.scheduler.config.num_train_timesteps
    ckpt_dir = run_dir / "checkpoints"
    loss_csv = run_dir / "loss.csv"
    eval_csv = run_dir / "prior_eval.csv"

    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_index", "epoch", "loss", "lr"])

    if eval_data is not None:
        with open(eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch_index", "epoch", "split", "paired_cosine", "top1", "num_samples"])

    for epoch_idx in range(args.epochs):
        loss_sum = 0.0
        pipe.diffusion_prior.train()
        for batch in dataloader:
            c_embeds = batch["c_embedding"].to(device) if "c_embedding" in batch.keys() else None
            h_embeds = batch["h_embedding"].to(device)
            n = h_embeds.shape[0]

            if torch.rand(1) < 0.1:
                c_embeds = None

            noise = torch.randn_like(h_embeds)
            timesteps = torch.randint(0, num_train_timesteps, (n,), device=device)
            perturbed_h_embeds = pipe.scheduler.add_noise(h_embeds, noise, timesteps)
            noise_pre = pipe.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
            loss = criterion(noise_pre, noise).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.diffusion_prior.parameters(), 1.0)
            lr_scheduler.step()
            optimizer.step()

            loss_sum += loss.item()

        loss_epoch = loss_sum / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch: {epoch_idx}, loss: {loss_epoch}")
        with open(loss_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch_idx, epoch_idx + 1, loss_epoch, current_lr])

        should_save = (epoch_idx + 1) % args.save_every == 0 or epoch_idx + 1 == args.epochs
        if should_save:
            save_state_dict(pipe, ckpt_dir / f"epoch_{epoch_idx + 1:03d}.pth")
            save_state_dict(pipe, run_dir / "latest.pth")

        if eval_data is not None and args.eval_every > 0 and (
            (epoch_idx + 1) % args.eval_every == 0 or epoch_idx + 1 == args.epochs
        ):
            test_cos, test_top1 = prior_retrieval_metrics(
                pipe,
                eval_data["test_eeg"],
                eval_data["test_img"],
                device,
                args.eval_num_samples,
                args.seed,
                args.eval_prior_steps,
                args.eval_guidance_scale,
            )
            print(
                f"eval epoch: {epoch_idx}, split: test, "
                f"paired_cosine: {test_cos}, top1: {test_top1}"
            )
            with open(eval_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch_idx, epoch_idx + 1, "test", test_cos, test_top1, args.eval_num_samples])


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
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eeg-test-embeds", default=None)
    parser.add_argument("--vit-test-features", default=None)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-num-samples", type=int, default=200)
    parser.add_argument("--eval-prior-steps", type=int, default=50)
    parser.add_argument("--eval-guidance-scale", type=float, default=5.0)
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
    args.vit_test_features = args.vit_test_features or str(Path(args.data_root) / "ViT-H-14_features_test.pt")
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

    raw_img_train = load_img_features(args.vit_train_features)
    emb_img_train_4 = raw_img_train.view(1654, 10, 1, 1024).repeat(1, 1, 4, 1).view(-1, 1024)
    validate_training_inputs(eeg_features_train, raw_img_train, emb_img_train_4, run_dir)
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

    eval_data = None
    if args.eeg_test_embeds is not None:
        eval_data = {
            "test_eeg": torch.load(args.eeg_test_embeds, map_location="cpu").float(),
            "test_img": load_img_features(args.vit_test_features),
        }
        print("loaded EEG test embeddings for prior eval:", args.eeg_test_embeds, tuple(eval_data["test_eeg"].shape))
        print("loaded image test embeddings for prior eval:", args.vit_test_features, tuple(eval_data["test_img"].shape))

    train_with_checkpoints(pipe, dataloader, args, run_dir, eval_data)

    save_path = run_dir / args.save_name
    save_state_dict(pipe, save_path)
    print("saved diffusion prior:", save_path)
    print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
