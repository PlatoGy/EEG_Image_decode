import argparse
import csv
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.loss import ClipLoss
from models.subject_layers.Embed import DataEmbedding
from models.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.subject_layers.Transformer_EncDec import Encoder, EncoderLayer


class Config:
    def __init__(self):
        self.task_name = "classification"
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = "timeF"
        self.freq = "h"
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = "gelu"
        self.enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            joint_train=False,
            num_subjects=num_subjects,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return enc_out[:, :63, :]


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.tsconv(x.unsqueeze(1)))


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FlattenHead(nn.Sequential):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(
        self,
        num_channels=63,
        sequence_length=250,
        num_subjects=2,
        num_features=64,
        num_latents=1024,
        num_blocks=1,
    ):
        super().__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)]
        )
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        return self.proj_eeg(eeg_embedding)


class FeatureDataset(Dataset):
    def __init__(self, eeg, labels, text_features, img_features):
        self.eeg = eeg
        self.labels = labels
        self.text_features = text_features
        self.img_features = img_features

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx], self.labels[idx], self.text_features[idx], self.img_features[idx]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_id_from_name(subject):
    match = re.search(r"\d+$", subject)
    if not match:
        raise ValueError(f"Cannot parse subject id from {subject}")
    return int(match.group())


def load_npy_dict(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def select_time_window(eeg, times, start=0.0, end=1.0):
    times = torch.as_tensor(times).float()
    if eeg.shape[-1] != len(times) and eeg.shape[-1] == len(times[50:]):
        times = times[50:]
    if eeg.shape[-1] != len(times):
        return eeg
    keep = (times >= start) & (times <= end)
    return eeg[..., keep]


def load_eeg_split(data_root, subject, train, time_window):
    filename = "preprocessed_eeg_training.npy" if train else "preprocessed_eeg_test.npy"
    path = Path(data_root) / "Preprocessed_data_250Hz" / subject / filename
    data = load_npy_dict(path)
    eeg = torch.from_numpy(data["preprocessed_eeg_data"]).float()
    eeg = select_time_window(eeg, data["times"], time_window[0], time_window[1])

    if train:
        if eeg.ndim != 4:
            raise ValueError(f"Expected train EEG [16540,4,63,250], got {tuple(eeg.shape)}")
        eeg = eeg.reshape(-1, eeg.shape[-2], eeg.shape[-1])
        labels = torch.arange(1654, dtype=torch.long).repeat_interleave(10 * 4)
    else:
        if eeg.ndim != 4:
            raise ValueError(f"Expected test EEG [200,80,63,250], got {tuple(eeg.shape)}")
        eeg = torch.stack([torch.mean(eeg[i].squeeze(0), dim=0) for i in range(eeg.shape[0])], dim=0)
        labels = torch.arange(200, dtype=torch.long)
    return eeg, labels


def load_clip_features(path):
    obj = torch.load(path, map_location="cpu")
    return obj["text_features"].float(), obj["img_features"].float()


def build_feature_datasets(args):
    train_eeg, train_labels = load_eeg_split(
        args.data_root, args.subject, True, (args.time_start, args.time_end)
    )
    test_eeg, test_labels = load_eeg_split(
        args.data_root, args.subject, False, (args.time_start, args.time_end)
    )
    text_train, img_train = load_clip_features(args.vit_train_features)
    text_test, img_test = load_clip_features(args.vit_test_features)

    train_text_per_sample = text_train[train_labels]
    train_img_per_sample = img_train.view(1654, 10, 1024).repeat_interleave(4, dim=1).view(-1, 1024)
    test_text_per_sample = text_test[test_labels]
    test_img_per_sample = img_test[test_labels]

    train_dataset = FeatureDataset(train_eeg, train_labels, train_text_per_sample, train_img_per_sample)
    test_dataset = FeatureDataset(test_eeg, test_labels, test_text_per_sample, test_img_per_sample)
    return train_dataset, test_dataset, text_train, img_train, text_test, img_test


def train_one_epoch(subject, model, dataloader, optimizer, device, img_features_all, subject_id):
    model.train()
    img_features_all = img_features_all[::10].to(device).float()
    total_loss = 0.0
    correct = 0
    total = 0
    alpha = 0.90
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, labels, text_features, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)
        img_features = img_features.to(device).float()

        optimizer.zero_grad()
        subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long, device=device)
        eeg_features = model(eeg_data, subject_ids).float()
        logit_scale = model.logit_scale

        img_loss = model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits = logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.numel()

    return total_loss / (batch_idx + 1), correct / total


@torch.no_grad()
def evaluate(subject, model, dataloader, device, text_features_all, img_features_all, subject_id, k=200):
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    all_labels = set(range(text_features_all.size(0)))
    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0
    alpha = 0.99
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, labels, text_features, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)
        img_features = img_features.to(device).float()
        subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long, device=device)
        eeg_features = model(eeg_data, subject_ids).float()
        logit_scale = model.logit_scale

        img_loss = model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
        total_loss += loss.item()

        for idx, label in enumerate(labels):
            possible_classes = list(all_labels - {label.item()})
            selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
            selected_img_features = img_features_all[selected_classes]
            logits = logit_scale * eeg_features[idx] @ selected_img_features.T
            predicted_label = selected_classes[torch.argmax(logits).item()]
            correct += int(predicted_label == label.item())
            if k >= 5:
                _, top5_indices = torch.topk(logits, 5, largest=True)
                top5_correct += int(label.item() in [selected_classes[i] for i in top5_indices.tolist()])
            total += 1

    return total_loss / (batch_idx + 1), correct / total, top5_correct / total if k >= 5 else 0.0


@torch.no_grad()
def extract_embeddings(model, dataset, batch_size, device, subject_id):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    features = []
    for eeg_data, labels, text_features, img_features in loader:
        eeg_data = eeg_data.to(device)
        subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long, device=device)
        features.append(model(eeg_data, subject_ids).float().cpu())
    return torch.cat(features, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ATMS using the author-style EEG/image contrast objective.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)
    parser.add_argument("--output-root", default="/data/gaoy/projects/EEG_Image_decode/models/contrast/ATMS")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--vit-train-features", default=None)
    parser.add_argument("--vit-test-features", default=None)
    parser.add_argument("--resume-ckpt", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.vit_train_features = args.vit_train_features or str(Path(args.data_root) / "ViT-H-14_features_train.pt")
    args.vit_test_features = args.vit_test_features or str(Path(args.data_root) / "ViT-H-14_features_test.pt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subject_id = subject_id_from_name(args.subject)

    timestamp = args.run_name or datetime.now().strftime("%m-%d_%H-%M")
    run_dir = Path(args.output_root) / args.subject / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_dataset, test_dataset, text_train, img_train, text_test, img_test = build_feature_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    model = ATMS(63, 250, num_subjects=args.num_subjects).to(device)
    if args.resume_ckpt:
        model.load_state_dict(torch.load(args.resume_ckpt, map_location="cpu"))
        print("loaded checkpoint:", args.resume_ckpt)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    print("ATMS parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("run_dir:", run_dir)

    results = []
    best_test_accuracy = -1.0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            args.subject, model, train_loader, optimizer, device, img_train, subject_id
        )
        test_loss, test_accuracy, top5_accuracy = evaluate(
            args.subject, model, test_loader, device, text_test, img_test, subject_id, k=200
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "top5_accuracy": top5_accuracy,
        }
        results.append(row)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
            f"Top5 Accuracy: {top5_accuracy:.4f}"
        )

        if (epoch + 1) % args.save_every == 0:
            path = ckpt_dir / f"{epoch + 1}.pth"
            torch.save(model.state_dict(), path)
            print("saved checkpoint:", path)
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), run_dir / "best.pth")

        with open(run_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    torch.save(model.state_dict(), run_dir / "final.pth")
    train_embeddings = extract_embeddings(model, train_dataset, args.batch_size, device, subject_id)
    test_embeddings = extract_embeddings(model, test_dataset, args.batch_size, device, subject_id)
    torch.save(train_embeddings, run_dir / f"ATM_S_eeg_features_{args.subject}_train.pt")
    torch.save(test_embeddings, run_dir / f"ATM_S_eeg_features_{args.subject}_test.pt")
    print("saved final:", run_dir / "final.pth")
    print("saved train embeddings:", tuple(train_embeddings.shape))
    print("saved test embeddings:", tuple(test_embeddings.shape))


if __name__ == "__main__":
    main()
