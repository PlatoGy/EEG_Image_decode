import argparse
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
from torch.utils.data import DataLoader, Dataset


GENERATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = GENERATION_DIR.parent
for path in (str(GENERATION_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diffusion_prior import DiffusionPriorUNet, Pipe
from custom_pipeline import Generator4Embeds
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


def load_test_eeg(data_root, subject, time_window):
    path = Path(data_root) / "Preprocessed_data_250Hz" / subject / "preprocessed_eeg_test.npy"
    data = load_npy_dict(path)
    eeg = torch.from_numpy(data["preprocessed_eeg_data"]).float()
    eeg = select_time_window(eeg, data["times"], time_window[0], time_window[1])

    data_list = []
    for class_idx in range(200):
        block = eeg[class_idx : class_idx + 1]
        data_list.append(torch.mean(block.squeeze(0), dim=0))
    return torch.cat(data_list, dim=0).view(-1, eeg.shape[-2], eeg.shape[-1])


def test_concept_names(test_image_dir):
    folders = [
        d for d in os.listdir(test_image_dir)
        if os.path.isdir(os.path.join(test_image_dir, d))
    ]
    folders.sort()
    names = []
    for folder in folders:
        names.append(folder[folder.index("_") + 1:] if "_" in folder else folder)
    return names


@torch.no_grad()
def extract_atms_embeddings(args, device):
    eeg = load_test_eeg(args.data_root, args.subject, (args.time_start, args.time_end))
    loader = DataLoader(EEGTensorDataset(eeg), batch_size=args.atms_batch_size, shuffle=False, num_workers=0)

    model = ATMS(63, 250, num_subjects=args.num_subjects)
    state_dict = torch.load(args.atms_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    all_features = []
    subject_id = subject_id_from_name(args.subject)
    for eeg_batch in loader:
        eeg_batch = eeg_batch.to(device)
        subject_ids = torch.full((eeg_batch.size(0),), subject_id, dtype=torch.long, device=device)
        all_features.append(model(eeg_batch, subject_ids).float().cpu())
    return torch.cat(all_features, dim=0)


def make_run_dir(output_root, run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or timestamp
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sub-08 reconstructions from ATMS/EEG embeddings.")
    parser.add_argument("--data-root", default="/data/gaoy/projects/datasets/EEG_Image_decode")
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-home", default="/data/gaoy/projects/.cache/huggingface")
    parser.add_argument("--torch-home", default="/data/gaoy/.cache/torch")

    parser.add_argument("--atms-ckpt", default=None)
    parser.add_argument("--eeg-test-embeds", default=None)
    parser.add_argument("--diffusion-prior-ckpt", required=True)
    parser.add_argument("--num-subjects", type=int, default=2)
    parser.add_argument("--atms-batch-size", type=int, default=1024)
    parser.add_argument("--time-start", type=float, default=0.0)
    parser.add_argument("--time-end", type=float, default=1.0)

    parser.add_argument("--output-root", default="/data/gaoy/projects/datasets/EEG_Image_decode/runs/generation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-concepts", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--prior-steps", type=int, default=50)
    parser.add_argument("--prior-guidance-scale", type=float, default=5.0)
    parser.add_argument("--sdxl-steps", type=int, default=4)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--test-image-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eeg_test_embeds is None and args.atms_ckpt is None:
        raise ValueError("Provide --atms-ckpt or --eeg-test-embeds.")

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_image_dir = args.test_image_dir or str(Path(args.data_root) / "images_set" / "test_images")
    concepts = test_concept_names(test_image_dir)
    if len(concepts) < args.start_index + args.num_concepts:
        args.num_concepts = len(concepts) - args.start_index

    run_dir = make_run_dir(args.output_root, args.run_name)
    image_dir = run_dir / "generated_imgs" / args.subject

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.eeg_test_embeds is not None:
        eeg_features_test = torch.load(args.eeg_test_embeds, map_location="cpu").float()
        print("loaded EEG test embeddings:", args.eeg_test_embeds, tuple(eeg_features_test.shape))
    else:
        eeg_features_test = extract_atms_embeddings(args, device)
        torch.save(eeg_features_test, run_dir / f"ATMS_eeg_features_{args.subject}_test.pt")
        print("extracted ATMS EEG test embeddings:", tuple(eeg_features_test.shape))

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(torch.load(args.diffusion_prior_ckpt, map_location=device))
    pipe.diffusion_prior.eval()
    print("loaded diffusion prior:", args.diffusion_prior_ckpt)

    end_index = min(args.start_index + args.num_concepts, len(concepts), eeg_features_test.shape[0])
    prior_embeds = []
    with torch.no_grad():
        for k in range(args.start_index, end_index):
            prior_generator = torch.Generator(device=device).manual_seed(args.seed + k)
            eeg_embeds = eeg_features_test[k : k + 1].to(device)
            h = pipe.generate(
                c_embeds=eeg_embeds,
                num_inference_steps=args.prior_steps,
                guidance_scale=args.prior_guidance_scale,
                generator=prior_generator,
            )
            prior_embeds.append(h.detach().cpu())

    del pipe
    del diffusion_prior
    if device.type == "cuda":
        torch.cuda.empty_cache()

    generator = Generator4Embeds(
        num_inference_steps=args.sdxl_steps,
        device=device,
        cpu_offload=args.cpu_offload or args.low_vram,
        sequential_cpu_offload=args.sequential_cpu_offload,
        attention_slicing=args.attention_slicing,
        vae_slicing=args.vae_slicing or args.low_vram,
        xformers=args.xformers,
        height=args.height,
        width=args.width,
    )

    for offset, k in enumerate(range(args.start_index, end_index)):
        h = prior_embeds[offset]
        for j in range(args.repeats):
            image_generator = torch.Generator(device=device).manual_seed(args.seed + k * 1000 + j)
            image = generator.generate(h.to(dtype=torch.float16), generator=image_generator)
            path = image_dir / concepts[k] / f"{j}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            print("Image saved to", path)

    print("run_dir:", run_dir)
    print("generated_dir:", image_dir)


if __name__ == "__main__":
    main()
