import argparse
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

from diffusion_prior import DiffusionPriorUNet, Pipe
from custom_pipeline import Generator4Embeds
from DC_chunk_router import FrozenFixedChunkATMS, make_fixed_chunk_router_modules
from DC_train_atms import ATMS, load_eeg_split, subject_id_from_name


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


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


def concept_from_folder(folder):
    return folder[folder.index("_") + 1:] if "_" in folder else folder


def test_concept_names(test_image_dir):
    folders = [
        d for d in os.listdir(test_image_dir)
        if os.path.isdir(os.path.join(test_image_dir, d))
    ]
    folders.sort()
    return [concept_from_folder(folder) for folder in folders]


def make_run_dir(output_root, run_name):
    name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@torch.no_grad()
def extract_original_embeddings(args, device):
    if args.eeg_test_embeds:
        embeddings = torch.load(args.eeg_test_embeds, map_location="cpu").float()
        print("loaded EEG test embeddings:", args.eeg_test_embeds, tuple(embeddings.shape))
        return embeddings

    if not args.atms_ckpt:
        raise ValueError("original mode requires --eeg-test-embeds or --atms-ckpt.")

    eeg, _ = load_eeg_split(args.data_root, args.subject, False, (args.time_start, args.time_end))
    loader = DataLoader(EEGTensorDataset(eeg), batch_size=args.atms_batch_size, shuffle=False, num_workers=0)
    model = ATMS(63, 250, num_subjects=args.num_subjects)
    model.load_state_dict(torch.load(args.atms_ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()

    subject_id = subject_id_from_name(args.subject)
    features = []
    for eeg_batch in loader:
        eeg_batch = eeg_batch.to(device)
        subject_ids = torch.full((eeg_batch.size(0),), subject_id, dtype=torch.long, device=device)
        features.append(model(eeg_batch, subject_ids).float().cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def extract_fixed_condition_cache(args, device, run_dir):
    if not args.atms_ckpt:
        raise ValueError("fixed_chunk_router mode requires --atms-ckpt.")
    eeg, _ = load_eeg_split(args.data_root, args.subject, False, (args.time_start, args.time_end))
    loader = DataLoader(EEGTensorDataset(eeg), batch_size=args.atms_batch_size, shuffle=False, num_workers=0)
    conditioner = FrozenFixedChunkATMS(args.atms_ckpt, num_subjects=args.num_subjects, device=device)
    subject_id = subject_id_from_name(args.subject)

    z_global = []
    z_chunks = []
    for eeg_batch in loader:
        eeg_batch = eeg_batch.to(device)
        subject_ids = torch.full((eeg_batch.size(0),), subject_id, dtype=torch.long, device=device)
        cache = conditioner(eeg_batch, subject_ids)
        z_global.append(cache["z_global"].float().cpu())
        z_chunks.append(cache["z_chunks"].float().cpu())

    condition_cache = {
        "z_global": torch.cat(z_global, dim=0),
        "z_chunks": torch.cat(z_chunks, dim=0),
    }
    cache_path = run_dir / f"fixed_chunk_condition_cache_{args.subject}.pt"
    torch.save(condition_cache, cache_path)
    print("saved fixed condition cache:", cache_path)
    print("z_global:", tuple(condition_cache["z_global"].shape))
    print("z_chunks:", tuple(condition_cache["z_chunks"].shape))
    return condition_cache


def load_original_pipe(args, device):
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=args.dropout)
    pipe = Pipe(diffusion_prior, device=device)
    state = torch.load(args.diffusion_prior_ckpt, map_location=device)
    if isinstance(state, dict) and "diffusion_prior" in state:
        state = state["diffusion_prior"]
    pipe.diffusion_prior.load_state_dict(state)
    pipe.diffusion_prior.eval()
    print("loaded original diffusion prior:", args.diffusion_prior_ckpt)
    return pipe


def load_fixed_pipe(args, device):
    pipe = make_fixed_chunk_router_modules(device, cond_dim=1024, dropout=args.dropout, init_gamma=args.init_gamma)
    state = torch.load(args.diffusion_prior_ckpt, map_location=device)
    if "diffusion_prior" not in state:
        raise ValueError("fixed_chunk_router checkpoint must contain diffusion_prior, router, and gamma.")
    pipe.diffusion_prior.load_state_dict(state["diffusion_prior"])
    pipe.router_condition.router.load_state_dict(state["router"])
    pipe.router_condition.gamma.data.copy_(state["gamma"].to(device))
    pipe.diffusion_prior.eval()
    pipe.router_condition.eval()
    print("loaded fixed chunk router checkpoint:", args.diffusion_prior_ckpt)
    print("gamma:", float(pipe.router_condition.gamma.detach().cpu().item()))
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with original or fixed raw-EEG chunk router prior.")
    parser.add_argument("--mode", choices=["original", "fixed_chunk_router"], default="fixed_chunk_router")
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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-gamma", type=float, default=0.1)

    parser.add_argument("--output-root", default="/data/gaoy/projects/datasets/EEG_Image_decode/runs/generation")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-concepts", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--prior-steps", type=int, default=50)
    parser.add_argument("--prior-guidance-scale", type=float, default=5.0)
    parser.add_argument("--sdxl-steps", type=int, default=4)
    parser.add_argument("--test-image-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TORCH_HOME", args.torch_home)
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(args.output_root, args.run_name)
    image_dir = run_dir / "generated_imgs" / args.subject
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    test_image_dir = args.test_image_dir or str(Path(args.data_root) / "images_set" / "test_images")
    concepts = test_concept_names(test_image_dir)
    end_index = min(args.start_index + args.num_concepts, len(concepts))

    generator = Generator4Embeds(num_inference_steps=args.sdxl_steps, device=device)

    if args.mode == "original":
        eeg_features_test = extract_original_embeddings(args, device)
        torch.save(eeg_features_test, run_dir / f"original_eeg_features_{args.subject}_test.pt")
        pipe = load_original_pipe(args, device)

        end_index = min(end_index, eeg_features_test.shape[0])
        for k in range(args.start_index, end_index):
            prior_generator = torch.Generator(device=device).manual_seed(args.seed + k)
            h = pipe.generate(
                c_embeds=eeg_features_test[k:k + 1].to(device),
                num_inference_steps=args.prior_steps,
                guidance_scale=args.prior_guidance_scale,
                generator=prior_generator,
            )
            for j in range(args.repeats):
                image_generator = torch.Generator(device=device).manual_seed(args.seed + k * 1000 + j)
                image = generator.generate(h.to(dtype=torch.float16), generator=image_generator)
                path = image_dir / concepts[k] / f"{j}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                print("Image saved to", path)
    else:
        condition_cache = extract_fixed_condition_cache(args, device, run_dir)
        pipe = load_fixed_pipe(args, device)

        end_index = min(end_index, condition_cache["z_global"].shape[0])
        for k in range(args.start_index, end_index):
            prior_generator = torch.Generator(device=device).manual_seed(args.seed + k)
            sample_cache = {
                "z_global": condition_cache["z_global"][k:k + 1].to(device),
                "z_chunks": condition_cache["z_chunks"][k:k + 1].to(device),
            }
            h = pipe.generate(
                sample_cache,
                num_inference_steps=args.prior_steps,
                guidance_scale=args.prior_guidance_scale,
                generator=prior_generator,
            )
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
