import argparse
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from path_config import (
    ATMS_CHECKPOINT,
    DIFFUSION_PRIOR_CKPT,
    EEG_DATA_PATH,
    GENERATED_IMAGE_DIR,
    RELEASED_EEG_TEST_FEATURES,
    TEST_IMAGE_DIR,
    ensure_repo_on_path,
)


def test_concept_names():
    folders = [d for d in os.listdir(TEST_IMAGE_DIR) if os.path.isdir(TEST_IMAGE_DIR / d)]
    folders.sort()
    names = []
    for folder in folders:
        try:
            names.append(folder[folder.index("_") + 1 :])
        except ValueError:
            names.append(folder)
    return names


def parse_subject_id(subject):
    match = re.search(r"\d+$", subject)
    if not match:
        raise ValueError(f"Cannot parse numeric subject id from {subject!r}")
    return int(match.group())


def resolve_device(device_name):
    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        if ":" in device_name:
            index = int(device_name.split(":", 1)[1])
            count = torch.cuda.device_count()
            if index >= count:
                raise RuntimeError(f"Requested {device_name}, but only {count} CUDA device(s) are visible.")
    return torch.device(device_name)


@torch.no_grad()
def extract_atms_test_embeddings(subject, checkpoint, device, batch_size, num_subjects):
    from ATMS_reconstruction import ATMS
    from eegdatasets_leaveone import EEGDataset

    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"ATMS checkpoint not found: {checkpoint}")

    dataset = EEGDataset(str(EEG_DATA_PATH), subjects=[subject], train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    eeg_model = ATMS(63, 250, num_subjects=num_subjects)
    eeg_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    eeg_model = eeg_model.to(device)
    eeg_model.eval()

    subject_id = parse_subject_id(subject)
    img_features_all = dataset.img_features.to(device).float()
    features = []
    correct = 0
    total = 0

    for eeg_data, labels, _text, _text_features, _img, _img_features in dataloader:
        eeg_data = eeg_data.to(device)
        labels = labels.to(device)
        subject_ids = torch.full((eeg_data.size(0),), subject_id, dtype=torch.long, device=device)
        eeg_features = eeg_model(eeg_data, subject_ids).float()
        features.append(eeg_features.cpu())

        logits = eeg_model.logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.numel()

    features = torch.cat(features, dim=0)
    print(f"ATMS test embeddings: {features.shape}")
    if total:
        print(f"ATMS test top-1 accuracy over {total} classes: {correct / total:.4f}")
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Generate Subject-08 reconstructions from released EEG embeddings or an ATMS 40.pth checkpoint."
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eeg-source", choices=["released", "atms"], default="released")
    parser.add_argument("--atms-checkpoint", default=str(ATMS_CHECKPOINT))
    parser.add_argument("--atms-batch-size", type=int, default=1024)
    parser.add_argument("--atms-num-subjects", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ensure_repo_on_path()
    device = resolve_device(args.device)

    if not DIFFUSION_PRIOR_CKPT.exists():
        raise FileNotFoundError(f"Diffusion prior checkpoint not found: {DIFFUSION_PRIOR_CKPT}")

    if args.eeg_source == "released":
        if not RELEASED_EEG_TEST_FEATURES.exists():
            raise FileNotFoundError(f"Released EEG embeddings not found: {RELEASED_EEG_TEST_FEATURES}")
        eeg_embeds_test = torch.load(RELEASED_EEG_TEST_FEATURES, map_location="cpu")
        print("Using released Subject-08 EEG embeddings; ATMS 40.pth is not loaded.")
        print(f"EEG embeddings: {RELEASED_EEG_TEST_FEATURES}")
    else:
        eeg_embeds_test = extract_atms_test_embeddings(
            subject=args.subject,
            checkpoint=args.atms_checkpoint,
            device=device,
            batch_size=args.atms_batch_size,
            num_subjects=args.atms_num_subjects,
        )
        print("Using EEG embeddings extracted from ATMS 40.pth.")
        print(f"ATMS checkpoint: {args.atms_checkpoint}")

    if list(eeg_embeds_test.shape) != [200, 1024]:
        raise ValueError(f"Expected EEG embedding shape [200, 1024], got {list(eeg_embeds_test.shape)}")

    names = test_concept_names()
    if len(names) < args.start + args.count:
        raise ValueError(f"Requested {args.start + args.count} concepts, but found {len(names)} in {TEST_IMAGE_DIR}")

    from custom_pipeline import Generator4Embeds
    from diffusion_prior import DiffusionPriorUNet, Pipe

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(torch.load(DIFFUSION_PRIOR_CKPT, map_location=device))

    generator = Generator4Embeds(num_inference_steps=4, device=device)
    output_subdir = args.output_subdir or args.subject
    output_root = GENERATED_IMAGE_DIR / output_subdir

    print(f"Diffusion prior: {DIFFUSION_PRIOR_CKPT}")
    print(f"Output: {output_root}")

    end = min(args.start + args.count, 200)
    for k in range(args.start, end):
        prior_generator = None
        if args.seed is not None:
            prior_generator = torch.Generator(device=device).manual_seed(args.seed + k)
        h = pipe.generate(
            c_embeds=eeg_embeds_test[k : k + 1].to(device),
            num_inference_steps=50,
            guidance_scale=5.0,
            generator=prior_generator,
        )
        for j in range(args.repeats):
            image_generator = None
            if args.seed is not None:
                image_generator = torch.Generator(device=device).manual_seed(args.seed + k * 1000 + j)
            image = generator.generate(h.to(dtype=torch.float16), generator=image_generator)
            path = output_root / names[k] / f"{j}.png"
            os.makedirs(path.parent, exist_ok=True)
            image.save(path)
            print(f"Image saved to {path}")


if __name__ == "__main__":
    main()
