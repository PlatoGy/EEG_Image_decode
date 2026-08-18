import argparse
import os

import torch

from custom_pipeline import Generator4Embeds
from diffusion_prior import DiffusionPriorUNet, Pipe
from path_config import (
    DIFFUSION_PRIOR_CKPT,
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Subject-08 reconstructions from the released EEG embeddings."
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--subject", default="sub-08")
    args = parser.parse_args()

    ensure_repo_on_path()
    device = torch.device(args.device)

    if not RELEASED_EEG_TEST_FEATURES.exists():
        raise FileNotFoundError(f"Released EEG embeddings not found: {RELEASED_EEG_TEST_FEATURES}")
    if not DIFFUSION_PRIOR_CKPT.exists():
        raise FileNotFoundError(f"Diffusion prior checkpoint not found: {DIFFUSION_PRIOR_CKPT}")

    eeg_embeds_test = torch.load(RELEASED_EEG_TEST_FEATURES, map_location=device)
    if list(eeg_embeds_test.shape) != [200, 1024]:
        raise ValueError(f"Expected EEG embedding shape [200, 1024], got {list(eeg_embeds_test.shape)}")

    names = test_concept_names()
    if len(names) < args.start + args.count:
        raise ValueError(f"Requested {args.start + args.count} concepts, but found {len(names)} in {TEST_IMAGE_DIR}")

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    pipe = Pipe(diffusion_prior, device=device)
    pipe.diffusion_prior.load_state_dict(torch.load(DIFFUSION_PRIOR_CKPT, map_location=device))

    generator = Generator4Embeds(num_inference_steps=4, device=device)
    output_root = GENERATED_IMAGE_DIR / args.subject

    print("Using released Subject-08 EEG embeddings; ATMS 40.pth is not loaded.")
    print(f"EEG embeddings: {RELEASED_EEG_TEST_FEATURES}")
    print(f"Diffusion prior: {DIFFUSION_PRIOR_CKPT}")
    print(f"Output: {output_root}")

    end = min(args.start + args.count, 200)
    for k in range(args.start, end):
        h = pipe.generate(c_embeds=eeg_embeds_test[k : k + 1], num_inference_steps=50, guidance_scale=5.0)
        for j in range(args.repeats):
            image = generator.generate(h.to(dtype=torch.float16))
            path = output_root / names[k] / f"{j}.png"
            os.makedirs(path.parent, exist_ok=True)
            image.save(path)
            print(f"Image saved to {path}")


if __name__ == "__main__":
    main()
