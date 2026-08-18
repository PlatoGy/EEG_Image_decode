import os
import urllib.error
from pathlib import Path


def torch_checkpoint_dir():
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")).expanduser()
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def print_weight_url(name, weights):
    filename = Path(weights.url).name
    print(f"{name} url={weights.url}")
    print(f"{name} cache target={torch_checkpoint_dir() / filename}")


def main():
    print(f"TORCH_HOME={os.environ.get('TORCH_HOME', str(Path.home() / '.cache' / 'torch'))}")
    print(f"checkpoint dir={torch_checkpoint_dir()}")
    print("HF_ENDPOINT only affects Hugging Face Hub downloads, not torchvision or OpenAI CLIP weights.")

    from torchvision.models import (
        AlexNet_Weights,
        EfficientNet_B1_Weights,
        Inception_V3_Weights,
        alexnet,
        efficientnet_b1,
        inception_v3,
    )

    print("Downloading/checking AlexNet ImageNet weights...")
    print_weight_url("AlexNet", AlexNet_Weights.IMAGENET1K_V1)
    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    print("Downloading/checking InceptionV3 ImageNet weights...")
    print_weight_url("InceptionV3", Inception_V3_Weights.DEFAULT)
    inception_v3(weights=Inception_V3_Weights.DEFAULT)

    print("Downloading/checking EfficientNet-B1 ImageNet weights...")
    print_weight_url("EfficientNet-B1", EfficientNet_B1_Weights.DEFAULT)
    efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

    print("Downloading/checking CLIP ViT-L/14 weights...")
    import clip

    try:
        print(f"CLIP cache target={torch_checkpoint_dir() / 'ViT-L-14.pt'}")
        clip.load("ViT-L/14", device="cpu", download_root=str(torch_checkpoint_dir()))
    except urllib.error.URLError as exc:
        target = torch_checkpoint_dir() / "ViT-L-14.pt"
        raise RuntimeError(
            "OpenAI CLIP ViT-L/14 is not a Hugging Face model and clip.load() "
            "downloads it from OpenAI's public URL. Network resolution failed. "
            f"If you already have this file from another environment, copy it to: {target}"
        ) from exc

    print("Done. Cached files:")
    for path in sorted(torch_checkpoint_dir().iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
