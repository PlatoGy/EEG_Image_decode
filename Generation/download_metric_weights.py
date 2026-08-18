import os
from pathlib import Path


def torch_checkpoint_dir():
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")).expanduser()
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def main():
    print(f"TORCH_HOME={os.environ.get('TORCH_HOME', str(Path.home() / '.cache' / 'torch'))}")
    print(f"checkpoint dir={torch_checkpoint_dir()}")

    from torchvision.models import (
        AlexNet_Weights,
        EfficientNet_B1_Weights,
        Inception_V3_Weights,
        alexnet,
        efficientnet_b1,
        inception_v3,
    )

    print("Downloading/checking AlexNet ImageNet weights...")
    alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

    print("Downloading/checking InceptionV3 ImageNet weights...")
    inception_v3(weights=Inception_V3_Weights.DEFAULT)

    print("Downloading/checking EfficientNet-B1 ImageNet weights...")
    efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

    print("Downloading/checking CLIP ViT-L/14 weights...")
    import clip

    clip.load("ViT-L/14", device="cpu", download_root=str(torch_checkpoint_dir()))

    print("Done. Cached files:")
    for path in sorted(torch_checkpoint_dir().iterdir()):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
