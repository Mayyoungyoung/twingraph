"""Cache frozen ImageNet ResNet-18 features once per shared initial observation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .plan import digest

ENCODER = "torchvision.resnet18.IMAGENET1K_V1"


class FrozenVision:
    def __init__(self, device="cuda"):
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights).to(device).eval()
        self.model.fc = torch.nn.Identity()
        self.model.requires_grad_(False)
        self.transform = weights.transforms()
        self.device = device

    @torch.inference_mode()
    def encode(self, paths):
        if not paths:
            raise ValueError("visual checkpoint requires observation images")
        batch = torch.stack(
            [self.transform(Image.open(p).convert("RGB")) for p in paths]
        ).to(self.device)
        return self.model(batch).cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    encoder = FrozenVision(a.device)
    count = 0
    for complete in sorted(Path(a.data).glob("group_*/complete.json")):
        directory = complete.parent
        inputs = json.loads((directory / "inputs.json").read_text())
        output = directory / "visual.npz"
        if output.exists():
            old = np.load(output, allow_pickle=False)
            if str(old["encoder"]) == ENCODER and str(old["input_sha256"]) == digest(
                inputs
            ):
                continue
        paths = [directory / f for f in inputs["images"]]
        features = encoder.encode(paths)
        np.savez_compressed(
            output,
            features=features,
            encoder=ENCODER,
            input_sha256=digest(inputs),
            image_sha256=np.array(
                [hashlib.sha256(f.read_bytes()).hexdigest() for f in paths]
            ),
        )
        count += 1
    print(json.dumps(dict(cached_groups=count, encoder=ENCODER, device=a.device)))


if __name__ == "__main__":
    main()
