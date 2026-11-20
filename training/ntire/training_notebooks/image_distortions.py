"""PIL image augmentations importable by Windows DataLoader workers."""

from io import BytesIO

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class RandomDownscale:
    """Lose resolution by shrinking, then restoring the original dimensions."""

    def __init__(self, scale_min=0.35, scale_max=0.9):
        if not 0 < scale_min <= scale_max <= 1:
            raise ValueError("Scale must satisfy 0 < min <= max <= 1.")
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, image):
        _, height, width = TF.get_dimensions(image)
        scale = float(torch.empty(()).uniform_(self.scale_min, self.scale_max).item())
        reduced_size = [max(1, round(height * scale)), max(1, round(width * scale))]
        small = TF.resize(
            image, reduced_size, interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return TF.resize(
            small, [height, width], interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )


class RandomJPEGCompression:
    """Re-encode an RGB image with a randomly chosen JPEG quality."""

    def __init__(self, quality_min=70, quality_max=95):
        if not 1 <= quality_min <= quality_max <= 100:
            raise ValueError("JPEG quality must satisfy 1 <= min <= max <= 100.")
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image):
        quality = int(torch.randint(self.quality_min, self.quality_max + 1, ()).item())
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                return compressed.convert("RGB")
