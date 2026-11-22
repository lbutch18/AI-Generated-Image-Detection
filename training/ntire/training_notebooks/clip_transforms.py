"""Picklable CLIP preprocessing for Windows DataLoader workers."""

import random

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize


class Downscale:
    """Shrink an image by a sampled fraction, then restore its input size."""

    def __init__(self, min_scale=0.30, max_scale=0.80):
        if not 0 < min_scale <= max_scale <= 1:
            raise ValueError("Expected 0 < min_scale <= max_scale <= 1.")
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image):
        height, width = image.shape[-2:]
        scale = random.uniform(self.min_scale, self.max_scale)
        smaller_size = [max(1, round(height * scale)), max(1, round(width * scale))]
        smaller = resize(image, smaller_size, InterpolationMode.BILINEAR, antialias=True)
        return resize(smaller, [height, width], InterpolationMode.BILINEAR, antialias=True)


class CLIPTransform:
    """Process a PIL image, optionally augment in [0, 1], then normalize."""

    def __init__(self, processor, augmentation=None):
        self.processor = processor
        self.augmentation = augmentation

    def __call__(self, image):
        # The processor handles RGB conversion, resize, crop, and /255 once.
        image = self.processor(
            images=image, return_tensors="pt", do_normalize=False,
        )["pixel_values"][0]

        if self.augmentation is not None:
            image = self.augmentation(image)

        return normalize(
            image, mean=self.processor.image_mean, std=self.processor.image_std,
        )
