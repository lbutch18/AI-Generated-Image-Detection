"""Picklable CLIP preprocessing for Windows DataLoader workers."""

from torchvision.transforms.functional import normalize


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
