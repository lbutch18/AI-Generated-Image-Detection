"""DataLoader-compatible wrapper around the organizers' released pipeline.

Source: https://drive.google.com/file/d/1oGr--PUOd11xy0ayYB6p2Mgg67n6eJPc/view
The caller controls the clean-image probability and applies normalization later.
"""

from torch import nn

from aug_utils_train import distort_images


class OfficialDistortions(nn.Module):
    """Apply 1..max_distortions operations to a CPU float [3,H,W] image in [0,1]."""

    def __init__(self, max_distortions=3, num_levels=5):
        super().__init__()
        if not 1 <= max_distortions <= 7:
            raise ValueError("max_distortions must be between 1 and 7 groups.")
        if not 1 <= num_levels <= 5:
            raise ValueError("num_levels must be between 1 and 5.")
        self.max_distortions = max_distortions
        self.num_levels = num_levels

    def forward(self, image):
        distorted, _, _ = distort_images(
            image, max_distortions=self.max_distortions,
            num_levels=self.num_levels,
        )
        return distorted
