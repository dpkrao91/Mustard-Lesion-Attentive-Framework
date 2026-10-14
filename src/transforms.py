"""Image transforms and on-the-fly augmentation.

The augmentation policy directly addresses the minority-class problem described
in the dataset documentation (Alternaria Blight = 783 images): aggressive but
label-preserving geometric and photometric perturbations expand the effective
support of the under-represented classes without synthesising unrealistic
samples.
"""
from __future__ import annotations

from torchvision import transforms

# ImageNet statistics (the timm backbones are ImageNet-pretrained).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224):
    """Training-time augmentation.

    Geometric: random resized crop, horizontal + vertical flips, small rotations
    and affine translation. Photometric: colour jitter to emulate the varied
    field lighting (direct sun, overcast, mist, canopy shadow) noted in the
    dataset description.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.20, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def build_eval_transform(image_size: int = 224):
    """Deterministic transform for validation / test / inference."""
    resize = int(round(image_size * 1.14))  # standard centre-crop ratio
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo ImageNet normalisation for visualisation. Returns a HxWx3 array in
    the [0, 1] range."""
    import numpy as np

    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return img.clip(0.0, 1.0)
