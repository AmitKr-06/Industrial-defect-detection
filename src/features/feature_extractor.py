"""
WideResNet50-2 feature extractor with PatchCore-style patch embeddings.

The backbone is frozen (no gradients, BatchNorm running stats preserved).
Two feature maps are returned for a 256x256 input:
    layer2 -> (B,  512, 32, 32)
    layer3 -> (B, 1024, 16, 16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

from src.logger import logging


# Names of the backbone stages we can read features from.
# Order matters: the first entry defines the output grid.
SUPPORTED_LAYERS = ("layer2", "layer3")


class FeatureExtractor(nn.Module):
    """
    Wrap a WideResNet50-2 pretrained on ImageNet and return the feature maps
    of layer2 and layer3.

    The backbone is FROZEN (no training, eval mode). PatchCore never trains a
    network; it only uses these features to describe each image patch. The
    torchvision weights are downloaded once and cached by torchvision.
    """

    def __init__(self):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        # Stem = conv1 + bn + relu + maxpool + layer1 (shared early layers).
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Freeze everything: no gradient, weights never change.
        for p in self.parameters():
            p.requires_grad = False

        # BatchNorm must use its stored running statistics, not batch statistics.
        self.eval()

    def train(self, mode: bool = True):
        """
        Force eval mode at all times.

        BatchNorm running statistics must never be updated for a frozen
        backbone, even if downstream code accidentally calls `.train()`.
        """
        return super().train(False)

    def forward(self, x):
        """Run the image through the backbone and return (layer2_map, layer3_map)."""
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        return f2, f3


def patch_embeddings(features, layers=("layer2", "layer3"),
                     pool_kernel=3, out_dim=None):
    """
    Turn backbone feature maps into one vector per image patch.

    Steps:
        1. avg-pool each selected layer with a `pool_kernel x pool_kernel`
           window, so each patch also "sees" its neighbours (more robust).
        2. resize every layer to the grid of the first selected layer
           (layer3 is 16x16, layer2 is 32x32, so layer3 is upsampled to 32x32).
        3. concatenate the layers along the channel axis
           (512 + 1024 = 1536 channels).
        4. optionally shrink the channels to `out_dim` with adaptive
           average pooling (out_dim=1024 is the baseline; None keeps all).

    Returns:
        embeddings: (B, H*W, D)
        grid:       (H, W)
    """
    if any(l not in SUPPORTED_LAYERS for l in layers):
        raise ValueError(
            f"layers must be a subset of {SUPPORTED_LAYERS}, got {list(layers)}"
        )
    if pool_kernel % 2 == 0:
        raise ValueError("pool_kernel must be odd so the grid size is preserved")

    fmap = dict(zip(SUPPORTED_LAYERS, features))
    pad = pool_kernel // 2

    # 1. Local average pooling per selected layer
    pooled = [
        F.avg_pool2d(fmap[l], pool_kernel, stride=1, padding=pad)
        for l in layers
    ]

    # 2. Align all layers to the first layer's spatial grid
    size = pooled[0].shape[-2:]
    pooled = [
        p if p.shape[-2:] == size
        else F.interpolate(p, size=size, mode="bilinear", align_corners=False)
        for p in pooled
    ]

    # 3. Concatenate along channels
    x = torch.cat(pooled, dim=1)                       # (B, C, H, W)
    B, C, H, W = x.shape

    # 4. Reshape to (B, H*W, C)
    x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)     # (B, N, C), N = H*W

    # 5. Optional channel-dimension reduction
    if out_dim is not None and out_dim != C:
        # We want to pool the CHANNEL axis (C) down to out_dim.
        # adaptive_avg_pool1d pools the LAST axis of a 3D input (batch, channels, length).
        # Treat each patch as an independent sample:
        #     x:         (B, N, C)      -- B images, N patches, C channels each
        #     reshape -> (B*N, 1, C)    -- each patch is now a "batch" element
        #     pool    -> (B*N, 1, out_dim)
        #     reshape -> (B, N, out_dim)
        B_, N, C_ = x.shape
        x = x.reshape(B_ * N, 1, C_)
        x = F.adaptive_avg_pool1d(x, out_dim)
        x = x.reshape(B_, N, out_dim)

    # Sanity check: the output must have the expected last dimension.
    expected_dim = out_dim if out_dim is not None else C
    assert x.shape == (B, H * W, expected_dim), \
        f"Unexpected output shape {tuple(x.shape)}, expected {(B, H*W, expected_dim)}"

    return x, (H, W)


@torch.inference_mode()
def extract_embeddings(model, loader, device, layers=SUPPORTED_LAYERS,
                       pool_kernel=3, out_dim=None):
    """
    Run a whole DataLoader through the backbone and collect patch embeddings.

    Used on the 198 fit images to get the raw patches that the coreset later
    reduces to the memory bank. Embeddings are moved to CPU batch by batch so
    GPU memory (4 GB) never fills up.

    Returns:
        embeddings: (N_images, patches_per_image, D) on CPU
        grid:       (H, W)
    """
    chunks, hw = [], None
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        emb, hw = patch_embeddings(model(x), layers, pool_kernel, out_dim)
        chunks.append(emb.cpu())

    assert chunks, "DataLoader produced no batches — check the loader / dataset."

    out = torch.cat(chunks)
    logging.debug("Extracted embeddings %s, grid %s", tuple(out.shape), hw)
    return out, hw