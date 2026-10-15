"""
Figures for error analysis: false positives and misses with their heatmaps.
"""

import matplotlib
matplotlib.use("Agg")      # draw to files only, no window (needed for scripts and DVC)
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.logger import logging
from src.constants import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denorm(t: torch.Tensor) -> np.ndarray:
    """
    What it does : turns a normalised image tensor back into a normal picture.
    What we do   : undo the ImageNet normalisation (x * std + mean), clamp to [0, 1] and
                   reorder (C, H, W) -> (H, W, C) so matplotlib can show it.
    """
    return (t.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def save_cases(indices, dataset, masks: np.ndarray, maps: np.ndarray, scores: np.ndarray,
               types, thr: float, title: str, out_path) -> bool:
    """
    What it does : saves one figure with a row per selected images.
    What we do   : each row shows 4 panels: the image with it's score, the ground-truth mask,
                   the heatmap overlay, and the predicted mask (map > thr). All heatmaps
                   share one colour scale (vmin/vmax over the whole set), so colours are
                   comparable between rows.
    Returns      : True if a figure was saved, False if there were no cases (e.g. zero
                   misses), in which cases nothing is written.
    """
    if len(indices) == 0:
        logging.info("[%s] no cases to show", title)
        return False

    vmin, vmax = float(maps.min()), float(maps.max())
    fig, axes = plt.subplots(len(indices), 4, figsize=(13, 3.2 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, i in zip(axes, indices):
        img = denorm(dataset[i]["image"])
        row[0].imshow(img)
        row[0].set_title(f"{types[i]} | score {scores[i]:.2f}")
        row[1].imshow(masks[i], cmap="gray")
        row[1].set_title("ground truth")
        row[2].imshow(img)
        row[2].imshow(maps[i], cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        row[2].set_title("overlay")
        row[3].imshow(maps[i] > thr, cmap="gray")
        row[3].set_title(f"pred mask (> {thr:.2f})")
        for a in row:
            a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)                     # free memory, important in a loop or long run
    logging.info("Saved figure %s (%d cases)", out_path, len(indices))
    return True