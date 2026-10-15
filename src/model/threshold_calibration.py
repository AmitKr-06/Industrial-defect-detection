"""
Pipeline stage: threshold calibration.

PatchCore gives every image a score. To say "defective / not defective" we need a
threshold. We set it from the 22 held-out NORMAL images (role == val_normal), which were
never used to build the memory bank, so no test data is involved.

Input : models/memory_bank.pt, data/processed/splits.csv
Output: models/threshold.json               (the threshold + how it was computed)
        reports/metrics/val_normal_scores.csv (the 22 scores, for plots/reports)
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.logger import logging
from src.exception import CustomException
from src.config import PARAMS_PATH, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, make_dirs
from src.constants import ROLE_VAL, SPLITS_FILE, THRESHOLD_FILE
from src.utils import load_params, save_json, seed_everything, get_device
from src.data.dataset import MVTecDataset, load_splits
from src.model.patchcore import PatchCore


def build_val_loader(splits: pd.DataFrame, image_size: int, batch_size: int) -> DataLoader:
    """
    What it does : builds the DataLoader over the held-out normal images only.
    What we do   : keep rows with role == "val_normal" (22 images). These images are
                   normal but were NOT used for the memory bank, so their scores show what
                   a normal image looks like to the model on unseen data.
    """
    val_df = splits[splits.role == ROLE_VAL]
    logging.info("Validation-normal images: %d", len(val_df))
    return DataLoader(MVTecDataset(val_df, image_size), batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=True)

def compute_threshold(scores: np.ndarray, policy: str, k_std: float, quantile: float) -> float:
    """
    What it does : converts the normal-image scores into one threshold.
    What we do   : three policies, chosen in params.yaml:
        "max"    : highest normal score. Refrence setting. No normal val image is flagged,
                   but test normals can still exceed it (2 of 22 did in Exp 6).
        "mean_std": mean + k_std * std. Higher threshold, fewer false alarms, may miss
                    weak defects (Exp 6: 0 false positives but 3 misses with k_std = 3).
        "quantile": the given quantile of normal scores (e.g. 0.99).
    Returns       : float threshold. Image is defective if score > threshold.
    """
    if policy == "max":
        return float(scores.max())
    if policy == "mean_std":
        return float(scores.mean() + k_std * scores.std())
    if policy == "quantile":
        return float(np.quantile(scores, quantile))
    raise ValueError(f"Unknown threshold policy: {policy} (use max | mean_std | quantile)")

def save_data(scores: np.ndarray, thr: float, th: dict) -> None:
    """
    What it does : saves the threshold and the scores it came from.
    What we do   : threshold.json is ready by evaluation and by the inference pipeline.
                   It also records the policy and the number of images, and is marked
                   "provisional", because 22 normals are too few to guarantee a
                   production false-alarm rate(same limiitation as in my Exp 6/8 notes).
    """
    pd.DataFrame({"score": scores}).to_csv(REPORT_DIR / "val_normal_scores.csv", index= False)
    save_json(dict(threshold = thr, policy = th["policy"], k_std = th["k_std"], quantile = th["quantile"],
                   n_val = int(len(scores)), val_mean = float(scores.mean()),
                   val_std = float(scores.std()), val_max = float(scores.max()),
                   status = "provisional (calibrated on 22 normal images)"),
                   MODEL_DIR / THRESHOLD_FILE)
    logging.debug("Threshold saved to %s", MODEL_DIR / THRESHOLD_FILE)


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        pre, th = params["preprocessing"], params["threshold"]
        seed_everything(params["data_splitting"]["seed"])
        device = get_device()
        logging.info("Device: %s", device)

        splits = load_splits(PROCESSED_DIR / SPLITS_FILE)
        loader = build_val_loader(splits, pre["image_size"], pre["batch_size"])

        model = PatchCore.from_params(params, device=device)
        scores = model.score_loader(loader)["scores"].numpy()

        thr = compute_threshold(scores, th["policy"], th["k_std"], th["quantile"])
        save_data(scores, thr, th)

        logging.info("Val-normal scores: mean %.3f | std %.3f | max %.4f",
                     scores.mean(), scores.std(), scores.max())
        logging.info("Threshold (%s) = %.4f", th["policy"], thr)
    except Exception as e:
        logging.error("Failed to calibrate the threshold: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()