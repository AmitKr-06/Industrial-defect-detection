"""
Evaluation metrics for anomaly detection.

Image level : image_metrics, threshold_metrics   (is the image defective?)
Pixel level : pixel_auroc, aupro, mean_dice      (where is the defect?)

All functions take plain numpy arrays. Higher score always means "more anomalous".
"""

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score, average_precision_score

# NumPy 1.x calls it trapz, NumPy 2.x calls it trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def image_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    What it does : measures how well image scores separate defective from normal images.
    What we do   : AUROC = chance that a random defective image scores higher than a random
                   normal one (1.0 = perfect ranking). AP = average precision, more
                   informative when classes are imbalanced (93 defective vs 22 normal).
                   Neither needs a threshold.
    """
    return dict(image_auroc=float(roc_auc_score(labels, scores)),
                image_ap=float(average_precision_score(labels, scores)))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """
    What it does : pixel-level AUROC over every pixel of every image.
    What we do   : flatten all ground-truth masks and anomaly maps, then compute one AUROC.
                   Large defects dominate this number, which is why we also report AUPRO.
    Args         : masks (N, H, W) bool, maps (N, H, W) float.
    """
    return float(roc_auc_score(masks.reshape(-1).astype(np.uint8), maps.reshape(-1)))


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, n_points: int = 1000) -> float:
    """
    What it does : Area Under the Per-Region-Overlap curve (MVTec AD standard), for false
                   positive rates up to fpr_limit, divided by fpr_limit (so 1.0 is perfect).
    What we do   :
      1. label every connected defect region (each region counts equally, small ones too)
      2. choose n_points thresholds so the false-positive rate on NORMAL pixels goes from
         0 to fpr_limit (thresholds come from the sorted normal-pixel scores)
      3. at each threshold: PRO = mean over regions of (fraction of the region above it)
      4. integrate PRO over FPR
    This is the dense implementation that agreed with the original to 0.0002 in Exp 8.
    Args         : masks (N, H, W) bool, maps (N, H, W) float.
    """
    masks = masks.astype(bool)
    maps = maps.astype(np.float64)

    normal = np.sort(maps[~masks])              # scores of all normal pixels, ascending
    n_norm = normal.size

    # Collect every defect region: its pixel scores and its area
    reg_ids, vals, areas, n_reg = [], [], [], 0
    for i in range(masks.shape[0]):
        lbl, k = ndimage.label(masks[i])
        for j in range(1, k + 1):
            sel = lbl == j
            reg_ids.append(np.full(int(sel.sum()), n_reg, dtype=np.int64))
            vals.append(maps[i][sel])
            areas.append(float(sel.sum()))
            n_reg += 1
    if n_norm == 0 or n_reg == 0:
        return 0.0

    reg_ids, vals, areas = np.concatenate(reg_ids), np.concatenate(vals), np.asarray(areas)

    # Thresholds that give FPR = 0 ... fpr_limit on the normal pixels
    targets = np.linspace(0.0, fpr_limit, n_points)
    pos = np.minimum(np.floor((1.0 - targets) * n_norm).astype(np.int64), n_norm - 1)
    thr = normal[pos]
    fprs = (n_norm - np.searchsorted(normal, thr, side="right")) / n_norm

    pros = np.empty(n_points)
    for a, t in enumerate(thr):
        hits = np.bincount(reg_ids[vals > t], minlength=n_reg)   # defect pixels above threshold
        pros[a] = (hits / areas).mean()

    order = np.argsort(fprs, kind="stable")
    fprs, pros = fprs[order], pros[order]

    # If many pixels tie, the curve can stop before fpr_limit. PRO can only grow with FPR,
    # so extend it flat to fpr_limit (keeps a perfect map at exactly 1.0).
    if fprs[-1] < fpr_limit:
        fprs = np.append(fprs, fpr_limit)
        pros = np.append(pros, pros[-1])
    return float(_trapz(pros, fprs) / fpr_limit)


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    """
    What it does : turns scores into decisions at one threshold and counts the outcomes.
    What we do   : image is "defective" if score > thr. Returns the confusion counts
                   (tp, fp, fn, tn) plus precision, recall, F1 and false-positive rate.
                   Every division is guarded so an empty class gives 0.0, not an error.
    """
    pred = scores > thr
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return dict(threshold=float(thr), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, fpr=fpr)


def mean_dice(masks: np.ndarray, maps: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """
    What it does : average segmentation overlap (Dice) between the predicted mask and the
                   ground-truth mask, over the DEFECTIVE images only.
    What we do   : predicted mask = anomaly map > thr. Dice = 2*overlap / (gt area + pred area).
                   Reported at the image threshold (a pixel threshold would be tuned separately).
    """
    pred_all, dices = maps > thr, []
    for i in np.where(labels == 1)[0]:
        gt, pr = masks[i].astype(bool), pred_all[i]
        s = gt.sum() + pr.sum()
        dices.append(2 * (gt & pr).sum() / s if s > 0 else 0.0)
    return float(np.mean(dices)) if dices else 0.0