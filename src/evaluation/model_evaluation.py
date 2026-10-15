"""
Pipeline stage: model evaluation.

Scores the 115 test images with the final PatchCore model and the threshold from
the calibration stage, then reports image-level and pixel-level metrics on:
    all test images | dev half | final half | each defect type (+ all normal images)

Input : models/memory_bank.pt, models/threshold.json, data/processed/splits.csv
Output: reports/metrics/metrics.json              (headline numbers, tracked by DVC)
        reports/metrics/metrics_by_subset.csv     (all subsets)
        reports/metrics/test_scores.csv           (per-image scores)
        reports/figures/false_positives.png       (only if there are false positives)
        reports/figures/misses.png
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import mlflow

from src.logger import logging
from src.exception import CustomException
from src.data.dataset import MVTecDataset, load_splits
from src.config import PARAMS_PATH, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR, make_dirs
from src.constants import (
    ROLE_TEST, EVAL_DEV, EVAL_FINAL, NORMAL_CLASS,
    SPLITS_FILE, THRESHOLD_FILE, EXPERIMENT_INFO_FILE,
)
from src.utils import load_params, save_json, seed_everything, get_device
from src.model.patchcore import PatchCore
from src.evaluation.benchmark import benchmark_latency
from src.visualization.visualize import save_cases
from src.evaluation.metrics import (
    image_metrics, pixel_auroc, aupro, threshold_metrics, mean_dice,
)
from src.tracking.mlflow_utils import setup_mlflow, flatten_params, numeric_metrics
from src.tracking.patchcore_pyfunc import log_patchcore_model


def build_test_set(splits: pd.DataFrame, image_size: int, batch_size: int):
    """
    Build the dataset and DataLoader over the 115 test images.

    Keeps rows with role == "test". reset_index keeps row i of the dataframe
    equal to sample i of the dataset, so scores, dev/final labels and
    defect types all line up. shuffle=False keeps the order fixed.
    """
    test_df = splits[splits.role == ROLE_TEST].reset_index(drop=True)
    ds = MVTecDataset(test_df, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    logging.info("Test images: %d", len(test_df))
    return test_df, ds, loader


def evaluate_subset(name: str, idx: np.ndarray, y: np.ndarray, scores: np.ndarray,
                    maps: np.ndarray, masks: np.ndarray, thr: float) -> dict:
    """
    Compute every metric for one subset of the test images.

    Image AUROC/AP, pixel AUROC and AUPRO (threshold-free), plus the
    confusion counts, precision, recall and F1 at the calibrated threshold.
    `idx` selects the images that belong to the subset.
    """
    row = dict(subset=name, n_images=int(len(idx)))
    row.update(image_metrics(y[idx], scores[idx]))
    row["pixel_auroc"] = pixel_auroc(masks[idx], maps[idx])
    row["aupro"] = aupro(masks[idx], maps[idx])
    tm = threshold_metrics(y[idx], scores[idx], thr)
    row.update({k: tm[k] for k in ("fp", "fn", "precision", "recall", "f1")})
    logging.info("%-14s n=%3d | img AUROC %.4f | pix AUROC %.4f | AUPRO %.4f | FP %d FN %d",
                 name, row["n_images"], row["image_auroc"], row["pixel_auroc"],
                 row["aupro"], row["fp"], row["fn"])
    return row


def evaluate_all(test_df: pd.DataFrame, y, scores, maps, masks, thr: float) -> pd.DataFrame:
    """
    Run evaluate_subset for every subset we report.

    "all" (115 images), "dev" and "final" halves (same stratified split as
    Exp 6/7/8), and one subset per defect type that also contains ALL normal
    images, so image-level metrics have both classes. Defect types are read
    from the data, not hard-coded.
    """
    types = test_df["defect_type"].values
    role = test_df["eval_split"].values

    rows = [evaluate_subset("all", np.arange(len(y)), y, scores, maps, masks, thr)]

    for name in (EVAL_DEV, EVAL_FINAL):
        idx = np.where(role == name)[0]
        if len(idx) == 0:
            logging.warning("Subset '%s' has no images; skipping.", name)
            continue
        rows.append(evaluate_subset(name, idx, y, scores, maps, masks, thr))

    for d in sorted(set(types) - {NORMAL_CLASS}):
        idx = np.where((types == d) | (types == NORMAL_CLASS))[0]
        rows.append(evaluate_subset(f"{d}+{NORMAL_CLASS}", idx,
                                    y, scores, maps, masks, thr))

    return pd.DataFrame(rows)


def reproduction_gate(row_all: dict, ref: dict, strict: bool = False) -> str:
    """
    Compare today's headline numbers with the notebook reference.

    Tolerances: image AUROC 0.001, pixel AUROC 0.002, AUPRO 0.005 (as in Exp 8).
    If `strict=True`, raise AssertionError when any gate fails. Otherwise log a
    warning and return "pass"/"changed".
    """
    tols = dict(image_auroc=0.001, pixel_auroc=0.002, aupro=0.005)
    ok = True
    for k, tol in tols.items():
        diff = abs(row_all[k] - ref[k])
        flag = "OK" if diff <= tol else "DIFFERENT"
        ok &= diff <= tol
        logging.info("Gate %-12s now %.4f | reference %.4f | diff %.4f [%s]",
                     k, row_all[k], ref[k], diff, flag)

    if not ok:
        msg = ("Results differ from the notebook reference. "
               "Expected only if params.yaml was changed.")
        if strict:
            raise AssertionError(msg)
        logging.warning(msg)

    return "pass" if ok else "changed"


def save_error_figures(ds, y, scores, maps, masks, types, thr: float) -> None:
    """
    Save heatmap figures for the images the model got wrong.

    False positives = normal images with score > threshold,
    misses           = defective images with score <= threshold.
    Nothing is written when a list is empty (e.g. zero misses).
    """
    fp_idx = list(np.where((y == 0) & (scores > thr))[0])
    fn_idx = list(np.where((y == 1) & (scores <= thr))[0])
    logging.info("False positives: %d | misses: %d", len(fp_idx), len(fn_idx))

    save_cases(fp_idx, ds, masks, maps, scores, types, thr,
               "False positives: normal above threshold",
               FIG_DIR / "false_positives.png")
    save_cases(fn_idx, ds, masks, maps, scores, types, thr,
               "Misses: defective below threshold",
               FIG_DIR / "misses.png")


def save_data(metrics_df: pd.DataFrame, test_df: pd.DataFrame, scores, y, thr: float,
              dice: float, bench: dict, gate: str) -> None:
    """
    Write all result files.

    metrics.json holds the headline numbers (all/dev/final) in a flat form
    that DVC can show and compare between experiments. The CSVs hold every
    subset and the per-image scores for later analysis and reports.
    """
    metrics_df.to_csv(REPORT_DIR / "metrics_by_subset.csv", index=False)
    pd.DataFrame(dict(
        path=test_df["path"].values,
        defect_type=test_df["defect_type"].values,
        label=y,
        score=scores,
    )).to_csv(REPORT_DIR / "test_scores.csv", index=False)

    r = {row["subset"]: row for row in metrics_df.to_dict("records")}
    save_json(dict(
        image_auroc=r["all"]["image_auroc"],
        image_ap=r["all"]["image_ap"],
        pixel_auroc=r["all"]["pixel_auroc"],
        aupro=r["all"]["aupro"],
        aupro_dev=r[EVAL_DEV]["aupro"],
        aupro_final=r[EVAL_FINAL]["aupro"],
        precision=r["all"]["precision"],
        recall=r["all"]["recall"],
        f1=r["all"]["f1"],
        fp=r["all"]["fp"],
        fn=r["all"]["fn"],
        mean_dice=dice,
        threshold=thr,
        reference_gate=gate,
        **bench,
    ), REPORT_DIR / "metrics.json")


def log_to_mlflow(params: dict, metrics_df: pd.DataFrame) -> None:
    """
    Record this evaluation as an MLflow run and store the model with it.

    1. params    : every value from params.yaml (except the mlflow section)
    2. metrics   : the headline numbers from metrics.json, plus AUROC/AUPRO of
                   every subset (e.g. aupro_bent_good)
    3. artifacts : metrics CSVs and error figures (only if they exist)
    4. model     : memory bank + threshold + code, as a pyfunc model
    5. run id    : written to reports/metrics/experiment_info.json, which the
                   registration stage reads to register exactly THIS model
    """
    cfg = params["mlflow"]
    setup_mlflow(cfg)
    headline = json.load(open(REPORT_DIR / "metrics.json"))

    with mlflow.start_run(run_name="patchcore-evaluation") as run:
        mlflow.set_tags(dict(
            stage="evaluation",
            category=params["data_ingestion"]["category"],
            reference_gate=headline.get("reference_gate", "unknown"),
        ))
        mlflow.log_params(flatten_params({k: v for k, v in params.items() if k != "mlflow"}))
        mlflow.log_metrics(numeric_metrics(headline))

        # Per-subset metrics
        for row in metrics_df.to_dict("records"):
            tag = row["subset"].replace("+", "_")
            for m in ("image_auroc", "pixel_auroc", "aupro"):
                mlflow.log_metric(f"{m}_{tag}", float(row[m]))

        # Artifacts
        for f in ("metrics.json", "metrics_by_subset.csv", "test_scores.csv"):
            mlflow.log_artifact(str(REPORT_DIR / f), artifact_path="reports")
        for f in ("false_positives.png", "misses.png"):
            if (FIG_DIR / f).exists():
                mlflow.log_artifact(str(FIG_DIR / f), artifact_path="figures")

        # Model
        log_patchcore_model("model")

        # Run ID for later registration
        save_json(dict(
            run_id=run.info.run_id,
            experiment_name=cfg["experiment_name"],
            model_name=cfg["registered_model_name"],
            model_uri=f"runs:/{run.info.run_id}/model",
        ), REPORT_DIR / EXPERIMENT_INFO_FILE)
        logging.info("MLflow run %s logged", run.info.run_id)


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        pre, ev = params["preprocessing"], params["evaluation"]
        seed_everything(params["data_splitting"]["seed"])
        device = get_device()
        logging.info("Device: %s", device)

        splits = load_splits(PROCESSED_DIR / SPLITS_FILE)
        test_df, ds, loader = build_test_set(splits, pre["image_size"], pre["batch_size"])

        model = PatchCore.from_params(params, device=device)
        thr = json.load(open(MODEL_DIR / THRESHOLD_FILE))["threshold"]
        logging.info("Threshold: %.4f", thr)

        res = model.score_loader(loader)
        assert [str(p) for p in res["paths"]] == [str(p) for p in test_df["path"]], \
            "Score order differs from the test dataframe"

        y, scores = res["labels"].numpy(), res["scores"].numpy()
        maps, masks = res["maps"].numpy(), res["masks"].numpy() > 0.5

        metrics_df = evaluate_all(test_df, y, scores, maps, masks, thr)
        gate = reproduction_gate(metrics_df.iloc[0].to_dict(), ev["reference"])
        dice = mean_dice(masks, maps, y, thr)

        images = torch.stack([ds[i]["image"] for i in range(len(ds))])
        bench = benchmark_latency(lambda x: model.predict(x), images, device,
                                  ev["bench_warmup"], ev["bench_reps"])

        save_data(metrics_df, test_df, scores, y, thr, dice, bench, gate)
        save_error_figures(ds, y, scores, maps, masks, res["defect_types"], thr)
        log_to_mlflow(params, metrics_df)

        logging.info("Evaluation finished. Mean Dice at threshold: %.4f", dice)
    except Exception as e:
        logging.error("Failed to complete model evaluation: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()