import sys
import tarfile
import pandas as pd
from pathlib import Path
from PIL import Image

from src.logger import logging
from src.exception import CustomException
from src.config import PROJECT_ROOT, PARAMS_PATH, RAW_DIR, PROCESSED_DIR, REPORT_DIR, make_dirs
from src.constants import IMG_EXT, NORMAL_CLASS, RECORDS_FILE
from src.utils import load_params, save_json


def rel(p: Path) -> str:
    """Convert an absolute path to a short project-relative 
    POSIX path (for clean logs, JSON, CSV).
    """
    return p.relative_to(PROJECT_ROOT).as_posix()

def extract_archive(archive: Path, raw_dir: Path, cat_dir: Path) -> None:
    """Extract the dataset archive if it is not already extracted."""
    if cat_dir.exists():
        logging.info("Already extracted: %s", cat_dir)
        return
    with tarfile.open(archive) as tar:
        tar.extractall(raw_dir, filter="data")
    logging.info("Extracted %s", archive)


def list_images(folder: Path):
    """Recursively find all image files in a folder, sorted (so order is deterministic)."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)

def build_records(cat_dir: Path) -> pd.DataFrame:
    """One row per image: split, defect_type, path, mask_path, label."""
    rows = []
    for p in list_images(cat_dir / "train" / NORMAL_CLASS):
        rows.append(dict(split = "train", defect_type = NORMAL_CLASS, path = rel(p), mask_path = "", label = 0))
    for d in sorted((cat_dir / "test").iterdir()):
        if not d.is_dir():
            continue
        for p in list_images(d):
            if d.name == NORMAL_CLASS:
                rows.append(dict(split = "test", defect_type = NORMAL_CLASS, path = rel(p), mask_path = "", label = 0))
            else:
                mp = cat_dir / "ground_truth" / d.name / f"{p.stem}_mask.png"
                rows.append(dict(split = "test", defect_type = d.name, path = rel(p), mask_path = rel(mp), label = 1))
    return pd.DataFrame(rows)

def audit_records(df: pd.DataFrame, cat_dir: Path) -> dict:
    """Unreadable images, missing masks, size mismatches, orphan masks, label rules."""
    bad, missing, mismatch = [], [], []

    for _, r in df.iterrows():
        img_path = PROJECT_ROOT / r["path"]
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                size = im.size
        except Exception:
            bad.append(r["path"])
            continue

        if r["label"] == 1:
            mp = PROJECT_ROOT / r["mask_path"]
            if not mp.exists():
                missing.append(r["mask_path"])
                continue
            with Image.open(mp) as m:          # compare the MASK size, not the image
                if m.size != size:
                    mismatch.append(r["path"])

    # Boolean-mask + column-name selection requires .loc
    expected = set(df.loc[df.label == 1, "mask_path"])
    orphans = [
        rel(p)
        for p in (cat_dir / "ground_truth").rglob("*_mask.png")
        if rel(p) not in expected
    ]

    # Train must be normal-only
    assert (df[df.split == "train"].label == 0).all(), \
        "Train set contains anomalous images"

    # Label must match defect_type-derived rule
    mask = (df.defect_type != NORMAL_CLASS).astype(int) == df.label
    assert mask.all(), (f"Label rule violated on {int((~mask).sum())} row(s): "
        f"{df.loc[~mask, ['split', 'defect_type', 'label']].head(10).to_dict('records')}")

    audit = dict(
        unreadable=bad,
        missing_masks=missing,
        size_mismatch=mismatch,
        orphan_masks=orphans,
    )
    if any(len(v) for v in audit.values()):
        raise ValueError(f"Dataset audit failed: { {k: len(v) for k, v in audit.items()} }")
    logging.info("Audit passed")
    return audit

def save_data(df: pd.DataFrame, audit: dict, category: str) -> None:
    """Persist the final records DataFrame + audit summary + audit counts to disk (CSV + JSON)."""
    df.to_csv(PROCESSED_DIR / RECORDS_FILE, index=False)
    summary = df.groupby(["split", "defect_type"]).size().rename("n_images").reset_index()
    summary.to_csv(REPORT_DIR / f"{category}_audit_summary.csv", index=False)
    save_json({k: len(v) for k, v in audit.items()}, REPORT_DIR / f"{category}_audit.json")
    logging.debug("Records saved: %d rows", len(df))


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        category = params["data_ingestion"]["category"]
        archive = RAW_DIR / params["data_ingestion"]["archive_name"]
        cat_dir = RAW_DIR / category

        extract_archive(archive, RAW_DIR, cat_dir)
        records = build_records(cat_dir)
        audit = audit_records(records, cat_dir)
        save_data(records, audit, category)

    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()