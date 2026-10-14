import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.config import PARAMS_PATH, PROCESSED_DIR, make_dirs
from src.constants import ROLE_FIT, ROLE_VAL, ROLE_TEST, EVAL_DEV, EVAL_FINAL, RECORDS_FILE, SPLITS_FILE
from src.utils import load_params


def split_records(records: pd.DataFrame, val_fraction: float, seed: int):
    """Split the training set into "fit" (memory bank) and "val_normal" (threshold calibration),
    keeping the test set separate. Called to create three disjoint data views for the pipeline. """
    train_all = records[records.split == "train"]      # grab only training rows
    val_normal = train_all.sample(frac=val_fraction, random_state=seed)     # randomly pick val_fraction of them, seed → same split every run
    fit = train_all.drop(val_normal.index)             # the rest = "fit" (memory bank)
    test = records[records.split == "test"].copy()     # test set kept untouched
    return fit.copy(), val_normal.copy(), test         # 3 clean DataFrames

def assign_dev_final(test_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratified 50/50 dev/final split (same rule as Exp 6/7/8)."""
    labels = test_df["defect_type"].values
    rng = np.random.default_rng(seed)
    role = np.empty(len(test_df), dtype=object)
    for g in np.unique(labels):
        idx = rng.permutation(np.where(labels == g)[0])
        half = len(idx) // 2
        role[idx[:half]] = EVAL_DEV
        role[idx[half:]] = EVAL_FINAL
    test_df["eval_split"] = role
    return test_df

def save_data(fit, val, test) -> None:
    """Tag each split with its "role" (fit / val / test), concatenate into one table,
    and save to disk. `eval_split` is added as an empty column so all rows share the schema."""
    fit["eval_split"] = ""
    val["eval_split"] = ""
    splits = pd.concat([fit.assign(role = ROLE_FIT), val.assign(role = ROLE_VAL), test.assign(role = ROLE_TEST)])
    splits.to_csv(PROCESSED_DIR / SPLITS_FILE, index = False)
    logging.info("Splits saved: fit=%d val=%d test=%d", len(fit), len(val), len(test))


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        p = params["data_splitting"]
        records = pd.read_csv(PROCESSED_DIR / RECORDS_FILE, keep_default_na=False)

        fit, val, test = split_records(records, p["val_fraction"], p["seed"])
        test = assign_dev_final(test, p["seed"])
        assert (len(fit), len(val), len(test)) == (198, 22, 115), "Unexpected split sizes"
        save_data(fit, val, test)
    except Exception as e:
        logging.error("Failed to complete the data splitting process: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()