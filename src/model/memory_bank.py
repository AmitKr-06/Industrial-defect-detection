"""
Pipeline stage: memory bank (coreset selection).

Takes all patch embeddings of the 198 fit images (202, 752 patches) and keeps only the
most representative 10% with a greedy k-center coreset. That subset is the PatchCore
"memory bank": at test time, a patch is anomalous if it is far from every bank patch.

Input : data/interim/fit_embeddings.pt
Output: models/memory_bank.pt       (bank tensor + indices + settings)
        reports/metrics/memory_bank_meta.json
"""
import sys
import time
import torch

from src.logger import logging
from src.exception import CustomException
from src.config import PARAMS_PATH, INTERIM_DIR, MODEL_DIR, REPORT_DIR, make_dirs
from src.constants import FIT_EMBEDDINGS_FILE, MEMORY_BANK_FILE
from src.utils import load_params, save_json, seed_everything, get_device, sync
from src.model.coreset import greedy_coreset


def build_bank(emb: torch.Tensor, ratio: float, proj_dim: int, seed: int, device):
    """
    What it does : selects the memory bank from all fit patches.
    What we do   :
       1. flatten (images, patches, D) into one big list of patches (n, D)
       2. n_select = ratio * N (10% of 202, 752 = 20,275 patches)
       3. greedy k-center: start from a random patch, then repeatedly add the patch that is 
          farthest from everything already chosen. Distances are computed on a random
          128-d projection to keep it fast. This covers the variety of normal patches with
          far fewer patches than the full set.
    Returns      : bank (n_select, D) on CPU, indices (n_select,), total patch count N.
    """
    flat = emb.reshape(-1, emb.shape[-1])
    n_select = int(ratio * flat.shape[0])
    logging.info("Selecting %d of %d patches (ratio %.2f)", n_select, flat.shape[0], ratio)

    flat_gpu = flat.to(device)        # needed on GPU for fast distance updates
    idx = greedy_coreset(flat_gpu, n_select, proj_dim, seed, device).cpu()
    del flat_gpu
    if device.type == "cuda":                   
        torch.cuda.empty_cache()                  # free ~830 MB before the next stage

    assert len(torch.unique(idx)) == n_select, "Duplicate coreset indices"
    return flat[idx].clone(), idx, flat.shape[0]

def validate_bank(bank: torch.Tensor, idx: torch.Tensor, n_total: int, emb: torch.Tensor) -> None:
    """
    What it does : sanity-checks the bank before it is saved.
    What we do   : confirms the embedding diemnsion is unchanged, the indices are inside the
                   valid range, and no value is NaN/Inf, so a broken never reaches the 
                   scoring stages.
    """
    assert bank.shape[1] == emb.shape[-1], "Bank dimension differs from embedding dimension"
    assert bank.shape[0] == len(idx), "Bank size and index count differ"
    assert int(idx.min()) >= 0 and int(idx.max()) < n_total, "Index out of range"
    assert torch.isfinite(bank).all(), "Bank contains NaN or Inf"


def compare_with_notebook_indices(idx: torch.Tensor, category: str) -> None:
    """
    What it does : optional reproducibility check against the Exp 4 notbook reuslt.
    What we do   : if the old file models/<category>_coreset_indices_baseline1024.pt exists,
                   compare our first n indices with it. With the baseline settings they
                   should be 100% identical (the fast greedy was verified in Exp 8). This
                   only logs, it never stops the pipline, because different params
                   (e.g. layer2-only) legitimately give different indices.
    """
    ref_path = MODEL_DIR / f"{category}_coreset_indices_baseline1024.pt"
    if not ref_path.exists():
        logging.info("No notebook reference file found, skipping index comparison")
        return
    ref = torch.load(ref_path, map_location="cpu")["indices"][:len(idx)]
    same = (idx == ref).float().mean().item() * 100
    logging.info("Indices identical to the notebook result: %.2f %%", same)
    if same < 99.9:
        logging.warning("Indices differ from Exp 4. Expected only if params.yaml was changed.")


def save_data(bank: torch.Tensor, idx: torch.Tensor, n_total: int, seconds: float, mb:dict) -> None:
    """
    What it does : saves the bank and a small metadata file.
    What we do   : models/memory_bank.pt holds the bank tensor itself (83 MB), so later
                   stages and the inference code never need the 830 MB embeddings file.
                   The JSON records size and selection time for the report.
    """
    torch.save(dict(bank=bank, indices = idx, n_toatl_patches =int(n_total),
                    ratio = mb["ratio"], seed=mb["seed"], proj_dim = mb["proj_dim"],
                    embed_dim = int(bank.shape[1])), MODEL_DIR / MEMORY_BANK_FILE)
    save_json(dict(bank_patches=int(bank.shape[0]), n_total_patches=int(n_total),
                   bank_mb=round(bank.numel() * 4 / 1e6, 1), greedy_seconds=round(seconds, 1),
                   ratio=mb["ratio"], proj_dim=mb["proj_dim"], seed=mb["seed"]),
                REPORT_DIR / "memory_bank_meta.json")
    logging.debug("Memory bank saved to %s", MODEL_DIR / MEMORY_BANK_FILE)


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        mb = params["memory_bank"]
        seed_everything(mb["seed"])
        device = get_device()
        logging.info("Device: %s", device)

        emb = torch.load(INTERIM_DIR / FIT_EMBEDDINGS_FILE, map_location="cpu")
        logging.info("Loaded fit embeddings %s", tuple(emb.shape))

        sync(device)
        t0 = time.time()
        bank, idx, n_total = build_bank(emb, mb["ratio"], mb["proj_dim"], mb["seed"], device)
        sync(device)
        seconds = time.time() - t0

        validate_bank(bank, idx, n_total, emb)
        compare_with_notebook_indices(idx, params["data_ingestion"]["category"])
        save_data(bank, idx, n_total, seconds, mb)

        logging.info("Memory bank %s built in %.1fs", tuple(bank.shape), seconds)
    except Exception as e:
        logging.error("Failed to build the memory bank: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()