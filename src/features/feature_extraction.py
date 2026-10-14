"""
Pipeline stage: feature extraction.

Reads the fit images listed in splits.csv, runs them through the frozen
WideResNet50-2 and saves one embedding per image patch.

Input  : data/processed/splits.csv
Output : data/interim/fit_embeddings.pt      (N_fit, patches, D)
         reports/metrics/feaure_extraction_meta.json
"""

import sys
import time
import torch
from torch.utils.data import DataLoader

from src.logger import logging
from src.exception import CustomException
from src.config import PARAMS_PATH, PROCESSED_DIR, INTERIM_DIR, REPORT_DIR, make_dirs
from src.constants import ROLE_FIT, SPLITS_FILE, FIT_EMBEDDINGS_FILE
from src.utils import load_params, save_json, seed_everything, get_device, sync
from src.data.dataset import MVTecDataset, load_splits
from src.features.feature_extractor import FeatureExtractor, extract_embeddings


def build_fit_loader(splits, image_size: int, batch_size: int) -> DataLoader:
    """
    What it does : builds the DataLoader over the FIT images only.
    What we do   : keeps only rows with role == "fit" (198 normal images). The val_normal
                   images (threshold) and test images are never used here, so the memory
                   bank cannot leak information from them. shuffle = False keeps the order
                   reproducible; num_workers = 0 is the safe setting on Windows + Jupyter.
    """
    fit_df = splits[splits.role == ROLE_FIT]
    logging.info("Fit images: %d", len(fit_df))
    return DataLoader(MVTecDataset(fit_df, image_size), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def run_extraction(loader: DataLoader, device, fe: dict):
    """
    What it does : runs every fit image through the frozen backbone and times it.
    What we do   : build the extractor, reset the GPU peak-memory counter, then call
                   extract_embeddings with the layers / pool_kernel / embed_dim from
                   params.yaml. sync() before and after makes the timing accurate on GPU.
    
    Returns      : embeddings (N, patches, D) on CPU, grid (H, W), seconds, peak GPU MB.
    """
    extractor = FeatureExtractor().to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    t0 = time.time()
    emb, hw = extract_embeddings(extractor, loader, device,
                                 layers=fe["layers"],
                                 pool_kernel=fe["pool_kernel"],
                                 out_dim=fe["embed_dim"])
    sync(device)
    seconds = time.time() - t0

    peak_mb = (round(torch.cuda.max_memory_allocated() / 1e6, 1)
               if device.type == "cuda" else None)
    return emb, hw, seconds, peak_mb

def validate_embeddings(emb: torch.Tensor, hw, n_images: int, fe: dict) -> None:
    """
    What it does : checks the result before it is saved.
    What we do   : stop the pipeline early (instead of failing 15 minutes later in the
                   coreset stage) if the number of images, the patch count, the embedding
                   dimension or the values are wrong.
    """
    n, p, d = emb.shape
    assert n == n_images, f"Expected {n_images} images, got {n}"
    assert p == hw[0] * hw[1], f"Patch count {p} does not match grid {hw}"
    if fe["embed_dim"] is not None:
        assert d == fe["embed_dim"], f"Expected dim {fe['embed_dim']}, got {d}"
    assert torch.isfinite(emb).all(), "Embeddings contain NaN or Inf"

def save_data(emb: torch.Tensor, hw, seconds: float, peak_mb, fe: dict) -> None:
    """
    What it does : saves the embeddings and a small metadata file.
    What we do   : the tensor goes to data/interim (large, tracked by DVC). The JSON in
                   reports/metrics records shape, grid, time and the settings used, so every
                   run is documented.
    """
    torch.save(emb, INTERIM_DIR / FIT_EMBEDDINGS_FILE)
    save_json(dict(shape = list(emb.shape), grid = list(hw), seconds = round(seconds, 2),
                   ms_per_image = round(1000 * seconds / emb.shape[0], 1),
                   peak_gpu_mb = peak_mb, layers = fe["layers"],
                   pool_kernel = fe["pool_kernel"], embed_dim = fe["embed_dim"]),
                   REPORT_DIR / "feature_extraction_meta.json")
    logging.debug("Embeddings saved to %s", INTERIM_DIR / FIT_EMBEDDINGS_FILE)


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        pre, fe = params["preprocessing"], params["feature_extraction"]
        seed_everything(params["data_splitting"]["seed"])
        device = get_device()
        logging.info("Device: %s", device)

        splits = load_splits(PROCESSED_DIR / SPLITS_FILE)
        loader = build_fit_loader(splits, pre["image_size"], pre["batch_size"])

        emb, hw, seconds, peak_mb = run_extraction(loader, device, fe)
        validate_embeddings(emb, hw, len(loader.dataset), fe)
        save_data(emb, hw, seconds, peak_mb, fe)

        logging.info("Fit embeddings %s | grid %s | %.1fs", tuple(emb.shape), hw, seconds)
    except Exception as e:
        logging.error("Failed to complete feature extraction: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()