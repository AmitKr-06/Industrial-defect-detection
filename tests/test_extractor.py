# tests/test_extractor.py
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR
from src.constants import SPLITS_FILE
from src.data.dataset import MVTecDataset, load_splits
from src.features.feature_extractor import (
    FeatureExtractor,
    patch_embeddings,
    extract_embeddings,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load_splits takes ONLY the CSV path.
    splits = load_splits(PROCESSED_DIR / SPLITS_FILE)
    fit = splits[splits.role == "fit"]
    print(f"Fit images: {len(fit)}")

    extractor = FeatureExtractor().to(device)
    n_params    = sum(p.numel() for p in extractor.parameters())
    n_trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.1f} M | trainable: {n_trainable}")
    print(f"Training mode: {extractor.training}   (must be False)")

    ds = MVTecDataset(fit.head(8), image_size=256)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    x = batch["image"].to(device)
    print(f"\nInput shape: {tuple(x.shape)}")

    with torch.inference_mode():
        f2, f3 = extractor(x)
    print(f"layer2 map : {tuple(f2.shape)}   (expect (8, 512, 32, 32))")
    print(f"layer3 map : {tuple(f3.shape)}   (expect (8, 1024, 16, 16))")

    # Baseline embeddings (out_dim=1024)
    with torch.inference_mode():
        emb_base, grid = patch_embeddings(
            (f2, f3), ("layer2", "layer3"), pool_kernel=3, out_dim=1024
        )
    print(f"\nBaseline embeddings: {tuple(emb_base.shape)}")
    print(f"Grid               : {grid}")
    assert emb_base.shape == (8, 32*32, 1024), f"Wrong shape: {emb_base.shape}"
    assert grid == (32, 32)
    assert torch.isfinite(emb_base).all()

    # Concatenated embeddings (out_dim=None)
    with torch.inference_mode():
        emb_cat, grid_c = patch_embeddings(
            (f2, f3), ("layer2", "layer3"), pool_kernel=3, out_dim=None
        )
    print(f"Concat embeddings  : {tuple(emb_cat.shape)}")
    assert emb_cat.shape == (8, 32*32, 1536), f"Wrong shape: {emb_cat.shape}"
    assert grid_c == (32, 32)

    # layer2-only
    with torch.inference_mode():
        emb_l2, _ = patch_embeddings(
            (f2, f3), ("layer2",), pool_kernel=3, out_dim=None
        )
    print(f"layer2-only emb    : {tuple(emb_l2.shape)}   (expect (8, 1024, 512))")
    assert emb_l2.shape == (8, 32*32, 512)

    # Full extraction via helper
    print("\nRunning extract_embeddings on the batch...")
    out, hw = extract_embeddings(
        extractor, loader, device,
        layers=("layer2", "layer3"),
        pool_kernel=3, out_dim=1024,
    )
    print(f"extract_embeddings output: {tuple(out.shape)} | grid {hw}")
    assert out.shape == (8, 32*32, 1024)
    assert hw == (32, 32)

    print("\n\u2713 All checks passed.")


if __name__ == "__main__":
    main()