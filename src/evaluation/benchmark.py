"""
Speed and memory benchmark for a model's predict function.

Protocol (same as Exp 8, so number stay comparable):
    batch size 1, warmp-up passes discarded, median of repeated passes,
    CUDA synchronised before and after every timed pass.
"""

import time
import numpy as np
import torch

from src.logger import logging
from src.utils import sync


@torch.inference_mode()
def benchmark_latency(predict_fn, images: torch.Tensor, device, warmup: int = 10, reps: int = 3) -> dict:
    """
    What it does : measures how long the model needs per image and how much GPU memory it uses.
    What we do   : 
        1. move all test images to the GPU first, so disk reading and CPU - GPU copies
           are not counted (we want the model's own speed)
        2. warm-up: run a few images and throw the timing away (the first calls are slow
           because of CUDA initialisation and kernel selection)
        3. reset the GPU peak-memory counter
        4. run the whole set one image at a time (batch size 1 = real inference), 'reps'
           times, with sync() around each pass so the GPU has really finished
        5. report the MEDIAN of the passes (robust against one slow pass)
    Peak memory includes the backbone, the memory bank and the pre-loaded images
    (~90 MB for 115 images), the same offset for every model.
    Args        : predict_fn(x) takes a (1, 3, H, W) tensor; images (N, 3, H, W) on CPU.
    Returns     : dict with latency_ms (per image), peak_gpu_mb, protocol text.
    """
    imgs = images.to(device)

    for i in range(min(warmup, len(imgs))):
        predict_fn(imgs[i:i + 1])

    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        for i in range(len(imgs)):
            predict_fn(imgs[i:i + 1])
        sync(device)
        times.append(1000 * (time.perf_counter() - t0) / len(imgs))

    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
    result = dict(latency_ms=float(np.median(times)),
                  latency_all_ms=[round(t, 2) for t in times],
                  peak_gpu_mb=float(peak),
                  protocol="bs=1, model only, images pre-loaded to GPU, median of reps")
    logging.info("Latency %.1f ms/image | peak GPU %.0f MB", result["latency_ms"], peak)
    return result
