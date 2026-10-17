"""
Inference pipeline: score ONE image with the trained PatchCore model.

Loads once (backbone + memory bank + threshold), then predict() can be called
many times. Used by the FastAPI app, the command line, and the tests.

Needs : models/memory_bank.pt, models/threshold.json  (produced by `dvc repro`)
        or explicit bank_path / threshold_path (used by the registry mode)
"""

import sys
import json
import time
import threading
import numpy as np
import torch
import matplotlib
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

from src.logger import logging
from src.exception import CustomException, InvalidInputError
from src.config import PARAMS_PATH, MODEL_DIR
from src.constants import IMAGENET_MEAN, IMAGENET_STD, THRESHOLD_FILE
from src.utils import load_params, get_device, sync
from src.model.patchcore import PatchCore

MIN_SIDE = 32             # images smaller than this cannot be analysed meaningfully
MAX_PIXELS = 50_000_000   # protects the server from decompression-bomb images
MIN_REGION_PIXELS = 20    # flagged regions smaller than this (on the 256x256 map) count as noise


def make_overlay(image: Image.Image, amap: np.ndarray, thr: float,
                 alpha: float = 0.5) -> Image.Image:
    """
    Draw the anomaly heatmap on top of the original image.

    1. resize the original to the map size (256x256)
    2. scale the map to 0..1 using a range tied to the threshold
       (thr*0.75 ... the larger of map max and thr*1.25), so blue = normal,
       red = clearly above threshold, and the colours mean the same thing
       for every image
    3. colour it with "jet" and blend with the original (alpha = heatmap strength)

    Returns a PIL RGB image, ready to be shown or saved as PNG.
    """
    base = image.convert("RGB").resize(amap.shape[::-1])
    lo, hi = thr * 0.75, max(float(amap.max()), thr * 1.25)
    norm = np.clip((amap - lo) / (hi - lo), 0.0, 1.0)
    heat = (matplotlib.colormaps["jet"](norm)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat
    return Image.fromarray(blended.astype(np.uint8))


def mask_statistics(amap: np.ndarray, mask: np.ndarray) -> dict:
    """
    Numbers that describe WHERE and HOW MUCH the model flagged.

    area_pct     : percentage of the image above the threshold
    n_regions    : number of separate flagged regions (tiny noise regions are ignored)
    max_location : (row, col) of the most anomalous pixel on the 256x256 map
    """
    labels, n = ndimage.label(mask)
    if n:
        sizes = np.asarray(ndimage.sum(mask, labels, index=range(1, n + 1)))
        n_regions = int((sizes >= MIN_REGION_PIXELS).sum())
    else:
        n_regions = 0
    row, col = np.unravel_index(int(amap.argmax()), amap.shape)
    return dict(area_pct=float(mask.mean() * 100), n_regions=n_regions,
                max_location=(int(row), int(col)))


class DefectPredictor:
    """
    Answers "is this image defective, and where?" for a single image.

    Builds the PatchCore model from params.yaml, reads the calibrated
    threshold, and applies the SAME preprocessing as training
    (resize to 256x256 bilinear, ImageNet normalisation). Any difference
    here would silently change the scores.

    bank_path / threshold_path default to the files in models/. The registry
    mode passes the files downloaded from the MLflow model version instead.
    """

    def __init__(self, params_path=PARAMS_PATH, bank_path=None, threshold_path=None, device=None):
        self.params = load_params(params_path)
        self.device = device or get_device()
        self.size = self.params["preprocessing"]["image_size"]

        self.model = PatchCore.from_params(self.params, bank_path=bank_path, device=self.device)
        with open(threshold_path or (MODEL_DIR / THRESHOLD_FILE)) as f:
            self.threshold = float(json.load(f)["threshold"])

        self.transform = T.Compose([
            T.Resize((self.size, self.size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.source = "local"
        self.model_version = None          # set by RegistryPredictor
        self._lock = threading.Lock()      # one prediction at a time (FastAPI uses a thread pool)
        logging.info("DefectPredictor ready | device=%s | threshold=%.4f",
                     self.device, self.threshold)

    def warm_up(self, n: int = 3) -> None:
        """
        Run a few dummy predictions so the first real request is fast.

        The first call on a GPU initialises CUDA kernels (hundreds of ms);
        on CPU it initialises the thread pool. Doing it at startup means the
        first user does not pay for it.
        """
        x = torch.zeros(1, 3, self.size, self.size, device=self.device)
        with self._lock:
            for _ in range(n):
                self.model.predict(x)
            sync(self.device)
        logging.info("Warm-up done (%d passes)", n)

    @staticmethod
    def load_image(image) -> Image.Image:
        """
        Open and validate the uploaded image.

        Accepts a file path or a file-like object. Anything that is not a
        readable image, is too small, or is absurdly large raises
        InvalidInputError with a message that is safe to show to the user.

        The size is checked BEFORE the pixels are decoded, so a huge image is
        rejected without using memory.
        """
        try:
            img = Image.open(image)
            width, height = img.size          # reads the header only
        except Image.DecompressionBombError:
            raise InvalidInputError("The image is too large to analyse.")
        except (UnidentifiedImageError, OSError, ValueError):
            raise InvalidInputError(
                "The uploaded file is not a readable image. "
                "Please use a PNG or JPG file."
            )

        if min(width, height) < MIN_SIDE:
            raise InvalidInputError(
                f"The image is too small (minimum {MIN_SIDE} pixels per side)."
            )
        if width * height > MAX_PIXELS:
            raise InvalidInputError("The image is too large to analyse.")

        try:
            img.load()                        # full decode, truncated files fail here
        except (OSError, ValueError, Image.DecompressionBombError):
            raise InvalidInputError(
                "The uploaded file is not a readable image. "
                "Please use a PNG or JPG file."
            )
        return img.convert("RGB")

    @torch.inference_mode()
    def predict(self, image) -> dict:
        """
        Score one image.

        Preprocess -> PatchCore.predict -> compare the score with the
        threshold. The decision rule is the same as in evaluation:
        defective if score > threshold.

        Returns a dict with score, threshold, ratio (score/threshold),
        is_defective, label, anomaly_map (256x256 array), mask
        (map > threshold, bool), overlay (PIL image), latency_ms, and the
        mask statistics (area_pct, n_regions, max_location).
        """
        try:
            img = self.load_image(image)
            x = self.transform(img).unsqueeze(0).to(self.device)

            with self._lock:
                sync(self.device)
                t0 = time.perf_counter()
                maps, scores = self.model.predict(x)
                sync(self.device)
                latency_ms = 1000 * (time.perf_counter() - t0)

            score = float(scores[0])
            amap = maps[0].cpu().numpy()
            mask = amap > self.threshold
            defective = score > self.threshold

            logging.info(
                "Prediction: score=%.4f threshold=%.4f defective=%s (%.1f ms)",
                score, self.threshold, defective, latency_ms,
            )
            return dict(
                score=score,
                threshold=self.threshold,
                ratio=score / self.threshold,
                is_defective=bool(defective),
                label="DEFECTIVE" if defective else "NORMAL",
                anomaly_map=amap,
                mask=mask,
                overlay=make_overlay(img, amap, self.threshold),
                latency_ms=latency_ms,
                **mask_statistics(amap, mask),
            )
        except InvalidInputError:
            raise  # a user mistake, not a system error
        except Exception as e:
            logging.error("Prediction failed: %s", e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Command line check:
    #   python -m src.pipeline.predict_pipeline path/to/image.png
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.pipeline.predict_pipeline <image_path>")
    try:
        result = DefectPredictor().predict(sys.argv[1])
    except InvalidInputError as e:
        sys.exit(f"Invalid input: {e}")
    print(f"{result['label']} | score {result['score']:.4f} "
          f"| threshold {result['threshold']:.4f} "
          f"| flagged {result['area_pct']:.2f}% in {result['n_regions']} region(s) "
          f"| {result['latency_ms']:.1f} ms")