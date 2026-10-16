"""
Inference pipeline: score ONE image with the trained PatchCore model.

Loads once (backbone + memory bank + threshold), then predict() can be called
many times. Used by the FastAPI app, the command line, and the tests.

Needs : models/memory_bank.pt, models/threshold.json  (produced by `dvc repro`)
"""

import sys
import json
import time
import numpy as np
import torch
import matplotlib
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from src.logger import logging
from src.exception import CustomException, InvalidInputError
from src.config import PARAMS_PATH, MODEL_DIR
from src.constants import IMAGENET_MEAN, IMAGENET_STD, THRESHOLD_FILE
from src.utils import load_params, get_device, sync
from src.model.patchcore import PatchCore

MIN_SIDE = 32            # images smaller than this cannot be analysed meaningfully
MAX_PIXELS = 50_000_000  # protects the server from decompression-bomb images


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


class DefectPredictor:
    """
    Answers "is this image defective, and where?" for a single image.

    Builds the PatchCore model from params.yaml, reads the calibrated
    threshold, and applies the SAME preprocessing as training
    (resize to 256x256 bilinear, ImageNet normalisation). Any difference
    here would silently change the scores.
    """

    def __init__(self, params_path=PARAMS_PATH, device=None):
        self.params = load_params(params_path)
        self.device = device or get_device()
        size = self.params["preprocessing"]["image_size"]

        self.model = PatchCore.from_params(self.params, device=self.device)
        with open(MODEL_DIR / THRESHOLD_FILE) as f:
            self.threshold = float(json.load(f)["threshold"])

        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        logging.info("DefectPredictor ready | device=%s | threshold=%.4f",
                     self.device, self.threshold)

    @staticmethod
    def load_image(image) -> Image.Image:
        """
        Open and validate the uploaded image.

        Accepts a file path or a file-like object. Anything that is not a
        readable image, is too small, or is absurdly large raises
        InvalidInputError with a message that is safe to show to the user.
        """
        try:
            img = Image.open(image)
            img.load()  # forces a full read, so truncated files fail here
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            raise InvalidInputError(
                "The uploaded file is not a readable image. "
                "Please use a PNG or JPG file."
            )
        if min(img.size) < MIN_SIDE:
            raise InvalidInputError(
                f"The image is too small (minimum {MIN_SIDE} pixels per side)."
            )
        if img.size[0] * img.size[1] > MAX_PIXELS:
            raise InvalidInputError("The image is too large to analyse.")
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
        (map > threshold, bool), overlay (PIL image), and latency_ms.
        """
        try:
            img = self.load_image(image)
            x = self.transform(img).unsqueeze(0).to(self.device)

            sync(self.device)
            t0 = time.perf_counter()
            maps, scores = self.model.predict(x)
            sync(self.device)
            latency_ms = 1000 * (time.perf_counter() - t0)

            score = float(scores[0])
            amap = maps[0].cpu().numpy()
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
                mask=amap > self.threshold,
                overlay=make_overlay(img, amap, self.threshold),
                latency_ms=latency_ms,
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
    result = DefectPredictor().predict(sys.argv[1])
    print(f"{result['label']} | score {result['score']:.4f} "
          f"| threshold {result['threshold']:.4f} "
          f"| {result['latency_ms']:.1f} ms")