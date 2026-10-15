"""
PatchCore model : turns an image into an anomaly map and image-level anomaly score.

How socring works:
    image -> frozen backbone -> patch embeddings -> distance of each patch to it's nearest
    memory-bank patch -> grid of distances -> upsample to image size -> Gaussian smoothing
    -> anomaly map. The image score is the aggregation (max) of that map.
    
This one class is used by threshold calibration, evaluation and the inference pipeline,
so all of them score images in exactly the same way.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.logger import logging
from src.config import MODEL_DIR
from src.constants import MEMORY_BANK_FILE
from src.features.feature_extractor import FeatureExtractor, patch_embeddings
from src.model.coreset import nn_distance

def make_maps(grids: torch.Tensor, size, sigma: float) -> torch.Tensor:
    """
    What it does : converts the coarse patch-distance grid into a full-resolution anomaly map.
    What we do   : 
      1. bilinear upsample the 32 x 32 grid to the image size (256 x 256)
      2. Gaussian blur with the given sigma (4.0 in the reference config) so the map is
         smooth and not blocky. sigma = 0 skips the blur.
         The kernel size is 2*int(4*sigma + 0.5) + 1, so it is always odd and covers
         about 4 sigma on each side.
    Args         : gids (B, 1, h, w), size = (H, W) of the output.
    Returns      : maps (B, H, W).
    """
    maps = F.interpolate(grids, size=size, mode="bilinear", align_corners=False)
    if sigma > 0:
        k = 2 * int(4 * sigma + 0.5) + 1
        maps = TF.gaussian_blur(maps, kernel_size=[k, k], sigma=[sigma, sigma])

    return maps[:, 0]

def image_score(maps: torch.Tensor, agg: str = "max") -> torch.Tensor:
    """
    What it does : reduces each anomaly map to one number, the image anomaly score.
    What we do   : "max" = the single most anomalous pixel (reference setting, best in Exp 5/8)
                   "to1" = mean of the top 1% pixels (less sensitive to one noisy spike)
    Returns      : tensor (B, ), higher = more anomalous.
    """
    flat = maps.flatten(1)
    if agg == "max":
        return flat.max(dim=1).values
    if agg == "top1":
        k = max(1, int(0.01 * flat.shape[1]))
        return flat.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"Unknown aggregation: {agg} (use 'max' or 'top1')")

class PatchCore:
    """
    What it does : holds the frozen backbone + memory bank and scores images.
    What we do   : nothing is trained here. The backbone is pretrained and frozen, the bank
                   was built by the memory_bank stage. cfg carries the scoring settings
                   (layers, pool_kernel, embed_dim, sigma, agg, chunk) that MUST be the same
                   as the ones used to build the bank, otherwise distances are meaningless.
    """
    def __init__(self, extractor: FeatureExtractor, bank: torch.Tensor, cfg: dict, device):
        self.extractor = extractor
        self.cfg = cfg
        self.device = device
        self.bank = bank.to(device)

    @classmethod
    def from_params(cls, params: dict, bank_path = None, device = None):
        """
        What it does : builds a ready-to-use model from params.yaml + the saved bank file.
        What we do   : read the settings, load models/memory_bank.pt, and check that the bank
                       dimension equals embed_dim, so a bank built with other settings is 
                        rejected immediately instead of giving wrong scores.
        """
        fe, sc = params["feature_extraction"], params["scoring"]
        cfg = dict(layers = fe["layers"], pool_kernel=fe["pool_kernel"], embed_dim=fe["embed_dim"],
                   sigma=sc["sigma"], agg=sc["aggregation"], chunk=sc["chunk"])
        bank_path = bank_path or (MODEL_DIR / MEMORY_BANK_FILE)
        ckpt = torch.load(bank_path, map_location="cpu")
        bank = ckpt["bank"]
        if cfg["embed_dim"] is not None and bank.shape[1] != cfg["embed_dim"]:
            raise ValueError(f"Bank dim {bank.shape[1]} != embed_dim {cfg['embed_dim']}. "
                             "Re-run the memory_bank stage.")

        extractor = FeatureExtractor().to(device)
        logging.info("PatchCore loaded: bank %s on %s", tuple(bank.shape), device)
        return cls(extractor, bank, cfg, device)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor):
        """
        What it does : scores a batch of images.
        What we do   : backbone -> patch embeddings -> nearest-neighbour distance to the 
                       bank -> reshape to the patch grid -> make_maps -> image_score.
        Args         : images (B, 3, H, W), already normalised, on self.device.
        Returns      : maps (B, H, W) and scores (B,)
        """
        c = self.cfg
        emb, (h, w) = patch_embeddings(self.extractor(images), c["layers"],
                                       c["pool_kernel"], c["embed_dim"])

        b, p, d = emb.shape
        dist = nn_distance(emb.reshape(b * p, d), self.bank, c["chunk"])
        maps = make_maps(dist.reshape(b, 1, h, w), images.shape[-2:], c["sigma"])
        return maps, image_score(maps, c["agg"])

    @torch.inference_mode()
    def score_loader(self, loader) -> dict:
        """
        What it does : scores a whole DataLoader (val_normal or test set).
        Wht we do    : run predict batch by batch, move results to CPU, and keep labels,
                       ground-truth masks, defect types and paths in the same order so
                       evaluation can compute metrics per image and per defect type.
        Returns      : dict with maps (N, H, W), scores (N,), labels (N,), masks (N, H, W),
                       defect_types (list), paths(list).
        """
        maps, scores, labels, masks, types, paths = [], [], [], [], [], []
        for batch in loader:
            m, s = self.predict(batch["image"].to(self.device, non_blocking = True))
            maps.append(m.cpu())
            scores.append(s.cpu())
            labels.append(batch["label"])
            masks.append(batch["mask"][:, 0])
            types += list(batch["defect_type"])
            paths += list(batch["path"])

        assert scores, "DataLoader produced no batches"
        return dict(maps = torch.cat(maps), scores=torch.cat(scores), labels=torch.cat(labels),
                    masks=torch.cat(masks), defect_types=types, paths=paths)
