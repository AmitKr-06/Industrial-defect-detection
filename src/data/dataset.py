import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.config import PROJECT_ROOT
from src.constants import IMAGENET_MEAN, IMAGENET_STD


def load_splits(csv_path) -> pd.DataFrame:
    """Read splits.csv and turn relative paths into absolute ones."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["path"] = df["path"].apply(lambda p: str(PROJECT_ROOT / p))
    df["mask_path"] = df["mask_path"].apply(lambda p: str(PROJECT_ROOT / p) if p else "")
    return df

class MVTecDataset(Dataset):
    """Custom PyTorch Dataset for MVTec anomaly detection — wraps a DataFrame of image paths + labels."""
    def __init__(self, records: pd.DataFrame, image_size: int = 256):
        """Store the records table and set up the image transform (resize → tensor → ImageNet normalize)."""
        self.records = records.reset_index(drop=True)
        self.image_size = image_size
        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # Tell PyTorch how many samples are in this dataset (used by DataLoader).
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        """Load one sample: read RGB image, load + binarize mask if it exists 
        (else all-zero mask), and return image, mask, label, defect type, and path as a dict.
        """
        r = self.records.iloc[idx]
        with Image.open(r["path"]) as im:
            image = self.img_tf(im.convert("RGB"))
        if r["mask_path"]:
            with Image.open(r["mask_path"]) as m:
                m = m.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)  # keep binary
            mask = torch.from_numpy((np.array(m) > 0).astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        return {"image": image, "mask": mask, "label": int(r["label"]),
                "defect_type": r["defect_type"], "path": r["path"]}