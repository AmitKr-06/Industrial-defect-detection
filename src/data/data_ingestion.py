import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_splits(csv_path, project_root):
    """Read the saved split file and turn relative paths back into absolute ones."""
    df = pd.read_csv(csv_path)
    root = Path(project_root)
    df["path"] = df["path"].apply(lambda p: str(root / p))
    df["mask_path"] = df["mask_path"].apply(lambda p: str(root / p) if isinstance(p, str) and p else None)
    return df

class MVTechDataset(Dataset):
    """Returns image, mask, label, defect_type, path for one MVTec AD category."""
    def __init__(self, records, image_size = 256):
        self.records = records.reset_index(drop = True)
        self.image_size = image_size
        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation= T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records.iloc[idx]
        image = self.img_tf(Image.open(r["path"]).convert("RGB"))

        mp = r["mask_path"]
        if isinstance(mp, str) and mp != "":
            m = Image.open(mp).convert("L")
            m = m.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask = torch.from_numpy((np.array(m) > 0).astype(np.float32)).unsqueeze(0)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image,
            "mask": mask,
            "label": int(r["label"]),
            "defect_type": r["defect_type"],
            "path": r["path"],
        }
    
        