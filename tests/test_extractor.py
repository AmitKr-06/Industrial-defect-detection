import numpy as np
import torch
from torch.utils.data import DataLoader
from src.config import PARAMS_PATH, PROCESSED_DIR, FIG_DIR
from src.constants import SPLITS_FILE, ROLE_TEST
from src.utils import load_params, get_device
from src.data.dataset import MVTecDataset, load_splits
from src.model.patchcore import PatchCore
from src.evaluation.benchmark import benchmark_latency
from src.visualization.visualize import save_cases

params, device = load_params(PARAMS_PATH), get_device()
test_df = load_splits(PROCESSED_DIR / SPLITS_FILE).query("role == @ROLE_TEST").reset_index(drop=True)
ds = MVTecDataset(test_df, 256)
model = PatchCore.from_params(params, device=device)

res = model.score_loader(DataLoader(ds, batch_size=8, shuffle=False))
print("test scores:", tuple(res["scores"].shape), "| max %.3f" % res["scores"].max())

imgs = torch.stack([ds[i]["image"] for i in range(len(ds))])
print(benchmark_latency(lambda x: model.predict(x), imgs, device))

s, y = res["scores"].numpy(), res["labels"].numpy()
fp = list(np.where((y == 0) & (s > 1.1666))[0])
print("false positives:", len(fp))
save_cases(fp, ds, res["masks"].numpy() > 0.5, res["maps"].numpy(), s,
           res["defect_types"], 1.1666, "False positives", FIG_DIR / "check_fp.png")