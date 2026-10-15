from src.utils import load_params
from src.config import PARAMS_PATH
from src.tracking.mlflow_utils import _resolve_tracking_uri

cfg = load_params(PARAMS_PATH)["mlflow"]
print("raw      :", cfg["tracking_uri"])
print("resolved :", _resolve_tracking_uri(cfg["tracking_uri"]))