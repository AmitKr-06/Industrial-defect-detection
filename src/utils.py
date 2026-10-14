import json
import random
import torch
import yaml
import numpy as np

from src.logger import logging


def load_params(params_path) -> dict:
    """Load parameters from YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML error: %s", e)
        raise

def save_json(obj: dict, path) -> None:
    """Save a dict as a JSON file (pretty format) 
    — used to dump configs, metrics, or results. """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logging.debug("Saved %s", path)

def seed_everything(seed: int) -> None:
    """Fix all random seeds so training gives the same results every run (reproducibility)."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Pick GPU if available, else CPU — so the code runs anywhere."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(device) -> None:
    """Wait for GPU to finish its work before timing or measuring 
    — gives accurate benchmark numbers."""
    if device.type == "cuda":
        torch.cuda.synchronize()

