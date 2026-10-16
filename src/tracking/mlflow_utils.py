"""
Small MLflow helpers shared by every file that talks to MLflow, so the tracking
settings live in ONE place (params.yaml -> mlflow section).
"""

import re
import os
import mlflow
from dotenv import load_dotenv

from pathlib import Path

from src.config import PROJECT_ROOT
from src.logger import logging
import warnings

# Silence the noisy MLflow pyfunc UserWarnings (input_example, artifact_path, etc.)
warnings.filterwarnings("ignore", category=UserWarning)

_ILLEGAL = re.compile(r"[^0-9A-Za-z_\-\. /]")

def resolve_tracking_uri(uri: str) -> str:
    """
    Anchor a relative sqlite tracking URI to PROJECT_ROOT so the database
    always lives at <project_root>/mlflow.db, no matter the current working
    directory when the script runs.

    Example: sqlite:///mlflow.db -> sqlite:///C:/.../project/mlflow.db
    """
    if not uri.startswith("sqlite:///"):
        return uri
    raw = uri[len("sqlite:///"):]
    if Path(raw).is_absolute():
        return uri
    return f"sqlite:///{(PROJECT_ROOT / raw).resolve().as_posix()}"

def use_local_tracking(cfg: dict) -> None:
    """
    LOCAL method.
    What it does : sends everything to a SQLite file in the project folder (mlflow.db)
    What we do   : no server and no credentials needed. The model registry needs a database
                   store, which is why this is sqlite and not a plain ./mlruns folder.
                   Use this on local tracking / laptop. Open the UI with
                   'mlflow ui --backend-store-uri sqlite:///mlflow.db'.
    """
    uri = resolve_tracking_uri(cfg["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    logging.info("MLflow mode: LOCAL (%s)", uri)


def use_production_tracking(cfg: dict) -> None:
    """
    PRODUCTION method.
    What it does : sends everything to a remote MLflow tracking server that we host
                   (EC2, a VM, Docker ....).
    What we do   : the server address comes from the environment (.env file locally, real
                   environment variables in CI or on the server), never from the code.
                   Username and password are optional and only needed if the server has
                   basic authentication; MLflow reads them from the environment itself.
    """

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI is not set. Add it to .env or the environment.")
    if not os.getenv("MLFLOW_TRACKING_USERNAME"):
        logging.warning("MLFLOW_TRACKING_USERNAME is not set. Fine only if the server has no authentication.")
    mlflow.set_tracking_uri(uri)
    logging.info("MLflow mode: PRODUCTION (%s)", uri)


def setup_mlflow(cfg: dict) -> None:
    """
    What it does : chooses local or production tracking and selects the experiment.
    What we do   : load .env first, then read the mode from the MLFLOW_MODE environment
                   variable, falling back to mlflow.mode in params.yaml. Because the mode
                   comes from the environment, switching it never changes params.yaml and
                   never triggers a DVC rerun. Every file that talks to MLflow calls this
                   fucntion, so they all use the same server.
    """
    # Load environment variables from .env file
    load_dotenv()
    mode = os.getenv("MLFLOW_MODE", cfg.get("mode", "local")).lower()
    if mode == "production":
        use_production_tracking(cfg)
    elif mode == "local":
        use_local_tracking(cfg)
    else:
        raise ValueError(f"Unknown MLflow mode: {mode} (use 'local or 'production')")
    mlflow.set_experiment(cfg["experiment_name"])


def flatten_params(d: dict, prefix: str = "") -> dict:
    """
    Turn the nested params.yaml dict into flat "section.key" -> value pairs.

    MLflow params are flat strings, so {"scoring": {"sigma": 4.0}} becomes
    {"scoring.sigma": "4.0"}. Keys are joined with "." (MLflow forbids ":"),
    any other illegal character is replaced by "_", and values are cut to 500
    characters (older MLflow versions limit param values to that).
    """
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten_params(v, prefix=f"{key}."))
        else:
            flat[_ILLEGAL.sub("_", key)] = str(v)[:500]
    return flat


def numeric_metrics(d: dict) -> dict:
    """Keep only the numeric entries of metrics.json (MLflow metrics must be numbers)."""
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}