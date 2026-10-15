"""
Small MLflow helpers shared by every file that talks to MLflow, so the tracking
settings live in ONE place (params.yaml -> mlflow section).
"""

import re
from pathlib import Path

import mlflow

from src.config import PROJECT_ROOT
from src.logger import logging


_ILLEGAL = re.compile(r"[^0-9A-Za-z_\-\. /]")


def _resolve_tracking_uri(uri: str) -> str:
    """
    Resolve a relative sqlite tracking URI to an absolute path so the database
    always lives under the project root, no matter where the script is launched.

    Example:
        "sqlite:///mlflow.db"  ->  "sqlite:////abs/path/to/project/mlflow.db"
    """
    if not uri.startswith("sqlite:///"):
        return uri

    # Strip the scheme. What remains is a relative or absolute file path.
    raw = uri[len("sqlite:///"):]

    # Already absolute? Leave it alone.
    if Path(raw).is_absolute():
        return uri

    # Otherwise, anchor to the project root.
    abs_path = (PROJECT_ROOT / raw).resolve()
    # SQLite URIs need four slashes when the path is absolute:
    #   sqlite:////home/user/project/mlflow.db
    return f"sqlite:///{abs_path.as_posix()}"


def setup_mlflow(cfg: dict) -> None:
    """
    Point MLflow at our local database and select the experiment.

    The tracking URI from params.yaml is resolved to an absolute path so the
    SQLite database always lives at <PROJECT_ROOT>/mlflow.db, regardless of the
    current working directory when this function is called.
    """
    uri = _resolve_tracking_uri(cfg["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["experiment_name"])
    logging.info("MLflow: %s | experiment %s", uri, cfg["experiment_name"])


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