"""
Predictor wrapper for the API.

Two modes, chosen by app.config.MODEL_SOURCE:
  - local:    DefectPredictor built from models/memory_bank.pt and models/threshold.json
  - registry: RegistryPredictor, the same DefectPredictor but built from the artifacts of
              the MLflow model version behind an alias (for example "champion")

Both return exactly the same result dict, so main.py never has to branch.
"""

from __future__ import annotations

from pathlib import Path

from src.logger import logging
from src.pipeline.predict_pipeline import DefectPredictor

from app import config


def load_predictor():
    """
    Build the predictor for the configured MODEL_SOURCE and warm it up.

    Exposes:
        .threshold, .source ("local" | "registry"), .model_version (int | None)
        .predict(image) -> dict (score, threshold, ratio, is_defective, label,
                           anomaly_map, mask, overlay, latency_ms,
                           area_pct, n_regions, max_location)
    """
    if config.MODEL_SOURCE == "local":
        logging.info("Loading predictor from LOCAL artifacts")
        predictor = DefectPredictor()
    elif config.MODEL_SOURCE == "registry":
        logging.info("Loading predictor from MLflow registry: models:/%s@%s",
                     config.MODEL_NAME, config.MODEL_ALIAS)
        predictor = RegistryPredictor(
            tracking_uri=config.MLFLOW_TRACKING_URI,
            model_name=config.MODEL_NAME,
            alias=config.MODEL_ALIAS,
        )
    else:
        raise ValueError(
            f"Unknown MODEL_SOURCE={config.MODEL_SOURCE!r}. Use 'local' or 'registry'."
        )

    predictor.warm_up()      # first real request is then fast
    return predictor


def _find_artifact(root: Path, name: str) -> Path:
    """
    Find a packaged file inside the downloaded registry model.

    MLflow stores the artifacts of a pyfunc model in an "artifacts" subfolder, but
    searching the whole folder is more robust than relying on that layout.
    """
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found in the downloaded registry model ({root}).")
    return hits[0]


class RegistryPredictor(DefectPredictor):
    """
    DefectPredictor whose memory bank, threshold and params come from the MLflow
    registry version behind the alias, not from the local models/ folder.

    Because the registered version carries its own threshold and settings, the
    container needs nothing from the project folder except the code.
    """

    def __init__(self, tracking_uri: str, model_name: str, alias: str):
        import mlflow
        from mlflow.tracking import MlflowClient

        if not tracking_uri:
            raise ValueError("MODEL_SOURCE=registry but MLFLOW_TRACKING_URI is empty.")
        mlflow.set_tracking_uri(tracking_uri)

        # Download the packaged model of the alias into a temporary folder
        root = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}@{alias}"))

        super().__init__(
            params_path=_find_artifact(root, "params.yaml"),
            bank_path=_find_artifact(root, "memory_bank.pt"),
            threshold_path=_find_artifact(root, "threshold.json"),
        )

        self.source = "registry"
        self.model_version = int(
            MlflowClient().get_model_version_by_alias(model_name, alias).version)
        logging.info("RegistryPredictor ready | %s@%s (version %d) | threshold=%.4f",
                     model_name, alias, self.model_version, self.threshold)