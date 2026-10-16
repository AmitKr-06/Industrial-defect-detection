"""
Pipeline stage: model registry.

Takes the model logged by the evaluation stage and puts it in the MLflow Model Registry.
    1. loads the run info written by the evaluation stage
    2. quality gate: refuses to register a model with metrics below the gate in params.yaml
    3. registers a new model VERSION and gives it the alias "staging"

Works against a local SQLite store or a production MLflow server, chosen by
MLFLOW_MODE (see src/tracking/mlflow_utils.py).

Input  : reports/metrics/experiment_info.json
Output : reports/metrics/registration_info.json
"""
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient

from src.logger import logging
from src.exception import CustomException
from src.config import REPORT_DIR,PARAMS_PATH, make_dirs
from src.constants import EXPERIMENT_INFO_FILE, REGISTRATION_INFO_FILE
from src.utils import load_params, save_json
from src.tracking.mlflow_utils import setup_mlflow


# Metrics written as tags onto every registered model version.
TRACKED_METRICS = ("image_auroc", "pixel_auroc", "aupro", "fp", "fn")

# Metrics that MUST exist in the MLflow run before the gate can be evaluated.
REQUIRED_METRICS = ("image_auroc", "pixel_auroc", "aupro", "fn")

def load_model_info(file_path) -> dict:
    """
    Load the run info written by the evaluation stage.
    
    Reads experiment_info.json, which holds run_id, model_name and model_rui
    (runs:/<run_id>/model). This is how the registry knows exactly WHICH
    evaluated model to register.
    """
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s (run the evaluation stage first)", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error while loading model info: %s", e)
        raise


def check_quality_gate(metrics: dict, gate: dict) -> None:
    """
    Stop a weak model from reaching the registry.
    
    Compares the metrics logged in the MLflow run with the minimum values in
    params.yaml. If any check fails, the stage stops with aan error and nothing
    is registered.
    """
    missing = [m for m in REQUIRED_METRICS if m not in metrics]
    if missing:
        raise ValueError(f"Missing metrics required by the quality gate: {missing}. "
                         f"Available metrics: {sorted(metrics.keys())}")
    checks = [("image_auroc", metrics["image_auroc"] >= gate["image_auroc"],
               f">= {gate['image_auroc']}"),
        ("pixel_auroc", metrics["pixel_auroc"] >= gate["pixel_auroc"],
         f">= {gate['pixel_auroc']}"),
        ("aupro",       metrics["aupro"]       >= gate["aupro"],
         f">= {gate['aupro']}"),
        ("fn",          metrics["fn"]          <= gate["max_fn"],
         f"<= {gate['max_fn']}"),]

    failed = []
    for name, ok, rule in checks:
        logging.info("Gate %-12s %.4f (%s) [%s]", name, metrics[name], rule, "ok" if ok else "FAILED")
        if not ok:
            failed.append(name)
    if failed:
        raise ValueError(f"Quality gate failed for {failed}. Model was NOT registered.")


def register_model(model_info: dict, metrics: dict) -> dict:
    """
    Register the model in the MLflow Model Registry.
    
    1. mlflow.register_model creates the registered model if it does not exist
       yet and adds a new version from the run's model (runs:/<run_id>/model).
    2. Attaches the key metrics, the run id and a description to the VERSION,
       so the registry page shows what each version is worth without opening
       the run.
    3. Sets the alias "staging" on the new version. Aliases replace the 
       deprecated stage-transition API (None / Staging / Production). A later
       step moves "cahmpion" to the version that should serve traffic.
    
    Returns a dict describing what was registered.
    """
    model_name = model_info["model_name"]
    try:
        client = MlflowClient()
        mv = mlflow.register_model(model_info["model_uri"], model_name)
        logging.info("Created model version %s for %s", mv.version, model_name)

        try:
            for key in TRACKED_METRICS:
                client.set_model_version_tag(model_name, mv.version, key, f"{metrics[key]:.4f}")
                client.set_model_version_tag(model_name, mv.version, "run_id", model_info["run_id"])
                client.update_model_version(model_name, mv.version, description=(
                    f"PatchCore (WideResNet50-2 layer2+3). "
                    f"AUPRO {metrics['aupro']:.4f}, "
                    f"image AUROC {metrics['image_auroc']:.4f}, "
                    f"pixel AUROC {metrics['pixel_auroc']:.4f}, "
                    f"FP {int(metrics['fp'])}, FN {int(metrics['fn'])}."
                ),
            )
            client.set_registered_model_alias(model_name, "staging", mv.version)
        except Exception as e:
            logging.error(
                "Partial registration: version %s was created but tagging/"
                "alias failed: %s. You may need to delete this version "
                "manually.", mv.version, e,
            )
            raise
        logging.info('Model %s version %s registered and aliased as "staging".', model_name, mv.version)
        return dict(model_name = model_name, version = int(mv.version), alias = "staging",
                    run_id = model_info["run_id"], model_uri = model_info["model_uri"],
                    metrics = {k: float(metrics[k]) for k in TRACKED_METRICS},)
    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise

def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)

        if "mlflow" not in params:
            raise KeyError("params.yaml is missing the 'mlflow' section. Add it under "
                "'mlflow:' with tracking_uri, experiment_name, "
                "registered_model_name, and registration_gate.")
        cfg = params["mlflow"]
        setup_mlflow(cfg)

        model_info = load_model_info(REPORT_DIR / EXPERIMENT_INFO_FILE)
        metrics = MlflowClient().get_run(model_info["run_id"]).data.metrics
        check_quality_gate(metrics, cfg["registration_gate"])

        result = register_model(model_info, metrics)
        save_json(result, REPORT_DIR / REGISTRATION_INFO_FILE)

    except Exception as e:
        logging.error("Failed to complete the model registraion process: %s", e)
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()