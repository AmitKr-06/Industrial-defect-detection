"""
Pipeline stage: model promotion.

Decides whether the version aliased "staging" should become the "champion"
that the API serves.
    - no champion yet                             -> staging becomes champion
    - staging AUPRO >= champion AUPRO - tolerance -> staging becomes champion
                                                     (a tie promotes)
    - otherwise                                   -> champion stays, staging
                                                     becomes "candidate"

Input  : MLflow registry (alias "staging"), registration_info.json
Output : reports/metrics/promotion_info.json
"""

import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from src.logger import logging
from src.exception import CustomException
from src.config import PARAMS_PATH, REPORT_DIR, make_dirs
from src.constants import PROMOTION_INFO_FILE
from src.utils import load_params, save_json
from src.tracking.mlflow_utils import setup_mlflow


def get_version_by_alias(client: MlflowClient, name: str, alias: str):
    """
     Find the model version that currently holds an alias.

    Returns the version object, or None when the alias does not exist
    (for example, no champion has ever been promoted).
    """
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def get_aupro(client: MlflowClient, version) -> float:
    """
    Read the AUPRO that belongs to a model version.

    Looks up the run that produced the version and returns its logged AUPRO
    metric. This is the number the promotion rule compares. Returns None if
    `version` is None (no champion yet). Raises ValueError with a clear
    message if the run is missing the metric.
    """
    if version is None:
        return None
    run = client.get_run(version.run_id)
    if "aupro" not in run.data.metrics:
        raise ValueError(f"Run {version.run_id} (version {version.version}) has no "
                         f"'aupro' metric. Cannot evaluate promotion.")
    return float(run.data.metrics["aupro"])


def decide_promotion(staging_aupro: float, champion_aupro, tolerance: float) -> tuple:
    """
    The promotion rule.

    Promote if there is no champion, or if the staging AUPRO is at least
    champion AUPRO minus the tolerance. A tie promotes, because the same code
    and data legitimately produce an equivalent model again.

    Returns (promote: bool, reason: str).
    """
    if champion_aupro is None:
        return True, "no champion exists yet"
    if staging_aupro >= champion_aupro - tolerance:
        return True, (f"staging AUPRO {staging_aupro:.4f} >= champion {champion_aupro:.4f}"
                      f"- tolerance {tolerance}")
    return False, (f"staging AUPRO {staging_aupro:.4f} is below champion {champion_aupro:.4f}"
                   f"- tolerance {tolerance}")


def promote_model(cfg: dict) -> dict:
    """
    Apply the promotion decision in the registry.

    If promoted, the alias "champion" moves to the staging version. Since an
    alias points to exactly one version at a time, the old champion loses the
    alias but is NEVER deleted — you can always move the alias back manually.

    If not promoted, the staging version instead gets the alias "candidate",
    so it remains discoverable for later inspection.

    Returns a dict describing the promotion decision and current state.
    """
    name = cfg["registered_model_name"]
    champion_alias = cfg["alias"]
    tolerance = cfg["champion_tolerance"]

    client = MlflowClient()

    staging = get_version_by_alias(client, name, "staging")
    if staging is None:
        logging.error("No version with alias 'staging' for model %r. "
            "Run 'python -m src.model.model_registry' first.", name)
        raise ValueError(f"No version with alias 'staging' for model '{name}'. "
            f"Run the registry stage first.")

    champion = get_version_by_alias(client, name, champion_alias)

    staging_aupro = get_aupro(client, staging)
    champion_aupro = get_aupro(client, champion)

    promote, reason = decide_promotion(staging_aupro, champion_aupro, tolerance)

    if promote:
        client.set_registered_model_alias(name, champion_alias, staging.version)
        logging.info("Version %s promoted to '%s': %s", staging.version, champion_alias, reason)
    else:
        client.set_registered_model_alias(name, "candidate", staging.version)
        logging.warning("Version %s NOT promoted: %s. Alias: 'candidate'", staging.version, reason)

    return dict(model_name = name, staging_version = int(staging.version),
                previous_champion_version=int(champion.version) if champion else None,
                staging_aupro=staging_aupro, promoted=bool(promote), reason=reason,
                serving_alias=champion_alias if promote else "candidate"
            )


def main():
    try:
        make_dirs()
        params = load_params(PARAMS_PATH)
        cfg = params["mlflow"]
        setup_mlflow(cfg)

        result = promote_model(cfg)
        save_json(result, REPORT_DIR / PROMOTION_INFO_FILE)
        logging.info("Promotion finished: promoted=%s", result["promoted"])
    except Exception as e:
        logging.error("Failed to complete the model promotion process: %s", e)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()