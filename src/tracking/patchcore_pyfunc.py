"""
MLflow wrapper that makes PatchCore a registrable model.

A registered version contains: the memory bank, the threshold, params.yaml and a copy of
the src/ code. Input: a table with a column "image_path". Output: score, is_defective, threshold.
"""

import json
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from src.config import PROJECT_ROOT, PARAMS_PATH, MODEL_DIR
from src.constants import MEMORY_BANK_FILE, THRESHOLD_FILE

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "image_path")]),
    outputs=Schema([ColSpec("double", "score"), ColSpec("boolean", "is_defective"),
                    ColSpec("double", "threshold")]),
)


class PatchCorePyfunc(mlflow.pyfunc.PythonModel):
    """
    What it does : lets MLflow load and call PatchCore like any other model.
    What we do   : load_context builds PatchCore from the packaged artifacts (same class
                   and preprocessing as training and evaluation). predict scores every
                   image path in the table and applies the stored threshold.
    """

    def load_context(self, context):
        import yaml
        import torchvision.transforms as T
        from src.utils import get_device
        from src.model.patchcore import PatchCore
        from src.constants import IMAGENET_MEAN, IMAGENET_STD

        with open(context.artifacts["params"]) as f:
            params = yaml.safe_load(f)
        with open(context.artifacts["threshold"]) as f:
            self.threshold = float(json.load(f)["threshold"])

        self.device = get_device()
        self.model = PatchCore.from_params(params, bank_path=context.artifacts["memory_bank"],
                                           device=self.device)
        size = params["preprocessing"]["image_size"]
        self.tf = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict(self, context, model_input, params=None):
        from PIL import Image
        rows = []
        for path in model_input["image_path"]:
            with Image.open(path) as im:
                x = self.tf(im.convert("RGB")).unsqueeze(0).to(self.device)
            _, scores = self.model.predict(x)
            score = float(scores[0])
            rows.append(dict(score=score, is_defective=bool(score > self.threshold),
                             threshold=self.threshold))
        return pd.DataFrame(rows)


def log_patchcore_model(artifact_path: str = "model"):
    """
    What it does : saves the current model into the active MLflow run.
    What we do   : packages memory_bank.pt, threshold.json and params.yaml, copies the src/
                   folder (code_paths) and writes the pip requirements, so the model is
                   self-contained and can be registered and loaded elsewhere.
    """
    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=PatchCorePyfunc(),
        artifacts=dict(memory_bank=str(MODEL_DIR / MEMORY_BANK_FILE),
                       threshold=str(MODEL_DIR / THRESHOLD_FILE),
                       params=str(PARAMS_PATH)),
        code_paths=[str(PROJECT_ROOT / "src")],
        signature=SIGNATURE,
        pip_requirements=["torch", "torchvision", "numpy", "pandas", "scipy",
                          "scikit-learn", "pillow", "pyyaml", "matplotlib"],
    )