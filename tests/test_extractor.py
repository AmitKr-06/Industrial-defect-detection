import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/patchcore-metal-nut@champion")
print(model.predict(pd.DataFrame({"image_path": [
    "data/raw/metal_nut/test/good/000.png",
    "data/raw/metal_nut/test/bent/000.png"]})))