"""
Interactive demo for the California Housing price model.

Run (after `python train_house.py` has produced artifacts/house_model.joblib):
    python app.py
Then open the local URL Gradio prints (default http://127.0.0.1:7860).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "artifacts" / "house_model.joblib"

if not MODEL_PATH.exists():
    print("artifacts/house_model.joblib not found - training it now "
          "(takes a few minutes, happens only once) ...", flush=True)
    subprocess.run([sys.executable, str(HERE / "train_house.py")], check=True)

BUNDLE = joblib.load(MODEL_PATH)
PIPE = BUNDLE["pipeline"]
MODEL_NAME = BUNDLE["model_name"]

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
]

BOUNDS = {
    "MedInc": (0.5, 15.0),
    "HouseAge": (1, 52),
    "AveRooms": (1, 12),
    "AveBedrms": (0.5, 4),
    "Population": (3, 5000),
    "AveOccup": (1, 10),
    "Latitude": (32.0, 42.0),
    "Longitude": (-125.0, -114.0),
}
DEFAULTS = {
    "MedInc": 4.0,
    "HouseAge": 28,
    "AveRooms": 5.5,
    "AveBedrms": 1.05,
    "Population": 1200,
    "AveOccup": 3.0,
    "Latitude": 34.5,
    "Longitude": -118.5,
}

TRAIN_MEANS: dict[str, float] = {
    "MedInc": 3.87, "HouseAge": 28.6, "AveRooms": 5.4, "AveBedrms": 1.1,
    "Population": 1425.0, "AveOccup": 3.07, "Latitude": 35.63,
    "Longitude": -119.57,
}


def predict(*values: float) -> tuple[float, str]:
    row = dict(zip(FEATURES, [float(v) for v in values]))
    x = pd.DataFrame([row])
    pred = float(PIPE.predict(x)[0])
    usd = pred * 100_000  # target units are $100k (1990 dollars)
    md = (
        f"### Predicted median house value: **${usd:,.0f}**  \n"
        f"({pred:.3f} in dataset units; block-group averages, 1990 dollars)"
    )
    return pred, md


def random_block_group() -> list[float]:
    """Sample a plausible block group around the training means."""
    rng = np.random.default_rng()
    vals = [
        rng.uniform(0.5, 12.0),                      # MedInc
        rng.uniform(1, 52),                          # HouseAge
        max(1.0, rng.normal(5.4, 2.0)),              # AveRooms
        max(0.5, rng.normal(1.1, 0.2)),              # AveBedrms
        max(3, rng.normal(1425, 800)),               # Population
        max(1.0, rng.normal(3.07, 1.3)),             # AveOccup
        rng.uniform(32.5, 41.9),                     # Latitude
        rng.uniform(-124.3, -114.5),                 # Longitude
    ]
    return [float(v) for v in vals]


def sample_and_predict() -> tuple:
    vals = random_block_group()
    pred, md = predict(*vals)
    return (*vals, pred, md)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="California House Price Predictor") as demo:
        gr.Markdown(
            "# California House Price Prediction\n"
            f"Trained model: **{MODEL_NAME}** (Linear / Ridge / Lasso / "
            "Random Forest comparison, 5-fold CV) deployed with Gradio.\n\n"
            "Inputs describe a *block group* (the dataset's unit), not a "
            "single house."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Block-group features")
                sliders: dict[str, gr.Slider] = {}
                for f in FEATURES:
                    lo, hi = BOUNDS[f]
                    sliders[f] = gr.Slider(lo, hi, value=DEFAULTS[f],
                                           step=(1 if f == "HouseAge" else 0.01),
                                           label=f)
                btn_rand = gr.Button("Random plausible block group")
                btn = gr.Button("Predict price", variant="primary")
            with gr.Column():
                pred_out = gr.Number(label="Predicted value (dataset units)")
                verdict = gr.Markdown("")

        inputs = [sliders[f] for f in FEATURES]
        btn.click(predict, inputs=inputs, outputs=[pred_out, verdict])
        btn_rand.click(sample_and_predict, inputs=None,
                       outputs=[*inputs, pred_out, verdict])
    return demo


if __name__ == "__main__":
    build_ui().launch()
