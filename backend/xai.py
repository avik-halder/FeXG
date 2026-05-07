from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

APP_TITLE = "MIT-BIH Federated ECG Classifier API"
MODEL_PATH = "fedavg_model.h5"

CLASS_NAMES = ["Normal (N)", "LBBB (L)", "RBBB (R)", "APB (A)", "VPB (V)"]

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # loaded on startup


class PredictRequest(BaseModel):
    signal: List[float] = Field(..., description="1D ECG beat samples (float list)")
    return_probabilities: Optional[bool] = Field(True, description="Return class probabilities")


class PredictResponse(BaseModel):
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: Optional[List[float]] = None


class ExplainRequest(BaseModel):
    signal: List[float] = Field(..., description="1D ECG beat samples (float list)")
    method: str = Field("integrated_gradients", description="saliency | integrated_gradients")
    steps: int = Field(50, ge=5, le=300, description="IG steps (only for integrated_gradients)")
    baseline: str = Field("zeros", description="zeros | mean")
    target_index: Optional[int] = Field(None, description="Class index to explain. Default: predicted class.")
    return_probabilities: Optional[bool] = Field(True, description="Return class probabilities")


class ExplainResponse(BaseModel):
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: Optional[List[float]] = None
    method: str
    target_index: int
    attribution: List[float]  # length = time_steps


@app.on_event("startup")
def _load():
    global model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_input_shape": getattr(model, "input_shape", None),
        "model_output_shape": getattr(model, "output_shape", None),
    }


@app.get("/model_info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "num_classes": model.output_shape[-1],
        "class_names": CLASS_NAMES,
    }


def _prepare_input(signal: List[float]) -> np.ndarray:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = np.asarray(signal, dtype=np.float32)

    if x.ndim != 1:
        raise HTTPException(status_code=400, detail="signal must be a 1D list of floats")

    expected_steps = model.input_shape[1]
    if expected_steps is not None and x.shape[0] != expected_steps:
        raise HTTPException(
            status_code=400,
            detail=f"signal length must be {expected_steps}, got {x.shape[0]}"
        )

    # NumPy-only z-score normalization (matches your training)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        raise HTTPException(status_code=400, detail="signal has near-zero variance; cannot normalize")
    x = (x - mean) / (std + 1e-8)

    # reshape to (1, time_steps, 1)
    x = x.reshape(1, x.shape[0], 1)
    return x


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = _prepare_input(req.signal)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    resp = {
        "predicted_index": pred_idx,
        "predicted_label": CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx),
        "confidence": conf
    }

    if req.return_probabilities:
        resp["probabilities"] = [float(p) for p in probs.tolist()]

    return resp


def _saliency_attribution(x_tf: tf.Tensor, target_index: int) -> np.ndarray:
    """
    Raw saliency: d score(target) / d input
    x_tf shape: (1, T, 1)
    """
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        preds = model(x_tf, training=False)  # (1, C)
        score = preds[:, target_index]       # (1,)
    grads = tape.gradient(score, x_tf)       # (1, T, 1)
    attr = tf.abs(grads)[0, :, 0]            # (T,)
    return attr.numpy()


def _integrated_gradients(x_tf: tf.Tensor, target_index: int, steps: int, baseline_mode: str) -> np.ndarray:
    """
    Integrated Gradients for 1D signal.
    Returns attribution of shape (T,)
    """
    x = x_tf
    if baseline_mode == "zeros":
        baseline = tf.zeros_like(x)
    elif baseline_mode == "mean":
        baseline = tf.ones_like(x) * tf.reduce_mean(x)
    else:
        raise HTTPException(status_code=400, detail="baseline must be 'zeros' or 'mean'")

    # interpolate between baseline and x
    alphas = tf.linspace(0.0, 1.0, steps + 1)  # steps+1 points
    attrs = tf.zeros_like(x)

    for a in alphas:
        x_interp = baseline + a * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x_interp)
            preds = model(x_interp, training=False)
            score = preds[:, target_index]
        grads = tape.gradient(score, x_interp)
        attrs += grads

    avg_grads = attrs / tf.cast(tf.size(alphas), tf.float32)
    ig = (x - baseline) * avg_grads
    attr = tf.abs(ig)[0, :, 0]
    return attr.numpy()


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    x = _prepare_input(req.signal)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    # which class to explain?
    target_index = pred_idx if req.target_index is None else int(req.target_index)
    if target_index < 0 or target_index >= len(probs):
        raise HTTPException(status_code=400, detail=f"target_index must be in [0, {len(probs)-1}]")

    method = req.method.lower().strip()
    if method == "saliency":
        attr = _saliency_attribution(x_tf, target_index)
    elif method == "integrated_gradients":
        attr = _integrated_gradients(x_tf, target_index, steps=req.steps, baseline_mode=req.baseline)
    else:
        raise HTTPException(status_code=400, detail="method must be 'saliency' or 'integrated_gradients'")

    # normalize attribution to [0,1] for visualization ease
    a_min, a_max = float(np.min(attr)), float(np.max(attr))
    if a_max - a_min > 1e-12:
        attr_norm = (attr - a_min) / (a_max - a_min)
    else:
        attr_norm = np.zeros_like(attr)

    resp = {
        "predicted_index": pred_idx,
        "predicted_label": CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx),
        "confidence": conf,
        "method": method,
        "target_index": target_index,
        "attribution": [float(v) for v in attr_norm.tolist()],
    }

    if req.return_probabilities:
        resp["probabilities"] = [float(p) for p in probs.tolist()]

    return resp
