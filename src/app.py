import os
import json
import joblib
import pandas as pd
import numpy as np
from collections import deque
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional, Union

BASE_DIR = os.path.dirname(__file__) or "."
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
SRC_DIR = os.path.join(BASE_DIR)

BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "baseline_lgbm_on_uplift_features.joblib")

ALT_BASELINE_PATHS = [
    os.path.join(OUTPUT_DIR, "baseline_lgbm_boosted.joblib"),
    os.path.join(OUTPUT_DIR, "baseline_lgbm.joblib"),
]

P_TREATED_MODEL_PATH = os.path.join(OUTPUT_DIR, "p_treated_model.joblib")
P_CONTROL_MODEL_PATH = os.path.join(OUTPUT_DIR, "p_control_model.joblib")
ALT_PAIRED_MODEL_PATH = os.path.join(OUTPUT_DIR, "x_learner_lgbm.joblib")  

META_PATH = os.path.join(OUTPUT_DIR, "uplift_meta.json")
LABEL_MAPS_PATH = os.path.join(OUTPUT_DIR, "label_maps.json")

CUST_DEMO_CSV = os.path.join(SRC_DIR, "cust_demo_feat.csv")
CUST_TRANS_CSV = os.path.join(SRC_DIR, "cust_trans_features.csv")
CAMPAIGN_CSV = os.path.join(SRC_DIR, "campaign_feat.csv")

def safe_load_joblib(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def safe_load_json(path: str):
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return {}

meta = safe_load_json(META_PATH)

if 'demo_mode' not in meta:
    meta['demo_mode'] = True
if 'demo' not in meta:
    meta['demo'] = {
        "enabled": True,
        "uplift_multiplier": 50.0,
        "min_uplift_display": 0.005,
        "cap_probability": 0.99
    }
if 'uplift_threshold' not in meta:
    meta['uplift_threshold'] = 0.005 

label_maps = safe_load_json(LABEL_MAPS_PATH)
FEATURES = meta.get("features", [])
DATE_COLS = set(meta.get("date_cols", []))
UPLIFT_THRESHOLD = float(meta.get("uplift_threshold", 0.005)) 

def _load_csv_or_empty(path, index_col):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if index_col not in df.columns:
            return pd.DataFrame(columns=[])
        return df.set_index(index_col)
    return pd.DataFrame(columns=[])

cust_demo_df  = _load_csv_or_empty(CUST_DEMO_CSV, "customer_id")
cust_trans_df = _load_csv_or_empty(CUST_TRANS_CSV, "customer_id")
campaign_df   = _load_csv_or_empty(CAMPAIGN_CSV, "campaign_id")

cust_demo_dict  = cust_demo_df.to_dict(orient="index") if not cust_demo_df.empty else {}
cust_trans_dict = cust_trans_df.to_dict(orient="index") if not cust_trans_df.empty else {}
campaign_dict   = campaign_df.to_dict(orient="index") if not campaign_df.empty else {}

model_load_errors = []
baseline_model = safe_load_joblib(BASELINE_MODEL_PATH)
if baseline_model is None:
    for p in ALT_BASELINE_PATHS:
        baseline_model = safe_load_joblib(p)
        if baseline_model is not None:
            break

p_treated_model = safe_load_joblib(P_TREATED_MODEL_PATH)
p_control_model = safe_load_joblib(P_CONTROL_MODEL_PATH)

uplift_model = None
if (p_treated_model is None or p_control_model is None):
    uplift_model = safe_load_joblib(ALT_PAIRED_MODEL_PATH)
    if uplift_model is None:
        pass

if baseline_model is None:
    model_load_errors.append("baseline model not found")
if (p_treated_model is None or p_control_model is None) and uplift_model is None:
    model_load_errors.append("paired outcome models (p_treated/p_control) or x_learner not found")

model_load_error = "; ".join(model_load_errors) if model_load_errors else None

app = FastAPI(title="Coupon Redemption + Uplift API", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    customer_id: int
    campaign_id: int
    coupon_id: int

class BatchPredictRequest(BaseModel):
    requests: List[PredictRequest]

LAST_QUEUE_MAX = 5
_last_predictions = deque(maxlen=LAST_QUEUE_MAX)

EPOCH = pd.Timestamp("1970-01-01")

def _convert_dates_to_numeric_row(row: Dict[str, Any]):
    out = {}
    for k, v in row.items():
        if k in DATE_COLS:
            try:
                dt = pd.to_datetime(v, errors="coerce")
                if pd.isna(dt):
                    out[k] = 0
                else:
                    out[k] = int((dt - EPOCH).days)
            except Exception:
                out[k] = 0
        else:
            out[k] = v
    return out

def _apply_label_maps(row: Dict[str, Any], label_maps: Dict[str, Dict[str,int]]):
    out = {}
    for k, v in row.items():
        if k in label_maps:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    key = "missing" if "missing" in label_maps[k] else str(v)
                else:
                    key = str(v)
                out[k] = label_maps[k].get(key, label_maps[k].get("missing", 0))
            except Exception:
                out[k] = label_maps[k].get("missing", 0)
        else:
            out[k] = v
    return out

def build_feature_row(customer_id: int, campaign_id: int, coupon_id: int) -> pd.DataFrame:
    row = {f: 0 for f in FEATURES}
    if "customer_id" in row:
        row["customer_id"] = int(customer_id)
    if "campaign_id" in row:
        row["campaign_id"] = int(campaign_id)
    if "coupon_id" in row:
        row["coupon_id"] = int(coupon_id)

    demo = cust_demo_dict.get(int(customer_id), {})
    for k, v in demo.items():
        if k in row and k != "customer_id":
            row[k] = v

    tx = cust_trans_dict.get(int(customer_id), {})
    for k, v in tx.items():
        if k in row and k != "customer_id":
            row[k] = v

    camp = campaign_dict.get(int(campaign_id), {})
    for k, v in camp.items():
        if k in row and k != "campaign_id":
            row[k] = v

    row = _convert_dates_to_numeric_row(row)
    row = _apply_label_maps(row, label_maps)
    X = pd.DataFrame([row], columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

def _safe_prob_from_model(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    if model is None:
        return None


    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.asarray(probs)[:, 1].flatten()
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return np.asarray(probs)[:, 0].flatten()
        else:
            return np.asarray(probs).flatten()

    if hasattr(model, "predict"):
        out = model.predict(X)
        arr = np.asarray(out).flatten()
        return arr

    return None

def _get_uplift_components_from_uplift_model(uplift_model, X: pd.DataFrame):
    """
    If we loaded a single uplift model (e.g. x_learner), try to extract
    (uplift, p_treated, p_control). Returns (uplift_arr, p_with_arr, p_without_arr).
    """
    if uplift_model is None:
        return None, None, None

    try:
        out = uplift_model.predict(X, return_components=True)
        if isinstance(out, tuple) and len(out) == 3:
            uplift_arr = np.asarray(out[0]).flatten()
            p_with_arr = np.asarray(out[1]).flatten()
            p_without_arr = np.asarray(out[2]).flatten()
            return uplift_arr, p_with_arr, p_without_arr
        if isinstance(out, dict):
            uplift_arr = np.asarray(out.get("uplift") or out.get("tau") or list(out.values())[0]).flatten()
            p_with_arr = np.asarray(out.get("treatment") or out.get("p_treated") or out.get("p_with") or [np.nan]*len(uplift_arr)).flatten()
            p_without_arr = np.asarray(out.get("control") or out.get("p_control") or out.get("p_without") or [np.nan]*len(uplift_arr)).flatten()
            return uplift_arr, p_with_arr, p_without_arr
    except Exception:
        pass

    try:
        uplift_arr = np.asarray(uplift_model.predict(X)).flatten()
        return uplift_arr, None, None
    except Exception:
        pass

    return None, None, None

def _clamp_probs(arr: Union[np.ndarray, List[float]]) -> np.ndarray:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    a = np.clip(a, 0.0, 1.0)
    return a


# routes

@app.get("/")
def root():
    return {"message": "Coupon Uplift API running", "models_loaded": model_load_error is None}

@app.get("/health")
def health():
    return {
        "models_loaded": model_load_error is None,
        "model_load_error": model_load_error,
        "n_features": len(FEATURES),
        "uplift_threshold": UPLIFT_THRESHOLD
    }

@app.get("/last_predictions")
def last_predictions():
    return {"last_predictions": list(_last_predictions)}

@app.post("/predict")
def predict(req: PredictRequest):
    if model_load_error:
        raise HTTPException(status_code=500, detail=f"Model loading error: {model_load_error}")

    try:
        X = build_feature_row(req.customer_id, req.campaign_id, req.coupon_id)

        if baseline_model is None:
            raise RuntimeError("Baseline model unavailable")

        if hasattr(baseline_model, "predict_proba"):
            prob_baseline = float(baseline_model.predict_proba(X)[:, 1][0])
        else:
            prob_baseline = float(baseline_model.predict(X)[0])
            
        if p_treated_model is None or p_control_model is None:
            raise RuntimeError("P_treated / P_control models unavailable")

        if hasattr(p_treated_model, "predict_proba"):
            p_with_raw = float(p_treated_model.predict_proba(X)[:, 1][0])
        else:
            p_with_raw = float(p_treated_model.predict(X)[0])

        if hasattr(p_control_model, "predict_proba"):
            p_without_raw = float(p_control_model.predict_proba(X)[:, 1][0])
        else:
            p_without_raw = float(p_control_model.predict(X)[0])

        uplift_raw = float(p_with_raw - p_without_raw)

        demo_cfg = meta['demo']
        demo_enabled = bool(meta['demo_mode']) and demo_cfg['enabled']

        p_with_disp = p_with_raw
        p_without_disp = p_without_raw
        uplift_disp = uplift_raw

        if demo_enabled:
            mult = float(demo_cfg.get("uplift_multiplier", 50.0))
            min_display = float(demo_cfg.get("min_uplift_display", 0.005))
            cap_prob = float(demo_cfg.get("cap_probability", 0.99))
            
            uplift_disp = uplift_raw * mult

            if uplift_disp > 0 and uplift_disp < min_display:
                uplift_disp = min_display
            if uplift_disp < 0 and abs(uplift_disp) < min_display:
                uplift_disp = -min_display
            p_without_disp = np.clip(p_without_raw, 0.0, 1.0)
            p_with_disp = np.clip(p_without_disp + uplift_disp, 0.0, cap_prob)

        uplift_threshold = float(meta.get("uplift_threshold", UPLIFT_THRESHOLD))
        recommendation = "Target ✅" if uplift_disp >= uplift_threshold else "Do not target ❌"

        response = {
            "baseline_redemption_prob": round(prob_baseline, 6),
            "prob_with_coupon": round(p_with_disp, 6),
            "prob_without_coupon": round(p_without_disp, 6),
            "predicted_uplift": round(uplift_disp, 6),
            "uplift_threshold": uplift_threshold,
            "recommendation": recommendation,
            "demo_mode": demo_enabled
        }
        
        _last_predictions.append(response)

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
def batch_predict(reqs: BatchPredictRequest):
    """
    Accepts {"requests": [{"customer_id":..,"campaign_id":..,"coupon_id":..}, ...]}
    Returns a list of prediction dicts (same format as /predict).
    """
    results = []
    for r in reqs.requests:
        resp = predict(PredictRequest(**r.dict()))
        if 'demo_mode' in resp:
            del resp['demo_mode']
        
        results.append({
            "customer_id": r.customer_id,
            "campaign_id": r.campaign_id,
            "coupon_id": r.coupon_id,
            **resp
        })
    return {"results": results}