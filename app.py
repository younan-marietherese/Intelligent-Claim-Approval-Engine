import os, json
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, redirect
import joblib

# ---------- Needed so unpickler can resolve the FunctionTransformer target ----------
def flatten_1d(X):
    """Flatten a (n,1) array/dataframe column to shape (n,)."""
    import numpy as _np
    return _np.asarray(X).ravel()

import sys, types
sys.modules.setdefault("__main__", types.ModuleType("__main__"))
sys.modules["__main__"].flatten_1d = flatten_1d

# ----------------------------
# Config
# ----------------------------
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "claim_approval_pipeline.joblib")
META_PATH     = os.path.join(ARTIFACTS_DIR, "metadata.json")
THR_PATH      = os.path.join(ARTIFACTS_DIR, "threshold.json")
CLIP_STATS_PATH = os.path.join(ARTIFACTS_DIR, "clip_stats.json")  # optional

# ----------------------------
# Load artifacts
# ----------------------------
try:
    clf = joblib.load(PIPELINE_PATH)  # ('pre', pre) + ('model', CalibratedClassifierCV)
except Exception as e:
    raise RuntimeError(f"Failed to load pipeline at {PIPELINE_PATH}: {e}")

with open(META_PATH, "r") as f:
    META = json.load(f)

with open(THR_PATH, "r") as f:
    BEST_THR = float(json.load(f).get("threshold", 0.5))

BASE_FEATURES: List[str] = META.get("base_features", [])
TEXT_COLS:    List[str] = META.get("text_cols", [])
CAT_COLS:     List[str] = META.get("cat_cols", [])
NUM_COLS:     List[str] = META.get("num_cols", [])

KNOWN_AMT_COLS = [
    "CLAIMED_AMOUNT","SYSTEM_CLAIMED_AMOUNT","PATIENT_SHARE",
    "BILLED_TAX","ACCEPTED_TAX","GROSS_CLAIMED_AMOUNT"
]
amt_for_feats = [c for c in KNOWN_AMT_COLS if c in BASE_FEATURES or c in NUM_COLS]

clip_stats = None
if os.path.exists(CLIP_STATS_PATH):
    try:
        with open(CLIP_STATS_PATH, "r") as f:
            clip_stats = json.load(f)
    except Exception:
        clip_stats = None  # continue without

# ----------------------------
# Helpers
# ----------------------------
def _norm_key(k: str) -> str:
    return "_".join(str(k).strip().upper().split())

def _apply_engineering_to_frame(df_like: pd.DataFrame) -> pd.DataFrame:
    """Recreate engineered features used during training."""
    eps = 1e-9
    df = df_like.copy()

    # Ratios
    if {"PATIENT_SHARE","CLAIMED_AMOUNT"}.issubset(df.columns):
        df["PATIENT_SHARE_PCT"] = (df["PATIENT_SHARE"] / (df["CLAIMED_AMOUNT"] + eps)).clip(0, 5.0)
    if {"ACCEPTED_TAX","BILLED_TAX"}.issubset(df.columns):
        df["TAX_ACCEPT_RATIO"] = (df["ACCEPTED_TAX"] / (df["BILLED_TAX"] + eps)).clip(0, 2.0)
    if {"SYSTEM_CLAIMED_AMOUNT","CLAIMED_AMOUNT"}.issubset(df.columns):
        df["SYSTEM_TO_CLAIMED_RATIO"] = (df["SYSTEM_CLAIMED_AMOUNT"] / (df["CLAIMED_AMOUNT"] + eps)).clip(0, 5.0)

    # Amount features
    for c in amt_for_feats:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if clip_stats and c in clip_stats:
                hi = float(clip_stats[c])
                clip_val = np.clip(vals, None, hi)
                log_val  = np.log1p(np.clip(vals, 0, hi))
            else:
                clip_val = vals
                log_val  = np.log1p(np.clip(vals, 0, None))

            df[f"{c}_CLIP"]      = clip_val
            df[f"{c}_LOG1P"]     = log_val
            df[f"{c}_NEG_FLAG"]  = (vals < 0).astype("uint8")
            df[f"{c}_ZERO_FLAG"] = (vals == 0).astype("uint8")

    return df

def _to_feature_frame(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Normalize JSON -> DataFrame with BASE_FEATURES, apply engineering."""
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Payload must be a JSON object or a list of JSON objects.")

    normed = []
    for r in records:
        nr = {_norm_key(k): v for k, v in r.items()}
        # alias if training used SRV_DESC
        if "SERVICE_DESC" in nr and "SRV_DESC" in BASE_FEATURES and "SRV_DESC" not in nr:
            nr["SRV_DESC"] = nr.pop("SERVICE_DESC")
        normed.append(nr)

    raw_df = pd.DataFrame(normed)
    enriched = _apply_engineering_to_frame(raw_df)

    for col in BASE_FEATURES:
        if col not in enriched.columns:
            enriched[col] = np.nan

    return enriched[BASE_FEATURES].copy()

def _predict_df(dfX: pd.DataFrame, threshold: float = None):
    thr = BEST_THR if threshold is None else float(threshold)
    proba = clf.predict_proba(dfX)[:, 1]
    return [{
        "proba_approved": round(float(p), 6),
        "threshold": round(thr, 6),
        "decision": int(p >= thr)
    } for p in proba]

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # redirect to the nice form UI
    return redirect("/web")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "artifacts_dir": ARTIFACTS_DIR,
        "pipeline_loaded": True,
        "features_expected": len(BASE_FEATURES),
        "threshold": BEST_THR,
        "clip_stats_loaded": bool(clip_stats)
    })

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({
        "base_features": BASE_FEATURES,
        "text_cols": TEXT_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "library_versions": META.get("library_versions", {})
    })

# UI form
@app.route("/web", methods=["GET"])
def web_ui():
    return """<!doctype html>
<html>
  <head>
    <meta charset="utf-8"><title>Claims Predictor</title>
    <style>
      body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;background:#f8fafc;margin:0;padding:2rem;}
      .card{max-width:960px;margin:0 auto;background:#fff;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.06);padding:1.25rem 1.25rem 2rem;}
      h2{margin:.5rem 0 1rem;}
      .grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;}
      label{font-size:.9rem;color:#334155;margin-bottom:6px;display:block}
      input,select{width:100%;padding:.6rem .7rem;border:1px solid #cbd5e1;border-radius:8px;background:#fff;font-size:.95rem}
      .row{margin-top:10px}
      button{appearance:none;background:#2563eb;border:0;color:#fff;padding:.7rem 1.1rem;border-radius:9px;font-weight:600;cursor:pointer}
      button:hover{background:#1e4fd6}
      .result{margin-top:18px;background:#0b1220;color:#bfe2ff;border-radius:8px;padding:12px;white-space:pre-wrap}
      .badge{display:inline-block;padding:.25rem .5rem;border-radius:9999px;font-size:.8rem;margin-left:.35rem}
      .ok{background:#d1fae5;color:#065f46}
      .warn{background:#fee2e2;color:#991b1b}
    </style>
  </head>
  <body>
    <div class="card">
      <h2>Claims Predictor</h2>
      <!-- form fields -->
      <div class="grid">
        <div><label>CLAIMED_AMOUNT</label><input id="CLAIMED_AMOUNT" type="number" step="0.01" value="560"></div>
        <div><label>SYSTEM_CLAIMED_AMOUNT</label><input id="SYSTEM_CLAIMED_AMOUNT" type="number" step="0.01" value="540"></div>
        <div><label>PATIENT_SHARE</label><input id="PATIENT_SHARE" type="number" step="0.01" value="50"></div>
        <div><label>BILLED_TAX</label><input id="BILLED_TAX" type="number" step="0.01" value="10"></div>
        <div><label>ACCEPTED_TAX</label><input id="ACCEPTED_TAX" type="number" step="0.01" value="10"></div>
        <div><label>GROSS_CLAIMED_AMOUNT</label><input id="GROSS_CLAIMED_AMOUNT" type="number" step="0.01" value="610"></div>
        <div><label>COUNTRY</label><input id="COUNTRY" value="LEB"></div>
        <div><label>INSURER</label><input id="INSURER" value="ARP"></div>
        <div><label>CLASS</label><input id="CLASS" value="AB"></div>
        <div><label>COVER</label><input id="COVER" value="STANDARD"></div>
        <div><label>CURRENCY</label><input id="CURRENCY" value="USD"></div>
        <div><label>ADM_TYPE</label><input id="ADM_TYPE" value="OUTPATIENT"></div>
        <div><label>SERVICE</label><input id="SERVICE" value="GENERAL"></div>
        <div><label>PRE-AUTHORIZED</label><select id="PRE-AUTHORIZED"><option>YES</option><option>NO</option></select></div>
        <div><label>PROD</label><input id="PROD" value="GEN"></div>
        <div class="row" style="grid-column:1/-1"><label>DIAGNOSIS_DESCRIPTION</label><input id="DIAGNOSIS_DESCRIPTION" value="routine check up"></div>
        <div class="row" style="grid-column:1/-1"><label>SERVICE_DESC</label><input id="SERVICE_DESC" value="consultation"></div>
      </div>
      <div class="row" style="display:flex;align-items:center;gap:10px;margin-top:16px">
        <label>Threshold</label>
        <input id="thr" type="number" step="0.01" min="0" max="1" placeholder="0.50" style="max-width:120px">
        <button onclick="predict()">Submit</button>
        <span id="badge"></span>
      </div>
      <div id="out" class="result" hidden></div>
    </div>
    <script>
      function readFields(){
        const ids=["CLAIMED_AMOUNT","SYSTEM_CLAIMED_AMOUNT","PATIENT_SHARE","BILLED_TAX","ACCEPTED_TAX","GROSS_CLAIMED_AMOUNT","COUNTRY","INSURER","CLASS","COVER","CURRENCY","ADM_TYPE","SERVICE","PRE-AUTHORIZED","PROD","DIAGNOSIS_DESCRIPTION","SERVICE_DESC"];
        const o={}; ids.forEach(id=>{const el=document.getElementById(id); if(!el) return; const v=el.value;
          o[id]=(["CLAIMED_AMOUNT","SYSTEM_CLAIMED_AMOUNT","PATIENT_SHARE","BILLED_TAX","ACCEPTED_TAX","GROSS_CLAIMED_AMOUNT"].includes(id))?(v===""?null:Number(v)):v;});
        return o;
      }
      async function predict(){
        const thr=document.getElementById('thr').value;
        const url= thr?(`/predict?threshold=${encodeURIComponent(thr)}`):'/predict';
        const payload=readFields();
        const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        const out=document.getElementById('out'); const badge=document.getElementById('badge');
        out.hidden=false; out.textContent="Loading...";
        const j=await res.json(); out.textContent=JSON.stringify(j,null,2);
        if(j.ok && j.predictions && j.predictions.length){
          const d=j.predictions[0]; const conf=d.proba_approved >= (d.threshold ?? 0.5);
          badge.innerHTML= conf?'<span class="badge ok">Approved</span>':'<span class="badge warn">Rejected</span>';
        } else { badge.innerHTML=''; }
      }
    </script>
  </body>
</html>"""

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        thr = request.args.get("threshold", default=None, type=float)
        X = _to_feature_frame(payload)
        preds = _predict_df(X, threshold=thr)
        for i, p in enumerate(preds):
            p["row_id"] = i
        return jsonify({"ok": True, "n": len(preds), "predictions": preds})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
