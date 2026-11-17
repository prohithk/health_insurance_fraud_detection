"""
fastapi_app.py - Health Insurance Fraud Detection (FastAPI backend)

REST API that loads the trained pipeline from `model.pkl` and serves predictions.
It never trains a model (run `python train.py` to create / refresh model.pkl).
The Streamlit frontend (app.py) is independent and is not changed by this file.

Local:   uvicorn fastapi_app:app --reload          ->  http://127.0.0.1:8000/docs
Render:  uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --workers 1

Environment variables (all optional):
    API_KEY          if set, /predict, /predict/batch and /model-info require the header  X-API-Key: <value>
    ALLOWED_ORIGINS  comma-separated browser origins for CORS (default "*")
    PORT             port used when running `python fastapi_app.py` (Render sets this automatically)

NOTE: this file must NOT be called fastapi.py - it would shadow the real `fastapi` library.
"""

import io
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("fraud-api")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
MODEL_PARTS_GLOB = "model.pkl.part*"   # model.pkl can be stored as model.pkl.part1, .part2, ... (GitHub web upload limit)
MAX_BATCH_SIZE = 1000
API_KEY = os.getenv("API_KEY") or None   # when set, prediction endpoints are protected
FRAUD_THRESHOLD = 0.5      # same rule as pipeline.predict() for this model: Fraud if P(fraud) > 50%


# --------------------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------------------
class ClaimInput(BaseModel):
    """One health-insurance claim. Field names are the model's input features.
    `previous_rejections`, `diagnosis_code`, `room_type` and `doctor_experience_years`
    may be omitted or null when unknown (the model imputes them, as in training).
    `claim_to_coverage_ratio`, `claim_year` and `claim_month` are calculated by the API."""

    model_config = ConfigDict(
        extra="forbid",   # unknown / misspelled fields are rejected instead of silently ignored
        json_schema_extra={"examples": [{
            "state": "Maharashtra", "gender": "Female", "patient_age": 42,
            "insurance_type": "Individual", "sum_insured": 500000, "policy_age_days": 700,
            "previous_claims": 1, "claims_last_12_months": 1, "previous_rejections": 0,
            "hospital_type": "Private", "admission_type": "Planned", "treatment_type": "Surgery",
            "diagnosis_code": "DIA-05", "room_type": "Semi-Private", "length_of_stay": 4,
            "doctor_experience_years": 13, "claim_date": "2026-06-15",
            "total_bill_amount": 175000.0, "claimed_amount": 150000.0, "payment_method": "Cashless",
        }]},
    )

    state: str
    gender: str
    patient_age: int
    insurance_type: str
    sum_insured: int
    policy_age_days: int
    previous_claims: int
    claims_last_12_months: int
    previous_rejections: Optional[int] = None
    hospital_type: str
    admission_type: str
    treatment_type: str
    diagnosis_code: Optional[str] = None
    room_type: Optional[str] = None
    length_of_stay: int
    doctor_experience_years: Optional[int] = None
    claim_date: date = Field(description="Format YYYY-MM-DD. Only the year and month are used by the model.")
    total_bill_amount: float
    claimed_amount: float
    payment_method: str


class PredictionResult(BaseModel):
    predicted_class: int = Field(description="1 = Fraud, 0 = Genuine")
    predicted_label: str
    fraud_probability: float
    genuine_probability: float
    threshold: float
    average_fraud_rate_in_training_data: float
    model_name: str


class BatchRequest(BaseModel):
    claims: list[ClaimInput] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResult(BaseModel):
    count: int
    predictions: list[PredictionResult]


# --------------------------------------------------------------------------------------
# Model loading (once, at start-up)
# --------------------------------------------------------------------------------------
def load_bundle() -> dict:
    """Load model.pkl, or - if it was uploaded in pieces - glue model.pkl.part1, part2, ... back together."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    parts = sorted(BASE_DIR.glob(MODEL_PARTS_GLOB))
    if parts:
        return joblib.load(io.BytesIO(b"".join(part.read_bytes() for part in parts)))
    raise FileNotFoundError(f"{MODEL_PATH} (or {MODEL_PARTS_GLOB}) not found. Run `python train.py` first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = None
    app.state.load_error = None
    try:
        bundle = load_bundle()
        for key in ("pipeline", "feature_order", "numeric_features", "categorical_features", "categories",
                    "numeric_ranges", "columns_with_missing", "sum_insured_options", "claim_year_range",
                    "class_names", "model_name", "fraud_rate_train"):
            if key not in bundle:
                raise KeyError(f"model.pkl is missing '{key}'. Run `python train.py` again.")
        # Single-row requests are fastest (and safest under concurrency) with one thread per request.
        bundle["pipeline"].named_steps["classifier"].n_jobs = 1
        for unused in ("test_claim_ids", "feature_importance", "confusion_matrix"):
            bundle.pop(unused, None)   # not needed by the API; saves a little memory
        app.state.bundle = bundle

        trained_with = bundle.get("versions", {}).get("scikit-learn")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning("model.pkl was trained with scikit-learn %s but %s is installed. "
                           "Run `python train.py` again if you see errors.", trained_with, sklearn.__version__)
        logger.info("Model loaded: %s", bundle["model_name"])
        if API_KEY is None:
            logger.warning("API_KEY is not set: anyone who knows the URL can call the prediction endpoints.")
    except Exception as error:   # the API still starts, so /health can explain what is wrong
        app.state.load_error = f"{type(error).__name__}: {error}"
        logger.error("Could not load model: %s", app.state.load_error)
    yield


app = FastAPI(
    title="Health Insurance Fraud Detection API",
    description="Predicts whether a health-insurance claim is fraudulent. "
                "A decision-support aid for prioritising claims for review - not for automatic rejection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed browser origins (only matters if a web page calls this API directly).
# Default "*" is fine for local development; set ALLOWED_ORIGINS="https://my-site.com" in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_bundle(request: Request) -> dict:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"Model is not available: {request.app.state.load_error}")
    return bundle


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If API_KEY is configured, the caller must send it in the `X-API-Key` header."""
    if API_KEY is not None and not (x_api_key and secrets.compare_digest(x_api_key.encode(), API_KEY.encode())):
        raise HTTPException(status_code=401, detail="Invalid or missing API key (send header X-API-Key).")


# --------------------------------------------------------------------------------------
# Validation + prediction helpers
# --------------------------------------------------------------------------------------
def validate_claim(claim: ClaimInput, bundle: dict) -> list[str]:
    """Business-rule checks based on what the model saw in training (same rules as the Streamlit app)."""
    problems = []
    data = claim.model_dump()

    for column in bundle["categorical_features"]:
        value = data.get(column)
        if value is None:
            if column not in bundle["columns_with_missing"]:
                problems.append(f"{column}: is required.")
        elif value not in bundle["categories"][column]:
            problems.append(f"{column}: '{value}' is not allowed. Allowed values: {bundle['categories'][column]}")

    for column, (low, high) in bundle["numeric_ranges"].items():
        if column in ("claim_year", "claim_month", "claim_to_coverage_ratio", "sum_insured"):
            continue   # calculated by the API (or checked against its allowed list below)
        value = data.get(column)
        if value is None:
            if column not in bundle["columns_with_missing"]:
                problems.append(f"{column}: is required.")
        elif not (low <= value <= high):
            problems.append(f"{column}: {value} is outside the range seen in training ({low:g} to {high:g}).")

    if claim.sum_insured not in bundle["sum_insured_options"]:
        problems.append(f"sum_insured: {claim.sum_insured} is not allowed. "
                        f"Allowed values: {bundle['sum_insured_options']}")

    year_low, year_high = bundle["claim_year_range"]
    if not (year_low <= claim.claim_date.year <= year_high):
        problems.append(f"claim_date: year must be between {year_low} and {year_high}.")

    if claim.claimed_amount > claim.total_bill_amount:
        problems.append("claimed_amount cannot be larger than total_bill_amount.")
    if claim.claimed_amount > claim.sum_insured:
        problems.append("claimed_amount cannot be larger than sum_insured.")
    return problems


def build_dataframe(claims: list[ClaimInput], bundle: dict) -> pd.DataFrame:
    """Turn claims into a DataFrame with the exact columns / order used in training."""
    rows = []
    for claim in claims:
        data = claim.model_dump()
        data["claim_year"] = claim.claim_date.year
        data["claim_month"] = claim.claim_date.month
        data["claim_to_coverage_ratio"] = round(claim.claimed_amount / claim.sum_insured, 4)
        rows.append({column: data[column] for column in bundle["feature_order"]})

    frame = pd.DataFrame(rows, columns=bundle["feature_order"])
    for column in bundle["categorical_features"]:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    for column in bundle["numeric_features"]:
        frame[column] = pd.to_numeric(frame[column])      # None -> NaN
    return frame


def run_model(frame: pd.DataFrame, bundle: dict) -> list[PredictionResult]:
    pipeline = bundle["pipeline"]
    classes = list(pipeline.classes_)
    predicted = pipeline.predict(frame)
    fraud_probability = pipeline.predict_proba(frame)[:, classes.index(1)]
    return [
        PredictionResult(
            predicted_class=int(label),
            predicted_label=bundle["class_names"][int(label)],
            fraud_probability=round(float(p), 6),
            genuine_probability=round(1.0 - float(p), 6),
            threshold=FRAUD_THRESHOLD,
            average_fraud_rate_in_training_data=round(float(bundle["fraud_rate_train"]), 6),
            model_name=bundle["model_name"],
        )
        for label, p in zip(predicted, fraud_probability)
    ]


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Health Insurance Fraud Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["Status"])
def health(request: Request):
    """Is the API running and is the model loaded?"""
    bundle = request.app.state.bundle
    if bundle is None:
        return JSONResponse(status_code=503, content={"status": "error", "model_loaded": False,
                                                      "detail": request.app.state.load_error})
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": bundle["model_name"],
        "trained_at": bundle.get("trained_at"),
        "trained_with_scikit_learn": bundle.get("versions", {}).get("scikit-learn"),
        "installed_scikit_learn": sklearn.__version__,
    }


@app.get("/model-info", tags=["Status"], dependencies=[Depends(require_api_key)])
def model_info(request: Request):
    """What the model expects: features, allowed values and ranges (taken from the training data)."""
    bundle = get_bundle(request)
    return {
        "model_name": bundle["model_name"],
        "hyperparameters": bundle.get("hyperparameters"),
        "class_names": bundle["class_names"],
        "feature_order": bundle["feature_order"],
        "calculated_by_api": ["claim_year", "claim_month", "claim_to_coverage_ratio"],
        "optional_when_unknown": bundle["columns_with_missing"],
        "categorical_allowed_values": bundle["categories"],
        "sum_insured_allowed_values": bundle["sum_insured_options"],
        "numeric_ranges": {c: {"min": lo, "max": hi} for c, (lo, hi) in bundle["numeric_ranges"].items()},
        "claim_year_range": bundle["claim_year_range"],
        "test_set_metrics": bundle.get("metrics"),
        "n_train": bundle.get("n_train"),
        "n_test": bundle.get("n_test"),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"], dependencies=[Depends(require_api_key)])
def predict(claim: ClaimInput, request: Request):
    """Predict one claim."""
    bundle = get_bundle(request)
    problems = validate_claim(claim, bundle)
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    try:
        return run_model(build_dataframe([claim], bundle), bundle)[0]
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(error).__name__}: {error}")


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"], dependencies=[Depends(require_api_key)])
def predict_batch(payload: BatchRequest, request: Request):
    """Predict up to 1000 claims at once. If any claim is invalid nothing is predicted and every problem is listed."""
    bundle = get_bundle(request)
    problems = []
    for index, claim in enumerate(payload.claims):
        problems += [f"claims[{index}] {message}" for message in validate_claim(claim, bundle)]
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    try:
        results = run_model(build_dataframe(payload.claims, bundle), bundle)
    except Exception as error:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(error).__name__}: {error}")
    return BatchResult(count=len(results), predictions=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), reload=False)
