
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train import build_final_model
from prepare import add_features
from casenote import run_case
from retrieval import retrieve, MIN_SCORE
from collections import Counter
import logging
from pathlib import Path

LOG_FILE = Path(__file__).with_name("service.log")

LOG_FILE = Path(__file__).with_name("service.log")

logger = logging.getLogger("fraud_triage")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)

logger.propagate = False


MODEL_ARTIFACTS = {}
REQUEST_COUNTERS = {
    "score_requests": 0,
    "case_requests": 0,
    "flagged": 0,
    "clear": 0,
    "hold_for_review": 0,
    "case_notes_safe_default": 0,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_ARTIFACTS.update(build_final_model())
    yield
    MODEL_ARTIFACTS.clear()


app = FastAPI(
    title="Razorpay Merchant Risk Desk",
    description="Fraud Triage Service API",
    version="1.0.0",
    lifespan=lifespan,
)


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float = Field(gt=0, le=1_000_000)
    hour_of_day: int = Field(ge=0, le=23)
    txn_count_last_24h: int = Field(ge=0)
    avg_amount_30d_inr: float = Field(gt=0)
    distance_from_home_km: float | None = Field(default=None, ge=0)
    is_new_device: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)

class CaseRequest(BaseModel):
    txn_id: str
    fraud_probability: float = Field(ge=0, le=1)
    amount_inr: float = Field(gt=0, le=1_000_000)
    question: str = Field(min_length=3)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fraud-triage-service",
    }


@app.post("/case")
def create_case(request: CaseRequest):
    REQUEST_COUNTERS["case_requests"] += 1
    logger.info("Case request received: txn_id=%s", request.txn_id)
    result = run_case(
        txn_id=request.txn_id,
        fraud_probability=request.fraud_probability,
        amount_inr=request.amount_inr,
        question=request.question,
    )
    if result.get("safe_default", False):
        REQUEST_COUNTERS["case_notes_safe_default"] += 1

    return result


@app.get("/metrics")
def metrics():
    return {
        "score_requests": REQUEST_COUNTERS["score_requests"],
        "case_requests": REQUEST_COUNTERS["case_requests"],
        "flagged": REQUEST_COUNTERS["flagged"],
        "clear": REQUEST_COUNTERS["clear"],
        "hold_for_review": REQUEST_COUNTERS["hold_for_review"],
        "case_notes_safe_default": REQUEST_COUNTERS["case_notes_safe_default"],
    }




@app.post("/score")
def score(txn: Txn):
    REQUEST_COUNTERS["score_requests"] += 1
    logger.info("Score request received: txn_id=%s", txn.txn_id)

    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=503,
            detail="Fraud model is not ready",
        )

    category_aliases = {
        "grocery": "Grocery",
        "food delivery": "Food Delivery",
        "fashion": "Fashion",
        "electronics": "Electronics",
        "utilities": "Utilities",
        "travel": "Travel",
    }

    city_aliases = {
        "bengaluru": "Bengaluru",
        "chennai": "Chennai",
        "kolkata": "Kolkata",
        "delhi": "Delhi",
        "pune": "Pune",
        "mumbai": "Mumbai",
    }

    category_key = " ".join(txn.merchant_category.strip().lower().split())
    city_key = " ".join(txn.city.strip().lower().split())
    mode_key = " ".join(txn.payment_mode.strip().lower().split())

    if category_key not in category_aliases:
        raise HTTPException(status_code=400, detail="Unknown merchant category")

    if city_key not in city_aliases:
        raise HTTPException(status_code=400, detail="Unknown city")

    mode_aliases = {
        "upi": "UPI",
        "card": "Card",
        "netbanking": "NetBanking",
        "wallet": "Wallet",
    }

    if mode_key not in mode_aliases:
        raise HTTPException(status_code=400, detail="Unknown payment mode")

    row = pd.DataFrame([{
        "txn_id": txn.txn_id,
        "merchant_category": category_aliases[category_key],
        "city": city_aliases[city_key],
        "payment_mode": mode_aliases[mode_key],
        "amount_inr": txn.amount_inr,
        "hour_of_day": txn.hour_of_day,
        "txn_count_last_24h": txn.txn_count_last_24h,
        "avg_amount_30d_inr": txn.avg_amount_30d_inr,
        "distance_from_home_km": txn.distance_from_home_km,
        "is_new_device": txn.is_new_device,
        "account_age_days": txn.account_age_days,
    }])

    row["distance_from_home_km"] = row["distance_from_home_km"].fillna(
        MODEL_ARTIFACTS["median"]
    )

    row = add_features(row)

    excluded = {"txn_id", "customer_id", "chargeback_filed", "is_fraud"}
    row = row.drop(
        columns=[col for col in excluded if col in row.columns],
        errors="ignore",
    )

    row = pd.get_dummies(
        row,
        columns=["merchant_category", "city", "payment_mode"],
        dtype=int,
    )

    expected_columns = MODEL_ARTIFACTS["columns"]
    row = row.reindex(columns=expected_columns, fill_value=0)

    model = MODEL_ARTIFACTS["model"]
    fraud_probability = float(model.predict_proba(row)[0][1])
    threshold = float(MODEL_ARTIFACTS["threshold"])

    decision = "flagged" if fraud_probability >= threshold else "clear"
    REQUEST_COUNTERS[decision] += 1
    if (
        decision == "clear"
        and txn.amount_inr >= 100_000
        and txn.is_new_device == 1
    ):
        decision = "hold_for_review"
        REQUEST_COUNTERS["hold_for_review"] += 1

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(fraud_probability, 4),
        "threshold": threshold,
        "decision": decision,
    }

