
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from train import build_final_model
from prepare import add_features
from casenote import run_case


LOG_FILE = Path(__file__).with_name("service.log")

logger = logging.getLogger("fraud_triage")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)


MODEL_ARTIFACTS = {}

METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "human_review_total": 0,
    "latency_total_ms": 0.0,
}

CATEGORY_ALIASES = {
    "grocery": "Grocery",
    "food delivery": "Food Delivery",
    "fashion": "Fashion",
    "electronics": "Electronics",
    "utilities": "Utilities",
    "travel": "Travel",
}

CITY_ALIASES = {
    "bengaluru": "Bengaluru",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "delhi": "Delhi",
    "pune": "Pune",
    "mumbai": "Mumbai",
}

MODE_ALIASES = {
    "upi": "UPI",
    "card": "Card",
    "netbanking": "NetBanking",
    "net banking": "NetBanking",
    "wallet": "Wallet",
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
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = Field(default=None, ge=0)
    is_new_device: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)


class CaseRequest(Txn):
    question: str = Field(min_length=3)


def normalized(value: str) -> str:
    return " ".join(value.strip().lower().split())


def transaction_features(txn: Txn) -> pd.DataFrame:
    category_key = normalized(txn.merchant_category)
    city_key = normalized(txn.city)
    mode_key = normalized(txn.payment_mode)

    if category_key not in CATEGORY_ALIASES:
        raise ValueError("merchant_category")
    if city_key not in CITY_ALIASES:
        raise ValueError("city")
    if mode_key not in MODE_ALIASES:
        raise ValueError("payment_mode")

    row = pd.DataFrame([{
        "txn_id": txn.txn_id,
        "merchant_category": CATEGORY_ALIASES[category_key],
        "city": CITY_ALIASES[city_key],
        "payment_mode": MODE_ALIASES[mode_key],
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
        columns=[column for column in excluded if column in row.columns],
        errors="ignore",
    )
    row = pd.get_dummies(
        row,
        columns=["merchant_category", "city", "payment_mode"],
        dtype=int,
    )
    return row.reindex(columns=MODEL_ARTIFACTS["columns"], fill_value=0)

def score_transaction(txn: Txn) -> dict:
    if not MODEL_ARTIFACTS:
        raise RuntimeError("Fraud model is not ready")

    # Reject impossible values before model scoring.
    if not 1 <= txn.amount_inr <= 1_000_000:
        raise ValueError("amount_inr")

    if not 0 <= txn.hour_of_day <= 23:
        raise ValueError("hour_of_day")

    if txn.txn_count_last_24h < 0:
        raise ValueError("txn_count_last_24h")

    if txn.avg_amount_30d_inr <= 0:
        raise ValueError("avg_amount_30d_inr")

    features = transaction_features(txn)

    probability = float(
        MODEL_ARTIFACTS["model"].predict_proba(features)[0][1]
    )
    threshold = float(MODEL_ARTIFACTS["threshold"])
    flagged = probability >= threshold

    # High-value payment on a new device needs human review.
    large_payment_hold = (
        not flagged
        and txn.amount_inr >= 100_000
        and txn.is_new_device == 1
    )

    human_review = flagged or large_payment_hold
    decision = "hold_for_review" if human_review else "approve"

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(probability, 4),
        "threshold": threshold,
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
    }


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {"error": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    errors = exc.errors()
    first = errors[0] if errors else {}
    location = first.get("loc", ())
    field = str(location[-1]) if location else "request"
    return JSONResponse(
        status_code=422,
        content={"error": "Request validation failed", "field": field},
    )


@app.middleware("http")
async def request_metrics_and_logging(request: Request, call_next):
    tracked = (
        request.method == "POST"
        and request.url.path in {"/score", "/case"}
    )
    if not tracked:
        return await call_next(request)

    started = time.perf_counter()
    body = b""
    txn_id = "unknown"
    probability = None

    try:
        body = await request.body()
        try:
            parsed = json.loads(body)
            txn_id = str(parsed.get("txn_id", "unknown"))
        except (ValueError, AttributeError):
            parsed = {}

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
        response = await call_next(request)

        status = response.status_code
        latency_ms = (time.perf_counter() - started) * 1000
        response_body = b""

        async for chunk in response.body_iterator:
            response_body += chunk

        try:
            result = json.loads(response_body) if response_body else {}
        except (ValueError, TypeError):
            result = {}

        METRICS["requests_total"] += 1
        METRICS["latency_total_ms"] += latency_ms

        if status < 400:
            METRICS["success_total"] += 1
        else:
            METRICS["error_total"] += 1

        if result.get("human_review") is True:
            METRICS["human_review_total"] += 1

        if "fraud_probability" in result:
            probability = result["fraud_probability"]

        level = logging.INFO
        if status >= 500:
            level = logging.ERROR
        elif status >= 400 or result.get("safe_default"):
            level = logging.WARNING
        elif result.get("human_review") and not result.get("flagged"):
            level = logging.WARNING

        logger.log(
            level,
            "method=%s endpoint=%s status=%s latency_ms=%.2f txn_id=%s fraud_probability=%s",
            request.method,
            request.url.path,
            status,
            latency_ms,
            txn_id,
            probability if probability is not None else "n/a",
        )

        from starlette.responses import Response
        return Response(
            content=response_body,
            status_code=status,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() not in {"content-length", "content-type"}
            },
            media_type="application/json",
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        METRICS["requests_total"] += 1
        METRICS["error_total"] += 1
        METRICS["latency_total_ms"] += latency_ms
        logger.error(
            "method=%s endpoint=%s status=503 latency_ms=%.2f txn_id=%s error=%s",
            request.method,
            request.url.path,
            latency_ms,
            txn_id,
            type(exc).__name__,
        )
        if request.url.path == "/case":
            return JSONResponse(
                status_code=503,
                content={
                    "error": "case-note service temporarily unavailable, try again later"
                },
            )
        return JSONResponse(
            status_code=503,
            content={"error": "service temporarily unavailable"},
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    total = METRICS["requests_total"]
    latency = (
        METRICS["latency_total_ms"] / total if total else 0.0
    )
    return {
        "requests_total": total,
        "success_total": METRICS["success_total"],
        "error_total": METRICS["error_total"],
        "success_rate": round(METRICS["success_total"] / total, 4) if total else 0.0,
        "error_rate": round(METRICS["error_total"] / total, 4) if total else 0.0,
        "human_review_total": METRICS["human_review_total"],
        "human_review_rate": round(METRICS["human_review_total"] / total, 4) if total else 0.0,
        "avg_latency_ms": round(latency, 2),
    }


@app.post("/score")
def score(txn: Txn):
    try:
        return score_transaction(txn)
    except ValueError as exc:
        field = str(exc)
        messages = {
            "amount_inr": "amount_inr must be between 1 and 1000000",
            "hour_of_day": "hour_of_day must be between 0 and 23",
            "txn_count_last_24h": "txn_count_last_24h must be non-negative",
            "avg_amount_30d_inr": "avg_amount_30d_inr must be greater than 0",
            "merchant_category": "Unknown merchant category",
            "city": "Unknown city",
            "payment_mode": "Unknown payment mode",
        }
        return JSONResponse(
            status_code=400,
            content={"error": messages.get(field, "Invalid transaction"), "field": field},
        )
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={"error": "Fraud model is not ready"},
        )


@app.post("/case")
def create_case(request: CaseRequest):
    try:
        scored = score_transaction(request)
    except ValueError as exc:
        field = str(exc)
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid transaction field: {field}", "field": field},
        )
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={"error": "Fraud model is not ready"},
        )

    result = run_case(
        txn_id=request.txn_id,
        fraud_probability=scored["fraud_probability"],
        amount_inr=request.amount_inr,
        question=request.question,
    )

    note = result.get("case_note")
    action = note.get("recommended_action") if note else None
    if action == "block_and_notify":
        scored["decision"] = "hold_for_review"
        scored["human_review"] = True
    elif action == "hold_for_review":
        scored["decision"] = "hold_for_review"
        scored["human_review"] = True
    elif result.get("safe_default"):
        scored["decision"] = "hold_for_review"
        scored["human_review"] = True

    return {
        **scored,
        "case_note": note,
        "status": result.get("status"),
        "attempts": result.get("attempts", 0),
        "failures": result.get("failures", []),
        "safe_default": result.get("safe_default", False),
        "guardrail_override": result.get("guardrail_override", False),
        "retrieved_policy_id": result.get("retrieved_policy_id"),
        "retrieval_score": result.get("retrieval_score"),
    }
