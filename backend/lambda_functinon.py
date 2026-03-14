import os
import json
import io
import math
import uuid
import traceback
from datetime import datetime, timedelta, timezone

import boto3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Env
# =========================
REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "flight-price-xgb-endpoint")

PLOT_BUCKET = os.getenv("PLOT_BUCKET", "").strip()
PLOT_PREFIX = os.getenv("PLOT_PREFIX", "plots/wingit").strip("/").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/").strip()

INR_TO_KRW = float(os.getenv("INR_TO_KRW", "16.5"))

# Matplotlib cache dir (Lambda read-only 홈 이슈 방지)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# KST
KST = timezone(timedelta(hours=9))

# Clients
smr = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


# =========================
# Helpers
# =========================
def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
        "Content-Type": "application/json",
    }


def _resp(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": _cors_headers(),
        "body": json.dumps(body, ensure_ascii=False),
    }


def _parse_event_body(event) -> dict:
    body = event.get("body", "{}") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        body = base64.b64decode(body).decode("utf-8")

    try:
        return json.loads(body) if body else {}
    except Exception:
        return {}


def _validate_user_input(p: dict) -> dict:
    required = ["origin", "destination", "departure_date", "arrival_date", "stops_count"]
    missing = [k for k in required if k not in p or p[k] in (None, "", [])]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    origin = str(p["origin"]).strip().upper()
    destination = str(p["destination"]).strip().upper()

    departure_date = str(p["departure_date"]).strip()  # YYYY-MM-DD
    arrival_date = str(p["arrival_date"]).strip()      # YYYY-MM-DD

    stops_count = int(p["stops_count"])
    if stops_count < 0:
        stops_count = 0

    return {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "arrival_date": arrival_date,
        "stops_count": stops_count,
    }


def _generate_30_candidates(now_kst: datetime):
    out = []
    for i in range(30):
        day = now_kst.date() + timedelta(days=i)
        purchase_dt = datetime(day.year, day.month, day.day, 9, 0, 0, tzinfo=KST)
        out.append((day.isoformat(), purchase_dt))
    return out


def _parse_prediction_to_float(raw: str) -> float:
    """
    Accept='text/csv' 기준으로 1개 값만 오는 케이스가 일반적이지만,
    JSON/다른 포맷으로 올 가능성에 대비해 방어.
    """
    s = (raw or "").strip()

    # CSV: "0.123" 또는 "0.123,..." 형태
    try:
        return float(s.split(",")[0].strip())
    except Exception:
        pass

    # JSON: {"predictions":[...]} or {"prediction":...} 등
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            if "prediction" in obj:
                return float(obj["prediction"])
            if "predictions" in obj and isinstance(obj["predictions"], list) and obj["predictions"]:
                return float(obj["predictions"][0])
        if isinstance(obj, list) and obj:
            return float(obj[0])
    except Exception:
        pass

    raise RuntimeError(f"Unexpected endpoint response: {s[:200]}")


def _invoke_predict_price_krw(user_input: dict, purchase_dt: datetime) -> float:
    record = {
        "Source": user_input["origin"],
        "Destination": user_input["destination"],
        "Departure Date": user_input["departure_date"],
        "Departure Time": "09:00",
        "Total Time": "0m",
        "Number Of Stops": user_input["stops_count"],
        "purchase_datetime": purchase_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }

    payload = {"records": [record]}

    resp = smr.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="text/csv",
        Body=json.dumps(payload).encode("utf-8"),
    )
    raw = resp["Body"].read().decode("utf-8", errors="replace")

    y_pred_log = _parse_prediction_to_float(raw)

    price_inr = math.expm1(y_pred_log)
    if (not math.isfinite(price_inr)) or price_inr < 0:
        price_inr = 0.0

    price_krw = price_inr * INR_TO_KRW
    return float(round(price_krw))


def _make_trend_plot_png(dates, prices, title="30-day price trend") -> bytes:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(dates, prices, linewidth=2)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("KRW", fontsize=9)

    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=120,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _upload_plot_and_get_url(png_bytes: bytes) -> str:
    if not PLOT_BUCKET:
        raise RuntimeError("PLOT_BUCKET env var is required to upload plot image.")

    key = f"{PLOT_PREFIX}/price_trend_30d_{uuid.uuid4().hex}.png"

    s3.put_object(
        Bucket=PLOT_BUCKET,
        Key=key,
        Body=png_bytes,
        ContentType="image/png",
        CacheControl="max-age=300",
    )

    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/{key}"

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": PLOT_BUCKET, "Key": key},
        ExpiresIn=3600,
    )


# =========================
# Handler
# =========================
def lambda_handler(event, context):
    # CORS preflight
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS" or event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": _cors_headers(), "body": ""}

    try:
        req = _parse_event_body(event)
        user_input = _validate_user_input(req)

        now_kst = datetime.now(tz=KST)
        candidates = _generate_30_candidates(now_kst)

        preds = []
        for date_str, purchase_dt in candidates:
            price = _invoke_predict_price_krw(user_input, purchase_dt)
            preds.append({"date": date_str, "predicted_price": price})

        top3 = sorted(preds, key=lambda x: x["predicted_price"])[:3]
        top3_ranked = [
            {
                "rank": i + 1,
                "date": it["date"],
                "predicted_price": it["predicted_price"],
                "currency": "KRW",
            }
            for i, it in enumerate(top3)
        ]

        # Trend plot (optional upload)
        dates = [p["date"] for p in preds]
        prices = [p["predicted_price"] for p in preds]

        img_url = ""
        trend_error = None
        try:
            if PLOT_BUCKET:
                png = _make_trend_plot_png(dates, prices)
                img_url = _upload_plot_and_get_url(png)
        except Exception as e:
            # 업로드/플롯 실패해도 서비스는 정상 응답 (프론트에서 url 없으면 숨김)
            trend_error = str(e)
            print("TREND_UPLOAD_ERROR:", trend_error)

        out = {
            "user_input": user_input,
            "top_3_cheapest_purchase_times": top3_ranked,
            "price_trend_30d": {
                "image": {"url": img_url},
                "metadata": {"horizon_days": 30},
            },
            "price_unit": "KRW (converted from INR via INR_TO_KRW env)",
            "debug": {
                "endpoint": ENDPOINT_NAME,
                "plot_bucket_set": bool(PLOT_BUCKET),
                "trend_error": trend_error,
            },
        }

        return _resp(200, out)

    except Exception as e:
        # CloudWatch에서 원인 추적
        print("ERROR:", str(e))
        print(traceback.format_exc())
        return _resp(
            400,
            {"message": str(e), "hint": "Check request body fields and endpoint status/permissions."},
        )