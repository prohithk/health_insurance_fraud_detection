"""
frontend/app.py - Health Insurance Fraud Detection (Streamlit FRONTEND)

This app contains NO model. It only shows a form and sends the claim to the FastAPI backend
(hosted on Render), then displays the answer. The dropdown options and number ranges are read from
the backend's /model-info endpoint, so the frontend and the model can never get out of sync.

Settings (Streamlit secrets, or environment variables when running locally):
    API_URL   URL of the backend, e.g. https://fraud-detection-api.onrender.com   (default: http://127.0.0.1:8000)
    API_KEY   optional; must be the same value as API_KEY on the backend

Run locally:  streamlit run app.py
"""

import os
import time
from datetime import date

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Health Insurance Fraud Detection", page_icon="🏥", layout="wide")

UNKNOWN = "Unknown (not available)"   # label for fields that can be missing in the training data


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
def get_setting(name, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:       # no secrets file when running locally
        pass
    return os.getenv(name, default)


API_URL = str(get_setting("API_URL", "http://127.0.0.1:8000")).strip().rstrip("/")
API_KEY = get_setting("API_KEY")


def auth_headers(api_key=None):
    key = api_key if api_key is not None else API_KEY
    return {"X-API-Key": str(key)} if key else {}


# --------------------------------------------------------------------------------------
# Talking to the backend
# --------------------------------------------------------------------------------------
def wake_backend(max_wait=120):
    """Free Render services go to sleep when idle and need about a minute to wake up.
    Poll /health until the model is loaded. Returns (ok, error_text)."""
    deadline = time.time() + max_wait
    last_error = "no response"
    while time.time() < deadline:
        try:
            response = requests.get(f"{API_URL}/health", timeout=15)
            if response.status_code == 200 and response.json().get("model_loaded"):
                return True, None
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except (requests.RequestException, ValueError) as error:   # ValueError: Render's "waking up" HTML page
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(3)
    return False, last_error


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_info(api_url, api_key):
    response = requests.get(f"{api_url}/model-info", headers=auth_headers(api_key), timeout=30)
    response.raise_for_status()
    return response.json()


def post_predict(payload):
    """POST /predict. If the free backend fell asleep meanwhile, wake it and try once more."""
    last_response = None
    for attempt in range(2):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, headers=auth_headers(), timeout=90)
        except (requests.ConnectionError, requests.Timeout):
            response = None
        if response is not None and response.status_code not in (502, 503, 504):
            return response
        last_response = response
        if attempt == 0:
            ok, _ = wake_backend()
            if not ok:
                break
    return last_response


def show_api_error(response):
    """Show a readable message for any non-200 answer from the backend."""
    if response is None:
        st.error("The prediction service did not answer. It may still be waking up - please try again in a minute.")
        return
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text[:500]
    if response.status_code == 401:
        st.error("The backend rejected the API key. Check that `API_KEY` in the Streamlit secrets "
                 "is the same as `API_KEY` on Render.")
    elif isinstance(detail, list):
        for item in detail:
            if isinstance(item, dict):                      # pydantic validation error
                where = ".".join(str(part) for part in item.get("loc", [])[1:])
                st.error(f"{where}: {item.get('msg')}")
            else:                                           # business-rule message from the API
                st.error(str(item))
    else:
        st.error(f"Backend error {response.status_code}: {detail}")


# --------------------------------------------------------------------------------------
# Connect to the backend and load the form options (once per browser session)
# --------------------------------------------------------------------------------------
if not st.session_state.get("backend_ready"):
    with st.spinner("Connecting to the prediction service... (if it was asleep this can take up to a minute)"):
        ready, problem = wake_backend()
    if not ready:
        st.error(f"**Cannot reach the backend** at `{API_URL}`.\n\nDetails: `{problem}`\n\n"
                 "Check that the Render service is running and that `API_URL` is set correctly in the secrets.")
        st.button("Try again")
        st.stop()
    st.session_state["backend_ready"] = True

try:
    info = fetch_model_info(API_URL, API_KEY)
except requests.HTTPError as error:
    if error.response is not None and error.response.status_code == 401:
        st.error("The backend rejected the API key. Check that `API_KEY` in the Streamlit secrets "
                 "is the same as `API_KEY` on Render.")
    else:
        st.error(f"Could not load the model information from the backend: `{error}`")
    st.stop()
except requests.RequestException as error:
    st.session_state["backend_ready"] = False
    st.error(f"Lost the connection to the backend: `{type(error).__name__}: {error}`. Refresh the page to retry.")
    st.stop()

CATEGORIES = info["categorical_allowed_values"]
RANGES = {column: (limits["min"], limits["max"]) for column, limits in info["numeric_ranges"].items()}
SUM_INSURED_OPTIONS = info["sum_insured_allowed_values"]
YEAR_MIN, YEAR_MAX = info["claim_year_range"]
MISSING_OK = info["optional_when_unknown"]
MISSING_NUMERIC = [c for c in MISSING_OK if c in RANGES]
MISSING_CATEGORICAL = [c for c in MISSING_OK if c in CATEGORIES]


def int_range(column):
    low, high = RANGES[column]
    return int(low), int(high)


def float_range(column):
    low, high = RANGES[column]
    return float(low), float(high)


# Starting values for the form (all inside the ranges the model saw in training)
DEFAULTS = {
    "state": CATEGORIES["state"][0],
    "gender": CATEGORIES["gender"][0],
    "patient_age": 42,
    "insurance_type": CATEGORIES["insurance_type"][0],
    "sum_insured": 500000 if 500000 in SUM_INSURED_OPTIONS else SUM_INSURED_OPTIONS[0],
    "policy_age_days": 700,
    "previous_claims": 1,
    "claims_last_12_months": 1,
    "previous_rejections": 0,
    "hospital_type": CATEGORIES["hospital_type"][0],
    "admission_type": CATEGORIES["admission_type"][0],
    "treatment_type": CATEGORIES["treatment_type"][0],
    "diagnosis_code": CATEGORIES["diagnosis_code"][0],
    "room_type": CATEGORIES["room_type"][0],
    "length_of_stay": 4,
    "doctor_experience_years": 13,
    "claim_date": date(YEAR_MAX, 6, 15),
    "total_bill_amount": 175000.0,
    "claimed_amount": 150000.0,
    "payment_method": CATEGORIES["payment_method"][0],
}
for column in MISSING_NUMERIC:
    DEFAULTS[f"unknown_{column}"] = False
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def number_with_unknown(label, column):
    """Integer input that can be switched to 'unknown' (the model imputes missing values)."""
    low, high = int_range(column)
    box, check = st.columns([3, 2])
    unknown = st.session_state.get(f"unknown_{column}", False)
    box.number_input(label, min_value=low, max_value=high, step=1, key=column, disabled=unknown)
    check.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    check.checkbox("Unknown", key=f"unknown_{column}")


def select_with_unknown(label, column):
    """Dropdown with an extra 'Unknown' option for columns that can be missing in the data."""
    options = [UNKNOWN] + CATEGORIES[column] if column in MISSING_CATEGORICAL else CATEGORIES[column]
    return st.selectbox(label, options, key=column)


# --------------------------------------------------------------------------------------
# Header + sidebar
# --------------------------------------------------------------------------------------
st.title("🏥 Health Insurance Fraud Detection")

with st.sidebar:
    st.header("About the model")
    st.write(f"**{info['model_name']}**")
    metrics = info.get("test_set_metrics") or {}
    if metrics:
        st.caption("Test-set results")
        st.table({name: f"{value:.3f}" for name, value in metrics.items()})
    st.caption("Decision-support aid for **prioritising claims for review**. "
               "Never use it alone to reject a claim or accuse anyone of fraud.")

# --------------------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------------------
st.subheader("1. Patient and policy")
a, b, c = st.columns(3)
with a:
    st.selectbox("State", CATEGORIES["state"], key="state")
    st.selectbox("Gender", CATEGORIES["gender"], key="gender")
    st.number_input("Patient age (years)", *int_range("patient_age"), step=1, key="patient_age")
with b:
    st.selectbox("Insurance type", CATEGORIES["insurance_type"], key="insurance_type")
    st.selectbox("Sum insured (₹)", SUM_INSURED_OPTIONS, key="sum_insured", format_func=lambda v: f"{v:,}")
    st.number_input("Policy age (days)", *int_range("policy_age_days"), step=1, key="policy_age_days")
with c:
    st.number_input("Previous claims", *int_range("previous_claims"), step=1, key="previous_claims")
    st.number_input("Claims in last 12 months", *int_range("claims_last_12_months"), step=1,
                    key="claims_last_12_months")
    number_with_unknown("Previous rejections", "previous_rejections")

st.subheader("2. Hospital and treatment")
a, b, c = st.columns(3)
with a:
    st.selectbox("Hospital type", CATEGORIES["hospital_type"], key="hospital_type")
    st.selectbox("Admission type", CATEGORIES["admission_type"], key="admission_type")
    st.date_input("Claim date", min_value=date(YEAR_MIN, 1, 1), max_value=date(YEAR_MAX, 12, 31),
                  key="claim_date", help="Only the year and month are used by the model.")
with b:
    st.selectbox("Treatment type", CATEGORIES["treatment_type"], key="treatment_type")
    select_with_unknown("Diagnosis code", "diagnosis_code")
    select_with_unknown("Room type", "room_type")
with c:
    st.number_input("Length of stay (days)", *int_range("length_of_stay"), step=1, key="length_of_stay")
    number_with_unknown("Doctor experience (years)", "doctor_experience_years")

st.subheader("3. Billing and claim")
a, b, c = st.columns(3)
with a:
    bill_low, bill_high = float_range("total_bill_amount")
    st.number_input("Total bill amount (₹)", min_value=bill_low, max_value=bill_high, step=1000.0,
                    format="%.2f", key="total_bill_amount")
with b:
    claim_low, claim_high = float_range("claimed_amount")
    st.number_input("Claimed amount (₹)", min_value=claim_low, max_value=claim_high, step=1000.0,
                    format="%.2f", key="claimed_amount")
with c:
    st.selectbox("Payment method", CATEGORIES["payment_method"], key="payment_method")

ratio = round(st.session_state["claimed_amount"] / st.session_state["sum_insured"], 4)
st.caption(f"Calculated by the API: claim-to-coverage ratio = claimed amount ÷ sum insured = **{ratio:.4f}**")

# ---------------- Input validation (the API checks again; this just gives instant feedback) ----------------
problems = []
if st.session_state["claimed_amount"] > st.session_state["total_bill_amount"]:
    problems.append("The claimed amount cannot be larger than the total bill amount "
                    "(this never happens in the training data).")
if st.session_state["claimed_amount"] > st.session_state["sum_insured"]:
    problems.append("The claimed amount cannot be larger than the sum insured "
                    "(this never happens in the training data).")
for message in problems:
    st.error(message)

predict_clicked = st.button("🔎 Predict", type="primary", disabled=bool(problems))


# --------------------------------------------------------------------------------------
# Prediction (calls the backend)
# --------------------------------------------------------------------------------------
def build_payload():
    state = st.session_state

    def value(column):
        if column in MISSING_NUMERIC and state.get(f"unknown_{column}", False):
            return None
        if column in MISSING_CATEGORICAL and state[column] == UNKNOWN:
            return None
        return state[column]

    def optional_int(column):
        v = value(column)
        return None if v is None else int(v)

    return {
        "state": state["state"],
        "gender": state["gender"],
        "patient_age": int(state["patient_age"]),
        "insurance_type": state["insurance_type"],
        "sum_insured": int(state["sum_insured"]),
        "policy_age_days": int(state["policy_age_days"]),
        "previous_claims": int(state["previous_claims"]),
        "claims_last_12_months": int(state["claims_last_12_months"]),
        "previous_rejections": optional_int("previous_rejections"),
        "hospital_type": state["hospital_type"],
        "admission_type": state["admission_type"],
        "treatment_type": state["treatment_type"],
        "diagnosis_code": value("diagnosis_code"),
        "room_type": value("room_type"),
        "length_of_stay": int(state["length_of_stay"]),
        "doctor_experience_years": optional_int("doctor_experience_years"),
        "claim_date": state["claim_date"].isoformat(),
        "total_bill_amount": float(state["total_bill_amount"]),
        "claimed_amount": float(state["claimed_amount"]),
        "payment_method": state["payment_method"],
    }


if predict_clicked:
    payload = build_payload()
    with st.spinner("Asking the model..."):
        response = post_predict(payload)

    if response is None or response.status_code != 200:
        show_api_error(response)
        st.stop()

    result = response.json()
    predicted_class = result["predicted_class"]
    fraud_probability = result["fraud_probability"]
    base_rate = result["average_fraud_rate_in_training_data"]
    threshold = result["threshold"]

    st.divider()
    st.subheader("Result")
    left, right = st.columns([1, 1])

    with left:
        if predicted_class == 1:
            st.error("### 🚩 Predicted: FRAUD\nThis claim looks suspicious and should be reviewed.")
        else:
            st.success("### ✅ Predicted: GENUINE\nThis claim does not look suspicious to the model.")

        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{fraud_probability:.1%}",
                  delta=f"{(fraud_probability - base_rate) * 100:+.1f} pts vs. average claim",
                  delta_color="inverse")
        m2.metric("Genuine probability", f"{result['genuine_probability']:.1%}")
        st.caption(f"In the training data, {base_rate:.1%} of all claims were fraudulent. "
                   f"The class is 'Fraud' when the fraud probability is above {threshold:.0%}.")

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Fraud probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728" if predicted_class == 1 else "#2ca02c"},
                "steps": [{"range": [0, threshold * 100], "color": "#e8f5e9"},
                          {"range": [threshold * 100, 100], "color": "#fdecea"}],
                "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
            },
        ))
        gauge.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(gauge)

    with st.expander("Show the request sent to the API and its answer"):
        st.caption("Request body (POST /predict)")
        st.json(payload)
        st.caption("Response")
        st.json(result)
