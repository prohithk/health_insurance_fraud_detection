"""
smoke_test.py - quick check that the deployed backend works.

    python smoke_test.py https://fraud-detection-api.onrender.com  YOUR_API_KEY

(The API key argument is optional - leave it out if you did not set API_KEY.)
The first call can take about a minute if the free Render service was asleep.
Needs only:  pip install requests
"""
import sys
import requests

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
headers = {"X-API-Key": sys.argv[2]} if len(sys.argv) > 2 else {}

claim = {
    "state": "Maharashtra", "gender": "Female", "patient_age": 42, "insurance_type": "Individual",
    "sum_insured": 500000, "policy_age_days": 700, "previous_claims": 1, "claims_last_12_months": 1,
    "previous_rejections": 0, "hospital_type": "Private", "admission_type": "Planned",
    "treatment_type": "Surgery", "diagnosis_code": "DIA-05", "room_type": "Semi-Private",
    "length_of_stay": 4, "doctor_experience_years": 13, "claim_date": "2026-06-15",
    "total_bill_amount": 175000.0, "claimed_amount": 150000.0, "payment_method": "Cashless",
}

print("1. /health      ->", end=" ")
r = requests.get(f"{base}/health", timeout=120)
print(r.status_code, r.json())

print("2. /predict     ->", end=" ")
r = requests.post(f"{base}/predict", json=claim, headers=headers, timeout=120)
print(r.status_code, r.json())

print("3. unknown values (diagnosis_code / room_type / doctor_experience_years / previous_rejections = null) ->", end=" ")
r = requests.post(f"{base}/predict", json={**claim, "diagnosis_code": None, "room_type": None,
                  "doctor_experience_years": None, "previous_rejections": None}, headers=headers, timeout=120)
print(r.status_code, r.json().get("predicted_label"))

print("4. invalid claim (claimed > bill) should be 422 ->", end=" ")
r = requests.post(f"{base}/predict", json={**claim, "claimed_amount": 400000.0}, headers=headers, timeout=120)
print(r.status_code, r.json())

if headers:
    print("5. wrong API key should be 401 ->", end=" ")
    r = requests.post(f"{base}/predict", json=claim, headers={"X-API-Key": "wrong"}, timeout=120)
    print(r.status_code)
