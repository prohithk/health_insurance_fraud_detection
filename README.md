# 🏥 Health Insurance Fraud Detection - FastAPI (Render) + Streamlit

```
 Browser ──► Streamlit app (Streamlit Community Cloud) ──HTTPS + X-API-Key──► FastAPI (Render) ──► model.pkl
              frontend/app.py                                                  backend/fastapi_app.py
```

## Folder structure
```
fraud-detection/
├── .python-version          # 3.12  (Render uses this; Streamlit Cloud: pick 3.12 in Advanced settings)
├── .gitignore
├── render.yaml              # optional Render Blueprint
├── smoke_test.py            # tests the deployed API
├── backend/                 # → Render
│   ├── fastapi_app.py
│   ├── model.pkl            # (≈12 MB, LZMA-compressed - joblib loads it transparently)
│   └── requirements.txt
└── frontend/                # → Streamlit Community Cloud
    ├── app.py
    ├── requirements.txt
    └── .streamlit/secrets.toml.example
```
`train.py` and the CSV are only needed for retraining. Keep them outside `backend/`/`frontend/` (or in a separate folder) and, after retraining, save the new model with `joblib.dump(bundle, "model.pkl", compress=("lzma", 9))` (keeps it under GitHub's 25 MB web-upload limit), put it into `backend/` and commit it.

## Run locally (two terminals)
```
# terminal 1 - backend
cd backend
pip install -r requirements.txt
uvicorn fastapi_app:app --reload                # http://127.0.0.1:8000/docs

# terminal 2 - frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py                            # uses API_URL=http://127.0.0.1:8000 by default
```

## Deploy
1. **GitHub** - push this folder as a repo (`git init`, `git add .`, `git commit`, `git push`).
2. **Render (backend)** - New → Web Service → pick the repo.
   Runtime `Python 3` · Region `Singapore` · Build `pip install -r backend/requirements.txt` ·
   Start `cd backend && uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT --workers 1` ·
   Health check path `/health` · Env var `API_KEY` = a long random string.
3. Check `https://<your-service>.onrender.com/health` and `/docs`, then run
   `python smoke_test.py https://<your-service>.onrender.com <API_KEY>`.
4. **Streamlit Community Cloud (frontend)** - New app → same repo → Main file path `frontend/app.py` →
   Advanced settings: Python 3.12 and Secrets:
   ```
   API_URL = "https://<your-service>.onrender.com"
   API_KEY = "<same key as on Render>"
   ```

## Notes
- **Cold starts:** a free Render service sleeps after 15 minutes without traffic and takes about a minute to wake. The frontend waits for it automatically.
- **Memory:** the loaded model needs ≈400 MB. If Render reports "out of memory", switch the service to the *Standard* (2 GB) instance, or retrain with fewer trees (e.g. `n_estimators=100`).
- **Security:** with `API_KEY` set, `/predict`, `/predict/batch` and `/model-info` need the `X-API-Key` header. `/health` stays open. Never commit the key.
- **Model caveats** (unchanged): precision ≈ 57 %, recall ≈ 54 % for fraud. Use it to *prioritise claims for human review*, never for automatic rejection.
