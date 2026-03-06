# 🚀 Chat Toxicity Detector — Production Deployment Guide

## Architecture

```
┌─────────────────────┐        API calls        ┌──────────────────────────┐
│   FRONTEND (React)  │  ───────────────────►    │   BACKEND (FastAPI)      │
│   Hosted on Vercel  │  ◄───────────────────    │   Hosted on Render       │
│   Port: 443 (HTTPS) │      JSON responses      │   Port: $PORT (HTTPS)    │
└─────────────────────┘                          └──────────┬───────────────┘
                                                            │
                                                   ┌────────▼────────┐
                                                   │  DistilBERT     │
                                                   │  PyTorch Model  │
                                                   │  (loaded once)  │
                                                   └─────────────────┘
```

---

## 📁 Final Project Structure

```
Chat Toxicity Detector2/
│
├── app/
│   ├── backend/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── config.py            # Reads $PORT, $CORS_ALLOWED env vars
│   │   ├── model.py             # ToxicityAnalyzer (DistilBERT)
│   │   ├── schemas.py           # Pydantic models
│   │   ├── requirements.txt     # Production Python deps
│   │   └── Dockerfile           # Container build (optional)
│   │
│   └── frontend/
│       ├── package.json
│       ├── vite.config.js
│       ├── index.html
│       ├── public/
│       │   └── tox_log.png      # App logo / favicon
│       └── src/
│           ├── App.jsx
│           ├── main.jsx
│           ├── hooks/
│           │   ├── useAnalyze.js # Uses VITE_API_URL env var
│           │   └── useStats.js   # Uses VITE_API_URL env var
│           ├── components/
│           └── styles/
│
├── ml/                          # Research scripts (NOT deployed)
│   ├── phase1_eda.py
│   ├── phase3_pipeline.py
│   ├── phase4_models.py
│   └── phase5_evaluation.py
│
├── data/                        # Raw datasets (NOT deployed)
├── outputs/                     # Model checkpoints & eval results
│   ├── phase3/
│   ├── phase4/
│   │   ├── saved_models/        # ← Backend loads weights from here
│   │   └── tuned_thresholds.json
│   └── phase5/
│
├── docs/
│   ├── DEPLOYMENT_GUIDE.md      # This file
│   └── images/
│
├── README.md
└── .gitignore
```

---

## 🖥️ STEP 1 — Deploy Backend to Render

### 1.1 Push to GitHub
```bash
cd "d:\Project\Chat\Chat Toxicity Detector2"
git init
git add .
git commit -m "production-ready: restructured for deployment"
git remote add origin https://github.com/YOUR_USERNAME/chat-toxicity-detector.git
git push -u origin main
```

### 1.2 Create Render Web Service
1. Go to [render.com](https://render.com) → **New +** → **Web Service**
2. Connect your GitHub repository
3. Configure these settings:

| Setting | Value |
|---|---|
| **Name** | `toxicity-detector-api` |
| **Region** | Oregon (US West) or closest to you |
| **Root Directory** | `app/backend` |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | **Starter+ ($7/mo)** or higher — needs ≥2GB RAM |

### 1.3 Set Environment Variables on Render
In Render Dashboard → Your Service → **Environment** tab:

| Variable | Value |
|---|---|
| `PYTHON_VERSION` | `3.11.6` |
| `CORS_ALLOWED` | `https://YOUR-APP.vercel.app` *(add after frontend deploy)* |

### 1.4 Verify Backend
After deploy completes (~5-10 min for first build), test:
```
curl https://toxicity-detector-api.onrender.com/health
```
Expected response:
```json
{"status":"ok","model_loaded":true,"device":"cpu","model_name":"distilbert-base-uncased"}
```

> ⚠️ **IMPORTANT:** Render free tier spins down after 15min of inactivity.
> The first request after sleep takes ~30s to cold-start the PyTorch model.
> Use the **Starter plan ($7/mo)** for always-on behavior.

---

## 🌐 STEP 2 — Deploy Frontend to Vercel

### 2.1 Create Vercel Project
1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repository
3. Configure:

| Setting | Value |
|---|---|
| **Framework Preset** | `Vite` |
| **Root Directory** | `app/frontend` |
| **Build Command** | `npm run build` (auto-detected) |
| **Output Directory** | `dist` (auto-detected) |

### 2.2 Set Environment Variables on Vercel
In Vercel Dashboard → Your Project → **Settings** → **Environment Variables**:

| Variable | Value |
|---|---|
| `VITE_API_URL` | `https://toxicity-detector-api.onrender.com` |

> 🔑 **The `VITE_` prefix is required.** Vite only exposes env vars starting with `VITE_` to the client bundle.

### 2.3 Deploy
Click **Deploy**. Vercel will:
1. Run `npm install` in `app/frontend`
2. Run `npm run build` (Vite compiles to static files)
3. Serve those static files on a CDN with HTTPS

Your live URL will be: `https://YOUR-APP.vercel.app`

### 2.4 Update CORS on Render
Go back to Render → Environment Variables → set:
```
CORS_ALLOWED=https://YOUR-APP.vercel.app
```
The backend will restart automatically and allow requests from your Vercel frontend.

---

## ✅ STEP 3 — Deployment Verification Checklist

Run through this checklist after both services are live:

| # | Check | Status |
|---|---|---|
| 1 | `GET /health` returns `{"model_loaded": true}` | ☐ |
| 2 | `POST /analyze` with `{"message": "hello"}` returns severity `"clean"` | ☐ |
| 3 | `POST /analyze` with `{"message": "you are terrible"}` returns `is_toxic: true` | ☐ |
| 4 | Frontend loads with logo and 4 tabs visible | ☐ |
| 5 | Status indicator shows "Model Ready" (green dot) | ☐ |
| 6 | Typing a message in Detector tab returns a result | ☐ |
| 7 | Session Stats tab shows live data after sending messages | ☐ |
| 8 | Model Perf tab renders charts | ☐ |
| 9 | Bias Audit tab renders gauges and cards | ☐ |
| 10 | No CORS errors in browser console | ☐ |

---

## 🔒 Security Hardening (Post-Deploy)

1. **Remove wildcard CORS** — Already done. `config.py` reads `CORS_ALLOWED` env var
2. **No `--reload`** — Start command uses plain `uvicorn`, no `--reload` flag
3. **Gradients disabled** — `torch.set_grad_enabled(False)` is set after `model.eval()`
4. **Model loads once** — Loaded in FastAPI `lifespan` startup, not per-request
5. **HTTPS everywhere** — Both Vercel and Render provide automatic SSL

---

## 🐳 Alternative: Docker Deployment

If you prefer Docker (for AWS ECS, GCP Cloud Run, etc.):

```bash
cd app/backend
docker build -t toxicity-api .
docker run -p 8000:8000 -e PORT=8000 toxicity-api
```

The existing `Dockerfile` in `app/backend/` handles this.
