# Deployment Guide

Two services to deploy:
1. **FastAPI** (the brain) → Railway or Render (free)
2. **Streamlit** (the face) → Streamlit Community Cloud (free)

---

## Step 1 — Pre-deployment check

```bash
# Seed clean demo data first
python scripts/reset_demo_data.py

# Run all deployment checks
python scripts/deploy_check.py
```

Fix any ❌ failures before continuing.

---

## Step 2 — Deploy FastAPI to Railway (recommended, ~10 min)

### 2a. Install Railway CLI
```bash
# Windows (PowerShell)
npm install -g @railway/cli

# Mac/Linux
brew install railway
```

### 2b. Login and deploy
```bash
cd C:\Users\harsh\healthcare-ai-agents

railway login        # opens browser, sign in with GitHub
railway init         # creates a new Railway project
railway up           # deploys using Dockerfile
```

### 2c. Set environment variables in Railway dashboard
Go to `railway.app` → your project → Variables → add these:

```
OPENAI_API_KEY      = sk-your-key-here
MODEL_NAME          = gpt-4o-mini
CONFIDENCE_THRESHOLD = 0.75
DATABASE_URL        = sqlite:///./data/healthcare_agents.db
USE_FHIR            = true
FHIR_BASE_URL       = https://hapi.fhir.org/baseR4
LANGCHAIN_TRACING_V2 = true
LANGCHAIN_API_KEY   = ls__your-key-here
```

### 2d. Get your Railway URL
Railway gives you a URL like:
`https://healthcare-ai-agents-production.up.railway.app`

Test it:
```bash
python scripts/deploy_check.py --api-url https://YOUR-URL.up.railway.app
```

You should see: `✅ Remote API health — Version: 1.0.0`

---

## Step 2 (alternative) — Deploy FastAPI to Render

1. Go to `render.com` → New → Web Service
2. Connect your GitHub repo `harshinireddy2204/healthcare-ai-agents`
3. Render detects `render.yaml` automatically — click **Apply**
4. Set `OPENAI_API_KEY` manually in the Environment tab (not in yaml for security)
5. First deploy takes ~5 minutes (installs deps, seeds database)
6. Free tier spins down after 15min inactivity — first request after sleep takes ~30s

---

## Step 3 — Deploy Streamlit to Community Cloud (~5 min)

### 3a. Go to share.streamlit.io
Sign in with your GitHub account.

### 3b. Deploy new app
- Repository: `harshinireddy2204/healthcare-ai-agents`
- Branch: `main`
- Main file path: `frontend/app.py`
- Click **Deploy**

### 3c. Set secrets in Streamlit Cloud
App Settings (⚙️ top right) → **Secrets** → paste:

```toml
API_BASE_URL = "https://YOUR-RAILWAY-OR-RENDER-URL.com"
```

That's it. Streamlit reads this automatically via `st.secrets`.

### 3d. Your public URL
Streamlit gives you a URL like:
`https://harshinireddy2204-healthcare-ai-agents-frontendapp-xxxxx.streamlit.app`

You can customize it in App Settings → **Custom subdomain**:
`healthcare-ai-demo.streamlit.app`

---

## Step 4 — Verify end-to-end

```bash
# Check both services are healthy
python scripts/deploy_check.py --api-url https://YOUR-RAILWAY-URL.com
```

Then open the Streamlit URL and verify:
- [ ] Sidebar shows "API Online" (green dot)
- [ ] Live Overview shows 10 demo patient runs
- [ ] Pending Reviews shows P004 and P014
- [ ] Knowledge Graph tab loads (93 nodes)
- [ ] FHIR tab can search patients

---

## Architecture diagram

```
LinkedIn visitors
      ↓
https://healthcare-ai-demo.streamlit.app
(Streamlit Community Cloud — free, always on)
      ↓  HTTP requests
https://healthcare-ai-xxxxx.up.railway.app
(Railway — FastAPI, free tier, always on)
      ↓
OpenFDA API     FHIR hapi.fhir.org     OpenAI API
(real-time)     (real-time FHIR R4)    (LLM calls)
```

---

## Cost breakdown

| Service | Cost | Limits |
|---|---|---|
| Railway (API) | Free | 500 hours/month (~20 days) |
| Render (API alt) | Free | Sleeps after 15min inactivity |
| Streamlit Cloud | Free | Unlimited |
| OpenAI API | Pay-per-use | ~$0.002/patient run (gpt-4o-mini) |
| OpenFDA | Free | 240 req/min, no key needed |
| FHIR (HAPI) | Free | Public test server |

For a LinkedIn demo with ~100-200 users triggering workflows, estimated OpenAI cost: **$2-5 total**.

---

## Keeping Railway alive past the free tier

Railway's free tier gives 500 hours/month. If you exceed it:

Option 1: Add a credit card (Railway charges ~$5/month for always-on)
Option 2: Use Render free tier (never expires, just sleeps)
Option 3: Upgrade to Railway Hobby ($5/month, unlimited hours)

For a LinkedIn post driving traffic for 1-2 weeks, the free tier is sufficient.

---

## Redeploying after code changes

```bash
# Make changes, commit, push
git add .
git commit -m "fix: update care gap report format"
git push origin main

# Railway auto-deploys on push (if connected to GitHub)
# OR manually:
railway up
```

Streamlit Cloud also auto-deploys when you push to `main`.