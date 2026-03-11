---
title: Travel Prophet Forecaster
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# ✈️ Travel Demand Forecaster — Prophet + Streamlit

A time series forecasting app for travel demand using Facebook Prophet, deployed via CI/CD to Hugging Face Spaces.


---

## 🚀 CI/CD Pipeline

```
Push code to GitHub
        ↓
GitHub Actions triggers
        ↓
🧪 Run all tests (CI)
        ↓
✅ Tests pass?
        ↓
🤗 Auto deploy to Hugging Face (CD)
        ↓
🌐 App is live!
```

---

## 🛠️ Setup Instructions

### Step 1 — Fork or clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/travel-prophet.git
cd travel-prophet
```

### Step 2 — Create Hugging Face Space
1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Choose **Streamlit** as SDK
4. Name it e.g. `travel-prophet-forecaster`

### Step 3 — Add HF Token to GitHub Secrets
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **write** access
3. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `HF_TOKEN`, Value: (paste your token)

### Step 4 — Update deploy.yml
In `.github/workflows/deploy.yml`, replace:
```
YOUR_HF_USERNAME  →  your Hugging Face username
YOUR_SPACE_NAME   →  your Space name
```

### Step 5 — Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 6 — Watch it deploy! 🎉
Go to your GitHub repo → **Actions** tab to watch the pipeline run.

---

## 📁 Project Structure

```
travel-prophet/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── tests/
│   └── test_app.py                 # All tests (CI runs these)
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD pipeline definition
└── README.md
```

---

## 🧪 Run Tests Locally

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/ -v
```

---

## 📊 Features

- Upload your own CSV travel data
- Sample data included for demo
- Adjustable forecast period (30–365 days)
- Yearly and weekly seasonality
- Confidence intervals
- Downloadable forecast CSV
- Interactive Plotly charts

---

## 🔄 How CI/CD Works
 # Travel Prophet Forecaster

| Event | What Happens |
|-------|-------------|
| Push to `main` | Tests run → if pass → deploy to HF |
| Pull Request | Tests run only → no deployment |
| Tests fail | Pipeline stops → no deployment |
| Tests pass | Auto deploys to Hugging Face Spaces |
