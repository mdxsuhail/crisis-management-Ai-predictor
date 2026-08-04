# Crisis Management LLM AI Predictor (100% Free)

An AI-powered financial crisis analysis and prediction tool built with Python, Machine Learning (`scikit-learn`), and Streamlit. This application analyzes historical financial crises, matches current economic situations to past events using NLP similarity matching, trains ML models on 11,000+ daily financial time-series observations, and predicts 6m/12m market drawdowns, oil prices, and crisis risk probabilities.

> **🟢 100% Free & Open Source:** Requires NO paid API keys, fees, or external cloud subscriptions. Operates fully offline using a built-in Local Smart AI Reasoning Engine.

---

## 🌟 Key Features

### 1. 🔮 LLM AI & ML Crisis Predictor
- Describe any economic scenario in plain text (e.g. *"Oil price shock due to geopolitical conflict, high inflation"*).
- **AI Executive Risk Report**: Generates structured executive risk assessments with severity levels, core drivers, historical analogs, and strategic portfolio hedging guidance.
- **Machine Learning Forecasts**: Quantitative 6-month & 12-month predictions for S&P 500 stock returns, Brent crude oil prices, inflation shifts, and systemic crisis risk probabilities.
- **Historical Crisis Analogs**: Ranks top matching historical crises (out of 20 major benchmark events from 1929 to 2020) with similarity scores and post-crisis performance metrics.

### 2. 🧪 AI Stress Testing & Custom Scenario Simulator
- Interactive sliders for oil shocks, market volatility, inflation rates, interest rates, debt default counts, and COVID stringency.
- Instant real-time ML prediction updates and AI executive synthesis reports for custom stress-tested scenarios.

### 3. 💬 AI Crisis Assistant (Interactive Q&A)
- Conversational chat interface to ask custom questions about crisis history, portfolio hedging strategies, inflation risks, or central bank policies.

### 4. 📈 Macroeconomic Timeline & News Explorer
- Explore 11,000+ daily records spanning 35+ years (1987–2022).
- Search 1.2M+ ABC News headlines by keyword to identify coverage peak dates and map them to market indicators.

### 5. 🤖 ML Model Insights & Data Learning
- Visualizes feature importances (e.g., market volatility, oil momentum, inflation, debt defaults) and model validation metrics ($R^2$ scores, MAE, AUC), proving how the AI learns directly from the dataset.

---

## 📁 Datasets Included

| Dataset | Description |
|---------|-------------|
| **Crisis Benchmark Events** | 20 historical crisis events (1929 Great Depression, 1973 Oil Shock, 1987 Black Monday, 2000 Dot-com, 2008 GFC, 2020 COVID-19, etc.) |
| **S&P 500 Stocks** | Daily OHLCV data for ~500 tickers |
| **Brent Oil Prices** | Daily crude oil prices (1987–2022) |
| **African Crises** | Banking, debt, inflation, and currency crises across 13 countries |
| **COVID-19 Economic Impact** | Consumption, investment, tourism decline, stringency index, and mobility data across 27 countries |
| **World Bank Indicators** | Global GDP, annual inflation, real interest rates, and governance metrics |
| **ABC News Headlines** | 1.2M+ headlines (2003–2020) for trend & peak analysis |

---

## 🚀 How to Run the App (Free)

### Method 1: One-Click Batch Launcher (Windows)
Double-click `run_app.bat` or run in terminal:
```bash
.\run_app.bat
```

### Method 2: Manual Terminal Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean and merge datasets into daily timeline
python clean_and_merge.py

# 3. Analyze 6m/12m historical crisis impacts
python analyze_crises.py

# 4. Train ML predictor models on dataset
python ml_predictor.py

# 5. Launch the dashboard
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Project Architecture

```
├── app.py                     # Streamlit web dashboard (main entry point)
├── llm_engine.py              # Dual-mode AI Predictor engine (Local Smart AI + optional API)
├── ml_predictor.py            # Machine Learning trainer (GradientBoosting & RandomForest)
├── clean_and_merge.py         # Data cleaning & 11K-row timeline merging pipeline
├── analyze_crises.py          # Historical crisis impact analyzer
├── find_similar_crises.py     # CLI tool for similarity matching & ML prediction
├── HOW_TO_RUN.txt             # Plain-text step-by-step instruction guide
├── run_app.bat                # One-click Windows batch execution script
├── dataset/                   # Raw economic and crisis datasets
├── processed/                 # Merged clean timeline and trained ML models
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## 📄 License

MIT License — 100% Free and Open Source.
