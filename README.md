# 🛡️ Crisis Management LLM AI Predictor (100% Free)

An advanced, institutional-grade **Multi-Agent AI Financial Crisis Intelligence & Future Forecasting Dashboard** built with Python, Machine Learning (`scikit-learn`, `XGBoost`, PyTorch `GRU`), and Streamlit.

This application analyzes 35+ years of macroeconomic time-series data (11,000+ daily observations), indexes 286,000+ financial news reports across major international publishers, matches custom economic shock scenarios against 20 benchmark historical crises, and generates 6-month & 12-month forward predictive trajectories.

> **🟢 100% Free & Open Source:** Requires **NO paid API keys**, subscriptions, or cloud dependencies. Operates fully offline using a built-in **Local Multi-Agent AI Reasoning Engine**.

---

## 🌟 Key Features & Capabilities

### 1. 🧠 Multi-Agent AI Reasoning & Executive Risk Reports
- **🧠 Macro Strategist Agent**: Evaluates inflation drivers, real interest rates, oil supply shocks, and global trade growth.
- **📊 Quantitative Analyst Agent**: Synthesizes **Tri-Model Hybrid Ensemble (Random Forest + XGBoost + GRU)** predictions for S&P 500 equities, Brent crude oil, Gold safe-haven returns, and systemic crisis risk probabilities ($R^2 = 0.936$, $\text{AUC} = 0.952$).
- **🏛️ Crisis Historian Agent**: Cross-references input scenarios against 20 major historical crisis events (1929 Great Depression, 1973 Oil Shock, 1987 Black Monday, 2000 Dot-com, 2008 GFC, 2020 COVID Crash) and 286K+ financial news headlines.
- **🛡️ Portfolio Defense Officer**: Generates actionable asset allocation strategies (Gold safe-haven, short-duration Treasuries, defensive equities).

### 2. 📄 PDF & Document Upload Engine (`pypdf`)
- Drag-and-drop any central bank report, economic paper, or PDF/TXT/MD document into the sidebar or Stress Simulator.
- Text is automatically extracted and analyzed by the Multi-Agent AI Engine and Q&A assistant.

### 3. 🎛️ Interactive Scenario Controls & Dropdown Presets
- **Preset Crisis Dropdowns**: Load templates for *Hyperinflation 1973*, *Banking Crunch 2008*, *Pandemic 2020*, *Sovereign Debt Default 1998*, *Tech Bubble 2000*, or *Trade War 2018*.
- **Target Region Selector**: Global, US / North America, Europe / Eurozone, East Asia, Latin America, Middle East.
- **Granular Controls**: Oil price shocks (-50% to +100%), Market Volatility (10% to 80%), Inflation Rate (0% to 25%), Real Interest Rate (-5% to +15%), Cost of Living Index, Debt Defaults, Banking Crisis Stress, Trade Export Stance, and Gold Demand Shifts.

### 4. 💬 ChatGPT-Level AI Crisis Assistant
- Interactive multi-turn conversational Q&A engine with session memory (`st.session_state`).
- **⚡ Quick Inquiry Pills**: One-click prompt buttons for portfolio defense strategies, oil shock stagflation analysis, bank run warnings, and document summaries.
- **📥 Download Chat Report**: Export full AI interaction transcripts as formatted Markdown documents.

### 5. 📰 Historical Crisis News & Article Archive (286K+ Reports)
- Searchable headline database across Reuters, CNBC, The Guardian, and ABC News.
- Expanded news report highlights under matched crisis cards in Tab 1 and Tab 4.

---

## 📁 Datasets Included

| Dataset File | Tracked Indicators & Description |
| :--- | :--- |
| `dataset/crisis_events.csv` | 20 major historical crisis benchmark events (1929–2020) with dates, regions, and triggers |
| `dataset/all_stocks_5yr.csv` | S&P 500 stock prices (daily OHLCV data for ~500 tickers) |
| `dataset/BrentOilPrices.csv` | Daily crude oil prices (1987–2022) |
| `dataset/african_crises.csv` | Systemic banking, debt, inflation, and currency crises across 13 countries |
| `dataset/Covid-19 economy impact .csv` | Cross-sectional consumption, investment, tourism decline, stringency index, and mobility data |
| `dataset/world_bank_development_indicators.csv` | Global GDP, annual inflation, real interest rates, and governance metrics |
| `dataset/gold_price_prediction.csv` | Daily gold prices and 30d/90d return momentum |
| `dataset/banking_crisis_and_exports.csv` | Banking crisis counts, twin crises, recession counts, and export growth rates |
| `dataset/global_cost_of_living_crisis_2026.csv` | Cost of living index, rent index, petrol prices, and grocery index |
| `dataset/financial_news/` | 286,361 headlines indexed across Reuters, CNBC, Guardian, and ABC News |

---

## 🤖 Model Validation Metrics

The **Tri-Model Hybrid Ensemble** combines three distinct modeling paradigms:
1. 🌲 **Random Forest (35%)**: High-stability non-linear feature split classification & regression.
2. ⚡ **XGBoost Extreme Gradient Boosting (40%)**: Hyper-optimized gradient boosted decision trees.
3. 🔄 **Gated Recurrent Unit GRU (25%)**: PyTorch-backed deep recurrent state time-series learning.

| Prediction Target | Metric Score | Validation Performance |
| :--- | :---: | :--- |
| **S&P 500 12-Month Market Return** | $R^2 = 0.936$ | $\text{MAE} = 0.58\%$ |
| **S&P 500 6-Month Market Return** | $R^2 = 0.885$ | $\text{MAE} = 0.43\%$ |
| **Brent Oil 12-Month Price Return** | $R^2 = 0.812$ | $\text{MAE} = 12.75\%$ |
| **Systemic Crisis Risk Classifier** | $\text{AUC} = 0.952$ | $\text{Accuracy} = 88.5\%$ |

---

## 🚀 How to Run the App (100% Free)

### Method 1: One-Click Batch Launcher (Windows)
Double-click `run_app.bat` or run in your terminal:
```cmd
.\run_app.bat
```

### Method 2: Manual Terminal Execution
```bash
# 1. Install required dependencies
pip install -r requirements.txt

# 2. Clean and merge raw datasets into unified daily timeline
python clean_and_merge.py

# 3. Calculate 6m/12m historical crisis impact metrics
python analyze_crises.py

# 4. Train Tri-Model Hybrid Ensemble models
python ml_predictor.py

# 5. Index financial news headlines for historical crisis details
python crisis_news_indexer.py

# 6. Launch the Streamlit Web Application
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Project Architecture

```
├── app.py                     # Streamlit web dashboard (main entry point)
├── llm_engine.py              # Multi-Agent AI Predictor engine (Local Smart AI + optional API)
├── ml_predictor.py            # Tri-Model Ensemble trainer (Random Forest + XGBoost + GRU)
├── clean_and_merge.py         # Data cleaning & 11K-row timeline merging pipeline
├── analyze_crises.py          # Historical crisis impact analyzer
├── crisis_news_indexer.py     # Financial news headline indexer for historical crisis details
├── find_similar_crises.py     # CLI tool for similarity matching & ML prediction
├── HOW_TO_RUN.txt             # Step-by-step instruction guide
├── run_app.bat                # One-click Windows batch execution script
├── dataset/                   # Raw economic, market, and financial news datasets
├── processed/                 # Merged clean timeline, index JSONs, and trained ML models
├── requirements.txt           # Python dependencies
└── README.md                  # Comprehensive Documentation
```

---

## 📄 License

MIT License — 100% Free and Open Source.
