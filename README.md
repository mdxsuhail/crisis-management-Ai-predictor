# Crisis Management AI Predictor

An AI-powered financial crisis analysis and prediction tool built with Python and Streamlit. This application analyzes historical financial crises, matches current economic situations to past events, and predicts potential impacts using real-world datasets.

## Features

### 1. Pre-defined Crisis Explorer
- Search by crisis type (dropdown) or describe a custom situation in plain text
- Jaccard similarity matching finds the top 3 most similar historical crises
- Interactive tabbed charts showing post-crisis performance (6-month and 12-month changes)
- Download individual crisis data as CSV

### 2. Headline News Impact Predictor
- Search 1.2M+ ABC News headlines by keyword (e.g. "oil price", "pandemic", "mortgage")
- Identifies peak coverage dates and maps them to economic indicators
- Displays S&P 500 stock data, Brent oil prices, World Bank indicators, COVID-19 impact metrics, and African crisis data side-by-side
- Comparison charts across peak dates

## Datasets Used

| Dataset | Description |
|---------|-------------|
| ABC News Headlines | 1.2M+ headlines (2003–2020) for keyword trend analysis |
| S&P 500 Stocks (5yr) | Daily OHLCV data for ~500 tickers (2013–2018) |
| Brent Oil Prices | Daily crude oil prices (1987–2019) |
| African Crises | Banking, debt, inflation, and currency crises by country |
| COVID-19 Economy Impact | Stringency index and mobility data across 27 countries |
| World Bank Development Indicators | GDP, inflation, life expectancy, and governance metrics |
| Daily Forex Rates | EUR/USD exchange rates |

> **Note:** Large CSV datasets are excluded from this repo via `.gitignore`. See the **Setup** section below to download them.

## Project Structure

```
├── app.py                    # Streamlit web dashboard (main entry point)
├── clean_and_merge.py        # Data cleaning & merging pipeline
├── analyze_crises.py         # Crisis impact analysis (6m/12m indicator changes)
├── find_similar_crises.py    # CLI tool for crisis similarity matching
├── dataset/
│   ├── crisis_events.csv     # Pre-filled template of 9 historical crises
│   ├── african_crises.csv    # African banking/systemic crises dataset
│   ├── daily_forex_rates.csv # EUR/USD daily exchange rates
│   └── ...                   # Other raw datasets (see .gitignore)
├── processed/
│   └── crisis_impact_summary.csv  # Pre-computed crisis impact metrics
├── .gitignore
└── README.md
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install pandas numpy streamlit altair
```

### Download Required Datasets

Download the following datasets and place them in the `dataset/` folder:

1. **ABC News Headlines** → [Kaggle](https://www.kaggle.com/therohk/million-headlines) → `abcnews-date-text.csv`
2. **S&P 500 Stocks** → [Kaggle](https://www.kaggle.com/camnugent/sandp500) → `all_stocks_5yr.csv`
3. **Brent Oil Prices** → [Kaggle](https://www.kaggle.com/mabusalah/brent-oil-prices) → `BrentOilPrices.csv`
4. **COVID-19 Economy Impact** → Place as `Covid-19 economy impact .csv`
5. **World Bank Indicators** → Place as `world_bank_development_indicators.csv`

### Run the Data Pipeline

```bash
# Step 1: Clean and merge all datasets
python clean_and_merge.py

# Step 2: Analyze crisis impacts
python analyze_crises.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## How It Works

1. **Data Cleaning Pipeline** (`clean_and_merge.py`): Standardizes date formats, aggregates multi-ticker stock data to daily averages, forward-fills sparse macro indicators, and merges everything into a single daily time series (`processed/clean_data.csv`).

2. **Crisis Impact Analysis** (`analyze_crises.py`): For each crisis in `crisis_events.csv`, finds the closest available trading dates at the start, 6 months, and 12 months post-event, then computes percentage changes across all indicators.

3. **Similarity Matching** (`find_similar_crises.py` / `app.py`): Tokenizes user input and crisis metadata, computes Jaccard similarity, and ranks the top matches.

4. **Headline Predictor** (`app.py`): Searches the ABC News corpus for keyword matches, groups by date to find peak coverage, and maps those dates to the cleaned economic timeline.

## License

MIT License
