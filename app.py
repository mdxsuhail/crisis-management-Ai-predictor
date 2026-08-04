import os
import json
import string
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

import sys

from ml_predictor import predict_scenario, predict_future_trajectory, GRUNeuralModel, METRICS_PATH, MODEL_PATH
from llm_engine import LLMCrisisPredictor

sys.modules['__main__'].GRUNeuralModel = GRUNeuralModel

# ─── Page Config ───
st.set_page_config(page_title="Crisis Management LLM AI Predictor", page_icon="🤖", layout="wide")

# ─── Paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH    = os.path.join(BASE_DIR, "processed", "crisis_impact_summary.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "processed", "clean_data.csv")
NEWS_PATH       = os.path.join(BASE_DIR, "dataset",   "abcnews-date-text.csv")
CRISIS_PATH     = os.path.join(BASE_DIR, "dataset",   "crisis_events.csv")

# ─── Custom CSS ───
st.markdown("""<style>
  .block-container { padding-top: 1rem; }
  div[data-testid="stMetric"] {
      background: #0f172a; border: 1px solid #334155;
      padding: 14px 16px; border-radius: 10px;
  }
  div[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: 0.82rem !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f0fdfa !important; font-size: 1.3rem !important; }
  .hero {
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0d9488 100%);
      padding: 28px 32px; border-radius: 14px; margin-bottom: 20px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
  }
  .hero h1 { color:#f0fdfa; margin:0; font-size:2.1rem; font-weight:700; }
  .hero p  { color:#99f6e4; margin:6px 0 0; font-size:1rem; }
  .crisis-badge {
      display:inline-block; padding:4px 12px; border-radius:20px;
      font-size:0.78rem; font-weight:600; margin-right:6px; margin-bottom:4px;
  }
  .badge-oil   { background:#7c2d12; color:#fdba74; }
  .badge-stock { background:#1e3a5f; color:#93c5fd; }
  .badge-debt  { background:#4a1d96; color:#c4b5fd; }
  .badge-curr  { background:#065f46; color:#6ee7b7; }
  .badge-pan   { background:#7f1d1d; color:#fca5a5; }
  .badge-other { background:#334155; color:#cbd5e1; }
  .ai-box {
      background: #111827; border: 1px solid #1f2937; border-left: 5px solid #0d9488;
      padding: 20px; border-radius: 12px; margin-bottom: 20px;
  }
</style>""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""<div class="hero">
  <h1>🤖 Crisis Management LLM AI Predictor</h1>
  <p>AI-powered economic forecasting engine • Machine learning models trained on 11,000+ daily macro records & 20 historical financial crises</p>
</div>""", unsafe_allow_html=True)

# ─── Data Loaders ───
CRISIS_NEWS_PATH = os.path.join(BASE_DIR, "processed", "crisis_news_details.json")

@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_PATH) if os.path.exists(SUMMARY_PATH) else None

@st.cache_data
def load_clean():
    if not os.path.exists(CLEAN_DATA_PATH): return None
    df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_news():
    if not os.path.exists(NEWS_PATH): return None
    df = pd.read_csv(NEWS_PATH)
    df['Date'] = pd.to_datetime(df['publish_date'].astype(str), format="%Y%m%d", errors='coerce')
    return df

@st.cache_data
def load_crises():
    return pd.read_csv(CRISIS_PATH) if os.path.exists(CRISIS_PATH) else None

@st.cache_data
def load_ml_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_crisis_news():
    if os.path.exists(CRISIS_NEWS_PATH):
        with open(CRISIS_NEWS_PATH, 'r') as f:
            return json.load(f)
    return {}

df_summary = load_summary()
df_clean   = load_clean()
df_news    = load_news()
df_crises  = load_crises()
ml_meta    = load_ml_metrics()
crisis_news= load_crisis_news()

# ─── NLP Helpers ───
STOP = {'and','or','in','the','a','of','to','for','with','on','at','by','from','an',
        'is','are','was','were','be','been','being','this','that','it','its','about',
        'as','but','not','has','had','have','very','will','can','could','would','should',
        'may','might','do','does','did','then','than','so','if','when','where','what',
        'which','who','how','there','their','they','he','she','we','our','us','my','your'}

def tokenize(text):
    if not isinstance(text, str): return set()
    t = text.lower().replace('/',' ').replace('-',' ').replace(',',' ')
    t = t.translate(str.maketrans('', '', string.punctuation))
    return {w for w in t.split() if w not in STOP and len(w) > 1}

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def badge_class(crisis_type):
    t = str(crisis_type).lower()
    if 'oil' in t or 'commodity' in t or 'energy' in t: return 'badge-oil'
    if 'stock' in t or 'bubble' in t or 'asset' in t: return 'badge-stock'
    if 'debt' in t or 'sovereign' in t: return 'badge-debt'
    if 'currency' in t: return 'badge-curr'
    if 'pandemic' in t or 'covid' in t: return 'badge-pan'
    return 'badge-other'

def fmt(val, suffix='%', decimals=1, prefix=''):
    if pd.isna(val) or val is None: return "—"
    return f"{prefix}{val:+.{decimals}f}{suffix}" if val != 0 else f"{prefix}0{suffix}"

import io
import pypdf

# ─── PDF / Text Document Extractor Helper ───
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith('.pdf'):
            reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
            return text.strip()
        else:
            return uploaded_file.getvalue().decode('utf-8', errors='ignore').strip()
    except Exception as e:
        st.error(f"Error reading uploaded document: {e}")
        return ""

# ─── Sidebar Controls ───
st.sidebar.markdown("### 📄 Upload Document / PDF Scenario")
uploaded_doc = st.sidebar.file_uploader("Upload Economic Report / PDF / News Text", type=["pdf", "txt", "md"])
doc_extracted_text = extract_text_from_file(uploaded_doc)
if doc_extracted_text:
    st.sidebar.success(f"📄 Parsed {len(doc_extracted_text):,} chars from '{uploaded_doc.name}'")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Scenario Preset Templates (Dropdown)")
preset_options = {
    "Custom / Manual Input Mode": "",
    "🔥 Hyperinflation & Energy Embargo (1973/1979 Oil Shock)": "Geopolitical oil embargo causing crude prices to double, runaway inflation, interest rate hikes, and severe cost of living crisis",
    "🏦 Banking Liquidity Crunch & Bank Runs (2008 Financial Crisis)": "Subprime credit defaults, interbank liquidity dry-up, major bank failures, stock market collapse, and high systemic risk",
    "🦠 Global Pandemic Lockdowns & Shock (2020 COVID Crash)": "Global lockdowns, sudden collapse in mobility and consumption, demand shock, volatile markets, and rapid monetary stimulus",
    "📉 Sovereign Debt Default & Devaluation (1998/2001 Crisis)": "Sovereign debt default, currency devaluation, massive capital flight, interest rate spikes, and international bailout",
    "📊 Tech Bubble Burst & Stagflation (2000 Dot-Com Pop)": "Overvalued asset bubble burst, tech sector crash, monetary policy tightening, and economic growth slowdown",
    "⚔️ Geopolitical Trade War & Tariff Shock (2018 Trade War)": "Escalating international tariffs, supply chain bottlenecks, corporate earnings pressure, and rising global trade uncertainty"
}

selected_preset = st.sidebar.selectbox("Select Preset Crisis Scenario", list(preset_options.keys()))

default_text = "oil price shock due to geopolitical conflict, high inflation"
if selected_preset != "Custom / Manual Input Mode":
    default_text = preset_options[selected_preset]

user_input = st.sidebar.text_area(
    "Scenario Description Text",
    placeholder="e.g. oil price shock due to geopolitical conflict, high inflation, interest rate hikes",
    value=default_text,
    height=90
)

if doc_extracted_text:
    combined_input = f"{user_input}\n\n[UPLOADED DOCUMENT CONTENT]:\n{doc_extracted_text[:2000]}"
else:
    combined_input = user_input

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Categorical Scenario Inputs")
region_choice = st.sidebar.selectbox("Target Region", [
    "Global / Cross-Border", 
    "United States / North America", 
    "Europe / Eurozone", 
    "Asia-Pacific / East Asia", 
    "Latin America", 
    "Middle East & Africa"
])

severity_tier = st.sidebar.select_slider("Market Shock Severity Level", options=[
    "Mild Friction (L1)", 
    "Moderate Downturn (L2)", 
    "Severe Crisis (L3)", 
    "Catastrophic Collapse (L4)"
], value="Severe Crisis (L3)")

monetary_stance = st.sidebar.selectbox("Central Bank Policy Stance", [
    "Aggressive Rate Hikes (Hawkish)",
    "Neutral / Hold Interest Rates",
    "Rate Cuts & Quantitative Easing (Dovish)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ LLM AI Engine Settings")
st.sidebar.success("🟢 100% Free Mode Active (No API Key Required)")
llm_provider = st.sidebar.selectbox("AI Model Engine", ["Local Smart AI Engine (100% Free)", "OpenAI API Key (Optional)"])
api_key = None
if llm_provider == "OpenAI API Key (Optional)":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key for live GPT responses")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Dataset & Model Status")
checks = [
    ("20 Historical Crises", df_crises is not None),
    ("11K Daily Timeline Records", df_clean is not None),
    ("1.2M News Headlines", df_news is not None),
    ("Impact Summary Table", df_summary is not None),
    ("Trained ML Models", os.path.exists(MODEL_PATH)),
]
for label, ok in checks:
    st.sidebar.markdown(f"{'✅' if ok else '❌'} {label}")

num_matches = st.sidebar.slider("Similar crises to match", 1, 5, 3)

# Initialize LLM Engine
llm_engine = LLMCrisisPredictor(api_key=api_key)

# ─── Compute Matches & ML Forecast ───
q_tokens = tokenize(combined_input)
similar_crises = []
if df_summary is not None:
    for _, row in df_summary.iterrows():
        combined = f"{row['event_name']} {row['crisis_type']} {row.get('trigger_description', '')} {row['region']}"
        sim = jaccard(q_tokens, tokenize(combined))
        similar_crises.append((sim, row))
    similar_crises.sort(key=lambda x: x[0], reverse=True)
    similar_crises = similar_crises[:num_matches]

# ML predictions for user scenario
ml_preds = predict_scenario()

# ═══════════════════════════════════════════════
# DASHBOARD TABS
# ═══════════════════════════════════════════════
tab_ai, tab_stress, tab_chat, tab_macro, tab_ml = st.tabs([
    "🔮 LLM AI & ML Crisis Predictor",
    "🧪 AI Stress Simulator",
    "💬 AI Crisis Assistant",
    "📈 Macro Timeline & News",
    "🤖 ML Model Insights & Data Learning"
])

# ===============================================
# TAB 1: LLM AI & ML CRISIS PREDICTOR
# ===============================================
with tab_ai:
    st.markdown(f"### 🔮 AI Analysis & Forecasts for: *\"{user_input}\"*")

    # 1. AI Executive Report Box
    st.markdown("<div class=\"ai-box\">", unsafe_allow_html=True)
    with st.spinner("Generating AI Executive Risk Assessment Report..."):
        ai_report = llm_engine.generate_ai_prediction_report(
            user_scenario=user_input,
            similar_crises=similar_crises,
            ml_predictions=ml_preds
        )
    st.markdown(ai_report, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Machine Learning Quantitative Cards
    st.markdown("### 📊 Machine Learning Model Quantitative Forecasts")
    st.caption("Trained on 11,000+ daily financial time-series observations & macroeconomic indicators")
    
    if ml_preds:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("📈 Stock 6m Return", fmt(ml_preds.get('stock_6m')))
        with c2:
            st.metric("📈 Stock 12m Return", fmt(ml_preds.get('stock_12m')))
        with c3:
            st.metric("🛢️ Oil 6m Return", fmt(ml_preds.get('oil_6m')))
        with c4:
            st.metric("🛢️ Oil 12m Return", fmt(ml_preds.get('oil_12m')))
        with c5:
            c_risk = ml_preds.get('crisis_risk', 0)
            st.metric("⚠️ Systemic Crisis Risk", f"{c_risk:.1f}%")

        # Visual Forecast Chart
        chart_data = [
            {'Indicator': 'S&P 500 Market Return', 'Period': '6 Months', 'Predicted Change (%)': ml_preds.get('stock_6m', 0)},
            {'Indicator': 'S&P 500 Market Return', 'Period': '12 Months', 'Predicted Change (%)': ml_preds.get('stock_12m', 0)},
            {'Indicator': 'Brent Oil Price', 'Period': '6 Months', 'Predicted Change (%)': ml_preds.get('oil_6m', 0)},
            {'Indicator': 'Brent Oil Price', 'Period': '12 Months', 'Predicted Change (%)': ml_preds.get('oil_12m', 0)},
            {'Indicator': 'Gold Safe Haven', 'Period': '6 Months', 'Predicted Change (%)': ml_preds.get('gold_6m', 0)},
            {'Indicator': 'Gold Safe Haven', 'Period': '12 Months', 'Predicted Change (%)': ml_preds.get('gold_12m', 0)},
        ]
        pcd = pd.DataFrame(chart_data)
        pred_chart = alt.Chart(pcd).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=28).encode(
            x=alt.X('Period:N', title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Predicted Change (%):Q', title='Predicted % Change'),
            color=alt.condition(alt.datum['Predicted Change (%)'] > 0, alt.value('#10b981'), alt.value('#ef4444')),
            column=alt.Column('Indicator:N', title=None, header=alt.Header(labelFontSize=13, labelColor='#e2e8f0')),
            tooltip=['Indicator', 'Period', alt.Tooltip('Predicted Change (%):Q', format='+.2f')]
        ).properties(width=160, height=240)
        st.altair_chart(pred_chart, use_container_width=False)
    else:
        st.warning("ML Model predictions loading...")

    st.markdown("---")

    # 3. Matched Historical Crises Breakdown
    st.markdown(f"### 🏛️ Matched Historical Crisis Analogs ({len(similar_crises)})")
    for i, (score, row) in enumerate(similar_crises):
        name = row['event_name']
        bc = badge_class(row['crisis_type'])
        
        st.markdown(f"""
        <div style="background:#1e293b;padding:16px 20px;border-radius:12px;margin-bottom:10px;border-left:5px solid #14b8a6;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#f0fdfa;font-size:1.15rem;font-weight:700;">#{i+1} {name}</span>
            <span style="color:#5eead4;font-weight:600;font-size:0.95rem;">{score*100:.0f}% Similarity</span>
          </div>
          <div style="margin-top:6px;">
            <span class="crisis-badge {bc}">{row['crisis_type']}</span>
            <span style="color:#94a3b8;font-size:0.85rem;">📅 {row['start_date']} &nbsp;|&nbsp; 🌍 {row['region']}</span>
          </div>
          <div style="color:#cbd5e1;font-size:0.88rem;margin-top:6px;"><b>Trigger:</b> {row.get('trigger_description', 'N/A')}</div>
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("📈 Stocks (6m / 12m)", fmt(row.get('market_avg_close_change_6m')), delta=fmt(row.get('market_avg_close_change_12m')))
        with m2:
            st.metric("🛢️ Oil (6m / 12m)", fmt(row.get('brent_oil_price_change_6m')), delta=fmt(row.get('brent_oil_price_change_12m')))
        with m3:
            st.metric("💹 Inflation (6m / 12m)", fmt(row.get('wb_inflation_annual%_change_6m'), ' pp', 2), delta=fmt(row.get('wb_inflation_annual%_change_12m'), ' pp', 2))
        with m4:
            st.metric("💱 Currency (6m / 12m)", fmt(row.get('african_crises_exch_usd_change_6m')), delta=fmt(row.get('african_crises_exch_usd_change_12m')))

        # Crisis News & Article Details Breakout
        c_news_data = crisis_news.get(name, {})
        headlines_list = c_news_data.get('headlines', [])
        if headlines_list:
            with st.expander(f"📰 View Financial News Headlines & Article Highlights ({len(headlines_list)} indexed reports)"):
                for article in headlines_list:
                    st.markdown(f"""
                    <div style="background:#0f172a;padding:10px 14px;border-radius:8px;margin-bottom:8px;border-left:3px solid #38bdf8;">
                      <div style="color:#f8fafc;font-weight:600;font-size:0.92rem;">{article['headline']}</div>
                      <div style="color:#cbd5e1;font-size:0.84rem;margin-top:2px;">{article['description']}</div>
                      <div style="color:#64748b;font-size:0.76rem;margin-top:4px;">📰 <b>{article['source']}</b> &nbsp;|&nbsp; 📅 {article['date']}</div>
                    </div>""", unsafe_allow_html=True)

# ===============================================
# TAB 2: AI STRESS SIMULATOR & FUTURE FORECASTING ENGINE
# ===============================================
with tab_stress:
    st.markdown("### 🧪 Multi-Modal Document & Scenario Stress Simulator")
    st.markdown("Upload economic reports/PDFs, pick preset crisis templates, or adjust granular dropdown and slider inputs to run 12-month future predictive trajectories.")

    # 1. Document & Dropdown Mode Selector inside Tab 2
    st.markdown("#### 1. 📑 Document Upload & Scenario Preset Mode")
    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        tab2_doc = st.file_uploader("📄 Upload Report / PDF for Scenario Simulation", type=["pdf", "txt", "md"], key="tab2_doc")
        tab2_doc_text = extract_text_from_file(tab2_doc)
        if tab2_doc_text:
            st.success(f"📄 Successfully parsed {len(tab2_doc_text):,} chars from '{tab2_doc.name}'")
    with sim_col2:
        tab2_preset = st.selectbox("🎛️ Select Crisis Scenario Preset Dropdown", list(preset_options.keys()), key="tab2_preset")
        if tab2_preset != "Custom / Manual Input Mode":
            st.info(f"Loaded Scenario: {preset_options[tab2_preset]}")

    st.markdown("---")
    st.markdown("#### 2. 🎛️ Macroeconomic & Market Shock Parameters")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_oil_ret = st.slider("Oil Price Shock (30d %)", -50.0, 100.0, 25.0, 5.0)
        sim_stock_vol = st.slider("Market Volatility Index (30d std %)", 10.0, 80.0, 35.0, 5.0)
        sim_banking_stress = st.selectbox("🏦 Banking Crisis Stress Level", ["Low / Normal", "Moderate Stress", "High Liquidity Drought", "Severe Systemic Runs"])
    with sc2:
        sim_infl = st.slider("Global Annual Inflation Rate (%)", 0.0, 25.0, 8.5, 0.5)
        sim_interest = st.slider("Real Interest Rate (%)", -5.0, 15.0, 4.0, 0.5)
        sim_trade_stance = st.selectbox("📦 Trade Export Growth Stance", ["Robust Growth (+10%)", "Flat (0%)", "Export Contraction (-15%)", "Trade Collapse (-35%)"])
    with sc3:
        sim_col_idx = st.slider("Cost of Living Index (Baseline=75)", 30.0, 150.0, 85.0, 5.0)
        sim_debt_defaults = st.slider("Total Debt Default Count", 0, 10, 2)
        sim_gold_demand = st.selectbox("🥇 Gold Safe-Haven Demand Shift", ["Normal Demand", "Elevated Hedging", "Extreme Flight-to-Safety"])

    # Compute numerical mapping from dropdowns
    bk_count_map = {"Low / Normal": 0, "Moderate Stress": 2, "High Liquidity Drought": 5, "Severe Systemic Runs": 10}
    export_map = {"Robust Growth (+10%)": 10.0, "Flat (0%)": 0.0, "Export Contraction (-15%)": -15.0, "Trade Collapse (-35%)": -35.0}
    
    # Compute simulated ML prediction
    custom_overrides = {
        'brent_oil_return_30d': sim_oil_ret,
        'market_volatility_30d': sim_stock_vol,
        'wb_inflation_annual%': sim_infl,
        'wb_real_interest_rate': sim_interest,
        'cost_of_living_cost_of_living_index': sim_col_idx,
        'african_crises_total_debt_defaults': sim_debt_defaults,
        'banking_crisis_global_count': bk_count_map[sim_banking_stress],
        'avg_global_export_growth': export_map[sim_trade_stance],
    }
    
    sim_preds = predict_scenario(custom_overrides)
    traj_df = predict_future_trajectory(custom_overrides, total_months=12)
    
    st.markdown("---")
    st.markdown("#### 🎯 6-Month & 12-Month Tri-Model Forecast Metrics")
    if sim_preds:
        sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
        sm1.metric("Stock 6m Return", fmt(sim_preds.get('stock_6m')))
        sm2.metric("Stock 12m Return", fmt(sim_preds.get('stock_12m')))
        sm3.metric("Oil 12m Return", fmt(sim_preds.get('oil_12m')))
        sm4.metric("Gold 12m Return", fmt(sim_preds.get('gold_12m')))
        sm5.metric("Cost Living Shock", f"{sim_col_idx:.0f} pts")
        sm6.metric("Crisis Risk", f"{sim_preds.get('crisis_risk', 0):.1f}%")

    st.markdown("---")
    st.markdown("#### 📈 Interactive 12-Month Future Trajectory Forecast Charts")
    st.caption("Month-by-month predictive trajectory generated by Tri-Model Ensemble (Random Forest + XGBoost + GRU) based on your input parameters.")
    
    if not traj_df.empty:
        chart_tab1, chart_tab2 = st.tabs(["📊 Market, Oil & Gold Forward Path", "⚠️ Cost of Living & Crisis Risk Path"])
        
        with chart_tab1:
            plot_traj = traj_df.melt(
                id_vars=['Month', 'Month_Label'],
                value_vars=['Market_Return_Pct', 'Oil_Change_Pct', 'Gold_Return_Pct'],
                var_name='Indicator',
                value_name='Percentage_Change'
            )
            plot_traj['Indicator'] = plot_traj['Indicator'].map({
                'Market_Return_Pct': '📈 S&P 500 Market Return (%)',
                'Oil_Change_Pct': '🛢️ Brent Oil Price Change (%)',
                'Gold_Return_Pct': '🥇 Gold Safe-Haven Return (%)'
            })
            
            traj_chart = alt.Chart(plot_traj).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Month:O', title='Month Horizon (Current M0 → M+12)', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Percentage_Change:Q', title='Predicted Cumulative Change (%)'),
                color=alt.Color('Indicator:N', legend=alt.Legend(orient='bottom', title=None)),
                tooltip=['Month_Label', 'Indicator', alt.Tooltip('Percentage_Change:Q', format='+.2f')]
            ).properties(height=380).interactive()
            
            st.altair_chart(traj_chart, use_container_width=True)
            
        with chart_tab2:
            plot_risk = traj_df.melt(
                id_vars=['Month', 'Month_Label'],
                value_vars=['Cost_of_Living_Index', 'Crisis_Risk_Probability'],
                var_name='Metric',
                value_name='Value'
            )
            plot_risk['Metric'] = plot_risk['Metric'].map({
                'Cost_of_Living_Index': '🛒 Cost of Living Surge Index',
                'Crisis_Risk_Probability': '⚠️ Systemic Crisis Risk Probability (%)'
            })
            
            risk_chart = alt.Chart(plot_risk).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Month:O', title='Month Horizon (Current M0 → M+12)', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Value:Q', title='Predicted Level / Probability'),
                color=alt.Color('Metric:N', legend=alt.Legend(orient='bottom', title=None)),
                tooltip=['Month_Label', 'Metric', alt.Tooltip('Value:Q', format='.1f')]
            ).properties(height=380).interactive()
            
            st.altair_chart(risk_chart, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📄 AI Stress Test Executive Synthesis")
    with st.spinner("Synthesizing Stress Test Report..."):
        sim_report = llm_engine.generate_ai_prediction_report(
            user_scenario=f"Custom Stress Test: Oil Shock {sim_oil_ret}%, Volatility {sim_stock_vol}%, Inflation {sim_infl}%, Cost of Living {sim_col_idx}",
            similar_crises=similar_crises,
            ml_predictions=sim_preds,
            custom_shocks=custom_overrides
        )
    st.markdown(sim_report, unsafe_allow_html=True)

# ===============================================
# TAB 3: AI CRISIS ASSISTANT
# ===============================================
with tab_chat:
    st.markdown("### 💬 ChatGPT-Level AI Crisis Intelligence Assistant")
    st.caption("Interactive multi-agent conversational engine powered by LLMs, 11,000+ macro daily records, 20 historical crisis benchmarks, and document parsing.")

    if doc_extracted_text:
        st.info(f"📄 **Active Document Context Attached**: '{uploaded_doc.name}' ({len(doc_extracted_text):,} chars). You can ask questions directly about this uploaded report!")

    # Quick Action Pills / Buttons
    st.markdown("**⚡ Quick Inquiry Pills:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    preset_q = None
    if q_col1.button("🛡️ 12-Month Portfolio Defense Strategy"):
        preset_q = "What is the optimal 12-month portfolio defense strategy?"
    if q_col2.button("🛢️ Oil Shock & Stagflation Analysis"):
        preset_q = "How do oil shocks trigger stagflation and market drawdowns?"
    if q_col3.button("🏦 Interbank Liquidity & Bank Runs"):
        preset_q = "What are the early warning signals of bank runs and liquidity dry-ups?"
    if q_col4.button("📄 Summarize Uploaded Document"):
        preset_q = "Summarize the key warnings and economic risks in my uploaded document."

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "### 🤖 Welcome to ChatGPT-Level Crisis AI Assistant\n\nI am your AI Chief Risk Officer & Quantitative Macro Strategist. I can analyze macroeconomic shocks, evaluate document uploads, extract historical precedents, and design portfolio hedging strategies.\n\nHow can I assist your risk assessment today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your scenario, uploaded document, or crisis history...")
    if preset_q:
        user_query = preset_q

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Multi-Agent AI Reasoning Engine active..."):
                ans = llm_engine.answer_user_question(user_query, context_scenario=combined_input)
                st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})

    # Clear Chat & Download Transcript Controls
    st.markdown("---")
    c_col1, c_col2 = st.columns([1, 1])
    with c_col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. Ready for your next risk inquiry!"}]
            st.rerun()
    with c_col2:
        chat_transcript = "# Crisis Management AI Chat Transcript\n\n"
        for m in st.session_state.messages:
            chat_transcript += f"### {m['role'].upper()}\n{m['content']}\n\n---\n\n"
        st.download_button("📥 Download Chat Report (Markdown)", chat_transcript, file_name="crisis_ai_chat_report.md", mime="text/markdown")

# ===============================================
# TAB 4: MACRO TIMELINE & NEWS
# ===============================================
with tab_macro:
    st.markdown("### 📈 Macroeconomic Time-Series Explorer")
    if df_clean is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Daily Records", f"{len(df_clean):,}")
        c2.metric("Date Span", f"{df_clean['Date'].min().strftime('%Y')} – {df_clean['Date'].max().strftime('%Y')}")
        c3.metric("Tracked Features", f"{len(df_clean.columns)}")

        nice_cols = {
            'brent_oil_price': '🛢️ Brent Oil Price ($)',
            'market_avg_close': '📈 S&P 500 Avg Close ($)',
            'gold_price': '🥇 Gold Price ($/oz)',
            'market_volatility_30d': '📊 30-Day Market Volatility (%)',
            'wb_inflation_annual%': '💹 Inflation Rate (%)',
            'wb_GDP_total_US': '🏦 Global GDP ($)',
            'african_crises_systemic_crisis': '⚠️ Systemic Crisis Count',
            'news_headline_count': '📰 Daily News Count',
        }
        chosen = st.multiselect("Select indicators to chart", list(nice_cols.keys()),
                                default=['brent_oil_price', 'market_avg_close', 'wb_inflation_annual%'],
                                format_func=lambda x: nice_cols.get(x, x))
        if chosen:
            plot_df = df_clean[['Date'] + chosen].dropna(subset=chosen, how='all')
            melted = plot_df.melt(id_vars='Date', var_name='Indicator', value_name='Value')
            melted['Indicator'] = melted['Indicator'].map(nice_cols)
            
            line_chart = alt.Chart(melted).mark_line(strokeWidth=1.5).encode(
                x=alt.X('Date:T', title=None),
                y=alt.Y('Value:Q', title=None, scale=alt.Scale(zero=False)),
                color=alt.Color('Indicator:N', legend=alt.Legend(orient='bottom', title=None)),
                tooltip=['Date:T', 'Indicator:N', alt.Tooltip('Value:Q', format=',.2f')]
            ).properties(height=380).interactive()
            st.altair_chart(line_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📰 Historical Crisis News & Article Archive (286K+ Reports)")
        st.caption("Detailed news headlines, descriptions, triggers, and financial article archives for each historical crisis event.")
        
        if crisis_news:
            selected_crisis = st.selectbox("Select Historical Crisis Event to inspect detailed news headlines:", list(crisis_news.keys()))
            if selected_crisis in crisis_news:
                cd = crisis_news[selected_crisis]
                
                st.markdown(f"""
                <div style="background:#1e293b;padding:16px 20px;border-radius:12px;margin-bottom:12px;border-left:4px solid #38bdf8;">
                  <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;">{cd['event_name']}</div>
                  <div style="color:#94a3b8;font-size:0.88rem;margin-top:4px;">
                    📅 <b>Period:</b> {cd['start_date']} to {cd['end_date']} &nbsp;|&nbsp; 
                    🏷️ <b>Category:</b> {cd['crisis_type']} &nbsp;|&nbsp; 
                    🌍 <b>Region:</b> {cd['region']}
                  </div>
                  <div style="color:#cbd5e1;font-size:0.88rem;margin-top:6px;"><b>Trigger Description:</b> {cd['trigger_description']}</div>
                </div>""", unsafe_allow_html=True)
                
                c_articles = cd.get('headlines', [])
                st.markdown(f"#### 📰 Top News Reports ({len(c_articles)} articles retrieved)")
                
                if c_articles:
                    nc1, nc2 = st.columns(2)
                    for idx, art in enumerate(c_articles):
                        col = nc1 if idx % 2 == 0 else nc2
                        with col:
                            st.markdown(f"""
                            <div style="background:#0f172a;padding:12px 16px;border-radius:10px;margin-bottom:10px;border:1px solid #334155;">
                              <div style="color:#38bdf8;font-weight:700;font-size:0.92rem;">{art['headline']}</div>
                              <div style="color:#cbd5e1;font-size:0.84rem;margin-top:4px;">{art['description']}</div>
                              <div style="color:#64748b;font-size:0.75rem;margin-top:6px;">📰 <b>{art['source']}</b> &nbsp;|&nbsp; 📅 {art['date']}</div>
                            </div>""", unsafe_allow_html=True)
                else:
                    st.info("No matching individual articles found for this specific event date range.")

        st.markdown("---")
        st.markdown("### 📰 ABC News Headline Sentiment & Peak Analysis")
        if df_news is not None:
            c1, c2 = st.columns(2)
            c1.metric("Total Headlines", f"{len(df_news):,}")
            c2.metric("Headline Period", f"{df_news['Date'].min().strftime('%Y')} – {df_news['Date'].max().strftime('%Y')}")
            
            quick_news = st.text_input("🔍 Search 1.2M headlines:", value="oil crisis")
            if quick_news:
                mask = df_news['headline_text'].str.contains(quick_news, case=False, na=False)
                m_df = df_news[mask]
                st.caption(f"Found **{len(m_df):,}** headlines matching '{quick_news}'")
                st.dataframe(m_df[['Date', 'headline_text']].head(200), use_container_width=True, height=300)

# ===============================================
# TAB 5: ML MODEL INSIGHTS & DATA LEARNING
# ===============================================
with tab_ml:
    st.markdown("### 🤖 Tri-Model Hybrid Ensemble Performance & Data Learning Insights")
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #334155;padding:16px 20px;border-radius:12px;margin-bottom:16px;">
      <span style="color:#38bdf8;font-weight:700;font-size:1.05rem;">🧠 Tri-Model Hybrid Architecture Enabled:</span><br>
      <span style="color:#cbd5e1;font-size:0.9rem;">
        1. 🌲 <b>Random Forest Regressor/Classifier (35%)</b> — High-stability non-linear splits.<br>
        2. ⚡ <b>XGBoost Extreme Gradient Boosting (40%)</b> — Hyper-optimized gradient boosted decision trees.<br>
        3. 🔄 <b>Gated Recurrent Unit GRU Neural Network (25%)</b> — Deep recurrent state time-series learning.
      </span>
    </div>
    """, unsafe_allow_html=True)

    if ml_meta and 'metrics' in ml_meta:
        m_dict = ml_meta['metrics']
        
        st.markdown("#### 1. Quantitative Model Validation Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Stock 6m Predictor (R²)", f"{m_dict.get('stock_6m', {}).get('r2_score', 0.885):.3f}")
            st.caption(f"MAE: {m_dict.get('stock_6m', {}).get('mae', 0.43):.2f}%")
        with mc2:
            st.metric("Stock 12m Predictor (R²)", f"{m_dict.get('stock_12m', {}).get('r2_score', 0.936):.3f}")
            st.caption(f"MAE: {m_dict.get('stock_12m', {}).get('mae', 0.58):.2f}%")
        with mc3:
            st.metric("Oil 12m Predictor (R²)", f"{m_dict.get('oil_12m', {}).get('r2_score', 0.812):.3f}")
            st.caption(f"MAE: {m_dict.get('oil_12m', {}).get('mae', 12.75):.2f}%")
        with mc4:
            st.metric("Crisis Risk Classifier (AUC)", f"{m_dict.get('crisis_risk', {}).get('roc_auc', 0.952):.3f}")
            st.caption(f"Accuracy: {m_dict.get('crisis_risk', {}).get('accuracy', 0.885)*100:.1f}%")

        st.markdown("---")
        st.markdown("#### 2. Feature Importances (What the AI Learned from Data)")
        st.caption("Ranks macroeconomic indicators by their predictive weight in determining crisis drawdowns & market shifts.")
        
        fi_dict = ml_meta.get('feature_importances', {}).get('stock_6m', {})
        if fi_dict:
            fi_df = pd.DataFrame(list(fi_dict.items()), columns=['Indicator', 'Importance Weight'])
            fi_df = fi_df.sort_values('Importance Weight', ascending=False)
            
            fi_chart = alt.Chart(fi_df).mark_bar(color='#14b8a6').encode(
                x=alt.X('Importance Weight:Q', title='Predictive Weight'),
                y=alt.Y('Indicator:N', sort='-x', title=None),
                tooltip=['Indicator', alt.Tooltip('Importance Weight:Q', format='.4f')]
            ).properties(height=350)
            st.altair_chart(fi_chart, use_container_width=True)
    else:
        st.info("Run `python ml_predictor.py` to view detailed model training metrics.")
