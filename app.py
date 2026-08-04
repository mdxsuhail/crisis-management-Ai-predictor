import os
import string
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import timedelta

# ─── Page Config ───
st.set_page_config(
    page_title="Crisis Management AI Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ───
BASE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(BASE, "processed", "crisis_impact_summary.csv")
CLEAN_DATA_PATH = os.path.join(BASE, "processed", "clean_data.csv")
NEWS_PATH = os.path.join(BASE, "dataset", "abcnews-date-text.csv")
CRISIS_PATH = os.path.join(BASE, "dataset", "crisis_events.csv")

# ─── Styling ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
        padding: 30px 35px; border-radius: 14px; margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .main-header h1 { color: #f0fdfa; margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { color: #99f6e4; margin: 6px 0 0 0; font-size: 1.05rem; }
    .pred-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 22px; border-radius: 12px; border-left: 5px solid #14b8a6;
        margin-bottom: 16px;
    }
    .pred-card h3 { color: #f0fdfa; margin: 0 0 4px 0; font-size: 1.25rem; }
    .pred-card .score { color: #5eead4; font-size: 14px; font-weight: 600; }
    .dataset-card {
        background: #1e293b; padding: 16px; border-radius: 10px;
        border: 1px solid #334155; margin-bottom: 12px;
    }
    .dataset-card h4 { color: #e2e8f0; margin: 0 0 6px 0; }
    .dataset-card p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }
    div[data-testid="stMetric"] {
        background: #0f172a; padding: 12px; border-radius: 8px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 Crisis Management AI Predictor</h1>
    <p>Describe any economic scenario — get matched historical crises, predicted impacts from real datasets</p>
</div>
""", unsafe_allow_html=True)

# ─── Data Loading ───
@st.cache_data
def load_summary():
    if os.path.exists(SUMMARY_PATH):
        return pd.read_csv(SUMMARY_PATH)
    return None

@st.cache_data
def load_clean():
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_news():
    if os.path.exists(NEWS_PATH):
        df = pd.read_csv(NEWS_PATH)
        df['Date'] = pd.to_datetime(df['publish_date'].astype(str), format="%Y%m%d", errors='coerce')
        return df
    return None

@st.cache_data
def load_crises():
    if os.path.exists(CRISIS_PATH):
        return pd.read_csv(CRISIS_PATH)
    return None

df_summary = load_summary()
df_clean   = load_clean()
df_news    = load_news()
df_crises  = load_crises()

# ─── NLP Helpers ───
STOPWORDS = {
    'and','or','in','the','a','of','to','for','with','on','at','by','from',
    'an','is','are','was','were','be','been','being','this','that','these',
    'those','it','its','about','as','but','not','has','had','have','very',
    'will','can','could','would','should','may','might','shall','do','does',
    'did','then','than','so','if','when','where','what','which','who','how'
}

def tokenize(text):
    if not isinstance(text, str):
        return set()
    text = text.lower().replace('/', ' ').replace('-', ' ').replace(',', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    return {w for w in text.split() if w not in STOPWORDS and len(w) > 1}

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def find_similar_crises(query_text, df, top_n=3):
    q_tokens = tokenize(query_text)
    scores = []
    for _, row in df.iterrows():
        combined = f"{row['event_name']} {row['crisis_type']} {row.get('trigger_description','')} {row['region']}"
        d_tokens = tokenize(combined)
        sim = jaccard(q_tokens, d_tokens)
        scores.append((sim, row))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_n]

def search_news(df_news, query_text, top_n=5):
    words = [w.strip().lower() for w in query_text.split() if len(w.strip()) > 1]
    if not words:
        return pd.DataFrame()
    mask = np.ones(len(df_news), dtype=bool)
    for w in words:
        mask &= df_news['headline_text'].str.contains(w, case=False, na=False)
    matched = df_news[mask]
    if matched.empty:
        return pd.DataFrame()
    daily = matched.groupby('Date').agg(
        headline_count=('headline_text', 'size'),
        sample_headline=('headline_text', 'first')
    ).reset_index().sort_values('headline_count', ascending=False)
    return daily.head(top_n)

def get_indicators_for_date(df_clean, target_date):
    if df_clean is None:
        return None
    diffs = (df_clean['Date'] - target_date).abs()
    return df_clean.loc[diffs.idxmin()]

# ─── Sidebar ───
st.sidebar.header("🔍 Your Scenario")
user_input = st.sidebar.text_area(
    "Describe the current crisis or economic situation:",
    placeholder="e.g. oil price shock due to geopolitical conflict, global supply chain disruption, currency devaluation in emerging markets",
    height=120
)
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Settings")
num_crises = st.sidebar.slider("Number of similar crises to find", 1, 5, 3)
num_news_dates = st.sidebar.slider("Number of peak news dates", 1, 10, 5)

# ─── Sidebar: Dataset Overview ───
st.sidebar.markdown("---")
st.sidebar.header("📁 Datasets Loaded")
datasets_status = {
    "Crisis Impact Summary": df_summary is not None,
    "Clean Daily Timeline": df_clean is not None,
    "ABC News Headlines (1.2M)": df_news is not None,
    "Crisis Events Template": df_crises is not None,
}
for name, ok in datasets_status.items():
    icon = "✅" if ok else "❌"
    st.sidebar.markdown(f"{icon} {name}")

# ─── Main Content ───
if not user_input:
    # Show dataset explorer when no input
    st.markdown("### 📁 Datasets Overview")
    st.info("👈 **Enter a scenario description in the sidebar** to get AI-powered crisis predictions. Below is an overview of the loaded datasets.")

    tab_crisis, tab_timeline, tab_news, tab_indicators = st.tabs([
        "📋 Crisis Events (20)", "📈 Daily Timeline", "📰 News Headlines", "🌍 World Bank & More"
    ])

    with tab_crisis:
        if df_crises is not None:
            st.markdown("#### Historical Crisis Events Database")
            st.dataframe(df_crises, use_container_width=True, height=500)
            st.caption(f"Total: {len(df_crises)} crisis events with dates, types, triggers, and regions")
        else:
            st.warning("Crisis events CSV not found.")

    with tab_timeline:
        if df_clean is not None:
            st.markdown("#### Merged Daily Economic Timeline")
            st.markdown(f"**Date Range:** `{df_clean['Date'].min().strftime('%Y-%m-%d')}` → `{df_clean['Date'].max().strftime('%Y-%m-%d')}`  |  **Rows:** {len(df_clean):,}  |  **Columns:** {len(df_clean.columns)}")
            col_list = [c for c in df_clean.columns if c != 'Date']
            selected_cols = st.multiselect("Select columns to display:", col_list, default=['brent_oil_price', 'market_avg_close', 'wb_inflation_annual%', 'wb_GDP_current_US'])
            if selected_cols:
                st.dataframe(df_clean[['Date'] + selected_cols].dropna(subset=selected_cols, how='all').tail(100), use_container_width=True, height=400)
        else:
            st.warning("Clean data CSV not found. Run `python clean_and_merge.py` first.")

    with tab_news:
        if df_news is not None:
            st.markdown("#### ABC News Headlines Dataset")
            st.markdown(f"**Total Headlines:** {len(df_news):,}  |  **Date Range:** `{df_news['Date'].min().strftime('%Y-%m-%d')}` → `{df_news['Date'].max().strftime('%Y-%m-%d')}`")
            st.dataframe(df_news[['Date', 'headline_text']].head(200), use_container_width=True, height=400)
        else:
            st.warning("ABC News headlines CSV not found.")

    with tab_indicators:
        if df_summary is not None:
            st.markdown("#### Crisis Impact Summary (Pre-computed)")
            st.dataframe(df_summary, use_container_width=True, height=500)
            st.caption("Shows 6-month and 12-month changes in stocks, oil, inflation, and currency after each crisis start date.")
        else:
            st.warning("Impact summary not found. Run `python analyze_crises.py` first.")

else:
    # ─── PREDICTION MODE ───
    st.markdown(f"### 🔮 Predictions for: *\"{user_input}\"*")

    with st.spinner("Matching your scenario against historical crises and 1.2M news headlines..."):
        # ── Section 1: Similar Historical Crises ──
        similar = []
        if df_summary is not None:
            similar = find_similar_crises(user_input, df_summary, top_n=num_crises)

        # ── Section 2: News Headline Matches ──
        news_peaks = pd.DataFrame()
        if df_news is not None:
            news_peaks = search_news(df_news, user_input, top_n=num_news_dates)

    # ═══ RESULTS ═══
    result_tab1, result_tab2, result_tab3 = st.tabs([
        "🏛️ Similar Historical Crises", "📰 News Headline Analysis", "📊 Predicted Impact Summary"
    ])

    # ── Tab 1: Similar Historical Crises ──
    with result_tab1:
        if not similar:
            st.warning("No crisis data available for matching.")
        else:
            for idx, (score, row) in enumerate(similar):
                event_name = row['event_name']
                st.markdown(f"""
                <div class="pred-card">
                    <h3>#{idx+1} — {event_name}</h3>
                    <span class="score">Match Score: {score*100:.1f}% &nbsp;|&nbsp; Type: {row['crisis_type']} &nbsp;|&nbsp; Region: {row['region']} &nbsp;|&nbsp; Start: {row['start_date']}</span>
                </div>
                """, unsafe_allow_html=True)

                col_metrics, col_chart = st.columns([1, 2])

                with col_metrics:
                    m1, m2 = st.columns(2)
                    with m1:
                        val = row.get('brent_oil_price_change_6m')
                        st.metric("Oil Price (6m)", f"{val:+.1f}%" if pd.notnull(val) else "N/A")
                        val = row.get('market_avg_close_change_6m')
                        st.metric("Stocks (6m)", f"{val:+.1f}%" if pd.notnull(val) else "N/A")
                    with m2:
                        val = row.get('wb_inflation_annual%_change_6m')
                        st.metric("Inflation (6m)", f"{val:+.2f} pp" if pd.notnull(val) else "N/A")
                        val = row.get('african_crises_exch_usd_change_6m')
                        st.metric("Currency (6m)", f"{val:+.1f}%" if pd.notnull(val) else "N/A")

                with col_chart:
                    metrics_data = []
                    metric_map = {
                        'Stock Prices': ('market_avg_close_change_6m', 'market_avg_close_change_12m'),
                        'Brent Oil': ('brent_oil_price_change_6m', 'brent_oil_price_change_12m'),
                        'Inflation': ('wb_inflation_annual%_change_6m', 'wb_inflation_annual%_change_12m'),
                        'Currency': ('african_crises_exch_usd_change_6m', 'african_crises_exch_usd_change_12m'),
                    }
                    for label, (col_6m, col_12m) in metric_map.items():
                        v6 = row.get(col_6m)
                        v12 = row.get(col_12m)
                        if pd.notnull(v6):
                            metrics_data.append({'Indicator': label, 'Change': v6, 'Period': '6 Months'})
                        if pd.notnull(v12):
                            metrics_data.append({'Indicator': label, 'Change': v12, 'Period': '12 Months'})

                    if metrics_data:
                        cdf = pd.DataFrame(metrics_data)
                        chart = alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                            x=alt.X('Period:N', title=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y('Change:Q', title='Change (% / pp)'),
                            color=alt.Color('Period:N', scale=alt.Scale(
                                domain=['6 Months', '12 Months'],
                                range=['#14b8a6', '#0d9488']
                            )),
                            column=alt.Column('Indicator:N', title=None,
                                header=alt.Header(labelFontSize=12, labelColor='#e2e8f0')),
                            tooltip=['Indicator', 'Period', alt.Tooltip('Change:Q', format='.2f')]
                        ).properties(width=130, height=220).configure_view(stroke=None)
                        st.altair_chart(chart, use_container_width=False)
                    else:
                        st.info("No indicator data available for this crisis period.")

                # Expandable detail
                with st.expander(f"📄 View all data for {event_name}"):
                    st.dataframe(pd.DataFrame([row]).T.rename(columns={row.name: "Value"}), use_container_width=True)

                st.markdown("---")

    # ── Tab 2: News Headline Analysis ──
    with result_tab2:
        if news_peaks.empty:
            st.warning(f"No news headlines found matching: **{user_input}**. Try broader keywords like 'oil', 'market', 'debt', 'war'.")
        else:
            st.success(f"Found matching headlines on **{len(news_peaks)}** peak dates.")

            for idx, (_, nrow) in enumerate(news_peaks.iterrows()):
                target_date = nrow['Date']
                headline_cnt = nrow['headline_count']
                sample = nrow['sample_headline']

                indicators = get_indicators_for_date(df_clean, target_date)

                st.markdown(f"""
                <div class="pred-card">
                    <h3>📅 {target_date.strftime('%B %d, %Y')}</h3>
                    <span class="score">{headline_cnt} matching headlines &nbsp;|&nbsp; Sample: "{sample[:80]}..."</span>
                </div>
                """, unsafe_allow_html=True)

                if indicators is not None:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        v = indicators.get('brent_oil_price')
                        st.metric("🛢️ Brent Oil", f"${v:.2f}" if pd.notnull(v) else "N/A")
                    with c2:
                        v = indicators.get('market_avg_close')
                        st.metric("📈 S&P 500 Avg", f"${v:.2f}" if pd.notnull(v) else "N/A")
                    with c3:
                        v = indicators.get('wb_inflation_annual%')
                        st.metric("💹 Inflation", f"{v:.2f}%" if pd.notnull(v) else "N/A")
                    with c4:
                        v = indicators.get('wb_GDP_current_US')
                        st.metric("🏦 World GDP", f"${v/1e12:.1f}T" if pd.notnull(v) else "N/A")
                    with c5:
                        v = indicators.get('african_crises_systemic_crisis')
                        st.metric("⚠️ Systemic Crises", f"{int(v)} countries" if pd.notnull(v) else "N/A")

                    # COVID row if applicable
                    cov = indicators.get('covid_impact_avg_stringency_index')
                    if pd.notnull(cov) and cov > 0:
                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            st.metric("🦠 COVID Stringency", f"{cov:.1f}")
                        with cc2:
                            mob = indicators.get('covid_impact_avg_mobility')
                            st.metric("🚶 COVID Mobility", f"{mob:.1f}" if pd.notnull(mob) else "N/A")
                        with cc3:
                            inv = indicators.get('covid_impact_avg_investment_shock_short')
                            st.metric("📉 Investment Shock", f"{inv:.1f}%" if pd.notnull(inv) else "N/A")

                    # Show matching headlines on this date
                    with st.expander(f"📰 View all {headline_cnt} headlines on {target_date.strftime('%Y-%m-%d')}"):
                        day_headlines = df_news[df_news['Date'] == target_date]['headline_text'].tolist()
                        for h in day_headlines[:20]:
                            st.markdown(f"- {h.capitalize()}")
                        if len(day_headlines) > 20:
                            st.caption(f"...and {len(day_headlines)-20} more")

                st.markdown("")

    # ── Tab 3: Predicted Impact Summary ──
    with result_tab3:
        st.markdown("### Predicted Impact Based on Historical Patterns")
        st.markdown(f"Based on the top **{len(similar)}** most similar historical crises to your scenario, here is the predicted range of impacts:")

        if similar:
            # Build aggregate prediction table
            pred_rows = []
            for score, row in similar:
                pred_rows.append({
                    'Crisis': row['event_name'],
                    'Match %': round(score * 100, 1),
                    'Oil 6m%': row.get('brent_oil_price_change_6m'),
                    'Oil 12m%': row.get('brent_oil_price_change_12m'),
                    'Stock 6m%': row.get('market_avg_close_change_6m'),
                    'Stock 12m%': row.get('market_avg_close_change_12m'),
                    'Inflation 6m (pp)': row.get('wb_inflation_annual%_change_6m'),
                    'Inflation 12m (pp)': row.get('wb_inflation_annual%_change_12m'),
                    'Currency 6m%': row.get('african_crises_exch_usd_change_6m'),
                    'Currency 12m%': row.get('african_crises_exch_usd_change_12m'),
                })
            pred_df = pd.DataFrame(pred_rows)
            st.dataframe(pred_df.style.format({
                'Match %': '{:.1f}',
                'Oil 6m%': '{:+.1f}', 'Oil 12m%': '{:+.1f}',
                'Stock 6m%': '{:+.1f}', 'Stock 12m%': '{:+.1f}',
                'Inflation 6m (pp)': '{:+.2f}', 'Inflation 12m (pp)': '{:+.2f}',
                'Currency 6m%': '{:+.1f}', 'Currency 12m%': '{:+.1f}',
            }, na_rep="—"), use_container_width=True)

            # Weighted average prediction
            st.markdown("---")
            st.markdown("### 🎯 Weighted Average Prediction")
            st.caption("Averages weighted by match score — higher-similarity crises contribute more.")

            weight_cols = ['Oil 6m%', 'Oil 12m%', 'Stock 6m%', 'Stock 12m%',
                          'Inflation 6m (pp)', 'Inflation 12m (pp)', 'Currency 6m%', 'Currency 12m%']
            weights = pred_df['Match %'].values
            w_sum = weights.sum()

            wa_results = {}
            for col in weight_cols:
                vals = pred_df[col].values
                valid_mask = ~pd.isna(vals)
                if valid_mask.any():
                    wa = np.average(vals[valid_mask].astype(float), weights=weights[valid_mask])
                    wa_results[col] = wa
                else:
                    wa_results[col] = None

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                v = wa_results.get('Oil 6m%')
                st.metric("🛢️ Oil Price (6m)", f"{v:+.1f}%" if v is not None else "—")
                v = wa_results.get('Oil 12m%')
                st.metric("🛢️ Oil Price (12m)", f"{v:+.1f}%" if v is not None else "—")
            with c2:
                v = wa_results.get('Stock 6m%')
                st.metric("📈 Stocks (6m)", f"{v:+.1f}%" if v is not None else "—")
                v = wa_results.get('Stock 12m%')
                st.metric("📈 Stocks (12m)", f"{v:+.1f}%" if v is not None else "—")
            with c3:
                v = wa_results.get('Inflation 6m (pp)')
                st.metric("💹 Inflation (6m)", f"{v:+.2f} pp" if v is not None else "—")
                v = wa_results.get('Inflation 12m (pp)')
                st.metric("💹 Inflation (12m)", f"{v:+.2f} pp" if v is not None else "—")
            with c4:
                v = wa_results.get('Currency 6m%')
                st.metric("💱 Currency (6m)", f"{v:+.1f}%" if v is not None else "—")
                v = wa_results.get('Currency 12m%')
                st.metric("💱 Currency (12m)", f"{v:+.1f}%" if v is not None else "—")

            # Visualization
            st.markdown("---")
            wa_chart_data = []
            nice_names = {
                'Oil 6m%': 'Oil Price', 'Oil 12m%': 'Oil Price',
                'Stock 6m%': 'Stocks', 'Stock 12m%': 'Stocks',
                'Inflation 6m (pp)': 'Inflation', 'Inflation 12m (pp)': 'Inflation',
                'Currency 6m%': 'Currency', 'Currency 12m%': 'Currency',
            }
            for col in weight_cols:
                v = wa_results.get(col)
                if v is not None:
                    period = '6 Months' if '6m' in col else '12 Months'
                    wa_chart_data.append({
                        'Indicator': nice_names[col],
                        'Period': period,
                        'Predicted Change': v
                    })

            if wa_chart_data:
                wa_df = pd.DataFrame(wa_chart_data)
                pred_chart = alt.Chart(wa_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X('Period:N', title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Predicted Change:Q', title='Predicted Change (% / pp)'),
                    color=alt.condition(
                        alt.datum['Predicted Change'] > 0,
                        alt.value('#10b981'),
                        alt.value('#ef4444')
                    ),
                    column=alt.Column('Indicator:N', title=None,
                        header=alt.Header(labelFontSize=13, labelColor='#e2e8f0')),
                    tooltip=['Indicator', 'Period', alt.Tooltip('Predicted Change:Q', format='.2f')]
                ).properties(width=140, height=250).configure_view(stroke=None)
                st.altair_chart(pred_chart, use_container_width=False)

            # Download
            st.markdown("---")
            csv_out = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Prediction Summary CSV",
                data=csv_out,
                file_name="crisis_prediction_summary.csv",
                mime="text/csv"
            )
        else:
            st.warning("No historical crisis data available for predictions.")
