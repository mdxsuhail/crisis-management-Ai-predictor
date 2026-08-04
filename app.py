import os, string
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ─── Page Config ───
st.set_page_config(page_title="Crisis Management AI Predictor", page_icon="📊", layout="wide")

# ─── Paths (relative to script) ───
BASE = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH   = os.path.join(BASE, "processed", "crisis_impact_summary.csv")
CLEAN_DATA_PATH= os.path.join(BASE, "processed", "clean_data.csv")
NEWS_PATH      = os.path.join(BASE, "dataset",   "abcnews-date-text.csv")
CRISIS_PATH    = os.path.join(BASE, "dataset",   "crisis_events.csv")

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
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d9488 100%);
      padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;
  }
  .hero h1 { color:#f0fdfa; margin:0; font-size:2rem; }
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
</style>""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""<div class="hero">
  <h1>📊 Crisis Management AI Predictor</h1>
  <p>Type any economic scenario → get matched historical crises, news signals, and predicted impacts from 6 real datasets</p>
</div>""", unsafe_allow_html=True)

# ─── Data Loaders ───
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

df_summary = load_summary()
df_clean   = load_clean()
df_news    = load_news()
df_crises  = load_crises()

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
    if pd.isna(val): return "—"
    return f"{prefix}{val:+.{decimals}f}{suffix}" if val != 0 else f"{prefix}0{suffix}"

def fmt_abs(val, suffix='', decimals=2, prefix='$'):
    if pd.isna(val): return "—"
    return f"{prefix}{val:,.{decimals}f}{suffix}"

# ─── Sidebar ───
st.sidebar.markdown("### 🔍 Describe Your Scenario")
user_input = st.sidebar.text_area(
    "What is happening?",
    placeholder="e.g. oil prices crashing due to oversupply, trade war between major economies, pandemic lockdowns",
    height=100
)
num_matches = st.sidebar.slider("Similar crises to find", 1, 5, 3)
num_news    = st.sidebar.slider("Peak news dates to show", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Data Status")
checks = [
    ("20 Historical Crises", df_crises is not None),
    ("Daily Timeline (11K rows)", df_clean is not None),
    ("1.2M News Headlines", df_news is not None),
    ("Impact Summary (20 crises)", df_summary is not None),
]
for label, ok in checks:
    st.sidebar.markdown(f"{'✅' if ok else '❌'} {label}")

# ═══════════════════════════════════════════════
# MODE 1: NO INPUT → Show Dataset Explorer
# ═══════════════════════════════════════════════
if not user_input:
    st.info("👈 **Type a scenario in the sidebar** to get predictions. Below is a preview of all loaded datasets.")

    t1, t2, t3, t4 = st.tabs(["📋 Crisis Events", "📈 Economic Timeline", "📰 News Headlines", "📊 Impact Analysis"])

    with t1:
        if df_crises is not None:
            st.markdown(f"**{len(df_crises)} historical crises** from 1929 to 2020 — each with type, trigger, dates and region.")
            for _, cr in df_crises.iterrows():
                bc = badge_class(cr['crisis_type'])
                st.markdown(f"""
                <div style="background:#1e293b;padding:14px 18px;border-radius:10px;margin-bottom:10px;border-left:4px solid #14b8a6;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#f0fdfa;font-size:1.05rem;font-weight:600;">{cr['event_name']}</span>
                    <span class="crisis-badge {bc}">{cr['crisis_type']}</span>
                  </div>
                  <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">
                    📅 {cr['start_date']} → {cr['end_date']} &nbsp;|&nbsp; 🌍 {cr['region']}
                  </div>
                  <div style="color:#cbd5e1;font-size:0.85rem;margin-top:4px;">{cr['trigger_description']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.warning("Crisis events file not found.")

    with t2:
        if df_clean is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", f"{len(df_clean):,}")
            c2.metric("Date Range", f"{df_clean['Date'].min().strftime('%Y')}–{df_clean['Date'].max().strftime('%Y')}")
            c3.metric("Columns", f"{len(df_clean.columns)}")

            st.markdown("**Select indicators to explore:**")
            nice_cols = {
                'brent_oil_price': '🛢️ Brent Oil Price',
                'market_avg_close': '📈 S&P 500 Avg Close',
                'wb_inflation_annual%': '💹 Inflation Rate',
                'wb_GDP_current_US': '🏦 Global GDP',
                'african_crises_exch_usd': '💱 Exchange Rate (USD)',
                'african_crises_systemic_crisis': '⚠️ Systemic Crisis Count',
                'news_headline_count': '📰 Daily News Count',
            }
            chosen = st.multiselect("Indicators", list(nice_cols.keys()),
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
                ).properties(height=350).interactive()
                st.altair_chart(line_chart, use_container_width=True)

                st.markdown("**Raw data (last 50 rows):**")
                st.dataframe(plot_df.tail(50), use_container_width=True, height=250)
        else:
            st.warning("Run `python clean_and_merge.py` first.")

    with t3:
        if df_news is not None:
            c1, c2 = st.columns(2)
            c1.metric("Total Headlines", f"{len(df_news):,}")
            c2.metric("Date Range", f"{df_news['Date'].min().strftime('%Y-%m-%d')} → {df_news['Date'].max().strftime('%Y-%m-%d')}")
            quick = st.text_input("🔍 Quick search headlines:", placeholder="e.g. oil, war, pandemic, market crash")
            display_df = df_news[['Date', 'headline_text']].copy()
            if quick:
                mask = display_df['headline_text'].str.contains(quick, case=False, na=False)
                display_df = display_df[mask]
                st.caption(f"Showing {len(display_df):,} matching headlines")
            st.dataframe(display_df.head(300), use_container_width=True, height=400)
        else:
            st.warning("ABC News headlines CSV not found.")

    with t4:
        if df_summary is not None:
            st.markdown("**Pre-computed 6-month and 12-month changes after each crisis start date:**")
            display_cols = ['event_name', 'crisis_type', 'start_date',
                           'brent_oil_price_change_6m', 'brent_oil_price_change_12m',
                           'market_avg_close_change_6m', 'market_avg_close_change_12m',
                           'wb_inflation_annual%_change_6m', 'wb_inflation_annual%_change_12m',
                           'african_crises_exch_usd_change_6m', 'african_crises_exch_usd_change_12m']
            avail = [c for c in display_cols if c in df_summary.columns]
            styled = df_summary[avail].style.format({
                c: '{:+.1f}' for c in avail if 'change' in c
            }, na_rep='—')
            st.dataframe(styled, use_container_width=True, height=500)
        else:
            st.warning("Run `python analyze_crises.py` first.")

# ═══════════════════════════════════════════════
# MODE 2: USER INPUT → Prediction Mode
# ═══════════════════════════════════════════════
else:
    st.markdown(f"### 🔮 Predictions for: *\"{user_input}\"*")

    # ── Compute matches ──
    q_tokens = tokenize(user_input)
    similar = []
    if df_summary is not None:
        for _, row in df_summary.iterrows():
            combined = f"{row['event_name']} {row['crisis_type']} {row['region']}"
            sim = jaccard(q_tokens, tokenize(combined))
            similar.append((sim, row))
        similar.sort(key=lambda x: x[0], reverse=True)
        similar = similar[:num_matches]

    news_peaks = pd.DataFrame()
    if df_news is not None:
        words = [w.strip().lower() for w in user_input.split() if len(w.strip()) > 1]
        if words:
            mask = np.ones(len(df_news), dtype=bool)
            for w in words:
                mask &= df_news['headline_text'].str.contains(w, case=False, na=False)
            matched = df_news[mask]
            if not matched.empty:
                news_peaks = matched.groupby('Date').agg(
                    count=('headline_text','size'),
                    sample=('headline_text','first')
                ).reset_index().sort_values('count', ascending=False).head(num_news)

    # ── Results Tabs ──
    tab_crisis, tab_news, tab_predict = st.tabs([
        f"🏛️ {len(similar)} Matched Crises",
        f"📰 {len(news_peaks)} News Peaks",
        "📊 Predicted Impact"
    ])

    # ═══ TAB 1: Matched Crises ═══
    with tab_crisis:
        if not similar:
            st.warning("No crisis data loaded.")
        else:
            for i, (score, row) in enumerate(similar):
                name = row['event_name']
                bc = badge_class(row['crisis_type'])

                st.markdown(f"""<div style="background:#1e293b;padding:16px 20px;border-radius:12px;
                    margin-bottom:6px;border-left:5px solid #14b8a6;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#f0fdfa;font-size:1.15rem;font-weight:700;">#{i+1} {name}</span>
                    <span style="color:#5eead4;font-weight:600;font-size:0.95rem;">{score*100:.0f}% match</span>
                  </div>
                  <div style="margin-top:6px;">
                    <span class="crisis-badge {bc}">{row['crisis_type']}</span>
                    <span style="color:#94a3b8;font-size:0.85rem;">📅 {row['start_date']} &nbsp;|&nbsp; 🌍 {row['region']}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Metric cards in 4 columns
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    v6 = row.get('brent_oil_price_change_6m')
                    v12 = row.get('brent_oil_price_change_12m')
                    st.metric("🛢️ Oil (6m)", fmt(v6))
                    st.metric("🛢️ Oil (12m)", fmt(v12))
                with m2:
                    v6 = row.get('market_avg_close_change_6m')
                    v12 = row.get('market_avg_close_change_12m')
                    st.metric("📈 Stocks (6m)", fmt(v6))
                    st.metric("📈 Stocks (12m)", fmt(v12))
                with m3:
                    v6 = row.get('wb_inflation_annual%_change_6m')
                    v12 = row.get('wb_inflation_annual%_change_12m')
                    st.metric("💹 Inflation (6m)", fmt(v6, ' pp', 2))
                    st.metric("💹 Inflation (12m)", fmt(v12, ' pp', 2))
                with m4:
                    v6 = row.get('african_crises_exch_usd_change_6m')
                    v12 = row.get('african_crises_exch_usd_change_12m')
                    st.metric("💱 Currency (6m)", fmt(v6))
                    st.metric("💱 Currency (12m)", fmt(v12))

                # Mini chart
                chart_rows = []
                pairs = [
                    ('Oil Price', 'brent_oil_price_change_6m', 'brent_oil_price_change_12m'),
                    ('Stocks',    'market_avg_close_change_6m','market_avg_close_change_12m'),
                    ('Inflation', 'wb_inflation_annual%_change_6m','wb_inflation_annual%_change_12m'),
                    ('Currency',  'african_crises_exch_usd_change_6m','african_crises_exch_usd_change_12m'),
                ]
                for label, c6, c12 in pairs:
                    if pd.notna(row.get(c6)):
                        chart_rows.append({'Indicator': label, 'Period': '6 Months', 'Change': row[c6]})
                    if pd.notna(row.get(c12)):
                        chart_rows.append({'Indicator': label, 'Period': '12 Months', 'Change': row[c12]})

                if chart_rows:
                    cdf = pd.DataFrame(chart_rows)
                    ch = alt.Chart(cdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                        x=alt.X('Period:N', title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Change:Q', title='Change (%)'),
                        color=alt.condition(
                            alt.datum.Change > 0, alt.value('#10b981'), alt.value('#ef4444')
                        ),
                        column=alt.Column('Indicator:N', title=None,
                            header=alt.Header(labelFontSize=12, labelColor='#cbd5e1')),
                        tooltip=['Indicator','Period', alt.Tooltip('Change:Q', format='+.2f')]
                    ).properties(width=120, height=180)
                    st.altair_chart(ch, use_container_width=False)

                with st.expander(f"📄 Full data for {name}"):
                    detail_df = pd.DataFrame([row]).T.reset_index()
                    detail_df.columns = ['Field', 'Value']
                    detail_df = detail_df[detail_df['Value'].notna()]
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                st.markdown("---")

    # ═══ TAB 2: News Headlines ═══
    with tab_news:
        if news_peaks.empty:
            st.warning(f"No news headlines matched **\"{user_input}\"**. Try simpler keywords like `oil`, `war`, `debt`, `market`.")
        else:
            st.success(f"Found **{len(news_peaks)}** peak coverage dates in 1.2M headlines.")
            for _, nrow in news_peaks.iterrows():
                dt = nrow['Date']
                cnt = nrow['count']

                st.markdown(f"""<div style="background:#1e293b;padding:14px 18px;border-radius:10px;
                    margin-bottom:6px;border-left:4px solid #f59e0b;">
                  <span style="color:#fbbf24;font-weight:700;font-size:1.05rem;">📅 {dt.strftime('%B %d, %Y')}</span>
                  <span style="color:#94a3b8;margin-left:12px;">{cnt} matching headlines</span>
                </div>""", unsafe_allow_html=True)

                ind = None
                if df_clean is not None:
                    diffs = (df_clean['Date'] - dt).abs()
                    ind = df_clean.loc[diffs.idxmin()]

                if ind is not None:
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("🛢️ Oil Price", fmt_abs(ind.get('brent_oil_price')))
                    with c2:
                        st.metric("📈 S&P 500", fmt_abs(ind.get('market_avg_close')))
                    with c3:
                        v = ind.get('wb_inflation_annual%')
                        st.metric("💹 Inflation", f"{v:.2f}%" if pd.notna(v) else "—")
                    with c4:
                        v = ind.get('wb_GDP_current_US')
                        st.metric("🏦 GDP", f"${v/1e12:.1f}T" if pd.notna(v) else "—")
                    with c5:
                        v = ind.get('african_crises_systemic_crisis')
                        st.metric("⚠️ Crises", f"{int(v)}" if pd.notna(v) else "—")

                    # COVID data if present
                    cov = ind.get('covid_impact_avg_stringency_index')
                    if pd.notna(cov) and cov > 0:
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.metric("🦠 COVID Stringency", f"{cov:.1f}")
                        with cc2:
                            mob = ind.get('covid_impact_avg_mobility')
                            st.metric("🚶 Mobility", f"{mob:.1f}" if pd.notna(mob) else "—")

                with st.expander(f"📰 Headlines on {dt.strftime('%Y-%m-%d')} ({cnt} total)"):
                    day_h = df_news[df_news['Date'] == dt]['headline_text'].tolist()
                    for h in day_h[:15]:
                        st.markdown(f"- {h.capitalize()}")
                    if len(day_h) > 15:
                        st.caption(f"…and {len(day_h)-15} more")
                st.markdown("")

    # ═══ TAB 3: Predicted Impact ═══
    with tab_predict:
        if not similar:
            st.warning("No historical data to generate predictions.")
        else:
            st.markdown("### 🎯 Weighted Prediction from Matched Crises")
            st.caption("Higher-similarity crises contribute more weight to the prediction.")

            # Build comparison table
            rows = []
            for score, row in similar:
                rows.append({
                    'Crisis': row['event_name'],
                    'Match': f"{score*100:.0f}%",
                    'Oil 6m': fmt(row.get('brent_oil_price_change_6m')),
                    'Oil 12m': fmt(row.get('brent_oil_price_change_12m')),
                    'Stocks 6m': fmt(row.get('market_avg_close_change_6m')),
                    'Stocks 12m': fmt(row.get('market_avg_close_change_12m')),
                    'Infl 6m': fmt(row.get('wb_inflation_annual%_change_6m'), ' pp', 2),
                    'Infl 12m': fmt(row.get('wb_inflation_annual%_change_12m'), ' pp', 2),
                    'FX 6m': fmt(row.get('african_crises_exch_usd_change_6m')),
                    'FX 12m': fmt(row.get('african_crises_exch_usd_change_12m')),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Compute weighted averages
            st.markdown("---")
            st.markdown("### 📈 Weighted Average Forecast")

            weight_keys = [
                ('🛢️ Oil Price',  'brent_oil_price_change_6m',        'brent_oil_price_change_12m'),
                ('📈 Stocks',     'market_avg_close_change_6m',       'market_avg_close_change_12m'),
                ('💹 Inflation',  'wb_inflation_annual%_change_6m',   'wb_inflation_annual%_change_12m'),
                ('💱 Currency',   'african_crises_exch_usd_change_6m','african_crises_exch_usd_change_12m'),
            ]
            weights = np.array([s for s, _ in similar])

            wa_6m = {}
            wa_12m = {}
            for label, col6, col12 in weight_keys:
                vals6  = np.array([r.get(col6)  for _, r in similar], dtype=float)
                vals12 = np.array([r.get(col12) for _, r in similar], dtype=float)
                m6  = ~np.isnan(vals6)
                m12 = ~np.isnan(vals12)
                wa_6m[label]  = np.average(vals6[m6],  weights=weights[m6])  if m6.any()  else None
                wa_12m[label] = np.average(vals12[m12], weights=weights[m12]) if m12.any() else None

            cols = st.columns(4)
            for idx, (label, _, _) in enumerate(weight_keys):
                with cols[idx]:
                    v6  = wa_6m.get(label)
                    v12 = wa_12m.get(label)
                    suf = ' pp' if 'Inflation' in label else '%'
                    dec = 2 if 'Inflation' in label else 1
                    st.metric(f"{label} (6m forecast)",
                              f"{v6:+.{dec}f}{suf}" if v6 is not None else "—")
                    st.metric(f"{label} (12m forecast)",
                              f"{v12:+.{dec}f}{suf}" if v12 is not None else "—")

            # Prediction chart
            st.markdown("---")
            chart_data = []
            for label, _, _ in weight_keys:
                nice = label.split(' ', 1)[1] if ' ' in label else label
                if wa_6m.get(label) is not None:
                    chart_data.append({'Indicator': nice, 'Period': '6 Months', 'Predicted': wa_6m[label]})
                if wa_12m.get(label) is not None:
                    chart_data.append({'Indicator': nice, 'Period': '12 Months', 'Predicted': wa_12m[label]})

            if chart_data:
                pcd = pd.DataFrame(chart_data)
                pred_ch = alt.Chart(pcd).mark_bar(
                    cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=28
                ).encode(
                    x=alt.X('Period:N', title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Predicted:Q', title='Predicted Change (% / pp)'),
                    color=alt.condition(
                        alt.datum.Predicted > 0,
                        alt.value('#10b981'),
                        alt.value('#ef4444')
                    ),
                    column=alt.Column('Indicator:N', title=None,
                        header=alt.Header(labelFontSize=13, labelColor='#e2e8f0')),
                    tooltip=['Indicator', 'Period', alt.Tooltip('Predicted:Q', format='+.2f')]
                ).properties(width=150, height=260)
                st.altair_chart(pred_ch, use_container_width=False)

            # Download
            st.markdown("---")
            dl_rows = []
            for score, row in similar:
                r = {'Crisis': row['event_name'], 'Match Score': f"{score*100:.0f}%"}
                for label, c6, c12 in weight_keys:
                    r[f'{label} 6m'] = row.get(c6)
                    r[f'{label} 12m'] = row.get(c12)
                dl_rows.append(r)
            st.download_button("📥 Download Full Prediction Report",
                data=pd.DataFrame(dl_rows).to_csv(index=False).encode('utf-8'),
                file_name="crisis_prediction_report.csv", mime="text/csv")
