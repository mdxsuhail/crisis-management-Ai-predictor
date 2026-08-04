import os
import string
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Set page config
st.set_page_config(page_title="Historical Crisis Explorer", layout="wide")

# App title and styling
st.markdown("""
<div style="background-color:#1e293b;padding:25px;border-radius:10px;margin-bottom:25px;box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);">
    <h1 style="color:#f8fafc;margin:0;font-family:sans-serif;font-size:2.2rem;font-weight:700">Historical Financial Crisis Impact Explorer</h1>
    <p style="color:#94a3b8;margin:8px 0 0 0;font-family:sans-serif;font-size:1.1rem">
        Analyze and compare past financial crises to understand the post-event effects on key economic indicators.
    </p>
</div>
""", unsafe_allow_html=True)

SUMMARY_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\processed\crisis_impact_summary.csv"
CLEAN_DATA_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\processed\clean_data.csv"
NEWS_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\dataset\abcnews-date-text.csv"

@st.cache_data
def load_summary_data():
    if os.path.exists(SUMMARY_PATH):
        return pd.read_csv(SUMMARY_PATH)
    return None

@st.cache_data
def load_clean_data():
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_news_data():
    if os.path.exists(NEWS_PATH):
        df = pd.read_csv(NEWS_PATH)
        df['Date'] = pd.to_datetime(df['publish_date'].astype(str), format="%Y%m%d", errors='coerce')
        return df
    return None

df_summary = load_summary_data()

if df_summary is None:
    st.error(f"Error: Impact summary dataset missing at `{SUMMARY_PATH}`. Please run `analyze_crises.py` first.")
else:
    # Sidebar Navigation Mode
    st.sidebar.header("Navigation Modes")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Pre-defined Crisis Explorer", "Headline News Predictor"])
    
    if app_mode == "Pre-defined Crisis Explorer":
        # Basic tokenize & similarity functions
        STOPWORDS = {
            'and', 'or', 'in', 'the', 'a', 'of', 'to', 'for', 'with', 'on', 'at', 
            'by', 'from', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'this', 'that', 'these', 'those', 'it', 'its', 'shock', 'crisis', 'crash', 
            'event', 'about', 'as', 'but', 'not', 'trigger', 'scope'
        }

        def tokenize(text):
            if not isinstance(text, str):
                return set()
            text_clean = text.lower().replace('/', ' ').replace('-', ' ')
            text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
            tokens = {word for word in text_clean.split() if word not in STOPWORDS}
            return tokens

        def find_similar(df, query_text, top_n=3):
            query_tokens = tokenize(query_text)
            scores = []
            for idx, row in df.iterrows():
                combined_text = f"{row['event_name']} {row['crisis_type']} {row['region']}"
                doc_tokens = tokenize(combined_text)
                if not query_tokens or not doc_tokens:
                    sim = 0.0
                else:
                    sim = len(query_tokens.intersection(doc_tokens)) / len(query_tokens.union(doc_tokens))
                scores.append((sim, row))
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            return scores[:top_n]

        st.sidebar.subheader("Search Parameters")
        crisis_types = sorted(list(df_summary['crisis_type'].dropna().unique()))
        search_mode = st.sidebar.radio("Select Search Input Mode", ["Dropdown Select", "Custom Description"])
        
        if search_mode == "Dropdown Select":
            selected_type = st.sidebar.selectbox("Choose Crisis Type", crisis_types)
            query = selected_type
        else:
            query = st.sidebar.text_input(
                "Describe the Situation",
                "oil price shock, geopolitical trigger, global scope"
            )
            
        top_n = st.sidebar.slider("Number of Similar Crises to Find", 1, 5, 3)
        
        # Find similar crises
        similar_crises = find_similar(df_summary, query, top_n=top_n)
        
        st.subheader(f"Analyzing Similar Historical Events for: '{query}'")
        
        # Create Tabs
        tab_titles = [f"Rank {i+1}: {row['event_name']} ({round(score * 100, 1)}% Match)" for i, (score, row) in enumerate(similar_crises)]
        tabs = st.tabs(tab_titles)
        
        for idx, (score, row) in enumerate(similar_crises):
            with tabs[idx]:
                event_name = row['event_name']
                col_meta, col_chart = st.columns([1, 2])
                
                with col_meta:
                    st.markdown(f"""
                    <div style="background-color:#0f172a;padding:20px;border-radius:8px;border-left:5px solid #3b82f6;margin-bottom:20px">
                        <h3 style="color:#f1f5f9;margin:0;font-size:1.4rem">{event_name}</h3>
                        <span style="color:#60a5fa;font-size:15px;font-weight:bold">Similarity Score: {round(score * 100, 1)}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    **Start Date:** `{row['start_date']}`  
                    **Crisis Type:** {row['crisis_type']}  
                    **Region affected:** {row['region']}  
                    """)
                    
                    csv_data = pd.DataFrame([row]).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Download {event_name} Data",
                        data=csv_data,
                        file_name=f"{event_name.lower().replace(' ', '_')}_data.csv",
                        mime="text/csv",
                        key=f"download_{idx}"
                    )
                    
                    show_details = st.checkbox("🔍 Show All Raw Parameters", key=f"details_{idx}")
                    if show_details:
                        st.write(row.dropna().to_frame(name="Value"))
                
                with col_chart:
                    st.markdown("#### Post-Crisis Performance Indicators")
                    
                    metrics = []
                    changes_6m = []
                    changes_12m = []
                    
                    if pd.notnull(row['market_avg_close_change_6m']):
                        metrics.append('Stock Prices')
                        changes_6m.append(row['market_avg_close_change_6m'])
                        changes_12m.append(row['market_avg_close_change_12m'])
                        
                    if pd.notnull(row['african_crises_exch_usd_change_6m']):
                        metrics.append('Currency vs USD')
                        changes_6m.append(row['african_crises_exch_usd_change_6m'])
                        changes_12m.append(row['african_crises_exch_usd_change_12m'])
                        
                    if pd.notnull(row['brent_oil_price_change_6m']):
                        metrics.append('Brent Oil Price')
                        changes_6m.append(row['brent_oil_price_change_6m'])
                        changes_12m.append(row['brent_oil_price_change_12m'])
                        
                    if pd.notnull(row['wb_inflation_annual%_change_6m']):
                        metrics.append('Global Inflation (pp)')
                        changes_6m.append(row['wb_inflation_annual%_change_6m'])
                        changes_12m.append(row['wb_inflation_annual%_change_12m'])
                    
                    if metrics:
                        chart_df = pd.DataFrame({
                            'Indicator': metrics * 2,
                            'Change (%)': changes_6m + changes_12m,
                            'Period': ['6 Months After'] * len(metrics) + ['12 Months After'] * len(metrics)
                        })
                        
                        chart = alt.Chart(chart_df).mark_bar().encode(
                            x=alt.X('Period:N', title=None, axis=alt.Axis(labels=True, ticks=True)),
                            y=alt.Y('Change (%):Q', title='Percentage Change (%) / pp Change'),
                            color=alt.Color('Period:N', scale=alt.Scale(domain=['6 Months After', '12 Months After'], range=['#60a5fa', '#2563eb'])),
                            column=alt.Column('Indicator:N', title=None, header=alt.Header(labelColor='#e2e8f0', labelFontSize=13)),
                            tooltip=['Indicator', 'Period', 'Change (%)']
                        ).properties(
                            width=140,
                            height=250
                        ).configure_view(
                            stroke=None
                        ).configure_legend(
                            orient='bottom',
                            title=None,
                            labelFontSize=11
                        )
                        
                        st.altair_chart(chart, use_container_width=False)
                    else:
                        st.info("💡 No post-crisis performance indicator records exist in our timeline for this historical era (the daily master timeline starts in May 1987).")

    else: # Headline News Predictor
        st.subheader("Headline News Impact Predictor")
        st.markdown("""
        Search for historical keywords in the **ABC News Headlines** dataset. This tool identifies the peak news dates, retrieves corresponding **S&P 500 stocks** and **Brent oil prices** data, and displays expected conditions based on **Covid-19**, **African Crises**, and **World Bank indicators**.
        """)
        
        # Load datasets
        df_news = load_news_data()
        df_clean = load_clean_data()
        
        if df_news is None or df_clean is None:
            st.error("Error: Missing required datasets. Ensure abcnews-date-text.csv and processed/clean_data.csv are present.")
        else:
            query = st.text_input("Enter News Headline Keywords (e.g. 'oil price', 'mortgage', 'default', 'pandemic', 'embargo')", "oil price")
            
            if query:
                # Tokenize & search
                words = [w.strip().lower() for w in query.split() if w.strip()]
                
                with st.spinner("Analyzing 1.2M news headlines and daily economic metrics..."):
                    # Find matching headlines
                    # To do this fast, search rows where text contains all words
                    mask = np.ones(len(df_news), dtype=bool)
                    for w in words:
                        mask &= df_news['headline_text'].str.contains(w, case=False, na=False)
                        
                    df_matched = df_news[mask]
                    
                    if df_matched.empty:
                        st.warning(f"No headlines found containing keywords: {words}")
                    else:
                        # Group by Date and sort to find peak dates
                        df_dates = df_matched.groupby('Date').size().reset_index(name='headline_count')
                        df_dates = df_dates.sort_values('headline_count', ascending=False)
                        top_dates = df_dates.head(3)
                        
                        st.success(f"Matched {len(df_matched)} headlines. Displaying the top {len(top_dates)} peak coverage dates:")
                        
                        # Comparison DataFrame
                        comparison_data = []
                        
                        tab_titles = [f"Peak Date {i+1}: {row['Date'].strftime('%Y-%m-%d')} ({row['headline_count']} news)" for i, row in top_dates.reset_index().iterrows()]
                        tabs = st.tabs(tab_titles)
                        
                        for idx, (index_val, date_row) in enumerate(top_dates.iterrows()):
                            target_date = date_row['Date']
                            headline_cnt = date_row['headline_count']
                            
                            # Find closest economic record in clean_data.csv
                            diffs = (df_clean['Date'] - target_date).abs()
                            closest_idx = diffs.idxmin()
                            indicator_row = df_clean.loc[closest_idx]
                            actual_date = indicator_row['Date']
                            
                            # Add to comparison data list
                            comparison_data.append({
                                'Peak Date': target_date.strftime('%Y-%m-%d'),
                                'S&P 500 Close': indicator_row['market_avg_close'],
                                'Brent Oil Price': indicator_row['brent_oil_price'],
                                'Global Inflation': indicator_row['wb_inflation_annual%'],
                                'Systemic Crises': indicator_row['african_crises_systemic_crisis']
                            })
                            
                            with tabs[idx]:
                                # Left: Sample headlines, Right: Mapped Indicators
                                col_left, col_right = st.columns([1, 1])
                                
                                with col_left:
                                    st.markdown("#### Sample Headlines on this Date")
                                    headlines = df_matched[df_matched['Date'] == target_date]['headline_text'].head(5).tolist()
                                    for h in headlines:
                                        st.markdown(f"- *{h.capitalize()}*")
                                        
                                    st.markdown("---")
                                    # Show raw values download button
                                    row_df = pd.DataFrame([indicator_row])
                                    st.download_button(
                                        label="📥 Download Mapped Date Indicators",
                                        data=row_df.to_csv(index=False).encode('utf-8'),
                                        file_name=f"indicators_{target_date.strftime('%Y-%m-%d')}.csv",
                                        mime="text/csv",
                                        key=f"dl_news_{idx}"
                                    )
                                    
                                with col_right:
                                    st.markdown(f"#### Economic Indicator Forecast (As of: `{actual_date.strftime('%Y-%m-%d')}`)")
                                    
                                    col_m1, col_m2 = st.columns(2)
                                    with col_m1:
                                        # Financials
                                        st.metric("S&P 500 Stock Close", f"${indicator_row['market_avg_close']:.2f}" if pd.notnull(indicator_row['market_avg_close']) else "N/A")
                                        st.metric("Brent Oil Price", f"${indicator_row['brent_oil_price']:.2f}" if pd.notnull(indicator_row['brent_oil_price']) else "N/A")
                                        st.metric("S&P 500 Volume", f"{indicator_row['market_total_volume']:,.0f}" if pd.notnull(indicator_row['market_total_volume']) else "N/A")
                                    with col_m2:
                                        # Macro & Shocks
                                        st.metric("World Bank Inflation", f"{indicator_row['wb_inflation_annual%']:.2f}%" if pd.notnull(indicator_row['wb_inflation_annual%']) else "N/A")
                                        st.metric("African Systemic Crises", f"{int(indicator_row['african_crises_systemic_crisis'])} active countries" if pd.notnull(indicator_row['african_crises_systemic_crisis']) else "N/A")
                                        st.metric("World Bank GDP", f"${indicator_row['wb_GDP_current_US']/1e12:.2f}T" if pd.notnull(indicator_row['wb_GDP_current_US']) else "N/A")
                                        
                                    # Covid-19 details if available
                                    cov_s = indicator_row['covid_impact_avg_stringency_index']
                                    if pd.notnull(cov_s) and cov_s > 0:
                                        st.markdown("---")
                                        col_c1, col_c2 = st.columns(2)
                                        with col_c1:
                                            st.metric("COVID average Stringency Index", f"{indicator_row['covid_impact_avg_stringency_index']:.2f}")
                                        with col_c2:
                                            st.metric("COVID average Mobility", f"{indicator_row['covid_impact_avg_mobility']:.2f}")
                                            
                        # Render comparison chart
                        st.markdown("---")
                        st.markdown("### Peak Interest Dates Comparison Charts")
                        df_comp = pd.DataFrame(comparison_data)
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            st.write(df_comp.set_index("Peak Date"))
                            
                        with col_chart2:
                            df_comp_melted = df_comp.melt(id_vars='Peak Date', value_vars=['S&P 500 Close', 'Brent Oil Price', 'Global Inflation'], var_name='Metric', value_name='Value')
                            
                            chart_comp = alt.Chart(df_comp_melted).mark_bar().encode(
                                x=alt.X('Peak Date:N', title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y('Value:Q', title=None),
                                color=alt.Color('Peak Date:N', scale=alt.Scale(range=['#3b82f6', '#10b981', '#f59e0b'])),
                                column=alt.Column('Metric:N', title=None, header=alt.Header(labelColor='#e2e8f0', labelFontSize=12)),
                                tooltip=['Peak Date', 'Metric', 'Value']
                            ).properties(
                                width=140,
                                height=200
                            ).configure_view(
                                stroke=None
                            )
                            
                            st.altair_chart(chart_comp, use_container_width=False)
