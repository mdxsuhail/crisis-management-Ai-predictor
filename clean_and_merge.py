import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR = os.path.join(BASE_DIR, "processed")

def parse_brent_date(val):
    val = str(val).strip().strip('"').strip("'")
    for fmt in ("%d-%b-%y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(val, format=fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(val)
    except Exception:
        return pd.NaT

def clean_and_merge():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset Directory: {DATA_DIR}")
    print(f"Processed Directory: {OUT_DIR}")

    # -------------------------------------------------------------
    # 1. Brent Oil Prices
    # -------------------------------------------------------------
    print("\nStep 1: Processing Brent Oil Prices...")
    brent_path = os.path.join(DATA_DIR, "BrentOilPrices.csv")
    if os.path.exists(brent_path):
        df_brent = pd.read_csv(brent_path)
        df_brent['Date'] = df_brent['Date'].apply(parse_brent_date)
        df_brent = df_brent.dropna(subset=['Date'])
        df_brent['Price'] = pd.to_numeric(df_brent['Price'], errors='coerce')
        df_brent = df_brent.sort_values('Date').reset_index(drop=True)
        df_brent['Price'] = df_brent['Price'].ffill().bfill()
        df_brent = df_brent.rename(columns={'Price': 'brent_oil_price'})
        df_brent = df_brent.groupby('Date')['brent_oil_price'].mean().reset_index()
        
        # Calculate oil returns and momentum
        df_brent['brent_oil_return_30d'] = df_brent['brent_oil_price'].pct_change(30) * 100
        df_brent['brent_oil_return_90d'] = df_brent['brent_oil_price'].pct_change(90) * 100
        print(f"Brent Oil processed: {len(df_brent)} daily records from {df_brent['Date'].min().strftime('%Y-%m-%d')} to {df_brent['Date'].max().strftime('%Y-%m-%d')}.")
    else:
        print(f"Warning: {brent_path} not found.")
        df_brent = pd.DataFrame(columns=['Date', 'brent_oil_price', 'brent_oil_return_30d', 'brent_oil_return_90d'])

    # -------------------------------------------------------------
    # 2. COVID-19 Economic Impact
    # -------------------------------------------------------------
    print("\nStep 2: Processing COVID-19 Economic Impact...")
    covid_path = os.path.join(DATA_DIR, "Covid-19 economy impact .csv")
    if os.path.exists(covid_path):
        cols = ['economy', 'code', 'consumption_shock_short', 'consumption_shock_long', 
                'investment_shock_short', 'investment_shock_long', 'tourism_decline_short', 
                'tourism_decline_long', 'stringency_index', 'mobility']
        df_covid = pd.read_csv(covid_path, skiprows=2, names=cols)
        
        # Convert all metric columns to numeric
        num_cols = ['consumption_shock_short', 'consumption_shock_long', 
                    'investment_shock_short', 'investment_shock_long', 
                    'tourism_decline_short', 'tourism_decline_long', 
                    'stringency_index', 'mobility']
        for col in num_cols:
            df_covid[col] = pd.to_numeric(df_covid[col], errors='coerce')
        
        covid_avg = df_covid[num_cols].mean().to_frame().T
        covid_avg = covid_avg.add_prefix('covid_impact_avg_')
        print("COVID-19 Economic Impact Averages:")
        print(covid_avg)
    else:
        print(f"Warning: {covid_path} not found.")
        covid_avg = pd.DataFrame([{
            'covid_impact_avg_consumption_shock_short': np.nan,
            'covid_impact_avg_consumption_shock_long': np.nan,
            'covid_impact_avg_investment_shock_short': np.nan,
            'covid_impact_avg_investment_shock_long': np.nan,
            'covid_impact_avg_tourism_decline_short': np.nan,
            'covid_impact_avg_tourism_decline_long': np.nan,
            'covid_impact_avg_stringency_index': np.nan,
            'covid_impact_avg_mobility': np.nan,
        }])

    # -------------------------------------------------------------
    # 3. ABC News Headlines
    # -------------------------------------------------------------
    print("\nStep 3: Processing ABC News Headlines...")
    news_path = os.path.join(DATA_DIR, "abcnews-date-text.csv")
    if os.path.exists(news_path):
        df_news = pd.read_csv(news_path)
        df_news['Date'] = pd.to_datetime(df_news['publish_date'].astype(str), format="%Y%m%d", errors='coerce')
        df_news = df_news.dropna(subset=['Date'])
        
        # Total headlines per day
        df_news_agg = df_news.groupby('Date').size().reset_index(name='news_headline_count')
        
        # Crisis-specific keyword counts
        crisis_pattern = r'crisis|crash|recession|oil|war|debt|bank|inflation|panic|lockdown|virus|outbreak'
        df_news['is_crisis'] = df_news['headline_text'].str.contains(crisis_pattern, case=False, na=False).astype(int)
        news_crisis_agg = df_news.groupby('Date')['is_crisis'].sum().reset_index(name='news_crisis_headline_count')
        
        df_news_merged = pd.merge(df_news_agg, news_crisis_agg, on='Date', how='left')
        print(f"ABC News processed: {len(df_news_merged)} daily records.")
    else:
        print(f"Warning: {news_path} not found.")
        df_news_merged = pd.DataFrame(columns=['Date', 'news_headline_count', 'news_crisis_headline_count'])

    # -------------------------------------------------------------
    # 4. African Crises
    # -------------------------------------------------------------
    print("\nStep 4: Processing African Crises...")
    crises_path = os.path.join(DATA_DIR, "african_crises.csv")
    if os.path.exists(crises_path):
        df_crises = pd.read_csv(crises_path)
        df_crises['Year'] = pd.to_numeric(df_crises['year'], errors='coerce')
        df_crises = df_crises.dropna(subset=['Year'])
        df_crises['Year'] = df_crises['Year'].astype(int)
        
        df_crises['banking_crisis'] = df_crises['banking_crisis'].apply(lambda x: 1 if str(x).strip().lower() == 'crisis' else 0)
        for col in ['systemic_crisis', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'currency_crises', 'inflation_crises']:
            if col in df_crises.columns:
                df_crises[col] = pd.to_numeric(df_crises[col], errors='coerce').fillna(0)
                
        df_crises['total_debt_defaults'] = df_crises['domestic_debt_in_default'] + df_crises['sovereign_external_debt_default']
        
        df_crises_agg = df_crises.groupby('Year').agg({
            'systemic_crisis': 'sum',
            'banking_crisis': 'sum',
            'total_debt_defaults': 'sum',
            'currency_crises': 'sum',
            'inflation_crises': 'sum',
            'exch_usd': 'mean',
            'inflation_annual_cpi': 'mean'
        }).reset_index()
        
        df_crises_agg = df_crises_agg.rename(columns={col: f"african_crises_{col}" for col in df_crises_agg.columns if col != 'Year'})
        print(f"African Crises processed: {len(df_crises_agg)} annual records.")
    else:
        print(f"Warning: {crises_path} not found.")
        df_crises_agg = pd.DataFrame(columns=['Year'])

    # -------------------------------------------------------------
    # 5. S&P 500 Stocks
    # -------------------------------------------------------------
    print("\nStep 5: Processing S&P 500 Stocks...")
    stocks_path = os.path.join(DATA_DIR, "all_stocks_5yr.csv")
    if os.path.exists(stocks_path):
        df_stocks = pd.read_csv(stocks_path)
        df_stocks['Date'] = pd.to_datetime(df_stocks['date'], format="%Y-%m-%d", errors='coerce')
        df_stocks = df_stocks.dropna(subset=['Date'])
        
        df_stocks = df_stocks.sort_values(['Name', 'Date'])
        for col in ['open', 'high', 'low', 'close']:
            df_stocks[col] = df_stocks.groupby('Name')[col].ffill().bfill()
            
        df_stocks_agg = df_stocks.groupby('Date').agg({
            'close': 'mean',
            'volume': 'sum',
            'Name': 'nunique'
        }).reset_index()
        
        df_stocks_agg = df_stocks_agg.rename(columns={
            'close': 'market_avg_close',
            'volume': 'market_total_volume',
            'Name': 'market_active_tickers'
        })
        
        df_stocks_agg = df_stocks_agg.sort_values('Date').reset_index(drop=True)
        df_stocks_agg['daily_return'] = df_stocks_agg['market_avg_close'].pct_change()
        df_stocks_agg['market_return_30d'] = df_stocks_agg['market_avg_close'].pct_change(30) * 100
        df_stocks_agg['market_return_90d'] = df_stocks_agg['market_avg_close'].pct_change(90) * 100
        df_stocks_agg['market_volatility_30d'] = df_stocks_agg['daily_return'].rolling(30).std() * np.sqrt(252) * 100
        df_stocks_agg = df_stocks_agg.drop(columns=['daily_return'])
        
        print(f"S&P Stocks processed: {len(df_stocks_agg)} daily records.")
    else:
        print(f"Warning: {stocks_path} not found.")
        df_stocks_agg = pd.DataFrame(columns=['Date', 'market_avg_close', 'market_total_volume', 'market_active_tickers', 'market_return_30d', 'market_return_90d', 'market_volatility_30d'])

    # -------------------------------------------------------------
    # 6. World Bank Indicators
    # -------------------------------------------------------------
    print("\nStep 6: Processing World Bank Indicators...")
    wb_path = os.path.join(DATA_DIR, "world_bank_development_indicators.csv")
    if os.path.exists(wb_path):
        df_wb = pd.read_csv(wb_path, low_memory=False)
        df_wb['Year'] = pd.to_datetime(df_wb['date'], errors='coerce').dt.year
        df_wb = df_wb.dropna(subset=['Year'])
        df_wb['Year'] = df_wb['Year'].astype(int)
        
        # Ensure key numeric columns
        wb_num_cols = [c for c in df_wb.columns if c not in ['country', 'date', 'Year']]
        for c in wb_num_cols:
            df_wb[c] = pd.to_numeric(df_wb[c], errors='coerce')
            
        # Compute global sums for scale indicators and means for rates
        sum_cols = ['GDP_current_US', 'population', 'rural_population', 'CO2_emisions', 'other_greenhouse_emisions']
        mean_cols = [c for c in wb_num_cols if c not in sum_cols]
        
        df_wb_sums = df_wb.groupby('Year')[sum_cols].sum(numeric_only=True).reset_index()
        df_wb_sums = df_wb_sums.rename(columns={
            'GDP_current_US': 'wb_GDP_total_US',
            'population': 'wb_population_total',
            'rural_population': 'wb_rural_population_total',
            'CO2_emisions': 'wb_CO2_emisions_total',
            'other_greenhouse_emisions': 'wb_other_greenhouse_emisions_total'
        })
        
        df_wb_means = df_wb.groupby('Year')[mean_cols].mean(numeric_only=True).reset_index()
        df_wb_means = df_wb_means.rename(columns={c: f"wb_{c}" for c in mean_cols if c != 'Year'})
        
        df_wb_agg = pd.merge(df_wb_sums, df_wb_means, on='Year', how='outer')
        print(f"World Bank indicators processed: {len(df_wb_agg)} annual records.")
    else:
        print(f"Warning: {wb_path} not found.")
        df_wb_agg = pd.DataFrame(columns=['Year'])

    # -------------------------------------------------------------
    # 7. Global Cost of Living Crisis (2026)
    # -------------------------------------------------------------
    print("\nStep 7: Processing Global Cost of Living Crisis 2026 Dataset...")
    col_path = os.path.join(DATA_DIR, "global_cost_of_living_crisis_2026.csv")
    if os.path.exists(col_path):
        df_col = pd.read_csv(col_path)
        col_num_cols = ['cost_of_living_index', 'rent_index', 'cost_of_living_plus_rent_index', 
                        'groceries_index', 'local_purchasing_power_index', 'avg_monthly_net_salary_usd', 
                        'petrol_price_usd_per_liter', 'annual_inflation_rate_2025_pct', 'rent_to_salary_ratio_pct']
        for col in col_num_cols:
            if col in df_col.columns:
                df_col[col] = pd.to_numeric(df_col[col], errors='coerce')
        
        col_avg = df_col[col_num_cols].mean().to_frame().T
        col_avg = col_avg.add_prefix('cost_of_living_')
        print("Global Cost of Living Indicators Averages:")
        print(col_avg)
    else:
        print(f"Warning: {col_path} not found.")
        col_avg = pd.DataFrame([{
            'cost_of_living_cost_of_living_index': 75.0,
            'cost_of_living_rent_index': 45.0,
            'cost_of_living_rent_to_salary_ratio_pct': 65.0,
            'cost_of_living_petrol_price_usd_per_liter': 1.65,
            'cost_of_living_annual_inflation_rate_2025_pct': 3.5,
            'cost_of_living_local_purchasing_power_index': 110.0
        }])

    # -------------------------------------------------------------
    # 8. Gold Price Prediction Dataset
    # -------------------------------------------------------------
    print("\nStep 8: Processing Gold Price Prediction Dataset...")
    gold_path = os.path.join(DATA_DIR, "gold_price_prediction.csv")
    if os.path.exists(gold_path):
        df_gold = pd.read_csv(gold_path)
        df_gold['Date'] = pd.to_datetime(df_gold['Date'], errors='coerce')
        df_gold = df_gold.dropna(subset=['Date'])
        df_gold = df_gold.sort_values('Date').reset_index(drop=True)
        
        # Extract Close as gold_price
        if 'Close' in df_gold.columns:
            df_gold['gold_price'] = pd.to_numeric(df_gold['Close'], errors='coerce')
        elif 'Adj Close' in df_gold.columns:
            df_gold['gold_price'] = pd.to_numeric(df_gold['Adj Close'], errors='coerce')
            
        df_gold['gold_price'] = df_gold['gold_price'].ffill().bfill()
        df_gold = df_gold.groupby('Date')['gold_price'].mean().reset_index()
        
        df_gold['gold_return_30d'] = df_gold['gold_price'].pct_change(30) * 100
        df_gold['gold_return_90d'] = df_gold['gold_price'].pct_change(90) * 100
        print(f"Gold Prices processed: {len(df_gold)} daily records from {df_gold['Date'].min().strftime('%Y-%m-%d')} to {df_gold['Date'].max().strftime('%Y-%m-%d')}.")
    else:
        print(f"Warning: {gold_path} not found.")
        df_gold = pd.DataFrame(columns=['Date', 'gold_price', 'gold_return_30d', 'gold_return_90d'])

    # -------------------------------------------------------------
    # 9. Banking Crisis and Exports Dataset
    # -------------------------------------------------------------
    print("\nStep 9: Processing Banking Crisis and Exports Dataset...")
    bk_path = os.path.join(DATA_DIR, "banking_crisis_and_exports.csv")
    if os.path.exists(bk_path):
        df_bk = pd.read_csv(bk_path, low_memory=False)
        df_bk['Year'] = pd.to_numeric(df_bk['year'], errors='coerce')
        df_bk = df_bk.dropna(subset=['Year'])
        df_bk['Year'] = df_bk['Year'].astype(int)
        
        for c in ['BANK', 'TWIN', 'recession', 'GDPgr', 'expgrowth', 'tradevalue']:
            if c in df_bk.columns:
                df_bk[c] = pd.to_numeric(df_bk[c], errors='coerce').fillna(0)
                
        df_bk_agg = df_bk.groupby('Year').agg({
            'BANK': 'sum',
            'TWIN': 'sum',
            'recession': 'sum',
            'GDPgr': 'mean',
            'expgrowth': 'mean',
            'tradevalue': 'sum'
        }).reset_index()
        
        df_bk_agg = df_bk_agg.rename(columns={
            'BANK': 'banking_crisis_global_count',
            'TWIN': 'twin_crisis_global_count',
            'recession': 'global_recession_count',
            'GDPgr': 'avg_global_gdp_growth',
            'expgrowth': 'avg_global_export_growth',
            'tradevalue': 'total_global_export_tradevalue'
        })
        print(f"Banking Crisis & Exports processed: {len(df_bk_agg)} annual records.")
    else:
        print(f"Warning: {bk_path} not found.")
        df_bk_agg = pd.DataFrame(columns=['Year'])

    # -------------------------------------------------------------
    # 10. Merging Datasets on Timeline
    # -------------------------------------------------------------
    print("\nStep 10: Merging Datasets into Daily Timeline...")
    daily_dfs = [df for df in [df_brent, df_news_merged, df_stocks_agg, df_gold] if 'Date' in df.columns and not df.empty]
    
    if daily_dfs:
        all_dates = pd.concat([df[['Date']] for df in daily_dfs]).drop_duplicates().sort_values('Date').reset_index(drop=True)
    else:
        print("Error: No daily date timelines available!")
        return

    df_merged = pd.merge(all_dates, df_brent, on='Date', how='left')
    df_merged = pd.merge(df_merged, df_news_merged, on='Date', how='left')
    df_merged = pd.merge(df_merged, df_stocks_agg, on='Date', how='left')
    df_merged = pd.merge(df_merged, df_gold, on='Date', how='left')

    df_merged['Year'] = df_merged['Date'].dt.year
    if not df_crises_agg.empty:
        df_merged = pd.merge(df_merged, df_crises_agg, on='Year', how='left')
    if not df_wb_agg.empty:
        df_merged = pd.merge(df_merged, df_wb_agg, on='Year', how='left')
    if not df_bk_agg.empty:
        df_merged = pd.merge(df_merged, df_bk_agg, on='Year', how='left')

    # Broadcast static metrics (COVID & Cost of Living)
    for col in covid_avg.columns:
        df_merged[col] = covid_avg[col].values[0]
    for col in col_avg.columns:
        df_merged[col] = col_avg[col].values[0]

    df_merged = df_merged.drop(columns=['Year'])
    df_merged = df_merged.sort_values('Date').reset_index(drop=True)

    # Forward fill / backward fill continuous time series columns
    ts_cols = ['brent_oil_price', 'market_avg_close', 'market_volatility_30d', 'gold_price', 'wb_inflation_annual%', 'wb_real_interest_rate', 'wb_GDP_total_US', 'cost_of_living_cost_of_living_index']
    for c in ts_cols:
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].ffill().bfill()

    out_path = os.path.join(OUT_DIR, "clean_data.csv")
    df_merged.to_csv(out_path, index=False)

    print(f"\nSUCCESS: Clean dataset saved to {out_path}")
    print(f"Dataset Shape: {df_merged.shape}")
    print(f"Date Range: {df_merged['Date'].min().strftime('%Y-%m-%d')} to {df_merged['Date'].max().strftime('%Y-%m-%d')}")
    print("Columns Overview:")
    for col in df_merged.columns[:15]:
        nulls = df_merged[col].isnull().sum()
        print(f" - {col}: {nulls} nulls ({nulls/len(df_merged)*100:.1f}%)")

if __name__ == "__main__":
    clean_and_merge()
