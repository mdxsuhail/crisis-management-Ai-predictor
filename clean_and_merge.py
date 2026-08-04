import os
import pandas as pd
import numpy as np

DATA_DIR = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\dataset"
OUT_DIR = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\processed"

def parse_brent_date(val):
    val = str(val).strip().strip('"').strip("'")
    for fmt in ("%d-%b-%y", "%b %d, %Y"):
        try:
            return pd.to_datetime(val, format=fmt)
        except ValueError:
            continue
    return pd.NaT

def clean_and_merge():
    # Make sure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Brent Oil Prices
    print("Step 1: Processing Brent Oil Prices...")
    brent_path = os.path.join(DATA_DIR, "BrentOilPrices.csv")
    df_brent = pd.read_csv(brent_path)
    df_brent['Date'] = df_brent['Date'].apply(parse_brent_date)
    df_brent = df_brent.dropna(subset=['Date'])
    df_brent['Price'] = pd.to_numeric(df_brent['Price'], errors='coerce')
    # Forward fill/interpolate any null prices if they exist
    df_brent['Price'] = df_brent['Price'].ffill().bfill()
    df_brent = df_brent.rename(columns={'Price': 'brent_oil_price'})
    df_brent = df_brent.groupby('Date')['brent_oil_price'].mean().reset_index()
    print(f"Brent Oil processed: {len(df_brent)} rows.")

    # 2. Covid-19 economy impact
    print("\nStep 2: Processing Covid-19 Economic Impact (cross-sectional averages)...")
    covid_path = os.path.join(DATA_DIR, "Covid-19 economy impact .csv")
    cols = ['economy', 'code', 'consumption_shock_short', 'consumption_shock_long', 
            'investment_shock_short', 'investment_shock_long', 'tourism_decline_short', 
            'tourism_decline_long', 'stringency_index', 'mobility']
    df_covid = pd.read_csv(covid_path, skiprows=2, names=cols)
    
    # Convert index and mobility to numeric (handling 'no data')
    df_covid['stringency_index'] = pd.to_numeric(df_covid['stringency_index'], errors='coerce')
    df_covid['mobility'] = pd.to_numeric(df_covid['mobility'], errors='coerce')
    
    # Calculate average shock values across all economies
    covid_avg = df_covid.mean(numeric_only=True).to_frame().T
    # Add prefix
    covid_avg = covid_avg.add_prefix('covid_impact_avg_')
    print("Covid-19 economic impact averages calculated:")
    print(covid_avg)

    # 3. ABC News Headlines
    print("\nStep 3: Processing ABC News Headlines...")
    news_path = os.path.join(DATA_DIR, "abcnews-date-text.csv")
    df_news = pd.read_csv(news_path)
    df_news['Date'] = pd.to_datetime(df_news['publish_date'].astype(str), format="%Y%m%d", errors='coerce')
    df_news = df_news.dropna(subset=['Date'])
    df_news_agg = df_news.groupby('Date').size().reset_index(name='news_headline_count')
    print(f"ABC News processed: {len(df_news_agg)} rows.")

    # 4. African Crises
    print("\nStep 4: Processing African Crises...")
    crises_path = os.path.join(DATA_DIR, "african_crises.csv")
    df_crises = pd.read_csv(crises_path)
    df_crises['Year'] = pd.to_numeric(df_crises['year'], errors='coerce')
    df_crises = df_crises.dropna(subset=['Year'])
    df_crises['Year'] = df_crises['Year'].astype(int)
    
    # Clean banking_crisis (convert to 1 if crisis, 0 if no_crisis)
    df_crises['banking_crisis'] = df_crises['banking_crisis'].apply(lambda x: 1 if str(x).strip().lower() == 'crisis' else 0)
    
    # Aggregate annually
    df_crises_agg = df_crises.groupby('Year').agg({
        'systemic_crisis': 'sum',
        'domestic_debt_in_default': 'sum',
        'sovereign_external_debt_default': 'sum',
        'banking_crisis': 'sum',
        'exch_usd': 'mean',
        'inflation_annual_cpi': 'mean'
    }).reset_index()
    
    # Add prefix
    df_crises_agg = df_crises_agg.rename(columns={col: f"african_crises_{col}" for col in df_crises_agg.columns if col != 'Year'})
    print(f"African Crises processed: {len(df_crises_agg)} annual records.")

    # 5. S&P 500 Stocks
    print("\nStep 5: Processing S&P 500 Stocks...")
    stocks_path = os.path.join(DATA_DIR, "all_stocks_5yr.csv")
    df_stocks = pd.read_csv(stocks_path)
    df_stocks['Date'] = pd.to_datetime(df_stocks['date'], format="%Y-%m-%d", errors='coerce')
    df_stocks = df_stocks.dropna(subset=['Date'])
    
    # Clean stock prices (forward-fill per ticker)
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
    print(f"S&P Stocks processed: {len(df_stocks_agg)} rows.")

    # 6. World Bank Indicators
    print("\nStep 6: Processing World Bank Indicators...")
    wb_path = os.path.join(DATA_DIR, "world_bank_development_indicators.csv")
    df_wb = pd.read_csv(wb_path)
    df_wb['Year'] = pd.to_datetime(df_wb['date'], errors='coerce').dt.year
    df_wb = df_wb.dropna(subset=['Year'])
    df_wb['Year'] = df_wb['Year'].astype(int)
    
    # Drop country and date columns before taking the mean per year
    df_wb_numeric = df_wb.drop(columns=['country', 'date'], errors='ignore')
    df_wb_agg = df_wb_numeric.groupby('Year').mean().reset_index()
    
    # Add prefix
    df_wb_agg = df_wb_agg.rename(columns={col: f"wb_{col}" for col in df_wb_agg.columns if col != 'Year'})
    print(f"World Bank indicators processed: {len(df_wb_agg)} annual records.")

    # 7. Merging Datasets
    print("\nStep 7: Merging Datasets on Date...")
    
    # Get the union of all daily dates from the daily datasets
    all_dates = pd.concat([
        df_brent[['Date']],
        df_news_agg[['Date']],
        df_stocks_agg[['Date']]
    ]).drop_duplicates().sort_values('Date').reset_index(drop=True)
    
    # Left join all daily data on the master timeline
    df_merged = pd.merge(all_dates, df_brent, on='Date', how='left')
    df_merged = pd.merge(df_merged, df_news_agg, on='Date', how='left')
    df_merged = pd.merge(df_merged, df_stocks_agg, on='Date', how='left')
    
    # Map annual indicators
    df_merged['Year'] = df_merged['Date'].dt.year
    df_merged = pd.merge(df_merged, df_crises_agg, on='Year', how='left')
    df_merged = pd.merge(df_merged, df_wb_agg, on='Year', how='left')
    
    # Broadcast COVID static averages
    for col in covid_avg.columns:
        df_merged[col] = covid_avg[col].values[0]
        
    df_merged = df_merged.drop(columns=['Year'])
    df_merged = df_merged.sort_values('Date')
    
    # Save output
    out_path = os.path.join(OUT_DIR, "clean_data.csv")
    df_merged.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS: Cleaned and merged dataset saved to: {out_path}")
    print(f"Final shape: {df_merged.shape}")
    print(f"Date range: {df_merged['Date'].min().strftime('%Y-%m-%d')} to {df_merged['Date'].max().strftime('%Y-%m-%d')}")
    print("\nColumns in final dataset:")
    for col in df_merged.columns[:15]: # show first 15 columns
        null_count = df_merged[col].isnull().sum()
        print(f" - {col}: {null_count} nulls ({round(null_count / len(df_merged) * 100, 2)}%)")
    if len(df_merged.columns) > 15:
        print(f" ... and {len(df_merged.columns) - 15} more columns.")

if __name__ == "__main__":
    clean_and_merge()
