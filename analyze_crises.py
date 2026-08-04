import os
import pandas as pd
import numpy as np

CLEAN_DATA_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\processed\clean_data.csv"
CRISES_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\dataset\crisis_events.csv"
OUT_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\processed\crisis_impact_summary.csv"

def get_closest_value(df, target_date, column, max_days_tol=14):
    """
    Finds the value of a column on the date closest to target_date in df.
    If the closest date is more than max_days_tol away, returns NaN.
    """
    if df.empty or target_date is pd.NaT:
        return np.nan, pd.NaT
    
    # Filter rows where the column is not null
    df_valid = df.dropna(subset=[column])
    if df_valid.empty:
        return np.nan, pd.NaT
    
    # Find closest date
    time_diffs = (df_valid['Date'] - target_date).abs()
    idx = time_diffs.idxmin()
    closest_date = df_valid.loc[idx, 'Date']
    diff_days = time_diffs.loc[idx].days
    
    if diff_days <= max_days_tol:
        return df_valid.loc[idx, column], closest_date
    else:
        return np.nan, pd.NaT

def main():
    print("Loading datasets...")
    if not os.path.exists(CLEAN_DATA_PATH) or not os.path.exists(CRISES_PATH):
        print("Error: Cleaned dataset or crisis events list missing.")
        return
        
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    df_crises = pd.read_csv(CRISES_PATH)
    df_crises['start_date'] = pd.to_datetime(df_crises['start_date'])
    
    summary_rows = []
    
    # Key columns to track
    # Stock: market_avg_close
    # Inflation: wb_inflation_annual% (Global avg) or african_crises_inflation_annual_cpi
    # Currency: african_crises_exch_usd
    # Commodity: brent_oil_price
    tracked_metrics = {
        'market_avg_close': 'percentage',
        'wb_inflation_annual%': 'absolute',
        'african_crises_exch_usd': 'percentage',
        'brent_oil_price': 'percentage'
    }
    
    for _, crisis in df_crises.iterrows():
        event = crisis['event_name']
        start_dt = crisis['start_date']
        
        print(f"\nAnalyzing impact of '{event}' (Started: {start_dt.strftime('%Y-%m-%d')})...")
        
        row_data = {
            'event_name': event,
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'crisis_type': crisis['crisis_type'],
            'region': crisis['region']
        }
        
        # Calculate offsets
        target_dates = {
            'start': start_dt,
            '6m': start_dt + pd.DateOffset(months=6),
            '12m': start_dt + pd.DateOffset(months=12)
        }
        
        # For each metric, look up values at start, 6m, and 12m
        for metric, calc_type in tracked_metrics.items():
            val_start, dt_start = get_closest_value(df_clean, target_dates['start'], metric)
            val_6m, dt_6m = get_closest_value(df_clean, target_dates['6m'], metric)
            val_12m, dt_12m = get_closest_value(df_clean, target_dates['12m'], metric)
            
            # Record base values
            row_data[f'{metric}_at_start'] = val_start
            row_data[f'{metric}_at_6m'] = val_6m
            row_data[f'{metric}_at_12m'] = val_12m
            
            # Calculate changes
            if pd.notnull(val_start):
                if pd.notnull(val_6m):
                    if calc_type == 'percentage':
                        row_data[f'{metric}_change_6m'] = round(((val_6m - val_start) / val_start) * 100, 2)
                    else:
                        row_data[f'{metric}_change_6m'] = round(val_6m - val_start, 2)
                else:
                    row_data[f'{metric}_change_6m'] = np.nan
                    
                if pd.notnull(val_12m):
                    if calc_type == 'percentage':
                        row_data[f'{metric}_change_12m'] = round(((val_12m - val_start) / val_start) * 100, 2)
                    else:
                        row_data[f'{metric}_change_12m'] = round(val_12m - val_start, 2)
                else:
                    row_data[f'{metric}_change_12m'] = np.nan
            else:
                row_data[f'{metric}_change_6m'] = np.nan
                row_data[f'{metric}_change_12m'] = np.nan
                
        summary_rows.append(row_data)
        
    df_summary = pd.DataFrame(summary_rows)
    
    # Save the output
    df_summary.to_csv(OUT_PATH, index=False)
    print(f"\nSUCCESS: Summary table saved to: {OUT_PATH}")
    print(f"Summary table shape: {df_summary.shape}")
    
    # Print a nice summary view
    print("\nCrisis Impact Summary Table (Changes in stock market, inflation, and currency):")
    display_cols = [
        'event_name',
        'market_avg_close_change_6m', 'market_avg_close_change_12m',
        'wb_inflation_annual%_change_6m', 'wb_inflation_annual%_change_12m',
        'african_crises_exch_usd_change_6m', 'african_crises_exch_usd_change_12m',
        'brent_oil_price_change_6m', 'brent_oil_price_change_12m'
    ]
    print(df_summary[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
