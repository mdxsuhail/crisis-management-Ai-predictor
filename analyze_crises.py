import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "processed", "clean_data.csv")
CRISES_PATH = os.path.join(BASE_DIR, "dataset", "crisis_events.csv")
OUT_PATH = os.path.join(BASE_DIR, "processed", "crisis_impact_summary.csv")

def get_closest_value(df, target_date, column, max_days_tol=180):
    """
    Finds the value of a column on the date closest to target_date in df.
    """
    if df.empty or target_date is pd.NaT or column not in df.columns:
        return np.nan, pd.NaT
    
    df_valid = df.dropna(subset=[column])
    if df_valid.empty:
        return np.nan, pd.NaT
    
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
        print(f"Error: Clean dataset or crisis events list missing.")
        print(f"CLEAN_DATA_PATH exists: {os.path.exists(CLEAN_DATA_PATH)}")
        print(f"CRISES_PATH exists: {os.path.exists(CRISES_PATH)}")
        return
        
    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    df_crises = pd.read_csv(CRISES_PATH)
    df_crises['start_date'] = pd.to_datetime(df_crises['start_date'])
    
    summary_rows = []
    
    tracked_metrics = {
        'market_avg_close': 'percentage',
        'wb_inflation_annual%': 'absolute',
        'african_crises_exch_usd': 'percentage',
        'brent_oil_price': 'percentage',
        'wb_GDP_total_US': 'percentage'
    }
    
    for _, crisis in df_crises.iterrows():
        event = crisis['event_name']
        start_dt = crisis['start_date']
        
        row_data = {
            'event_name': event,
            'start_date': start_dt.strftime('%Y-%m-%d'),
            'end_date': str(crisis.get('end_date', '')),
            'crisis_type': crisis['crisis_type'],
            'trigger_description': str(crisis.get('trigger_description', '')),
            'region': crisis['region']
        }
        
        target_dates = {
            'start': start_dt,
            '6m': start_dt + pd.DateOffset(months=6),
            '12m': start_dt + pd.DateOffset(months=12)
        }
        
        for metric, calc_type in tracked_metrics.items():
            val_start, dt_start = get_closest_value(df_clean, target_dates['start'], metric)
            val_6m, dt_6m = get_closest_value(df_clean, target_dates['6m'], metric)
            val_12m, dt_12m = get_closest_value(df_clean, target_dates['12m'], metric)
            
            row_data[f'{metric}_at_start'] = val_start
            row_data[f'{metric}_at_6m'] = val_6m
            row_data[f'{metric}_at_12m'] = val_12m
            
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
    df_summary.to_csv(OUT_PATH, index=False)
    print(f"\nSUCCESS: Crisis Impact Summary saved to {OUT_PATH}")
    print(f"Summary Table Shape: {df_summary.shape}")

if __name__ == "__main__":
    main()
