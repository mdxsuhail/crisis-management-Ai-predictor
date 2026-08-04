import os
import sys
import string
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(BASE_DIR, "processed", "crisis_impact_summary.csv")

try:
    from ml_predictor import predict_scenario
except ImportError:
    predict_scenario = None

STOPWORDS = {
    'and', 'or', 'in', 'the', 'a', 'of', 'to', 'for', 'with', 'on', 'at', 
    'by', 'from', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'this', 'that', 'these', 'those', 'it', 'its', 'shock', 'crisis', 'crash', 
    'event', 'about', 'as', 'but', 'not', 'trigger', 'scope', 'due', 'cause',
    'has', 'had', 'have', 'very', 'will', 'can', 'could', 'would', 'should'
}

def tokenize(text):
    if not isinstance(text, str):
        return set()
    text_clean = text.lower().replace('/', ' ').replace('-', ' ').replace(',', ' ')
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    tokens = {word for word in text_clean.split() if word not in STOPWORDS and len(word) > 1}
    return tokens

def find_similar_crises(query_description, top_n=3):
    if not os.path.exists(SUMMARY_PATH):
        print(f"Error: Impact summary dataset missing at {SUMMARY_PATH}")
        return []
        
    df = pd.read_csv(SUMMARY_PATH)
    query_tokens = tokenize(query_description)
    
    if not query_tokens:
        print("Warning: Query did not contain any valid keywords.")
        
    scores = []
    for idx, row in df.iterrows():
        name_tokens = tokenize(str(row.get('event_name', '')))
        type_tokens = tokenize(str(row.get('crisis_type', '')))
        trigger_tokens = tokenize(str(row.get('trigger_description', '')))
        region_tokens = tokenize(str(row.get('region', '')))
        
        # Weighted matching
        sim_name = len(query_tokens & name_tokens) / len(query_tokens | name_tokens) if (query_tokens | name_tokens) else 0
        sim_type = len(query_tokens & type_tokens) / len(query_tokens | type_tokens) if (query_tokens | type_tokens) else 0
        sim_trigger = len(query_tokens & trigger_tokens) / len(query_tokens | trigger_tokens) if (query_tokens | trigger_tokens) else 0
        sim_region = len(query_tokens & region_tokens) / len(query_tokens | region_tokens) if (query_tokens | region_tokens) else 0
        
        weighted_score = (sim_name * 0.3) + (sim_type * 0.3) + (sim_trigger * 0.3) + (sim_region * 0.1)
        
        # Simple overall Jaccard fallback
        all_tokens = name_tokens | type_tokens | trigger_tokens | region_tokens
        jaccard_sim = len(query_tokens & all_tokens) / len(query_tokens | all_tokens) if (query_tokens | all_tokens) else 0
        
        final_sim = max(weighted_score, jaccard_sim)
        scores.append((final_sim, row))
        
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_matches = scores[:top_n]
    
    print(f"\nQuery Situation: '{query_description}'")
    print(f"Extracted Keywords: {list(query_tokens)}")
    print(f"\nTop {min(top_n, len(top_matches))} Most Similar Historical Crises:")
    print("=" * 80)
    
    for i, (score, row) in enumerate(top_matches):
        event = row['event_name']
        print(f"{i+1}. {event} (Match Confidence: {round(score * 100, 1)}%)")
        print(f"   Crisis Type: {row.get('crisis_type', 'N/A')}")
        print(f"   Region:      {row.get('region', 'N/A')}")
        print(f"   Start Date:  {row.get('start_date', 'N/A')}")
        print(f"   Trigger:     {row.get('trigger_description', 'N/A')}")
        print("-" * 80)
        
        # Stock market
        stock_6m = row.get('market_avg_close_change_6m')
        stock_12m = row.get('market_avg_close_change_12m')
        stock_str = f"6m: {stock_6m}% | 12m: {stock_12m}%" if pd.notnull(stock_6m) else "Data Unavailable"
        print(f"   - S&P Stock Market Change: {stock_str}")
        
        # Inflation
        inf_6m = row.get('wb_inflation_annual%_change_6m')
        inf_12m = row.get('wb_inflation_annual%_change_12m')
        inf_str = f"6m: {inf_6m} pp | 12m: {inf_12m} pp" if pd.notnull(inf_6m) else "Data Unavailable"
        print(f"   - Inflation Rate Change:   {inf_str}")
        
        # Brent Oil
        oil_6m = row.get('brent_oil_price_change_6m')
        oil_12m = row.get('brent_oil_price_change_12m')
        oil_str = f"6m: {oil_6m}% | 12m: {oil_12m}%" if pd.notnull(oil_6m) else "Data Unavailable"
        print(f"   - Brent Oil Price Change:  {oil_str}")
        print("=" * 80)

    # If ML Predictor available, print ML forecasts
    if predict_scenario:
        print("\nMachine Learning Quantitative Predictions:")
        ml_preds = predict_scenario()
        if ml_preds:
            print(f" - Predicted 6m Stock Change: {ml_preds.get('stock_6m', 'N/A')}%")
            print(f" - Predicted 12m Stock Change: {ml_preds.get('stock_12m', 'N/A')}%")
            print(f" - Predicted 6m Oil Change: {ml_preds.get('oil_6m', 'N/A')}%")
            print(f" - Predicted 12m Oil Change: {ml_preds.get('oil_12m', 'N/A')}%")
            print(f" - Crisis Risk Probability: {ml_preds.get('crisis_risk', 'N/A')}%")
            
    return top_matches

if __name__ == "__main__":
    query = "oil price shock, geopolitical conflict, global inflation"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    find_similar_crises(query)
