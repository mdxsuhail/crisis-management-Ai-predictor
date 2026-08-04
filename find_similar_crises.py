import os
import sys
import string
import pandas as pd
import numpy as np

SUMMARY_PATH = r"c:\Users\ADMIN\Desktop\problem statement\problem statement 1\processed\crisis_impact_summary.csv"

# Basic list of stop words to filter out noise
STOPWORDS = {
    'and', 'or', 'in', 'the', 'a', 'of', 'to', 'for', 'with', 'on', 'at', 
    'by', 'from', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'this', 'that', 'these', 'those', 'it', 'its', 'shock', 'crisis', 'crash', 
    'event', 'about', 'as', 'but', 'not', 'trigger', 'scope'
}

def tokenize(text):
    if not isinstance(text, str):
        return set()
    # Lowercase, replace slashes and punctuation with space, then split
    text_clean = text.lower().replace('/', ' ').replace('-', ' ')
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    tokens = {word for word in text_clean.split() if word not in STOPWORDS}
    return tokens

def find_similar_crises(query_description, top_n=3):
    if not os.path.exists(SUMMARY_PATH):
        print(f"Error: Impact summary dataset missing at {SUMMARY_PATH}")
        return
        
    df = pd.read_csv(SUMMARY_PATH)
    query_tokens = tokenize(query_description)
    
    if not query_tokens:
        print("Warning: Query did not contain any valid keywords.")
        
    scores = []
    
    for idx, row in df.iterrows():
        # Combine descriptive fields for similarity matching
        combined_text = f"{row['event_name']} {row['crisis_type']} {row['region']}"
        doc_tokens = tokenize(combined_text)
        
        # Calculate Jaccard similarity: size of intersection / size of union
        if not query_tokens or not doc_tokens:
            sim = 0.0
        else:
            intersection = query_tokens.intersection(doc_tokens)
            union = query_tokens.union(doc_tokens)
            sim = len(intersection) / len(union)
            
        scores.append((sim, row))
        
    # Sort by similarity score descending
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    
    print(f"\nQuery Situation: '{query_description}'")
    print(f"Query Keywords: {list(query_tokens)}")
    print(f"\nTop {top_n} Most Similar Historical Crises:")
    print("=" * 80)
    
    for i in range(min(top_n, len(scores))):
        score, row = scores[i]
        event = row['event_name']
        print(f"{i+1}. {event} (Similarity: {round(score * 100, 1)}%)")
        print(f"   Crisis Type: {row['crisis_type']}")
        print(f"   Region:      {row['region']}")
        print(f"   Start Date:  {row['start_date']}")
        print("-" * 80)
        
        # Print indicator impacts
        print("   Indicator changes after the event:")
        
        # Stock market
        stock_6m = row['market_avg_close_change_6m']
        stock_12m = row['market_avg_close_change_12m']
        stock_str = f"6 months: {stock_6m}% | 12 months: {stock_12m}%" if pd.notnull(stock_6m) else "Data Unavailable (Outside 2013-2018 window)"
        print(f"   - Stock Prices (avg close): {stock_str}")
        
        # Inflation
        inf_6m = row['wb_inflation_annual%_change_6m']
        inf_12m = row['wb_inflation_annual%_change_12m']
        inf_str = f"6 months: {inf_6m} pp | 12 months: {inf_12m} pp" if pd.notnull(inf_6m) else "Data Unavailable"
        print(f"   - Inflation (global avg):   {inf_str}")
        
        # Currency exchange rate
        curr_6m = row['african_crises_exch_usd_change_6m']
        curr_12m = row['african_crises_exch_usd_change_12m']
        curr_str = f"6 months: {curr_6m}% | 12 months: {curr_12m}%" if pd.notnull(curr_6m) else "Data Unavailable"
        print(f"   - Currency vs USD (avg):    {curr_str}")
        
        # Brent Oil
        oil_6m = row['brent_oil_price_change_6m']
        oil_12m = row['brent_oil_price_change_12m']
        oil_str = f"6 months: {oil_6m}% | 12 months: {oil_12m}%" if pd.notnull(oil_6m) else "Data Unavailable"
        print(f"   - Brent Oil Price:          {oil_str}")
        print("=" * 80)

if __name__ == "__main__":
    query = "oil price shock, geopolitical trigger, global scope"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    find_similar_crises(query)
