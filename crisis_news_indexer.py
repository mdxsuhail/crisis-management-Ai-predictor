import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRISIS_PATH = os.path.join(BASE_DIR, "dataset", "crisis_events.csv")
FIN_DIR = os.path.join(BASE_DIR, "dataset", "financial_news")
ABC_PATH = os.path.join(BASE_DIR, "dataset", "abcnews-date-text.csv")
OUT_JSON = os.path.join(BASE_DIR, "processed", "crisis_news_details.json")

def load_news_datasets():
    print("Loading financial news datasets...")
    records = []
    
    # 1. Reuters Headlines
    reuters_path = os.path.join(FIN_DIR, "reuters_headlines.csv")
    if os.path.exists(reuters_path):
        df_reuters = pd.read_csv(reuters_path)
        for _, r in df_reuters.iterrows():
            head = str(r.get('Headlines', '')).strip()
            desc = str(r.get('Description', '')).strip()
            dt_str = str(r.get('Time', '')).strip()
            if head and head != 'nan':
                records.append({
                    'headline': head,
                    'description': desc if desc != 'nan' else head,
                    'date': dt_str,
                    'source': 'Reuters Financial'
                })
        print(f"Loaded {len(df_reuters)} Reuters headlines.")
        
    # 2. CNBC Headlines
    cnbc_path = os.path.join(FIN_DIR, "cnbc_headlines.csv")
    if os.path.exists(cnbc_path):
        df_cnbc = pd.read_csv(cnbc_path)
        for _, r in df_cnbc.iterrows():
            head = str(r.get('Headlines', '')).strip()
            desc = str(r.get('Description', '')).strip()
            dt_str = str(r.get('Time', '')).strip()
            if head and head != 'nan':
                records.append({
                    'headline': head,
                    'description': desc if desc != 'nan' else head,
                    'date': dt_str,
                    'source': 'CNBC News'
                })
        print(f"Loaded {len(df_cnbc)} CNBC headlines.")

    # 3. Guardian Headlines
    guardian_path = os.path.join(FIN_DIR, "guardian_headlines.csv")
    if os.path.exists(guardian_path):
        df_guard = pd.read_csv(guardian_path)
        for _, r in df_guard.iterrows():
            head = str(r.get('Headlines', '')).strip()
            dt_str = str(r.get('Time', '')).strip()
            if head and head != 'nan':
                records.append({
                    'headline': head,
                    'description': head,
                    'date': dt_str,
                    'source': 'The Guardian'
                })
        print(f"Loaded {len(df_guard)} Guardian headlines.")

    # 4. ABC News Headlines
    if os.path.exists(ABC_PATH):
        df_abc = pd.read_csv(ABC_PATH)
        df_abc['Date'] = pd.to_datetime(df_abc['publish_date'].astype(str), format='%Y%m%d', errors='coerce')
        # Filter for crisis keywords to keep memory lightweight
        keywords = ['crisis', 'crash', 'recession', 'inflation', 'bank', 'debt', 'oil', 'stock', 'market', 'covid', 'virus', 'default', 'tariff', 'war', 'bubble', 'fed', 'rate']
        pattern = '|'.join(keywords)
        mask = df_abc['headline_text'].str.contains(pattern, case=False, na=False)
        df_abc_filtered = df_abc[mask]
        
        for _, r in df_abc_filtered.iterrows():
            head = str(r.get('headline_text', '')).strip().title()
            dt_str = r['Date'].strftime('%b %d %Y') if pd.notnull(r['Date']) else ''
            if head:
                records.append({
                    'headline': head,
                    'description': f"Global news report on economic developments: {head}",
                    'date': dt_str,
                    'source': 'ABC News Archive'
                })
        print(f"Loaded {len(df_abc_filtered)} ABC News crisis headlines.")
        
    df_all = pd.DataFrame(records)
    print(f"Total News Articles Indexed: {len(df_all):,}")
    return df_all

def extract_crisis_news():
    if not os.path.exists(CRISIS_PATH):
        print(f"Error: {CRISIS_PATH} not found.")
        return
        
    df_crises = pd.read_csv(CRISIS_PATH)
    df_news = load_news_datasets()
    
    crisis_details = {}
    
    # Precompute text series once outside loop
    if not df_news.empty:
        text_series = (df_news['headline'].fillna('').astype(str) + " " + df_news['description'].fillna('').astype(str)).str.lower()
    else:
        text_series = pd.Series(dtype=str)

    for _, crisis in df_crises.iterrows():
        event = str(crisis.get('event_name', ''))
        start_date = str(crisis.get('start_date', ''))
        end_date = "" if pd.isna(crisis.get('end_date')) else str(crisis.get('end_date', ''))
        crisis_type = str(crisis.get('crisis_type', ''))
        trigger = "" if pd.isna(crisis.get('trigger_description')) else str(crisis.get('trigger_description', ''))
        region = str(crisis.get('region', ''))
        
        # Build keywords for filtering
        words = set(event.lower().replace('-', ' ').split() + crisis_type.lower().replace('/', ' ').split() + trigger.lower().split())
        words = {w for w in words if len(w) > 3 and w not in {'crisis', 'crash', 'burst', 'global', 'shock'}}
        
        top_articles = []
        if not text_series.empty and words:
            scores = pd.Series(0, index=df_news.index)
            for w in words:
                scores += text_series.str.contains(w, regex=False, na=False).astype(int)
                
            match_indices = scores[scores > 0].sort_values(ascending=False).index
            seen = set()
            for idx in match_indices:
                r = df_news.loc[idx]
                h = r.get('headline', '')
                if h not in seen:
                    seen.add(h)
                    top_articles.append({
                        'headline': r.get('headline', ''),
                        'description': r.get('description', ''),
                        'date': r.get('date', 'N/A'),
                        'source': r.get('source', 'News')
                    })
                if len(top_articles) >= 8:
                    break
                
        crisis_details[event] = {
            'event_name': event,
            'start_date': start_date,
            'end_date': end_date,
            'crisis_type': crisis_type,
            'trigger_description': trigger,
            'region': region,
            'matched_news_count': len(matches),
            'headlines': top_articles
        }
        
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(crisis_details, f, indent=2)
        
    print(f"\nSUCCESS: Crisis news & details indexed to {OUT_JSON}")

if __name__ == '__main__':
    extract_crisis_news()
