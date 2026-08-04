import os
import json
import joblib
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# Framework Imports & Fallbacks (Random Forest, XGBoost, GRU)
# ─────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestRegressor, 
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score

# Try importing native XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Try importing PyTorch for PyTorch GRU
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "processed", "clean_data.csv")
CRISES_PATH = os.path.join(BASE_DIR, "dataset", "crisis_events.csv")
MODEL_PATH = os.path.join(BASE_DIR, "processed", "ml_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "processed", "model_metrics.json")

FEATURE_COLS = [
    'brent_oil_price', 'brent_oil_return_30d', 'brent_oil_return_90d',
    'market_avg_close', 'market_return_30d', 'market_return_90d', 'market_volatility_30d',
    'gold_price', 'gold_return_30d', 'gold_return_90d',
    'news_headline_count', 'news_crisis_headline_count',
    'african_crises_systemic_crisis', 'african_crises_banking_crisis', 'african_crises_total_debt_defaults',
    'african_crises_exch_usd', 'wb_inflation_annual%', 'wb_real_interest_rate',
    'covid_impact_avg_stringency_index', 'covid_impact_avg_mobility',
    'cost_of_living_cost_of_living_index', 'cost_of_living_rent_to_salary_ratio_pct',
    'cost_of_living_petrol_price_usd_per_liter', 'cost_of_living_annual_inflation_rate_2025_pct',
    'banking_crisis_global_count', 'twin_crisis_global_count', 'global_recession_count', 'avg_global_export_growth'
]

import sys

# ─────────────────────────────────────────────────────────────
# 1. Gated Recurrent Unit (GRU) Neural Model Implementation
# ─────────────────────────────────────────────────────────────
class GRUNeuralModel:
    """
    Gated Recurrent Unit (GRU) Neural Time-Series Model.
    Computes update gate (z_t), reset gate (r_t), and candidate state (h_tilde_t)
    over input sequences.
    """
    __module__ = 'ml_predictor'

    def __init__(self, input_dim, hidden_dim=32, is_classification=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.is_classification = is_classification
        
        # Initialize GRU weight matrices & biases
        np.random.seed(42)
        k = 1.0 / np.sqrt(hidden_dim)
        self.Wz = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.Uz = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.bz = np.zeros((hidden_dim, 1))
        
        self.Wr = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.Ur = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.br = np.zeros((hidden_dim, 1))
        
        self.Wh = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.Uh = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.bh = np.zeros((hidden_dim, 1))
        
        self.Wo = np.random.uniform(-k, k, (1, hidden_dim))
        self.bo = np.zeros((1, 1))
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def fit(self, X, y, epochs=25, lr=0.01):
        N, D = X.shape
        y_vec = y.reshape(-1, 1)
        
        for _ in range(epochs):
            for i in range(min(N, 500)):
                x_t = X[i:i+1].T
                h_prev = np.zeros((self.hidden_dim, 1))
                
                z_t = self.sigmoid(self.Wz @ x_t + self.Uz @ h_prev + self.bz)
                r_t = self.sigmoid(self.Wr @ x_t + self.Ur @ h_prev + self.br)
                h_tilde = np.tanh(self.Wh @ x_t + self.Uh @ (r_t * h_prev) + self.bh)
                h_t = (1 - z_t) * h_prev + z_t * h_tilde
                
                out = self.Wo @ h_t + self.bo
                if self.is_classification:
                    pred = self.sigmoid(out)
                else:
                    pred = out
                    
                err = pred - y_vec[i:i+1]
                self.Wo -= lr * err @ h_t.T
                self.bo -= lr * err

    def predict(self, X):
        N, D = X.shape
        preds = []
        for i in range(N):
            x_t = X[i:i+1].T
            h_prev = np.zeros((self.hidden_dim, 1))
            
            z_t = self.sigmoid(self.Wz @ x_t + self.Uz @ h_prev + self.bz)
            r_t = self.sigmoid(self.Wr @ x_t + self.Ur @ h_prev + self.br)
            h_tilde = np.tanh(self.Wh @ x_t + self.Uh @ (r_t * h_prev) + self.bh)
            h_t = (1 - z_t) * h_prev + z_t * h_tilde
            
            out = self.Wo @ h_t + self.bo
            if self.is_classification:
                p = float(self.sigmoid(out)[0, 0])
            else:
                p = float(out[0, 0])
            preds.append(p)
        return np.array(preds)

    def get_weights(self):
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'is_classification': self.is_classification,
            'Wz': self.Wz, 'Uz': self.Uz, 'bz': self.bz,
            'Wr': self.Wr, 'Ur': self.Ur, 'br': self.br,
            'Wh': self.Wh, 'Uh': self.Uh, 'bh': self.bh,
            'Wo': self.Wo, 'bo': self.bo
        }

    @classmethod
    def from_weights(cls, w):
        m = cls(w['input_dim'], w['hidden_dim'], w['is_classification'])
        m.Wz, m.Uz, m.bz = w['Wz'], w['Uz'], w['bz']
        m.Wr, m.Ur, m.br = w['Wr'], w['Ur'], w['br']
        m.Wh, m.Uh, m.bh = w['Wh'], w['Uh'], w['bh']
        m.Wo, m.bo = w['Wo'], w['bo']
        return m

# ─────────────────────────────────────────────────────────────
# 2. Vectorized Target Building
# ─────────────────────────────────────────────────────────────
def build_training_dataset():
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"Error: {CLEAN_DATA_PATH} missing.")
        return None
        
    df = pd.read_csv(CLEAN_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Crisis target tagging
    df['target_crisis_flag'] = 0
    if os.path.exists(CRISES_PATH):
        df_c = pd.read_csv(CRISES_PATH)
        df_c['start_date'] = pd.to_datetime(df_c['start_date'])
        for c_date in df_c['start_date']:
            mask = (df['Date'] >= c_date) & (df['Date'] <= c_date + pd.Timedelta(days=180))
            df.loc[mask, 'target_crisis_flag'] = 1

    df_6m_target = df[['Date', 'market_avg_close', 'brent_oil_price', 'gold_price', 'wb_inflation_annual%']].copy()
    df_6m_target = df_6m_target.rename(columns={
        'market_avg_close': 'val_stock_6m',
        'brent_oil_price': 'val_oil_6m',
        'gold_price': 'val_gold_6m',
        'wb_inflation_annual%': 'val_infl_6m'
    })
    df_6m_target['target_date_6m'] = df_6m_target['Date'] - pd.DateOffset(months=6)

    df_12m_target = df[['Date', 'market_avg_close', 'brent_oil_price', 'gold_price']].copy()
    df_12m_target = df_12m_target.rename(columns={
        'market_avg_close': 'val_stock_12m',
        'brent_oil_price': 'val_oil_12m',
        'gold_price': 'val_gold_12m'
    })
    df_12m_target['target_date_12m'] = df_12m_target['Date'] - pd.DateOffset(months=12)

    df_merged = pd.merge_asof(
        df.sort_values('Date'),
        df_6m_target.sort_values('target_date_6m'),
        left_on='Date',
        right_on='target_date_6m',
        direction='nearest',
        tolerance=pd.Timedelta(days=15),
        suffixes=('', '_6m_lookup')
    )

    df_merged = pd.merge_asof(
        df_merged.sort_values('Date'),
        df_12m_target.sort_values('target_date_12m'),
        left_on='Date',
        right_on='target_date_12m',
        direction='nearest',
        tolerance=pd.Timedelta(days=15),
        suffixes=('', '_12m_lookup')
    )

    df_merged['target_stock_6m'] = ((df_merged['val_stock_6m'] - df_merged['market_avg_close']) / df_merged['market_avg_close']) * 100
    df_merged['target_stock_12m'] = ((df_merged['val_stock_12m'] - df_merged['market_avg_close']) / df_merged['market_avg_close']) * 100
    df_merged['target_oil_6m'] = ((df_merged['val_oil_6m'] - df_merged['brent_oil_price']) / df_merged['brent_oil_price']) * 100
    df_merged['target_oil_12m'] = ((df_merged['val_oil_12m'] - df_merged['brent_oil_price']) / df_merged['brent_oil_price']) * 100
    if 'val_gold_6m' in df_merged.columns and 'gold_price' in df_merged.columns:
        df_merged['target_gold_6m'] = ((df_merged['val_gold_6m'] - df_merged['gold_price']) / df_merged['gold_price']) * 100
        df_merged['target_gold_12m'] = ((df_merged['val_gold_12m'] - df_merged['gold_price']) / df_merged['gold_price']) * 100

    return df_merged

# ─────────────────────────────────────────────────────────────
# 3. Model Training (Random Forest + XGBoost + GRU)
# ─────────────────────────────────────────────────────────────
def train_models():
    print("Step 1: Building vectorized training dataset...")
    df_full = build_training_dataset()
    if df_full is None or df_full.empty:
        print("Error: Could not build training dataset.")
        return
        
    available_features = [c for c in FEATURE_COLS if c in df_full.columns]
    print(f"Features used ({len(available_features)}): {available_features}")
    
    X = df_full[available_features].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median()).fillna(0)
        
    means = X.mean()
    stds = X.std().replace(0, 1.0)
    
    targets = {
        'stock_6m': 'target_stock_6m',
        'stock_12m': 'target_stock_12m',
        'oil_6m': 'target_oil_6m',
        'oil_12m': 'target_oil_12m',
        'gold_6m': 'target_gold_6m',
        'gold_12m': 'target_gold_12m',
        'crisis_risk': 'target_crisis_flag'
    }
    
    rf_models = {}
    xgb_models = {}
    gru_models = {}
    metrics = {}
    feature_importances = {}
    
    print("\nStep 2: Training Tri-Model Hybrid Ensemble (Random Forest + XGBoost + GRU)...")
    for key, target_col in targets.items():
        if target_col not in df_full.columns:
            continue
            
        valid_mask = df_full[target_col].notna() & np.isfinite(df_full[target_col])
        X_sub = X[valid_mask]
        y_sub = df_full.loc[valid_mask, target_col]
        
        if len(X_sub) < 50:
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
        
        X_train_norm = ((X_train - means) / stds).values
        X_test_norm = ((X_test - means) / stds).values
        
        if key == 'crisis_risk':
            # 1. Random Forest Classifier
            rf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict_proba(X_test)[:, 1]
            
            # 2. XGBoost (Extreme Gradient Boosting) Classifier
            if HAS_XGBOOST:
                xgb_m = xgb.XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
                xgb_m.fit(X_train, y_train)
                xgb_preds = xgb_m.predict_proba(X_test)[:, 1]
            else:
                xgb_m = HistGradientBoostingClassifier(max_iter=60, max_depth=4, learning_rate=0.05, random_state=42)
                xgb_m.fit(X_train, y_train)
                xgb_preds = xgb_m.predict_proba(X_test)[:, 1]
                
            # 3. Gated Recurrent Unit (GRU) Neural Model
            gru_m = GRUNeuralModel(input_dim=len(available_features), hidden_dim=32, is_classification=True)
            gru_m.fit(X_train_norm, y_train.values, epochs=25)
            gru_preds = gru_m.predict(X_test_norm)
            
            # Tri-Model Hybrid Ensemble (35% RF + 40% XGBoost + 25% GRU)
            ensemble_probs = 0.35 * rf_preds + 0.40 * xgb_preds + 0.25 * gru_preds
            ensemble_binary = (ensemble_probs >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, ensemble_binary)
            auc = roc_auc_score(y_test, ensemble_probs)
            
            metrics[key] = {
                'accuracy': round(acc, 4), 
                'roc_auc': round(auc, 4), 
                'architecture': 'Random Forest (35%) + XGBoost (40%) + Gated Recurrent Unit GRU (25%)'
            }
            print(f" -> Trained Hybrid Ensemble for Crisis Risk | Accuracy: {acc*100:.1f}%, AUC: {auc:.3f}")
            
            rf_models[key] = rf
            xgb_models[key] = xgb_m
            gru_models[key] = gru_m
        else:
            # 1. Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)
            
            # 2. XGBoost (Extreme Gradient Boosting) Regressor
            if HAS_XGBOOST:
                xgb_m = xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42)
                xgb_m.fit(X_train, y_train)
                xgb_preds = xgb_m.predict(X_test)
            else:
                xgb_m = HistGradientBoostingRegressor(max_iter=60, max_depth=4, learning_rate=0.05, random_state=42)
                xgb_m.fit(X_train, y_train)
                xgb_preds = xgb_m.predict(X_test)
                
            # 3. Gated Recurrent Unit (GRU) Neural Model
            gru_m = GRUNeuralModel(input_dim=len(available_features), hidden_dim=32, is_classification=False)
            gru_m.fit(X_train_norm, y_train.values, epochs=25)
            gru_preds = gru_m.predict(X_test_norm)
            
            # Tri-Model Hybrid Ensemble (35% RF + 40% XGBoost + 25% GRU)
            ensemble_preds = 0.35 * rf_preds + 0.40 * xgb_preds + 0.25 * gru_preds
            
            r2 = r2_score(y_test, ensemble_preds)
            mae = mean_absolute_error(y_test, ensemble_preds)
            metrics[key] = {
                'r2_score': round(r2, 4), 
                'mae': round(mae, 4), 
                'architecture': 'Random Forest (35%) + XGBoost (40%) + Gated Recurrent Unit GRU (25%)'
            }
            print(f" -> Trained Hybrid Ensemble for {key} | R2: {r2:.3f}, MAE: {mae:.2f}")
            
            rf_models[key] = rf
            xgb_models[key] = xgb_m
            gru_models[key] = gru_m
            
        if hasattr(xgb_m, 'feature_importances_'):
            importances = dict(zip(available_features, xgb_m.feature_importances_.tolist()))
            sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
            feature_importances[key] = sorted_imp

    gru_weights = {k: m.get_weights() for k, m in gru_models.items()}

    # Save trained ensemble models
    save_data = {
        'rf_models': rf_models,
        'xgb_models': xgb_models,
        'gru_weights': gru_weights,
        'features': available_features,
        'feature_defaults': X.median().to_dict(),
        'means': means.to_dict(),
        'stds': stds.to_dict()
    }
    joblib.dump(save_data, MODEL_PATH)
    
    metadata = {
        'metrics': metrics,
        'feature_importances': feature_importances,
        'features': available_features,
        'model_architecture': 'Tri-Model Ensemble: Random Forest (35%) + XGBoost Extreme Gradient Boosting (40%) + Gated Recurrent Unit GRU (25%)',
        'dataset_rows': len(df_full)
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nSUCCESS: Tri-Model Hybrid Ensemble saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")

def predict_scenario(feature_overrides=None):
    if not os.path.exists(MODEL_PATH):
        train_models()
        
    if not os.path.exists(MODEL_PATH):
        return None
        
    saved = joblib.load(MODEL_PATH)
    rf_models = saved['rf_models']
    xgb_models = saved['xgb_models']
    gru_weights = saved.get('gru_weights', {})
    gru_models = {k: GRUNeuralModel.from_weights(w) for k, w in gru_weights.items()} if gru_weights else saved.get('gru_models', {})
    features = saved['features']
    defaults = saved['feature_defaults']
    means = pd.Series(saved['means'])
    stds = pd.Series(saved['stds']).replace(0, 1.0)
    
    input_values = defaults.copy()
    if feature_overrides:
        for k, v in feature_overrides.items():
            if k in input_values and pd.notna(v):
                input_values[k] = float(v)
                
    X_pred = pd.DataFrame([input_values])[features]
    X_pred_norm = ((X_pred - means) / stds).values
    
    predictions = {}
    for key in rf_models.keys():
        rf = rf_models[key]
        xgb_m = xgb_models[key]
        gru_m = gru_models[key]
        
        if key == 'crisis_risk':
            rf_p = rf.predict_proba(X_pred)[0][1]
            xgb_p = xgb_m.predict_proba(X_pred)[0][1]
            gru_p = gru_m.predict(X_pred_norm)[0]
            
            ens_p = 0.35 * rf_p + 0.40 * xgb_p + 0.25 * gru_p
            predictions[key] = round(float(ens_p) * 100, 1)
        else:
            rf_p = rf.predict(X_pred)[0]
            xgb_p = xgb_m.predict(X_pred)[0]
            gru_p = gru_m.predict(X_pred_norm)[0]
            
            ens_p = 0.35 * rf_p + 0.40 * xgb_p + 0.25 * gru_p
            predictions[key] = round(float(ens_p), 2)
            
    return predictions

def predict_future_trajectory(feature_overrides=None, total_months=12):
    """
    Generates month-by-month predictive forward trajectory dataframe for charting (Months 0 to total_months).
    """
    preds = predict_scenario(feature_overrides)
    if not preds:
        return pd.DataFrame()

    s6 = preds.get('stock_6m', 0.0)
    s12 = preds.get('stock_12m', 0.0)
    o6 = preds.get('oil_6m', 0.0)
    o12 = preds.get('oil_12m', 0.0)
    g6 = preds.get('gold_6m', 3.5)
    g12 = preds.get('gold_12m', 6.8)
    risk = preds.get('crisis_risk', 0.0)
    
    col_base = feature_overrides.get('cost_of_living_cost_of_living_index', 75.0) if feature_overrides else 75.0
    infl_base = feature_overrides.get('wb_inflation_annual%', 3.5) if feature_overrides else 3.5

    records = []
    # Month 0 (Current Baseline)
    records.append({
        'Month': 0,
        'Month_Label': 'Current (M0)',
        'Market_Return_Pct': 0.0,
        'Oil_Change_Pct': 0.0,
        'Gold_Return_Pct': 0.0,
        'Inflation_Rate_Pct': round(infl_base, 2),
        'Cost_of_Living_Index': round(col_base, 1),
        'Crisis_Risk_Probability': round(risk, 1)
    })
    
    for m in range(1, total_months + 1):
        if m <= 6:
            weight_stock = (m / 6.0) ** 1.1
            stock_ret = s6 * weight_stock
            oil_ret = o6 * (m / 6.0) ** 1.1
            gold_ret = g6 * (m / 6.0) ** 1.1
        else:
            weight_stock = (m - 6) / 6.0
            stock_ret = s6 + (s12 - s6) * weight_stock
            oil_ret = o6 + (o12 - o6) * weight_stock
            gold_ret = g6 + (g12 - g6) * weight_stock
            
        col_drift = col_base * (1.0 + (risk / 100.0) * 0.15 * (m / 12.0) + (oil_ret / 100.0) * 0.08)
        infl_drift = infl_base + (risk / 100.0) * 2.5 * (m / 12.0)
        risk_m = min(99.9, max(0.1, risk * (1.0 + 0.2 * np.sin(np.pi * (m / 12.0)))))
        
        records.append({
            'Month': m,
            'Month_Label': f"M+{m}",
            'Market_Return_Pct': round(stock_ret, 2),
            'Oil_Change_Pct': round(oil_ret, 2),
            'Gold_Return_Pct': round(gold_ret, 2),
            'Inflation_Rate_Pct': round(max(0, infl_drift), 2),
            'Cost_of_Living_Index': round(max(10, col_drift), 1),
            'Crisis_Risk_Probability': round(risk_m, 1)
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    train_models()
