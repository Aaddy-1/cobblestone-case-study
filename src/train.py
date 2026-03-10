import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

def train_and_predict(df):
    """
    Implements a baseline and an Improved Model (LightGBM)
    using Time-Series Cross-Validation for rigorous evaluation.
    """
    target = 'day_ahead_price'
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]
    
    # --- 1. Rigorous Validation: Time-Series Cross Validation ---
    tscv = TimeSeriesSplit(n_splits=3)
    cv_baseline_maes = []
    cv_improved_maes = []
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        importance_type='gain',
        verbosity=-1
    )
    
    for train_idx, test_idx in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # Baseline CV (Yesterday's price)
        cv_baseline_preds = X_test_cv['price_lag_24']
        cv_baseline_maes.append(mean_absolute_error(y_test_cv, cv_baseline_preds))
        
        # Improved CV
        model.fit(X_train_cv, y_train_cv)
        cv_improved_preds = model.predict(X_test_cv)
        cv_improved_maes.append(mean_absolute_error(y_test_cv, cv_improved_preds))
        
    # --- 2. Final Forecast (Last 48 hours for the Trading View) ---
    train_df = df.iloc[:-48]
    test_df = df.iloc[-48:]
    
    X_train_final, y_train_final = train_df[features], train_df[target]
    X_test_final, y_test_final = test_df[features], test_df[target]
    
    model.fit(X_train_final, y_train_final)
    final_improved_preds = model.predict(X_test_final)
    final_baseline_preds = X_test_final['price_lag_24']
    
    results = pd.DataFrame({
        'actual': y_test_final,
        'baseline': final_baseline_preds,
        'improved': final_improved_preds
    }, index=y_test_final.index)
    
    # Use the robust CV MAE averages for reporting
    baseline_mae_avg = np.mean(cv_baseline_maes)
    improved_mae_avg = np.mean(cv_improved_maes)
    
    metrics = {
        "baseline_mae": baseline_mae_avg,
        "improved_mae": improved_mae_avg,
        "improvement_pct": ((baseline_mae_avg - improved_mae_avg) / baseline_mae_avg) * 100
    }
    
    return results, metrics