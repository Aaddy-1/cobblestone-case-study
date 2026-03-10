import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

def train_and_predict(df):
    """
    Implements a baseline and an Improved Model (LightGBM)
    using Time-Series Cross-Validation for rigorous evaluation.
    Finally, generates a TRUE forecast for the next 24 hours (Tomorrow).
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
        
    # --- 2. Final Forecast: Predicting the TRUE Next 24 Hours ---
    # Train the model on the ENTIRE dataset to use the most recent information
    model.fit(X, y)
    
    # Construct the datetime index for tomorrow (Next 24 hours)
    last_timestamp = df.index[-1]
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=24, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    
    # Populate Time Features
    if 'hour' in features: future_df['hour'] = future_dates.hour
    if 'day_of_week' in features: future_df['day_of_week'] = future_dates.dayofweek
    if 'is_weekend' in features: future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Populate Lags (Using the known prices from the end of our dataset)
    if 'price_lag_24' in features: future_df['price_lag_24'] = df['day_ahead_price'].iloc[-24:].values
    if 'price_lag_168' in features: future_df['price_lag_168'] = df['day_ahead_price'].iloc[-168:-144].values
    
    # Populate Fundamental Forecasts 
    # PROTOTYPE ASSUMPTION: We use a "naive persistence" forecast, assuming tomorrow's 
    # wind/solar/load profile will perfectly match today's outturn.
    # In production, these columns would be populated by external forecast APIs.
    for col in ['load', 'renewables', 'residual_load', 'wind_onshore', 'wind_offshore', 'solar']:
        if col in features:
            future_df[col] = df[col].iloc[-24:].values

    # Ensure the column order matches exactly what the model was trained on
    future_df = future_df[features]
    
    # Generate the prediction for Tomorrow
    future_preds = model.predict(future_df)
    
    # Ensure baseline is included for the visualization
    results = pd.DataFrame({
        'baseline': future_df['price_lag_24'].values, # Tomorrow = Today
        'improved': future_preds
    }, index=future_df.index)

    
    
    # Use the robust CV MAE averages for reporting
    baseline_mae_avg = np.mean(cv_baseline_maes)
    improved_mae_avg = np.mean(cv_improved_maes)
    
    metrics = {
        "baseline_mae": baseline_mae_avg,
        "improved_mae": improved_mae_avg,
        "improvement_pct": ((baseline_mae_avg - improved_mae_avg) / baseline_mae_avg) * 100
    }
    
    return results, metrics