import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pandas as pd

def train_and_predict(df):
    """
    Implements a baseline (Persistence) and an Improved Model (LightGBM).
    """
    # 1. Feature Prep
    # We use 'day_ahead_price' as target. t
    # Features include lags, residual load, and time variables.
    target = 'day_ahead_price'
    features = [col for col in df.columns if col != target]
    
    # 2. Simple Time-Based Split (Last 48 hours for testing)
    train_df = df.iloc[:-48]
    test_df = df.iloc[-48:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # 3. Baseline Model: Naive Persistence (Tomorrow = Today)
    # Predicted price is just the price from 24h ago
    baseline_preds = X_test['price_lag_24']
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    
    # 4. Improved Model: LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        importance_type='gain',
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    improved_preds = model.predict(X_test)
    improved_mae = mean_absolute_error(y_test, improved_preds)
    
    # 5. Output results
    results = pd.DataFrame({
        'actual': y_test,
        'baseline': baseline_preds,
        'improved': improved_preds
    }, index=y_test.index)
    
    metrics = {
        "baseline_mae": baseline_mae,
        "improved_mae": improved_mae,
        "improvement_pct": ((baseline_mae - improved_mae) / baseline_mae) * 100
    }
    
    return results, metrics