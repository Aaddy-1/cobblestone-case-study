import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_feature_set(df):
    """
    Transforms master data into a LightGBM-ready matrix.
    Captures the 'Merit Order Effect' via Residual Load.
    """
    # 1. Fundamental Drivers: Residual Load
    if 'load' in df.columns and 'renewables' in df.columns:
        df['residual_load'] = df['load'] - df['renewables']

    # 2. Time-Series Lags (24h and 168h)
    df['price_lag_24'] = df['day_ahead_price'].shift(24)
    df['price_lag_168'] = df['day_ahead_price'].shift(168)

    # 3. Calendar Seasonality
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # --- Explicitly Log Data Truncation ---
    initial_len = len(df)
    df_clean = df.dropna()
    dropped_rows = initial_len - len(df_clean)
    
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} initial rows due to NaNs created by lag features (e.g., 168h lag).")

    return df_clean