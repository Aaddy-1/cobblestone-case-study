import pandas as pd

def build_feature_set(df):
    """
    Transforms master data into a LightGBM-ready matrix.
    Captures the 'Merit Order Effect' via Residual Load.
    """
    # 1. Fundamental Drivers: Residual Load [cite: 17]
    # Essential for models that outperform baseline LightGBM [cite: 22]
    if 'load' in df.columns and 'renewables' in df.columns:
        df['residual_load'] = df['load'] - df['renewables']

    # 2. Time-Series Lags (24h and 168h) [cite: 430]
    # Addresses the 'Temporal Awareness' gap in basic tree models [cite: 93]
    df['price_lag_24'] = df['day_ahead_price'].shift(24)
    df['price_lag_168'] = df['day_ahead_price'].shift(168)

    # 3. Calendar Seasonality [cite: 452]
    # Differentiates between working days, weekends, and holidays [cite: 453]
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df.dropna()