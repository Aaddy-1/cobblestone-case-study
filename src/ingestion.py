import pandas as pd
import os

def load_exact_column(filepath, exact_col_name, new_col_name):
    """Reads a file and extracts exactly one renamed column."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")

    df = pd.read_csv(filepath, sep=';', thousands=',')
    df['timestamp'] = pd.to_datetime(df['Start date'], format='%b %d, %Y %I:%M %p')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    return df[[exact_col_name]].rename(columns={exact_col_name: new_col_name})

def main_ingestion():
    # 1. Consumption
    df_con = load_exact_column(
        "data/consumption.csv",
        "grid load [MWh] Calculated resolutions",
        "load"
    )

    # 2. Prices
    df_price = load_exact_column(
        "data/prices.csv",
        "Germany/Luxembourg [€/MWh] Calculated resolutions",
        "day_ahead_price"
    )

    # 3. Generation (Extracts three specific columns)
    df_gen_raw = pd.read_csv("data/generation.csv", sep=';', thousands=',')
    df_gen_raw['timestamp'] = pd.to_datetime(df_gen_raw['Start date'], format='%b %d, %Y %I:%M %p')
    df_gen_raw.set_index('timestamp', inplace=True)
    df_gen_raw = df_gen_raw[~df_gen_raw.index.duplicated(keep='first')]
    
    df_gen = df_gen_raw[[
        'Wind onshore [MWh] Calculated resolutions',
        'Wind offshore [MWh] Calculated resolutions',
        'Photovoltaics [MWh] Calculated resolutions'
    ]].rename(columns={
        'Wind onshore [MWh] Calculated resolutions': 'wind_onshore',
        'Wind offshore [MWh] Calculated resolutions': 'wind_offshore',
        'Photovoltaics [MWh] Calculated resolutions': 'solar'
    })

    # Merge them all based on the datetime index
    master_df = pd.concat([df_gen, df_con, df_price], axis=1)

    # Calculate renewables and handle missing values
    master_df['renewables'] = master_df['wind_onshore'].fillna(0) + master_df['wind_offshore'].fillna(0) + master_df['solar'].fillna(0)
    master_df = master_df.ffill().bfill()

    print(f"Ingestion complete. Dataset shape: {master_df.shape}")
    return master_df