import pandas as pd
import os
import logging

# Set up logging for the QA process
logger = logging.getLogger(__name__)

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
    """
    Ingests raw data, performs Quality Assurance (QA) checks, and prepares the master dataframe.
    
    Data Sources (Example):
    - Consumption: ENTSO-E Transparency Platform
    - Generation: SMARD.de (Bundesnetzagentur)
    - Prices: EPEX SPOT / ENTSO-E
    """
    logger.info("Starting Data Ingestion...")

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
    master_df['renewables'] = master_df['wind_onshore'].fillna(0) + master_df['wind_offshore'].fillna(0) + master_df['solar'].fillna(0)

    # --- QUALITY ASSURANCE (QA) CHECKS ---
    logger.info("Running QA Checks...")
    qa_alerts = [] # NEW: List to store warnings
    
    # QA Check 1: Quantify Missing Data before imputation
    missing_counts = master_df.isna().sum()
    logger.info(f"Missing values per column before imputation:\n{missing_counts[missing_counts > 0]}")
    
    total_rows = len(master_df)
    for col in master_df.columns:
        missing_pct = master_df[col].isna().sum() / total_rows
        if missing_pct > 0.05:
            msg = f"High missing data ratio in {col}: {missing_pct:.1%}"
            logger.warning(msg)
            qa_alerts.append(msg)

    # QA Check 2: Physical bounds (Load should be positive)
    if (master_df['load'] <= 0).any():
        logger.warning("QA Alert: Found zero or negative grid load values.")
        qa_alerts.append("Found zero or negative grid load values.")
        
    # QA Check 3: Price Outliers
    if (master_df['day_ahead_price'] < -500).any() or (master_df['day_ahead_price'] > 4000).any():
        logger.warning("QA Alert: Extreme Day-Ahead prices detected (<-500 or >4000 EUR).")
        qa_alerts.append("Extreme Day-Ahead prices detected.")

    # --- IMPUTATION ---
    logger.info("Imputing missing values using forward/backward fill.")
    master_df = master_df.ffill().bfill()

    # Final verification
    assert master_df.isna().sum().sum() == 0, "QA Failed: NaNs remain after imputation!"

    logger.info(f"Ingestion complete. Final Dataset shape: {master_df.shape}")
    
    # NEW: Create a summary string to pass to the LLM
    qa_summary = "All QA checks passed cleanly." if not qa_alerts else "WARNINGS: " + "; ".join(qa_alerts)
    
    return master_df, qa_summary # NEW: Returning a tuple now