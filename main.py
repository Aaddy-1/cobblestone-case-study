import logging
import pandas as pd
from src.ingestion import main_ingestion
from src.features import build_feature_set
from src.train import train_and_predict
from src.llm_report import generate_summary

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("--- Step 1: Data Ingestion & QA ---")
    master_df = main_ingestion()
    # In main.py
    # master_df = main_ingestion()
    print("Columns after ingestion:", master_df.columns.tolist()) # ADD THIS LINE
    df_with_features = build_feature_set(master_df)

    logger.info("--- Step 2: Feature Engineering ---")
    # Note: Ensure you have created src/features.py based on previous discussion
    df_with_features = build_feature_set(master_df)

    logger.info("--- Step 3: Forecasting & Validation ---")
    results, metrics = train_and_predict(df_with_features)

    logger.info("--- Step 4: Prompt Curve Translation ---")
    # Assume a dummy Front-Month/Week price of 65.0 EUR for the view
    market_prompt_price = 65.0 
    predicted_fair_value = results['improved'].mean()
    
    # Logic: If fair value is higher than market, we expect DA to settle higher (Long)
    signal = "LONG DA / SHORT CURVE" if predicted_fair_value > market_prompt_price else "SHORT DA / LONG CURVE"

    logger.info("--- Step 5: AI/LLM Integration ---")
    report_context = {
        "metrics": metrics,
        "fair_value": predicted_fair_value,
        "signal": signal
    }
    ai_insight = generate_summary(report_context)
    
    print("\n" + "="*30)
    print("DAILY POWER MARKET REPORT")
    print("="*30)
    print(f"Fair Value Prediction: {predicted_fair_value:.2f} EUR/MWh")
    print(f"Trading View: {signal}")
    print("\nAI INSIGHT:")
    print(ai_insight)

if __name__ == "__main__":
    main()