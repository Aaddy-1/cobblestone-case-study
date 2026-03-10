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
    master_df, qa_summary = main_ingestion()
    print("Columns after ingestion:", master_df.columns.tolist())

    logger.info("--- Step 2: Feature Engineering ---")
    df_with_features = build_feature_set(master_df)

    logger.info("--- Step 3: Forecasting & Validation ---")
    results, metrics = train_and_predict(df_with_features)

    # ... (Step 1, 2, 3 remain the same) ...
    logger.info("--- Step 4: Prompt Curve Translation ---")
    market_prompt_price = 65.0 
    predicted_fair_value = results['improved'].mean()
    signal = "LONG DA / SHORT CURVE" if predicted_fair_value > market_prompt_price else "SHORT DA / LONG CURVE"

    logger.info("--- Step 5: AI/LLM Integration ---")
    report_context = {
        "metrics": metrics,
        "fair_value": predicted_fair_value,
        "signal": signal,
        "qa_summary": qa_summary
    }
    ai_insight = generate_summary(report_context)
    
    print("\n" + "="*30)
    print("DAILY POWER MARKET REPORT")
    print("="*30)
    print(f"Fair Value Prediction: {predicted_fair_value:.2f} EUR/MWh")
    print(f"Trading View: {signal}")
    print("\n[SYSTEM WARNING] Invalidate this view if intraday renewable generation "
          "deviates by >15% from day-ahead forecasts, or if major plant outages occur.")
    print("\nAI INSIGHT:")
    print(ai_insight)

if __name__ == "__main__":
    main()