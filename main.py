import logging
import pandas as pd
import os
from src.ingestion import main_ingestion
from src.features import build_feature_set
from src.train import train_and_predict
from src.llm_report import generate_summary
from src.visualization import save_forecast_plot

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
    
    # Log the cross-validation performance
    logger.info(f"Baseline CV MAE: {metrics['baseline_mae']:.2f} EUR/MWh")
    logger.info(f"Improved CV MAE: {metrics['improved_mae']:.2f} EUR/MWh")
    logger.info(f"Improvement: {metrics['improvement_pct']:.1f}%")

    # Generate Visual Artifact
    save_forecast_plot(results)
    logger.info("Forecast visualization saved to outputs/forecast_comparison.png")

    # --- NEW: Generate submission.csv ---
    submission_df = results[['improved']].copy()
    submission_df.reset_index(inplace=True)
    submission_df.columns = ['id', 'y_pred'] # Renaming to match case study requirements
    
    os.makedirs('outputs', exist_ok=True)
    submission_df.to_csv('outputs/submission.csv', index=False)
    logger.info("Out-of-sample predictions saved to outputs/submission.csv")

    logger.info("--- Step 4: Prompt Curve Translation ---")
    market_prompt_price = 65.0 
    predicted_fair_value = results['improved'].mean()
    signal = "LONG DA / SHORT CURVE" if predicted_fair_value > market_prompt_price else "SHORT DA / LONG CURVE"
    
    # Log the pricing logic so the user can see how the signal was derived
    logger.info(f"Current Market Prompt Price: {market_prompt_price:.2f} EUR/MWh")
    logger.info(f"Model Predicted Fair Value:  {predicted_fair_value:.2f} EUR/MWh")
    logger.info(f"Generated Signal: {signal}")

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