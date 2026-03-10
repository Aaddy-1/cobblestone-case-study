import os
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def generate_summary(report_context):
    """
    Uses the modern Google GenAI SDK to transform metrics into a trading summary,
    and logs the exact prompt/output for audit purposes.
    """
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    
    prompt = f"""
    You are a Quant Trading Assistant for a European Power Desk. 
    Review the following daily model results and provide a short, punchy 3-sentence summary:
    1. Overall model health (Baseline vs. Improved MAE).
    2. The suggested trading direction (Fair Value vs. Curve).
    3. Any data quality concerns.

    Context:
    - Baseline CV MAE: {report_context['metrics']['baseline_mae']:.2f}
    - Improved CV MAE: {report_context['metrics']['improved_mae']:.2f}
    - Improvement: {report_context['metrics']['improvement_pct']:.1f}%
    - Predicted Fair Value (Avg): {report_context['fair_value']:.2f} EUR/MWh
    - Trading Signal: {report_context['signal']}
    - QA Summary: {report_context['qa_summary']}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        # --- NEW: Logging the prompt and output to a file ---
        os.makedirs('logs', exist_ok=True)
        with open('logs/llm_audit.log', 'a') as f:
            f.write("========== NEW LLM RUN ==========\n")
            f.write(f"PROMPT:\n{prompt}\n")
            f.write(f"OUTPUT:\n{response.text}\n")
            f.write("=================================\n\n")
            
        logger.info("LLM prompt and response successfully logged to logs/llm_audit.log")
        
        return response.text
    except Exception as e:
        logger.error(f"LLM API Error: {str(e)}")
        return f"AI Summary unavailable: {str(e)}"