import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Setup your API Key in environment variables or hardcode for the prototype
genai.configure(api_key=os.environ["GEMINI_KEY"])

def generate_summary(report_context):
    """
    Uses an LLM to transform technical metrics into a trading summary.
    """
    prompt = f"""
    You are a Quant Trading Assistant for a European Power Desk. 
    Review the following daily model results and provide a 3-sentence summary:
    1. Overall model health (Baseline vs. Improved MAE).
    2. The suggested trading direction (Fair Value vs. Curve).
    3. Any data quality concerns.

    Context:
    - Baseline MAE: {report_context['metrics']['baseline_mae']:.2f}
    - Improved MAE: {report_context['metrics']['improved_mae']:.2f}
    - Improvement: {report_context['metrics']['improvement_pct']:.1f}%
    - Predicted Fair Value (Avg): {report_context['fair_value']:.2f} EUR/MWh
    - Trading Signal: {report_context['signal']}
    """
    
    try:
        # Initializing the model (Gemini 3 Flash)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Summary unavailable: {str(e)}"