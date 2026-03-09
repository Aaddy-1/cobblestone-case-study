import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def generate_summary(report_context):
    """
    Uses the modern Google GenAI SDK to transform metrics into a trading summary.
    """
    # Key is now passed directly to the Client object
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    
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
        # Modern syntax for content generation
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Summary unavailable: {str(e)}"