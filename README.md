### **README.md**

# European Power Day-Ahead Fair Value Pipeline

This repository contains a prototype pipeline for forecasting European Power Day-Ahead (DA) prices and translating those forecasts into tradable prompt curve views.

## **1. Methodology & Rigor**

* **Time-Series Cross-Validation:** Uses `TimeSeriesSplit` to calculate a robust Mean Absolute Error (MAE) across multiple historical windows, ensuring the model's performance isn't a fluke of a single test period.
* **Out-of-Sample Forecasting:** Unlike basic prototypes that predict the past, this system generates a true 24-hour forecast for the "Next Day" using naive persistence for fundamental drivers.
* **Leakage Prevention:** Strict adherence to forward-filling (`ffill`) for missing data to ensure future information never leaks into the training sets.
* **Fundamental Drivers:** Captures the **Merit Order Effect** by calculating **Residual Load** (Total Load - Renewables) as a primary feature.

## **2. Setup & Installation**

### **Prerequisites**

* Python 3.9+
* A Google Gemini API Key (stored in a `.env` file)

### **Installation**

1. **Clone the repository:**
```bash
git clone <repo-url>
cd cobblestone-case-study

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Environment:**
Create a `.env` file in the root directory and add your API key:
```env
GEMINI_KEY=your_api_key_here

```



## **3. Usage**

To run the full pipeline (Ingestion -> Features -> Training -> Visualization -> AI Summary):

```bash
python main.py

```

### **Pipeline Outputs**

* **Terminal:** Logs detailed QA checks, CV metrics, and the final AI Market Insight.
* **`logs/llm_audit.log`:** A full audit trail of every prompt sent to the LLM and its raw response.
* **`outputs/forecast_comparison.png`:** A visual comparison of the persistence baseline vs. the LightGBM fair value forecast.
* **`outputs/submission.csv`:** (Optional) The raw hourly predictions for the next 24 hours.

## **4. Quality Assurance (QA)**

The pipeline implements three automated layers of QA:

1. **Missing Data Analysis:** Quantifies and logs all NaNs before deciding to fill or drop.
2. **Physical Bound Checks:** Flags non-physical values (e.g., negative grid load).
3. **Price Outlier Detection:** Identifies extreme volatility that may skew model training.

## **5. Assumptions & Limitations**

* **Perfect Foresight:** This prototype uses actual outturns for renewables and load. In a production environment, these would be replaced by Day-Ahead weather and demand forecasts.
* **Market Price Benchmark:** The "Curve" price is currently set to a dummy value (65.0 EUR/MWh) to demonstrate signal translation logic.