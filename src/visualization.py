import matplotlib.pyplot as plt
import os

def save_forecast_plot(results, output_path='outputs/forecast_comparison.png'):
    """
    Generates and saves a plot comparing the Baseline vs. Improved model forecasts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plotting the baseline (Persistence) vs the model prediction
    plt.plot(results.index, results['baseline'], label='Baseline (Persistence)', linestyle='--', color='gray', alpha=0.7)
    plt.plot(results.index, results['improved'], label='Improved (LightGBM)', color='blue', linewidth=2)
    
    plt.title('Day-Ahead Price Forecast: Next 24 Hours', fontsize=14)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Price (EUR/MWh)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()