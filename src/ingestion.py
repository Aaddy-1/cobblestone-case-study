import requests
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class SmardDataIngestor:
    BASE_URL = "https://www.smard.de/app/chart_data"
    
    FILTERS = {
        "day_ahead_price": 4169,
        "wind_onshore_forecast": 4067,
        "wind_offshore_forecast": 4068,
        "solar_forecast": 4066,
        "load_forecast": 4127
    }

    def fetch_data(self, filter_id, region="DE", resolution="hour"):
        index_url = f"{self.BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
        
        try:
            response = requests.get(index_url)
            if response.status_code != 200:
                return pd.DataFrame()
            
            indices = response.json().get('timestamps', [])
            if not indices:
                return pd.DataFrame()

            # To get enough data for training, we take the last 3 chunks of data
            # Each chunk is usually a week or a month of data
            all_chunks = []
            for ts in indices[-3:]: 
                data_url = f"{self.BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"
                data_res = requests.get(data_url)
                if data_res.status_code == 200:
                    chunk_data = data_res.json().get('series', [])
                    all_chunks.extend(chunk_data)
                time.sleep(0.1) # Be kind to the API

            if not all_chunks:
                return pd.DataFrame()

            df = pd.DataFrame(all_chunks, columns=['timestamp', 'value'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

        except Exception as e:
            logger.error(f"Request failed for {filter_id}: {e}")
            return pd.DataFrame()

    def run_qa_checks(self, df, name):
        """QA Logic with safety for empty DataFrames"""
        if df.empty or 'value' not in df.columns:
            logger.error(f"QA Failed: No data for {name}")
            return pd.DataFrame(), {"status": "Critical Failure", "count": 0}

        checks = {
            "missing_values": df['value'].isnull().sum(),
            "duplicate_timestamps": df['timestamp'].duplicated().sum(),
            "count": len(df)
        }
        
        # Clean the data
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df['value'] = df['value'].interpolate(method='linear')
        
        return df, checks

def main_ingestion():
    ingestor = SmardDataIngestor()
    data_frames = {}

    for key, filter_id in ingestor.FILTERS.items():
        logger.info(f"Fetching {key}...")
        raw_df = ingestor.fetch_data(filter_id)
        clean_df, qa_results = ingestor.run_qa_checks(raw_df, key)
        
        if not clean_df.empty:
            data_frames[key] = clean_df.set_index('timestamp')

    if not data_frames:
        raise ValueError("No data was collected. Check your internet connection or SMARD status.")

    # Combine all data into one DataFrame
    master_df = pd.concat(data_frames.values(), axis=1)
    master_df.columns = data_frames.keys()
    
    # Fill any remaining NaNs across the whole table
    master_df = master_df.ffill().bfill()
    
    logger.info("Ingestion Complete.")
    return master_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = main_ingestion()
    print(df.tail())