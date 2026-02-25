import os
import pandas as pd
import pickle
from typing import Dict

# Configuration for caching
CACHE_DIR = os.path.join(os.getcwd(), 'cache')
DATASET_BASE_PATH = r'Dataset\LBNL_FDD_Data_Sets_DDAHU'

def load_anomaly_datasets(
    base_path: str = DATASET_BASE_PATH, 
    use_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Extracts all .csv files from the specified folder and returns them as a dictionary.
    Includes a persistent caching mechanism to avoid redundant CSV parsing.
    
    Args:
        base_path (str): The path to the data folder.
        use_cache (bool): Whether to use the disk cache. Defaults to True.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames {filename: dataframe}.
    """
    target_dir = os.path.abspath(base_path)
    csv_data_dict = {}

    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"The directory '{target_dir}' was not found.")

    # Sensibility check: Verify cache presence and population status
    if use_cache:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            print(f"📁 Created cache directory at: {CACHE_DIR}")
            print("🚀 First run detected. Cache will be populated from CSV files.")
        elif not any(f.endswith('.pkl') for f in os.listdir(CACHE_DIR)):
            print("📁 Cache directory exists but is empty. Populating cache...")
        else:
            print(f"✅ Cache detected at {CACHE_DIR}. Syncing datasets...")

    print(f"🚀 Initializing data extraction (Mode: {'Cache-First' if use_cache else 'Direct CSV'})")
    
    # List all .csv files in source
    files = [f for f in os.listdir(target_dir) if f.lower().endswith('.csv')]
    
    if not files:
        print("⚠️ No .csv files found in the source directory.")
        return csv_data_dict

    for filename in files:
        cache_file = os.path.join(CACHE_DIR, f"{filename}.pkl")
        file_path = os.path.join(target_dir, filename)
        
        # Try loading from cache first
        if use_cache and os.path.exists(cache_file):
            print(f"⚡ Loading from cache: {filename}")
            try:
                csv_data_dict[filename] = pd.read_pickle(cache_file)
                continue
            except Exception as e:
                print(f"⚠️ Cache read error for {filename}, falling back to CSV: {e}")

        # Loading from CSV
        print(f"🔍 Parsing CSV: {filename}...")
        try:
            df = pd.read_csv(file_path)
            csv_data_dict[filename] = df
            
            # Save to cache for next time
            if use_cache:
                print(f"💾 Saving {filename} to cache...")
                df.to_pickle(cache_file)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✨ Successfully loaded {len(csv_data_dict)} datasets.")
    return csv_data_dict

def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Loads a specific dataset by name. Checks cache first, then the source directory.
    
    Args:
        file_name (str): The name of the file to load (e.g., 'Sample_Data.csv').
        
    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        FileNotFoundError: If the file is not found in cache or source directory.
    """
    cache_file = os.path.join(CACHE_DIR, f"{file_name}.pkl")
    source_file = os.path.join(os.path.abspath(DATASET_BASE_PATH), file_name)
    
    # 1. Check Cache
    if os.path.exists(cache_file):
        print(f"⚡ Loading from cache: {file_name}")
        try:
            return pd.read_pickle(cache_file)
        except Exception as e:
            print(f"⚠️ Cache read error for {file_name}, falling back to CSV: {e}")
            
    # 2. Check Directory
    if os.path.exists(source_file):
        print(f"🔍 Parsing CSV: {file_name} from source directory...")
        try:
            df = pd.read_csv(source_file)
            # Optional: Populate cache if missing
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
            df.to_pickle(cache_file)
            return df
        except Exception as e:
            raise Exception(f"❌ Error processing {file_name}: {e}")
            
    # 3. Raise Error if not found in both
    raise FileNotFoundError(f"File not found: '{file_name}' was not found in cache or '{DATASET_BASE_PATH}'")

if __name__ == "__main__":
    # Test block to verify loader functionality
    try:
        """# 1. Test batch loading
        data = load_anomaly_datasets()
        print("\nSummary of loaded datasets:")
        for name, df in data.items():
            print(f" - {name}: {len(df)} rows, {len(df.columns)} columns")"""
        
        # 2. Test single file loading
        test_file = "DualDuct_FaultFree.csv"
        print(f"\n--- Testing load_dataset with '{test_file}' ---")
        df_single = load_dataset(test_file)
        print(f"✅ Successfully loaded '{test_file}': {len(df_single)} rows")
        
        # 3. Test non-existent file
        print(f"\n--- Testing load_dataset with non-existent file ---")
        try:
            load_dataset("non_existent.csv")
        except FileNotFoundError as e:
            print(f"✅ Caught expected error: {e}")

    except Exception as error:
        print(f"Critical Failure: {error}")
  