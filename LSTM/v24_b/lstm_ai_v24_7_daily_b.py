"""
Author: QuaziBit

Advanced LSTM Cryptocurrency Price Prediction System
Version: 24.7
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.lines as mlines
from sklearn.metrics import mean_absolute_error, r2_score
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline
from functools import lru_cache
import warnings
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import json
import time
import pickle
import torch.optim as optim
import mplfinance as mpf
import time

warnings.filterwarnings('ignore')

# ========================= Get directory path ========================= #
def set_ticker_name() -> str:
    # Get cryptocurrency name from the data filename (removing file extension)
    ticker_name = os.path.splitext(Config.STOCK_DATA_FILENAME)[0].split('_')[0]
    Config.TICKER_NAME = ticker_name

def get_script_based_dir() -> str:
    """Get directory path based on current script name and cryptocurrency"""
    # Get current script name without extension
    current_script = os.path.splitext(os.path.basename(__file__))[0]
    
    # Get cryptocurrency name from the data filename (removing file extension)
    crypto_name = os.path.splitext(Config.STOCK_DATA_FILENAME)[0].split('_')[0]
    
    # Create root directory with script name and crypto
    root_dir = os.path.join(os.path.dirname(__file__), current_script, crypto_name)
    
    # Create standard subdirectories
    subdirs = {
        'visualizations': ['charts', 'logs'],  # Removed 'models' from here
        'data': [],  # empty list means no subdirectories
        'temp': [],
        'models': ['BasicStockModel']  # Added new top-level models directory
    }
    
    # Create all directories
    for main_dir, sub_dirs in subdirs.items():
        main_path = os.path.join(root_dir, main_dir)
        os.makedirs(main_path, exist_ok=True)
        
        # Create subdirectories if any
        for sub_dir in sub_dirs:
            sub_path = os.path.join(main_path, sub_dir)
            os.makedirs(sub_path, exist_ok=True)
    
    return root_dir

def get_script_version() -> str:
    """Extract version from script filename"""
    try:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        # Look for v{number} pattern
        version_parts = [part for part in script_name.split('_') if part.startswith('v')]
        if version_parts:
            # Extract version number (e.g., 'v23_2' -> '2.0')
            version = version_parts[0].replace('v', '')
            subversion = script_name.split(version)[1].split('_')[1]
            return f"{version}.{subversion}"
    except Exception as e:
        logger.warning(f"Could not extract version from filename: {e}")
    return "1.0"  # Default version if extraction fails

def save_temp_file(data: Any, filename: str):
    """Save temporary data to the temp directory
    
    Args:
        data (Any): Data to save
        filename (str): Filename with extension (.json or .pkl)
    """
    root_dir = get_script_based_dir()
    temp_dir = os.path.join(root_dir, 'temp')
    filepath = os.path.join(temp_dir, filename)
    
    # Save based on file type
    if filename.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(data, f)
    elif filename.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    logger.info(f"Saved temporary file to: {filepath}")

# ========================= Constants ========================= #
# In the Config class, add:
class Config:
    SLEEP = 7

    TICKER_NAME = ""
    SCRIPT_TYPE = "Cryptocurrency"
    SCRIPT_VERSION = get_script_version()
    
    FILE_PATH = ""
    
    # Data paths
    CRYPTO_BASE_PATH = "../../data/crypto"
    STOCK_BASE_PATH = "../../data/stocks"
    
    # Default data configuration (fallback values)
    DATA_PATH = "../../data/crypto/shiba"
    STOCK_DATA_FILENAME = 'Shiba_6_18_2022_10_28_2024.csv'
    
    # Define required column types for validation
    BASE_REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
    CRYPTO_SPECIFIC_COLUMNS = ['marketcap']
    STOCK_SPECIFIC_COLUMNS = ['adj_close']
    OPTIONAL_COLUMNS = ['volume']
    
    USE_RAW_FEATURES_ONLY = False

    # Feature columns configuration
    FEATURE_COLUMNS = []
    
    # Calculate input size based on feature columns
    INPUT_SIZE = 0
    
    USE_WALK_FORWARD_VALIDATION = False
    
    HIDDEN_LAYER_SIZE = 200  # Reduced from 400
    DROPOUT_PROB = 0.4      # Increased from 0.4
    USE_ATTENTION = False
    TIME_STEP = 30
    FUTURE_STEPS = 30       # Reduced from 90 for better accuracy
    BATCH_SIZE = 32         # Reduced from 64
    LEARNING_RATE = 0.001
    EPOCHS = 64
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_SEED = 42
    
    # Validation thresholds
    MAX_SEQUENCE_VALUE = 1.0  # Maximum allowed value in normalized sequences
    MIN_SEQUENCE_VALUE = 0.0  # Minimum allowed value in normalized sequences

    # Add directory paths
    @classmethod
    def get_paths(cls):
        """Get all directory paths based on script name"""
        root_dir = get_script_based_dir()
        return {
            'root': root_dir,
            'visualizations': os.path.join(root_dir, 'visualizations'),
            'charts': os.path.join(root_dir, 'visualizations', 'charts'),
            'models': os.path.join(root_dir, 'models'),
            'logs': os.path.join(root_dir, 'visualizations', 'logs'),
            'data': os.path.join(root_dir, 'data'),
            'temp': os.path.join(root_dir, 'temp')
        }

    @classmethod
    def save_config(cls) -> None:
        """Save current configuration to JSON file in script-named directory"""
        try:
            save_dir = get_script_based_dir()
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            config_file = os.path.join(save_dir, f"training_config_{timestamp}.json")

            config_dict = {
                attr: getattr(cls, attr)
                for attr in dir(cls)
                if not attr.startswith('__') and attr.isupper()
            }

            for key, value in config_dict.items():
                if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    config_dict[key] = str(value)

            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)
            
            logger.info(f"Saved training configuration to: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            
    @classmethod
    def get_available_subdirs(cls, base_path: str) -> List[str]:
        """Get list of available subdirectories"""
        try:
            if os.path.exists(base_path):
                return sorted([d for d in os.listdir(base_path) 
                             if os.path.isdir(os.path.join(base_path, d))])
            return []
        except Exception as e:
            logger.error(f"Error reading directory {base_path}: {str(e)}")
            return []

    @classmethod
    def get_available_files(cls, path: str) -> List[str]:
        """Get list of available CSV files in the directory"""
        try:
            if os.path.exists(path):
                return sorted([f for f in os.listdir(path) 
                             if f.endswith('.csv')])
            return []
        except Exception as e:
            logger.error(f"Error reading directory {path}: {str(e)}")
            return []

    @classmethod
    def select_data_type(cls):
        """Interactive method to select data type and update configuration"""
        print("\nData Type Selection:")
        print("1. Cryptocurrency Data")
        print("2. Stock Market Data")
        
        while True:
            try:
                data_type_choice = int(input("Select data type (1-2): "))
                
                if data_type_choice == 1:
                    cls.SCRIPT_TYPE = "Cryptocurrency"
                    base_path = cls.CRYPTO_BASE_PATH
                elif data_type_choice == 2:
                    cls.SCRIPT_TYPE = "Stock"
                    base_path = cls.STOCK_BASE_PATH
                else:
                    print("Please enter 1 or 2")
                    continue
                
                # Get available subdirectories
                available_dirs = cls.get_available_subdirs(base_path)
                if not available_dirs:
                    logger.error(f"No {cls.SCRIPT_TYPE.lower()} directories found in {base_path}")
                    continue
                
                # Display available directories
                print(f"\nAvailable {cls.SCRIPT_TYPE} directories:")
                for idx, dir_name in enumerate(available_dirs, 1):
                    print(f"{idx}. {dir_name.title()}")
                
                # Let user select directory
                while True:
                    try:
                        dir_choice = int(input(f"Select directory (1-{len(available_dirs)}): "))
                        if 1 <= dir_choice <= len(available_dirs):
                            selected_dir = available_dirs[dir_choice - 1]
                            cls.DATA_PATH = os.path.join(base_path, selected_dir)
                            break
                        print(f"Please enter a number between 1 and {len(available_dirs)}")
                    except ValueError:
                        print("Please enter a valid number")
                
                # Get available files from the selected path
                available_files = cls.get_available_files(cls.DATA_PATH)
                if not available_files:
                    logger.error(f"No CSV files found in {cls.DATA_PATH}")
                    continue
                
                # Display available files
                print(f"\nAvailable {cls.SCRIPT_TYPE} files:")
                for idx, file in enumerate(available_files, 1):
                    print(f"{idx}. {file}")
                
                # Let user select file
                while True:
                    try:
                        file_choice = int(input(f"Select file (1-{len(available_files)}): "))
                        if 1 <= file_choice <= len(available_files):
                            cls.STOCK_DATA_FILENAME = available_files[file_choice - 1]
                            break
                        print(f"Please enter a number between 1 and {len(available_files)}")
                    except ValueError:
                        print("Please enter a valid number")
                break
                
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                logger.error(f"Error: {str(e)}")
        
        logger.info(f"\nSelected Data Type: {cls.SCRIPT_TYPE}")
        logger.info(f"Data Path: {cls.DATA_PATH}")
        logger.info(f"Data File: {cls.STOCK_DATA_FILENAME}\n")
        
        # Add full path log
        full_path = os.path.abspath(os.path.join(cls.DATA_PATH, cls.STOCK_DATA_FILENAME))
        cls.FILE_PATH = full_path
        logger.info(f"Full File Path: {cls.FILE_PATH}\n")
        
        set_ticker_name()
        logger.info(f"TICKER_NAME: {Config.TICKER_NAME}\n")
      
    # Dynamic feature columns (will be populated based on data type)
    @classmethod
    def get_required_columns(cls) -> List[str]:
        """Get required columns based on script type"""
        required = cls.BASE_REQUIRED_COLUMNS.copy()
        if cls.SCRIPT_TYPE.lower() == "cryptocurrency":
            required.extend(cls.CRYPTO_SPECIFIC_COLUMNS)
        else:
            required.extend(cls.STOCK_SPECIFIC_COLUMNS)
        return required

    @classmethod
    def get_all_possible_columns(cls) -> List[str]:
        """Get all possible columns including optional ones"""
        return cls.get_required_columns() + cls.OPTIONAL_COLUMNS
 

# ========================= Configure Logging ========================= #
class LoggerSetup:

    @staticmethod
    def setup_logging(name: Optional[str] = None) -> logging.Logger:
        """
        Set up logging with optional custom logger name
        
        Args:
            name: Optional name for the logger. If None, uses Config.SCRIPT_TYPE
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Get script-based directory and use logs subdirectory
        root_dir = get_script_based_dir()
        log_dir = os.path.join(root_dir, 'visualizations', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger first
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Use provided name or default to SCRIPT_TYPE
        logger_name = name if name else Config.SCRIPT_TYPE
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Only add handlers if they don't exist
        if not root_logger.handlers:
            # File handler - keep detailed logs in file
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f'crypto_prediction_{timestamp}.log')
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(file_formatter)
            
            # Console handler - simplified format
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(message)s')  # Simplified console output
            ch.setFormatter(console_formatter)
            
            root_logger.addHandler(fh)
            root_logger.addHandler(ch)
            
            logger.info(f"Initialized logging for {logger_name}. Log file: {log_file}")
        
        return logger

logger = LoggerSetup.setup_logging()

# =============== Set random seeds for reproducibility =============== #
# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ========================= Data Processing ========================= #
class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_crypto = Config.SCRIPT_TYPE == "Cryptocurrency"
        self.column_mapping = {}  # Will store the detected column mapping
        self.df = None
        self.processing_time = 0
        
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect and map dataset columns to standardized names
        Returns mapping of standardized names to actual column names
        """
        try:
            # Convert all column names to lowercase for case-insensitive matching
            columns_lower = {col.lower().replace(' ', '_'): col for col in df.columns}
            
            # Initialize column mapping
            mapping = {}
            
            # Detect date column with expanded candidates
            date_candidates = ['date', 'time', 'timeopen', 'timestamp', 'timeclose', 'datetime']
            for candidate in date_candidates:
                if candidate in columns_lower:
                    mapping['date'] = columns_lower[candidate]
                    break
            
            if not mapping.get('date'):
                raise ValueError("No valid date column found in dataset")
            
            # Detect price columns (case-insensitive) with validation
            price_variants = {
                'open': ['open', 'opening', 'openingprice', 'timeopen'],
                'high': ['high', 'highest', 'highprice', 'timehigh'],
                'low': ['low', 'lowest', 'lowprice', 'timelow'],
                'close': ['close', 'closing', 'closingprice', 'timeclose']
            }
            
            for std_col, variants in price_variants.items():
                found = False
                for variant in variants:
                    if variant in columns_lower:
                        mapping[std_col] = columns_lower[variant]
                        found = True
                        break
                if not found:
                    raise ValueError(f"Required price column '{std_col}' not found in dataset")
            
            # Detect volume column with expanded variants
            volume_variants = ['volume', 'vol', 'quantity', 'tradingvolume']
            volume_found = False
            for variant in volume_variants:
                if variant in columns_lower:
                    mapping['volume'] = columns_lower[variant]
                    volume_found = True
                    break
            if not volume_found:
                logger.warning("Volume column not found in dataset")
                mapping['volume'] = None
            
            # Detect market cap with variants (including camelCase)
            marketcap_variants = ['marketcap', 'market_cap', 'marketcapitalization', 'marketcap']
            for variant in marketcap_variants:
                if variant in columns_lower:
                    mapping['marketcap'] = columns_lower[variant]
                    break
            
            # Detect optional columns with expanded variants
            optional_variants = {
                'adj_close': ['adjclose', 'adj_close', 'adjustedclose', 'adj close'],
                'vwap': ['vwap', 'volume_weighted_avg_price'],
                'trades': ['trades', 'numberoftrades', 'num_trades']
            }
            
            for std_name, variants in optional_variants.items():
                for variant in variants:
                    if variant in columns_lower:
                        mapping[std_name] = columns_lower[variant]
                        break
            
            # Log detected mapping with improved formatting
            logger.info("\nDetected column mapping:")
            logger.info("-" * 40)
            for category, cols in {
                "Required Columns": ['date', 'open', 'high', 'low', 'close'],
                "Volume Column": ['volume'],
                "Optional Columns": ['marketcap', 'adj_close', 'vwap', 'trades']
            }.items():
                logger.info(f"\n{category}:")
                for col in cols:
                    if col in mapping:
                        logger.info(f"  {col:<12}: {mapping[col]}")
                    else:
                        logger.info(f"  {col:<12}: Not found")
            logger.info("-" * 40)
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error in column detection: {str(e)}")
            logger.error("Available columns: {}".format(", ".join(df.columns)))
            raise

    def _convert_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Convert specified columns to numeric type"""
        try:
            for col in columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error converting numeric columns: {str(e)}")
            raise
    
    def _print_dataframe_info(self, df: pd.DataFrame, title: str = "Sample of loaded data:"):
        """Helper method to print DataFrame information with borders"""
        try:
            border = "+" + "-" * 100 + "+"
            logger.info(f"\n{title}")
            logger.info(border)
            
            sample_data = df.head().to_string().split('\n')
            for line in sample_data:
                logger.info(f"|{line:<100}|")
                
            logger.info(border)
        except Exception as e:
            logger.error(f"Error printing DataFrame info: {str(e)}")

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate the processed dataset before model training"""
        try:
            # Check for minimum required data points
            min_required = Config.TIME_STEP + Config.FUTURE_STEPS
            if len(df) < min_required:
                raise ValueError(f"Dataset too small. Need at least {min_required} points")
                
            # Check for infinite values
            if np.any(np.isinf(df.values)):
                raise ValueError("Dataset contains infinite values")
                
            # Check feature ranges
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    if df[col].max() > Config.MAX_SEQUENCE_VALUE or df[col].min() < Config.MIN_SEQUENCE_VALUE:
                        logger.warning(f"Feature {col} has values outside expected range")
            
            # Check for high correlations
            correlations = df.corr()
            high_corr = np.where(np.abs(correlations) > 0.95)
            high_corr = [(correlations.index[x], correlations.columns[y], correlations.iloc[x, y])
                        for x, y in zip(*high_corr) if x != y]
            if high_corr:
                logger.warning("High correlations detected between features:")
                for feat1, feat2, corr in high_corr:
                    logger.warning(f"{feat1} - {feat2}: {corr:.3f}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.warning("Missing values detected in dataset:")
                for col in missing_values[missing_values > 0].index:
                    logger.warning(f"{col}: {missing_values[col]} missing values")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False

    def process_data(self, feature_columns: List[str], time_step: int = Config.TIME_STEP) -> Tuple[np.ndarray, np.ndarray]:
        """Process data through the entire pipeline with validation"""
        try:
            # Load data
            df = self.load_data()
            
            # Engineer features
            df = self.feature_engineering(df)
            
            # Validate dataset
            if not self.validate_dataset(df):
                raise ValueError("Dataset validation failed")
            
            # Create sequences
            return self.create_sequences(df, feature_columns, time_step)
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the data from CSV file"""
        try:
            # Try both separators with better error handling
            for separator in [',', ';']:
                try:
                    self.df = pd.read_csv(self.file_path, sep=separator)
                    if len(self.df.columns) > 1:  # Valid separation
                        logger.info(f"Successfully loaded data with '{separator}' separator")
                        break
                except Exception as e:
                    logger.debug(f"Failed to load with separator '{separator}': {str(e)}")
                    continue
            
            if self.df is None or len(self.df.columns) <= 1:
                raise ValueError("Failed to load data with any separator")

            logger.info(f"Available columns in dataset: {self.df.columns.tolist()}")
            
            # Detect and map columns
            self.column_mapping = self._detect_columns(self.df)
            
            # Handle date column first (special case for crypto data)
            date_col = self.column_mapping.get('date')
            if not date_col:
                raise ValueError("No date column mapping found")
            
            try:
                if self.is_crypto and 'timestamp' in self.df.columns:
                    # For crypto data, prefer timestamp if available
                    self.df['date'] = pd.to_datetime(self.df['timestamp'])
                else:
                    self.df['date'] = pd.to_datetime(self.df[date_col])
                
                # Set date as index and sort
                self.df.set_index('date', inplace=True)
                self.df.sort_index(inplace=True)
            except Exception as e:
                logger.error(f"Error processing date column '{date_col}': {str(e)}")
                logger.error(f"Date sample values: {self.df[date_col].head()}")
                raise

            # Rename remaining columns to standardized names
            inverse_mapping = {v: k for k, v in self.column_mapping.items() 
                            if v is not None and v != date_col}
            if not inverse_mapping:
                raise ValueError("No valid column mappings detected")
            self.df.rename(columns=inverse_mapping, inplace=True)
            
            # Handle adjusted close if available (mainly for stock data)
            if 'adj_close' in self.df.columns and hasattr(Config, 'USE_ADJ_CLOSE') and Config.USE_ADJ_CLOSE:
                logger.info("Using adjusted close prices")
                self.df['close'] = self.df['adj_close']
            
            # Convert price columns to numeric with validation
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'marketcap']
            for col in numeric_columns:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Error converting {col} to numeric: {str(e)}")
            
            # Handle missing and invalid values with format-specific approach
            if 'volume' in self.df.columns:
                self.df['volume'] = self.df['volume'].replace([0, float('inf'), float('-inf')], np.nan)
                if self.is_crypto:
                    # Crypto specific: handle very large volume numbers
                    self.df['volume'] = self.df['volume'].replace(1e20, np.nan)
            
            # Fill missing values with more sophisticated approach
            for col in self.df.columns:
                if self.df[col].isnull().any():
                    if col in ['volume', 'marketcap']:
                        # Use rolling mean for volume and marketcap
                        self.df[col] = self.df[col].fillna(self.df[col].rolling(window=5, min_periods=1).mean())
                    else:
                        # Use forward fill then backward fill for price data
                        self.df[col] = self.df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Select and validate required columns using new Config method
            required_columns = Config.get_required_columns()
            available_columns = [col for col in required_columns if col in self.df.columns]
            if len(available_columns) != len(required_columns):
                missing = set(required_columns) - set(available_columns)
                raise ValueError(f"Missing required price columns: {missing}")
            
            # Add optional columns if available
            for col in Config.OPTIONAL_COLUMNS:
                if col in self.df.columns:
                    available_columns.append(col)
            
            # Select final columns and validate data
            self.df = self.df[available_columns]
            if self.df.empty:
                raise ValueError("Processed DataFrame is empty")
            
            # Remove any remaining invalid data
            self.df = self.df[~self.df.isin([np.inf, -np.inf]).any(axis=1)]
            self.df = self.df.dropna()
            
            # Log data information
            logger.info(f"\nLoaded {Config.SCRIPT_TYPE} data:")
            logger.info(f"Shape: {self.df.shape}")
            logger.info(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
            logger.info(f"Final columns: {self.df.columns.tolist()}")
            
            # Print sample of loaded data
            self._print_dataframe_info(self.df)
            
            return self.df
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(f"File path: {self.file_path}")
            if hasattr(self, 'df') and self.df is not None:
                logger.error(f"DataFrame head:\n{self.df.head()}")
            raise

    def log_feature_statistics(self, df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> None:
        """Log detailed statistics about features"""
        try:
            # Print header
            logger.info("\n" + "="*100)
            logger.info(f"{'Feature Analysis Summary':^100}")
            logger.info("="*100)

            # Original data features
            original_data_features = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
            logger.info("\nOriginal Data Features:")
            logger.info("-"*100)
            for i in range(0, len(original_data_features), 4):
                chunk = original_data_features[i:i+4]
                logger.info("    " + " | ".join(f"{feat:<20}" for feat in chunk))

            # All current features
            engineered_features = [col for col in df.columns if col not in original_data_features]
            logger.info(f"\nEngineered Features ({len(engineered_features)}):")
            logger.info("-"*100)
            for i in range(0, len(engineered_features), 4):
                chunk = engineered_features[i:i+4]
                logger.info("    " + " | ".join(f"{feat:<20}" for feat in chunk))
            
            # Price statistics
            logger.info("\n" + "-"*100)
            logger.info("Price Features (8):")
            logger.info("-"*100)
            price_features = ['open', 'high', 'low', 'close',
                            'open_norm', 'high_norm', 'low_norm', 'close_norm']
            for col in price_features:
                if col in df.columns:
                    if original_df is not None and col in original_df.columns:
                        orig_mean = original_df[col].mean()
                        orig_std = original_df[col].std()
                        logger.info(f"{col:.<30} mean: {orig_mean:>12.8f}, std: {orig_std:>12.8f}")
                    stats = df[col].describe()
                    prefix = "(Original)" if col in original_data_features else "(Engineered)"
                    logger.info(f"{col:.<30} mean: {stats['mean']:>12.8f}, std: {stats['std']:>12.8f}    {prefix:>12}")

            # Feature groups statistics
            feature_groups = {
                'Volume Features (5)': [
                    'volume', 'volume_norm', 'Volume_ROC', 'Volume_MA_5', 'Volume_MA_20'
                ],
                'Momentum Features (8)': [
                    'ROC_5', 'ROC_20', 'ROC_7', 'Momentum_7', 
                    'ROC_14', 'Momentum_14', 'ROC_21', 'Momentum_21'
                ],
                'Technical Indicators (7)': [
                    'MACD', 'Signal_Line', 'MACD_Histogram', 
                    'RSI', 'EMA_12', 'EMA_26', 'Volatility'
                ],
                'Volatility Features (3)': [
                    'Volatility', 'Volatility_5', 'Volatility_20'
                ],
                'Pattern Features (5)': [
                    'Body', 'Upper_Shadow', 'Lower_Shadow', 
                    'Candle_Pattern', 'Pattern_Strength'
                ],
                'Decomposition Features': [
                    'Trend', 'Seasonal', 'Residual'
                ]
            }

            # Log statistics for each feature group
            for group_name, features in feature_groups.items():
                logger.info("\n" + "-"*100)
                logger.info(f"{group_name}:")
                logger.info("-"*100)
                available_features = [f for f in features if f in df.columns]
                for feature in available_features:
                    if df[feature].dtype in [np.float64, np.int64]:
                        stats = df[feature].describe()
                        prefix = "(Original)" if feature in original_data_features else "(Engineered)"
                        logger.info(f"{feature:.<30} mean: {stats['mean']:>12.4f}, std: {stats['std']:>12.4f}    {prefix:>12}")
                    else:
                        unique_count = df[feature].nunique()
                        prefix = "(Original)" if feature in original_data_features else "(Engineered)"
                        logger.info(f"{feature:.<30} categorical with {unique_count:>6} unique values    {prefix:>12}")

            # Additional Features
            logger.info("\n" + "-"*100)
            logger.info("Additional Features:")
            logger.info("-"*100)
            all_listed_features = sum([features for features in feature_groups.values()], []) + price_features
            other_features = [col for col in df.columns if col not in all_listed_features]
            for feature in other_features:
                if df[feature].dtype in [np.float64, np.int64]:
                    stats = df[feature].describe()
                    prefix = "(Original)" if feature in original_data_features else "(Engineered)"
                    logger.info(f"{feature:.<30} mean: {stats['mean']:>12.4f}, std: {stats['std']:>12.4f}    {prefix:>12}")
                else:
                    unique_count = df[feature].nunique()
                    prefix = "(Original)" if feature in original_data_features else "(Engineered)"
                    logger.info(f"{feature:.<30} categorical with {unique_count:>6} unique values    {prefix:>12}")

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.info("\n" + "-"*100)
                logger.warning("Missing Values Detected:")
                logger.info("-"*100)
                for col in missing_values[missing_values > 0].index:
                    logger.warning(f"{col}: {missing_values[col]} missing values")
            else:
                logger.info("\n" + "-"*100)
                logger.info("Validation Summary:")
                logger.info("-"*100)
                logger.info("All features validated successfully")

            # Log processing summary
            logger.info("\n" + "="*100)
            logger.info(f"Total Features: {len(df.columns)} ({len(original_data_features)} original + {len(engineered_features)} engineered)")
            if hasattr(self, 'processing_time'):
                logger.info(f"Feature engineering completed in {self.processing_time:.2f} seconds")
            logger.info("="*100 + "\n")

        except Exception as e:
            logger.error(f"Error logging feature statistics: {str(e)}")
            logger.error("Stack trace:", exc_info=True)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features based on available columns"""
        try:
            # Store start time
            start_time = time.time()
            
            # Store original DataFrame for statistics
            original_df = df.copy()
            
            # Get available columns
            available_columns = df.columns.tolist()
            
            # Basic price features (require 'close' column)
            if 'close' in available_columns:
                # Lag Features (normalized)
                df['Lag_1'] = df['close'].pct_change(1)
                df['Lag_2'] = df['close'].pct_change(2)
                df['Lag_5'] = df['close'].pct_change(5)
                
                # Rolling Statistics (normalized)
                df['Rolling_Mean_5'] = df['close'].rolling(window=5).mean() / df['close'] - 1
                df['Rolling_Std_5'] = df['close'].rolling(window=5).std() / df['close']
                
                # Volatility
                df['Volatility'] = df['close'].pct_change().rolling(window=5).std()
                df['Volatility_5'] = df['close'].pct_change().rolling(window=5).std()
                df['Volatility_20'] = df['close'].pct_change().rolling(window=20).std()
                
                # MACD (normalized)
                df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean() / df['close'] - 1
                df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean() / df['close'] - 1
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands (normalized)
                rolling_mean = df['close'].rolling(window=5).mean()
                rolling_std = df['close'].rolling(window=5).std()
                df['Bollinger_Upper'] = (rolling_mean + 2 * rolling_std) / df['close'] - 1
                df['Bollinger_Lower'] = (rolling_mean - 2 * rolling_std) / df['close'] - 1
                
                # STL Decomposition (normalized)
                try:
                    stl = STL(df['close'], period=30)
                    result = stl.fit()
                    df['Trend'] = result.trend / df['close'] - 1
                    df['Seasonal'] = result.seasonal / df['close']
                    df['Residual'] = result.resid / df['close']
                except Exception as e:
                    logger.warning(f"STL Decomposition failed: {str(e)}. Skipping these features.")
                
                # Momentum Features
                df['ROC_5'] = df['close'].pct_change(5)
                df['ROC_20'] = df['close'].pct_change(20)
                df['ROC_7'] = df['close'].pct_change(7)
                df['Momentum_7'] = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
                df['ROC_14'] = df['close'].pct_change(14)
                df['Momentum_14'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
                df['ROC_21'] = df['close'].pct_change(21)
                df['Momentum_21'] = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
            else:
                logger.error("'close' column not found. Cannot calculate price-based features.")
                raise ValueError("Missing required 'close' column")
                
            # Volume Features (if volume column exists)
            if 'volume' in available_columns:
                df['volume_norm'] = df['volume'].pct_change()
                df['Volume_ROC'] = df['volume'].pct_change(5)
                df['Volume_MA_5'] = df['volume'].rolling(window=5).mean() / df['volume'] - 1
                df['Volume_MA_20'] = df['volume'].rolling(window=20).mean() / df['volume'] - 1
            else:
                logger.warning("'volume' column not found. Skipping volume-based features.")
                
            # Pattern Features (if OHLC columns exist)
            required_pattern_cols = ['open', 'high', 'low', 'close']
            if all(col in available_columns for col in required_pattern_cols):
                df['Body'] = (df['close'] - df['open']) / df['open']
                df['Upper_Shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
                df['Lower_Shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
                
                # Normalize price features
                for col in required_pattern_cols:
                    df[f'{col}_norm'] = df[col].pct_change()
            else:
                logger.warning("Not all OHLC columns found. Skipping pattern-based features.")
                missing_cols = [col for col in required_pattern_cols if col not in available_columns]
                logger.warning(f"Missing columns: {missing_cols}")
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            # Store processing time
            self.processing_time = time.time() - start_time
            
            # Log feature statistics after engineering
            self.log_feature_statistics(df, original_df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            logger.error(f"Available columns: {available_columns}")
            raise
    
    def create_sequences(self, 
                        df: pd.DataFrame, 
                        feature_columns: List[str],
                        time_step: int = Config.TIME_STEP) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        try:
            # Validate feature columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required feature columns: {missing_cols}")
                
            # Validate dataset before sequence creation
            if not self.validate_dataset(df):
                raise ValueError("Dataset validation failed before sequence creation")
                
            features = df[feature_columns].values
            target = df[['close']].values
            
            # Scale features and target
            scaled_features = self.scaler_X.fit_transform(features)
            scaled_target = self.scaler_y.fit_transform(target)
            
            X, y = [], []
            for i in range(len(scaled_features) - time_step):
                X.append(scaled_features[i:(i + time_step)])
                y.append(scaled_target[i + time_step])
            
            # Validate sequence shapes
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No sequences created. Check time_step and data length.")
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
 

# ========================= Basic Stock Model ========================= #
class BasicStockModel:
    """Basic LSTM model for stock price prediction with visualization capabilities"""
    
    BASE_RAW_FEATURES = []

    STOCK_RAW_FEATURES = ['Close', 'Volume', 'High', 'Low', 'Open', 'Adj Close']
    CRYPTO_RAW_FEATURES = ['Close', 'Volume', 'High', 'Low', 'Open', 'Marketcap']

    # Base features common to both stock and crypto
    BASE_FEATURES = ['Close', 'Volume', 'High', 'Low', 'Lag_1', 'Lag_2', 'Lag_5',
                    'Rolling_Mean_5', 'Rolling_Std_5', 'Volatility', 'MACD',
                    'Signal_Line', 'MACD_Histogram', '%K', '%D', 'Bollinger_Upper',
                    'Bollinger_Lower', 'RSI', 'Trend', 'Seasonal', 'Residual']
    
    # Crypto-specific features
    CRYPTO_FEATURES = ['Market_Cap', 'Market_Cap_MA', 'Market_Cap_Std', 
                      'Market_Cap_Change', 'Market_Cap_ROC', 'Price_to_MC_Ratio',
                      'Market_Cap_RSI']
    
    def __init__(self, config: Config, epochs: Optional[int] = None, batch_size: Optional[int] = None, learning_rate: Optional[float] = None):
        """Initialize BasicStockModel with configuration and optional parameters
        
        Args:
            config (Config): Configuration object with model parameters
            epochs (Optional[int]): Number of training epochs, overrides config value
            batch_size (Optional[int]): Batch size for training, overrides config value
            learning_rate (Optional[float]): Learning rate for optimization, overrides config value
        """
        try:
            self._data_processed = False
            
            # Store configuration
            self.config = config
            
            #TRAIN_TEST_SPLIT
            self.traine_test_split = config.TRAIN_TEST_SPLIT

            self.use_raw_features_only = config.USE_RAW_FEATURES_ONLY

            # Initialize data type flag
            self.is_crypto = False
            
            # Setup logging
            paths = self.config.get_paths()
            logs_dir = os.path.join(paths['logs'], 'BasicStockModel')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create logger with appropriate name
            self.logger = logging.getLogger(f"{Config.SCRIPT_TYPE}.BasicStock.{config.TICKER_NAME}")
            self.logger.setLevel(logging.DEBUG)
            
            # Create file handler with timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{self.config.TICKER_NAME}_model_{timestamp}.log"
            file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)
            
            # Log initialization
            self.logger.info(f"Initializing BasicStockModel for {config.TICKER_NAME}")
            self.logger.info(f"Model version: {config.SCRIPT_VERSION}")
            
            # Initialize model components
            self.model = None
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Setup directories
            self.charts_dir = os.path.join(paths['charts'], 'BasicStockModel')
            os.makedirs(self.charts_dir, exist_ok=True)
            self.logger.info(f"Charts directory: {self.charts_dir}")
            
            self.features = None

            # Model architecture parameters
            self.input_size = self.get_input_size(self.is_crypto)  # Initial size with base features
            self.hidden_layer_size = config.HIDDEN_LAYER_SIZE
            self.dropout_prob = config.DROPOUT_PROB
            self.use_attention = config.USE_ATTENTION
            self.time_step = config.TIME_STEP
            
            # Training parameters with fallback logic
            # Epochs
            self.epochs = epochs or getattr(config, 'EPOCHS', 150)
            if self.epochs <= 0:
                self.epochs = 150
                self.logger.warning("Invalid epochs value, using default: 150")
                
            # Batch size
            self.batch_size = batch_size or getattr(config, 'BATCH_SIZE', 64)
            if self.batch_size <= 0:
                self.batch_size = 64
                self.logger.warning("Invalid batch_size value, using default: 64")
                
            # Learning rate
            self.lr = learning_rate or getattr(config, 'LEARNING_RATE', 0.001)
            if self.lr <= 0:
                self.lr = 0.001
                self.logger.warning("Invalid learning_rate value, using default: 0.001")
            
            # Log model parameters
            self.logger.info("Model parameters:")
            self.logger.info(f"Initial input size: {self.input_size}")
            self.logger.info(f"Hidden layer size: {self.hidden_layer_size}")
            self.logger.info(f"Dropout probability: {self.dropout_prob}")
            self.logger.info(f"Use attention: {self.use_attention}")
            self.logger.info(f"Time step: {self.time_step}")
            self.logger.info(f"Epochs: {self.epochs}")
            self.logger.info(f"Batch size: {self.batch_size}")
            self.logger.info(f"Learning rate: {self.lr}")
            
            # Initialize model
            self.initialize_model()
            
            # Data attributes
            self.df = None
            self.scaled_data = None
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            self.train_loader = None
            self.train_size = None
            self.test_index = None
            
            # Predictions
            self.train_predictions = None
            self.test_predictions = None
            self.train_predictions_rescaled = None
            self.test_predictions_rescaled = None
            self.train_prices_rescaled = None
            self.test_prices_rescaled = None
            
            self.logger.info("BasicStockModel initialization completed successfully")
            
        except Exception as e:
            # Ensure logging of initialization errors
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during BasicStockModel initialization: {str(e)}")
                self.logger.exception("Stack trace:")
            else:
                print(f"Error during BasicStockModel initialization: {str(e)}")
            raise
        
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_dim: int, dropout_prob: float = 0.2):
            super().__init__()
            self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
            self.dropout = nn.Dropout(dropout_prob)

        def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
            attn_scores = self.attention_weights(lstm_output)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context_vector = torch.sum(attn_weights * lstm_output, dim=1)
            return self.dropout(context_vector)

    class LSTMModel(nn.Module):
        def __init__(self, input_size: int, hidden_layer_size: int = 50, 
                     output_size: int = 1, dropout_prob: float = 0.2, 
                     use_attention: bool = False):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
            self.dropout = nn.Dropout(dropout_prob)
            self.use_attention = use_attention
            
            # Initialize attention as None first
            self.attention = None
            if use_attention:
                self.attention = BasicStockModel.AttentionLayer(
                    hidden_dim=hidden_layer_size, 
                    dropout_prob=dropout_prob
                )
            
            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)

            if self.use_attention:
                lstm_out = self.attention(lstm_out)
            else:
                lstm_out = lstm_out[:, -1]

            return self.linear(lstm_out)
        
        def update_attention(self, use_attention: bool):
            """Update attention layer based on configuration"""
            self.use_attention = use_attention
            if use_attention and self.attention is None:
                self.attention = BasicStockModel.AttentionLayer(
                    hidden_dim=self.hidden_layer_size, 
                    dropout_prob=self.dropout.p
                )
            elif not use_attention:
                self.attention = None
        
    def _setup_logging(self) -> None:
        """Setup logging configuration for BasicStockModel"""
        try:
            # Use existing Config paths
            paths = self.config.get_paths()
            logs_dir = os.path.join(paths['logs'], 'BasicStockModel')
            os.makedirs(logs_dir, exist_ok=True)

            # Generate log filename with timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{self.config.TICKER_NAME}_model_{timestamp}.log"
            log_filepath = os.path.join(logs_dir, log_filename)

            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Create file handler
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)

            # Create logger
            self.logger = logging.getLogger(f'BasicStockModel_{self.config.TICKER_NAME}')
            self.logger.setLevel(logging.DEBUG)
            
            # Remove any existing handlers
            if self.logger.hasHandlers():
                self.logger.handlers.clear()
                
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            # Log initial information
            self.logger.info(f"Initialized BasicStockModel for {self.config.TICKER_NAME}")
            self.logger.info(f"Log file created at: {log_filepath}")
            self.logger.info(f"Model version: {self.config.SCRIPT_VERSION}")
            
            # Log configuration settings
            self.logger.debug("Configuration settings:")
            for key, value in vars(self.config).items():
                if key.isupper():  # Only log uppercase attributes (constants)
                    self.logger.debug(f"{key}: {value}")

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise
        
    @classmethod
    def get_input_size(cls, is_crypto: bool = False) -> int:
        """
        Get the input size based on data type
        
        Args:
            is_crypto (bool): Whether to include crypto-specific features
            
        Returns:
            int: Number of input features
        """
        features = cls.BASE_FEATURES.copy()
        if is_crypto:
            features.extend(cls.CRYPTO_FEATURES)
        return len(features)
        
    def _detect_data_format(self, filepath: str) -> bool:
        """Detect if the data is cryptocurrency format by checking headers"""
        try:
            # Try reading first line with different delimiters
            with open(filepath, 'r') as f:
                header = f.readline().strip()
            
            # Check if crypto-specific columns are present
            crypto_columns = {'timeOpen', 'timeClose', 'marketCap'}
            return any(col in header for col in crypto_columns)
        except Exception as e:
            self.logger.warning(f"Error detecting data format: {str(e)}")
            return False
        
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names between crypto and stock data"""
        if self.is_crypto:
            # Crypto data mapping
            column_map = {
                'timeOpen': 'Date',
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'marketCap': 'Market_Cap'
            }
        else:
            # Stock data mapping
            column_map = {
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Adj Close': 'Adj_Close',
                'Volume': 'Volume'
            }
        
        # Rename columns (case-insensitive)
        for old_col, new_col in column_map.items():
            matching_cols = [col for col in df.columns if col.lower() == old_col.lower()]
            if matching_cols:
                df = df.rename(columns={matching_cols[0]: new_col})
        
        return df
        
    def _get_model_path(self) -> str:
        """
        Get the path for saving/loading model weights with model parameters in filename
        
        Returns:
            str: Complete path for model file
        """
        try:
            # Get base names
            asset_name = self.config.STOCK_DATA_FILENAME.split('_')[0]
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            
            # Create path including BasicStockModel directory
            model_dir = os.path.join(
                get_script_based_dir(),  # Get root dir
                'models',                # Use models directory
                'BasicStockModel'        # BasicStockModel subdirectory
            )
            
            # Ensure directory exists
            os.makedirs(model_dir, exist_ok=True)
            
            # Create filename with model parameters
            filename = (
                f"{script_name}_"
                f"{asset_name}_"
                f"{'crypto' if self.is_crypto else 'stock'}_"  # Add data type
                f"h.{self.hidden_layer_size}_"
                f"d.{str(self.dropout_prob).replace('.', '')}_"
                f"att.{int(self.use_attention)}_"
                f"t.{self.time_step}_"
                f"f.{len(self.features)}_"  # Add feature count
                f"{'raw' if self.config.USE_RAW_FEATURES_ONLY else 'full'}_"  # Indicate feature type
                f"v.{self.config.SCRIPT_VERSION}_"  # Add version
                f"basic_model.pth"
            )
            
            # Get complete path
            model_path = os.path.join(model_dir, filename)
            
            # Log path creation
            self.logger.debug(f"Generated model path: {model_path}")
            
            return model_path
            
        except Exception as e:
            error_msg = f"Error generating model path: {str(e)}"
            self.logger.error(error_msg)
            raise
   
    def _add_technical_indicators(self) -> None:
        """Add technical indicators to the dataframe"""
        try:
            # Basic price indicators
            self.df['Lag_1'] = self.df['Close'].shift(1)
            self.df['Lag_2'] = self.df['Close'].shift(2)
            self.df['Lag_5'] = self.df['Close'].shift(5)
            
            # Moving averages and volatility
            self.df['Rolling_Mean_5'] = self.df['Close'].rolling(window=5).mean()
            self.df['Rolling_Std_5'] = self.df['Close'].rolling(window=5).std()
            self.df['Volatility'] = self.df['Close'].pct_change().rolling(window=5).std()
            
            # MACD
            self.df['EMA_12'] = self.df['Close'].ewm(span=12, adjust=False).mean()
            self.df['EMA_26'] = self.df['Close'].ewm(span=26, adjust=False).mean()
            self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
            self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
            self.df['MACD_Histogram'] = self.df['MACD'] - self.df['Signal_Line']
            
            # Stochastic Oscillator
            low_14 = self.df['Low'].rolling(window=14).min()
            high_14 = self.df['High'].rolling(window=14).max()
            self.df['%K'] = 100 * ((self.df['Close'] - low_14) / (high_14 - low_14))
            self.df['%D'] = self.df['%K'].rolling(window=3).mean()
            
            # Bollinger Bands
            self.df['Bollinger_Upper'] = self.df['Rolling_Mean_5'] + 2 * self.df['Rolling_Std_5']
            self.df['Bollinger_Lower'] = self.df['Rolling_Mean_5'] - 2 * self.df['Rolling_Std_5']
            
            def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
                """
                Calculate RSI using Wilder's method properly
                """
                # Calculate price changes
                delta = prices.diff()
                
                # Split gains and losses
                gains = delta.copy()
                losses = delta.copy()
                gains[gains < 0] = 0.0
                losses[losses > 0] = 0.0
                losses = abs(losses)
                
                # Calculate initial SMA values
                first_avg_gain = gains.iloc[:period].mean()
                first_avg_loss = losses.iloc[:period].mean()
                
                # Initialize series
                avg_gain = pd.Series(index=prices.index, dtype=float)
                avg_loss = pd.Series(index=prices.index, dtype=float)
                
                # Set first values
                avg_gain.iloc[period - 1] = first_avg_gain
                avg_loss.iloc[period - 1] = first_avg_loss
                
                # Calculate subsequent values using Wilder's smoothing
                for i in range(period, len(prices)):
                    avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gains.iloc[i]) / period
                    avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + losses.iloc[i]) / period
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                return rsi

            # Apply RSI calculation
            self.df['RSI'] = calculate_rsi(self.df['Close'])
            
            # STL Decomposition
            stl = STL(self.df['Close'], period=365)
            result = stl.fit()
            self.df['Trend'] = result.trend
            self.df['Seasonal'] = result.seasonal
            self.df['Residual'] = result.resid
            
            # Crypto-specific indicators (if available)
            if self.is_crypto and 'Market_Cap' in self.df.columns:
                # Market Cap indicators
                self.df['Market_Cap_MA'] = self.df['Market_Cap'].rolling(window=5).mean()
                self.df['Market_Cap_Std'] = self.df['Market_Cap'].rolling(window=5).std()
                self.df['Market_Cap_Change'] = self.df['Market_Cap'].pct_change()
                
                # Market Cap momentum
                self.df['Market_Cap_ROC'] = self.df['Market_Cap'].pct_change(5)  # Rate of Change
                
                # Price to Market Cap ratio
                self.df['Price_to_MC_Ratio'] = self.df['Close'] / self.df['Market_Cap']
                
                # Market Cap RSI
                mc_delta = self.df['Market_Cap'].diff(1)
                mc_gain = mc_delta.where(mc_delta > 0, 0)
                mc_loss = -mc_delta.where(mc_delta < 0, 0)
                mc_avg_gain = mc_gain.rolling(window=14).mean()
                mc_avg_loss = mc_loss.rolling(window=14).mean()
                mc_rs = mc_avg_gain / mc_avg_loss
                self.df['Market_Cap_RSI'] = 100 - (100 / (1 + mc_rs))
            
            # Drop NaN values
            self.df.dropna(inplace=True)
            
            # Log the number of indicators added
            n_indicators = len(self.df.columns) - (6 if not self.is_crypto else 7)  # Basic columns + indicators
            self.logger.info(f"Added {n_indicators} technical indicators to the dataset")
            
        except Exception as e:
            error_msg = f"Error adding technical indicators: {str(e)}"
            self.logger.error(error_msg)
            raise
        
    def _prepare_features(self) -> None:
        """Prepare and scale features"""
        try:
            # Validate features exist in DataFrame
            missing_features = [f for f in self.FEATURES if f not in self.df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Extract features and store as numpy array
            self.features = self.df[self.FEATURES].values
            self.logger.info(f"Features array shape: {self.features.shape}")
            
            # Initialize scaler if not already done
            if not hasattr(self, 'scaler'):
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                
            # Scale features
            self.scaled_data = self.scaler.fit_transform(self.features)
            
            self.logger.info(f"Features prepared and scaled. Shape: {self.scaled_data.shape}")
            self.logger.info(f"Using features: {self.FEATURES}")
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            self.logger.exception("Stack trace:")
            raise

    def _create_sequences(self) -> None:
        """Create sequences for training"""
        try:
            self.logger.info("\nCreating sequences for training:")
            self.logger.info(f"Initial scaled data shape: {self.scaled_data.shape}")
            self.logger.info(f"Time step: {self.time_step}")
            
            X, y = [], []
            sequence_length = len(self.scaled_data) - self.time_step
            
            # Create progress bar for sequence creation
            for i in tqdm(range(sequence_length), desc="Creating sequences"):
                try:
                    sequence = self.scaled_data[i:i + self.time_step]
                    target = self.scaled_data[i + self.time_step, 0]
                    
                    # Validate sequence
                    if sequence.shape != (self.time_step, len(self.FEATURES)):
                        raise ValueError(f"Invalid sequence shape at index {i}: {sequence.shape}")
                    
                    X.append(sequence)
                    y.append(target)
                except Exception as e:
                    self.logger.error(f"Error creating sequence at index {i}: {str(e)}")
                    raise
            
            # Convert to numpy arrays
            try:
                X, y = np.array(X), np.array(y)
                self.logger.info(f"Sequence shapes - X: {X.shape}, y: {y.shape}")
            except Exception as e:
                self.logger.error(f"Error converting sequences to numpy arrays: {str(e)}")
                raise
            
            # Train-test split
            try:
                self.train_size = int(len(X) * self.traine_test_split)
                self.logger.info(f"Train-test split ratio: {self.traine_test_split}")
                self.logger.info(f"Train size: {self.train_size}, Test size: {len(X) - self.train_size}")
                
                self.X_train, self.X_test = X[:self.train_size], X[self.train_size:]
                self.y_train, self.y_test = y[:self.train_size], y[self.train_size:]
                
                self.logger.info(f"Training set shapes - X: {self.X_train.shape}, y: {self.y_train.shape}")
                self.logger.info(f"Test set shapes - X: {self.X_test.shape}, y: {self.y_test.shape}")
            except Exception as e:
                self.logger.error(f"Error performing train-test split: {str(e)}")
                raise
            
            # Convert to tensors
            try:
                self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
                self.y_train = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1)
                self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
                self.y_test = torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1)
                
                self.logger.info("\nTensor shapes:")
                self.logger.info(f"X_train: {self.X_train.shape}")
                self.logger.info(f"y_train: {self.y_train.shape}")
                self.logger.info(f"X_test: {self.X_test.shape}")
                self.logger.info(f"y_test: {self.y_test.shape}")
            except Exception as e:
                self.logger.error(f"Error converting to PyTorch tensors: {str(e)}")
                raise
            
            # Create DataLoader
            try:
                self.train_loader = torch.utils.data.DataLoader(
                    list(zip(self.X_train, self.y_train)), 
                    batch_size=self.batch_size, 
                    shuffle=True
                )
                self.logger.info(f"\nDataLoader created with batch size: {self.batch_size}")
                self.logger.info(f"Number of batches: {len(self.train_loader)}")
            except Exception as e:
                self.logger.error(f"Error creating DataLoader: {str(e)}")
                raise
            
            # Set input size for model
            try:
                self.input_size = self.X_train.shape[2]
                self.logger.info(f"\nFinal input size set to: {self.input_size}")
            except Exception as e:
                self.logger.error(f"Error setting input size: {str(e)}")
                raise
            
            self.logger.info("Sequence creation completed successfully")
            
        except Exception as e:
            self.logger.error("Error in sequence creation process")
            self.logger.exception("Stack trace:")
            raise
        
    def _select_features(self) -> None:
        """Select and prepare features based on configuration"""
        try:
            # Debug print current DataFrame state
            print("\nDebug: DataFrame State")
            print(f"Current columns: {list(self.df.columns)}")
            print(f"DataFrame head:\n{self.df.head()}")
            
            # Log all available feature sets at the start
            self.logger.info("\nAvailable Feature Sets:")
            self.logger.info(f"Base Raw Features: {self.BASE_RAW_FEATURES}")
            self.logger.info(f"Technical Features: {self.BASE_FEATURES}")
            self.logger.info(f"Crypto Features: {self.CRYPTO_FEATURES}\n")
            
            if self.config.USE_RAW_FEATURES_ONLY:
                # Use only raw features without adding any indicators
                self.features = self.BASE_RAW_FEATURES.copy()
                self.logger.info("\nUsing raw features only")
            else:
                # Add technical indicators
                self._add_technical_indicators()
                
                # Initialize with raw features
                self.features = self.BASE_RAW_FEATURES.copy()
                
                # Add technical features
                self.features.extend(self.BASE_FEATURES)
                
                # Add crypto features if they were calculated
                if self.is_crypto and 'Market_Cap' in self.df.columns:
                    self.features.extend(self.CRYPTO_FEATURES)

                self.logger.info("\nUsing full feature set with technical indicators")
            
            # Update input size
            self.input_size = len(self.features)
            
            self.FEATURES = self.features

            # Check for missing features
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                print("\nDebug: Feature Matching")
                print(f"Looking for features: {self.features}")
                print(f"Available columns: {list(self.df.columns)}")
                print(f"Missing features: {missing_features}")
                raise ValueError(f"Missing features in dataframe: {missing_features}")

            # Log detailed feature selection
            self.logger.info("\nFeature Selection Summary:")
            self.logger.info(f"Total features selected: {len(self.features)}")
            self.logger.info(f"Selected features: {self.features}")
            self.logger.info(f"Current columns in dataframe: {list(self.df.columns)}")
            
            # Prepare and scale the selected features
            self._prepare_features()
            
            # Create sequences for training
            self._create_sequences()

        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            self.logger.exception("Stack trace:")
            raise

    def _inverse_transform_predictions(self) -> None:
        """Inverse transform predictions to original scale"""
        try:
            self.logger.info("Starting inverse transformation of predictions...")
            
            # Helper function to pad and inverse transform
            def inverse_transform_with_padding(data):
                self.logger.info(f"Input data shape: {data.shape}")
                padded = np.concatenate((
                    data, 
                    np.zeros((data.shape[0], self.features.shape[1] - 1))
                ), axis=1)
                self.logger.info(f"Padded data shape: {padded.shape}")
                inverted = self.scaler.inverse_transform(padded)[:, 0]
                self.logger.info(f"Inverted data shape: {inverted.shape}")
                return inverted
            
            # Transform all predictions and actual values
            self.logger.info("Transforming train predictions...")
            self.train_predictions_rescaled = inverse_transform_with_padding(self.train_predictions)
            self.logger.info(f"Train predictions rescaled shape: {self.train_predictions_rescaled.shape}")
            
            self.logger.info("Transforming test predictions...")
            self.test_predictions_rescaled = inverse_transform_with_padding(self.test_predictions)
            self.logger.info(f"Test predictions rescaled shape: {self.test_predictions_rescaled.shape}")
            
            self.logger.info("Transforming train prices...")
            self.train_prices_rescaled = inverse_transform_with_padding(self.y_train.numpy())
            self.logger.info(f"Train prices rescaled shape: {self.train_prices_rescaled.shape}")
            
            self.logger.info("Transforming test prices...")
            self.test_prices_rescaled = inverse_transform_with_padding(self.y_test.numpy())
            self.logger.info(f"Test prices rescaled shape: {self.test_prices_rescaled.shape}")
            
            self.logger.info("Inverse transformation completed successfully.")
            
        except Exception as e:
            self.logger.error(f"Error in inverse transformation: {str(e)}")
            self.logger.exception("Stack trace:")
            raise

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic stock/crypto data from predictions"""
        try:
            # Create synthetic DataFrame from last portion of real data
            df_synthetic = self.df.iloc[-len(self.test_predictions_rescaled):].copy()
            
            # Replace close prices with predictions
            df_synthetic['Close'] = self.test_predictions_rescaled
            
            # Generate synthetic OHLV data
            df_synthetic['Open'] = df_synthetic['Close'] * (1 + np.random.uniform(-0.02, 0.02, 
                                                        size=len(df_synthetic)))
            df_synthetic['High'] = df_synthetic[['Open', 'Close']].max(axis=1) * \
                                (1 + np.random.uniform(0, 0.01, size=len(df_synthetic)))
            df_synthetic['Low'] = df_synthetic[['Open', 'Close']].min(axis=1) * \
                                (1 - np.random.uniform(0, 0.01, size=len(df_synthetic)))
            
            # Generate synthetic volume with safeguards for crypto's large volumes
            if 'Volume' in self.df.columns:
                avg_volume = self.df['Volume'].mean()
                
                if hasattr(self, 'is_crypto') and self.is_crypto:
                    # Handle crypto volumes which might be very large
                    try:
                        # Scale down volume if needed
                        if avg_volume > np.iinfo(np.int32).max / 10:
                            scale_factor = avg_volume / (np.iinfo(np.int32).max / 10)
                            scaled_avg = avg_volume / scale_factor
                            volume_noise = np.random.uniform(-0.2, 0.2, size=len(df_synthetic)) * scaled_avg
                            volume_noise = volume_noise * scale_factor
                        else:
                            volume_noise = np.random.uniform(-0.2, 0.2, size=len(df_synthetic)) * avg_volume
                    except Exception as e:
                        self.logger.warning(f"Error generating crypto volume noise: {str(e)}. Using simplified calculation.")
                        volume_noise = np.random.uniform(-0.2, 0.2, size=len(df_synthetic)) * avg_volume
                else:
                    # Original stock volume calculation
                    volume_noise = np.random.randint(-avg_volume * 0.2, avg_volume * 0.2, 
                                                size=len(df_synthetic))
                
                df_synthetic['Volume'] = avg_volume + volume_noise
            else:
                df_synthetic['Volume'] = 0
            
            return df_synthetic
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise RuntimeError(f"Error generating synthetic data: {str(e)}")

    def _save_plot_data(self, plot_name: str, data: Dict):
        """Save plot data to CSV file with flexible structure"""

        # Debug input
        self.logger.info(f"\n=== Starting _save_plot_data for {plot_name} ===")
        self.logger.info(f"Input data keys: {list(data.keys())}")

        root_dir = get_script_based_dir()
        
        # Create BasicStockModel subdirectories
        data_dir = os.path.join(root_dir, 'data', 'BasicStockModel')
        charts_dir = os.path.join(root_dir, 'visualizations', 'charts', 'BasicStockModel')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(charts_dir, exist_ok=True)
        
        self.logger.info(f"Directories created/verified: \n- {data_dir}\n- {charts_dir}")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Generated timestamp: {timestamp}")

        try:
            # Handle interactive plotly figure if present
            if 'figure' in data:
                self.logger.info("Processing Plotly figure...")
                fig = data.pop('figure')
                self.logger.debug(f"Figure type: {type(fig)}")

                html_path = os.path.join(charts_dir, f'{plot_name}_{timestamp}.html')
                self.logger.info(f"Attempting to save HTML to: {html_path}")
                fig.write_html(html_path)
                
                try:
                    png_path = os.path.join(charts_dir, f'{plot_name}_{timestamp}.png')
                    self.logger.info(f"Attempting to save PNG to: {png_path}")

                    # Debug and convert datetime objects in figure data
                    self.logger.info("Analyzing figure data...")
                    for i, trace in enumerate(fig.data):
                        self.logger.info(f"Processing trace {i}")
                        if hasattr(trace, 'x'):
                            self.logger.info(f"X data type: {type(trace.x)}")
                            if isinstance(trace.x, np.ndarray):
                                # Debug the array contents
                                self.logger.info(f"First few elements of X: {trace.x[:5]}")
                                self.logger.info(f"X dtype: {trace.x.dtype}")

                    fig.write_image(png_path)
                    self.logger.info(f"Saved interactive plot as PNG: {png_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save PNG (kaleido might not be installed): {str(e)}")
                    self.logger.exception("PNG save error details:")
                    
                    # Additional debug information
                    self.logger.error("=== Figure Debug Information ===")
                    self.logger.error(f"Figure type: {type(fig)}")
                    self.logger.error(f"Number of traces: {len(fig.data)}")
                    for i, trace in enumerate(fig.data):
                        self.logger.error(f"\nTrace {i}:")
                        self.logger.error(f"Trace type: {type(trace)}")
                        if hasattr(trace, 'x'):
                            self.logger.error(f"X type: {type(trace.x)}")
                            if isinstance(trace.x, (list, np.ndarray)) and len(trace.x) > 0:
                                self.logger.error(f"First X element: {trace.x[0]}")
                                self.logger.error(f"First X element type: {type(trace.x[0])}")
                
                self.logger.info(f"Saved interactive plot as HTML: {html_path}")
            
            # Process different plot types

            elif plot_name == "combined_predictions":
                # New combined predictions handling
                if all(k in data for k in ['df_with_indicators', 'training_df', 'test_real_df', 'test_pred_df', 'future_df']):
                    # Save each DataFrame separately
                    for name, df in {
                        'indicators': data['df_with_indicators'],
                        'training': data['training_df'],
                        'test_real': data['test_real_df'],
                        'test_pred': data['test_pred_df'],
                        'future': data['future_df']
                    }.items():
                        # Save CSV version
                        df_path = os.path.join(data_dir, f"{plot_name}_{name}_{timestamp}.csv")
                        df.to_csv(df_path)
                        self.logger.info(f"Saved {name} data: {df_path}")
                        
                        # Save pickle version
                        pkl_path = os.path.join(data_dir, f"{plot_name}_{name}_{timestamp}.pkl")
                        df.to_pickle(pkl_path)
                        self.logger.info(f"Saved {name} pickle: {pkl_path}")
                    
                    # Update metadata with prediction start date
                    metadata = data.get('metadata', {})
                    if 'prediction_start_date' in data:
                        metadata['prediction_start_date'] = data['prediction_start_date'].strftime('%Y-%m-%d')
                    
                    # Save metadata
                    metadata_file = os.path.join(data_dir, f"{plot_name}_metadata_{timestamp}.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    self.logger.info(f"Saved metadata: {metadata_file}")
                    
                    # Create combined data for standard CSV save
                    combined_data = pd.concat([
                        data['training_df'],
                        data['test_real_df'],
                        data['test_pred_df'],
                        data['future_df']
                    ]).sort_index()
            
            elif plot_name == "predictions":
                train_data = pd.DataFrame({
                    'date': [d.strftime('%Y-%m-%d') for d in self.df.index[:self.train_size]],
                    'actual': self.train_prices_rescaled,
                    'predicted': self.train_predictions_rescaled,
                    'segment': ['train'] * len(self.train_prices_rescaled)
                })
                
                test_data = pd.DataFrame({
                    'date': [d.strftime('%Y-%m-%d') for d in self.test_index],
                    'actual': self.test_prices_rescaled,
                    'predicted': self.test_predictions_rescaled,
                    'segment': ['test'] * len(self.test_prices_rescaled)
                })
                
                combined_data = pd.concat([train_data, test_data], ignore_index=True)
                
                metrics = self.evaluate()
                metrics_file = os.path.join(data_dir, f"{plot_name}_metrics_{timestamp}.json")
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
                    
            elif plot_name == "technical_indicators":
                combined_data = pd.DataFrame({
                    'date': [d.strftime('%Y-%m-%d') for d in self.df.index],
                    'close': self.df['Close'].values,
                    'volume': self.df['Volume'].values,
                    'macd': self.df['MACD'].values,
                    'signal_line': self.df['Signal_Line'].values,
                    'macd_histogram': self.df['MACD_Histogram'].values,
                    'stochastic_k': self.df['%K'].values,
                    'stochastic_d': self.df['%D'].values,
                    'rsi': self.df['RSI'].values,
                    'bollinger_upper': self.df['Bollinger_Upper'].values,
                    'bollinger_lower': self.df['Bollinger_Lower'].values
                })
                
                stats = {
                    'mean_volume': float(self.df['Volume'].mean()),
                    'std_volume': float(self.df['Volume'].std()),
                    'mean_close': float(self.df['Close'].mean()),
                    'std_close': float(self.df['Close'].std())
                }
                stats_file = os.path.join(data_dir, f"{plot_name}_stats_{timestamp}.json")
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=4)
                    
            elif plot_name == "residuals":
                combined_data = pd.DataFrame({
                    'date': [d.strftime('%Y-%m-%d') for d in pd.concat([pd.Series(data['train_dates']), pd.Series(data['test_dates'])])],
                    'residuals': np.concatenate([data['train_residuals'], data['test_residuals']]),
                    'segment': ['train'] * len(data['train_residuals']) + ['test'] * len(data['test_residuals'])
                })
                
                if 'metadata' in data:
                    metadata_file = os.path.join(data_dir, f"{plot_name}_metadata_{timestamp}.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(data['metadata'], f, indent=4)
                        
            elif plot_name == "candlestick":
                synthetic_df = self._generate_synthetic_data()
                synthetic_df = synthetic_df.reindex(self.df.index, method='ffill')
                
                combined_data = pd.DataFrame({
                    'date': [d.strftime('%Y-%m-%d') for d in self.df.index],
                    'actual_open': self.df['Open'].values,
                    'actual_high': self.df['High'].values,
                    'actual_low': self.df['Low'].values,
                    'actual_close': self.df['Close'].values,
                    'actual_volume': self.df['Volume'].values,
                    'predicted_close': synthetic_df['Close'].values,
                    'predicted_volume': synthetic_df['Volume'].values
                })
                
                stats = {
                    'avg_daily_range': float((self.df['High'] - self.df['Low']).mean()),
                    'avg_volume': float(self.df['Volume'].mean()),
                    'price_volatility': float(self.df['Close'].pct_change().std()),
                    'prediction_mae': float(np.mean(np.abs(
                        synthetic_df['Close'] - self.df['Close']
                    )))
                }
                stats_file = os.path.join(data_dir, f"{plot_name}_stats_{timestamp}.json")
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=4)
            
            # Add this case in the elif chain, before the else clause
            elif plot_name == "residuals_diagnostics":
                # Create DataFrame for residuals
                train_data = pd.DataFrame({
                    'residuals': data['train_residuals'],
                    'segment': ['train'] * len(data['train_residuals'])
                })
                
                test_data = pd.DataFrame({
                    'residuals': data['test_residuals'],
                    'segment': ['test'] * len(data['test_residuals'])
                })
                
                combined_data = pd.concat([train_data, test_data], ignore_index=True)
                
                # Save statistics separately
                stats_file = os.path.join(data_dir, f"{plot_name}_stats_{timestamp}.json")
                with open(stats_file, 'w') as f:
                    json.dump(data['statistics'], f, indent=4)
                self.logger.info(f"Saved residuals diagnostics statistics: {stats_file}")
            
            else:
                # Default handling for other data types
                processed_data = {}
                for key, value in data.items():
                    if isinstance(value, (pd.DatetimeIndex, pd.Index)):
                        processed_data[key] = [d.strftime('%Y-%m-%d') for d in value]
                    elif isinstance(value, (np.ndarray, pd.Series)):
                        processed_data[key] = value.tolist()
                    else:
                        processed_data[key] = value
                
                combined_data = pd.DataFrame(processed_data)
                
                if 'metadata' in data:
                    metadata_file = os.path.join(data_dir, f"{plot_name}_metadata_{timestamp}.json")
                    with open(metadata_file, 'w') as f:
                        json.dump(data['metadata'], f, indent=4)
            
            # Save to CSV without index
            if 'combined_data' in locals():
                filename = f"{plot_name}_{timestamp}.csv"
                filepath = os.path.join(data_dir, filename)
                combined_data.to_csv(filepath, index=False)
                self.logger.info(f"Saved plot data: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving plot data for {plot_name}: {str(e)}")
            self.logger.error("=== Debug Information ===")
            self.logger.error(f"Plot name: {plot_name}")
            self.logger.error(f"Available data keys: {list(data.keys())}")
            if 'figure' in locals():
                self.logger.error("Figure information:")
                for i, trace in enumerate(fig.data):
                    self.logger.error(f"Trace {i} type: {type(trace)}")
                    if hasattr(trace, 'x'):
                        self.logger.error(f"X data type: {type(trace.x)}")
                        if isinstance(trace.x, (list, np.ndarray)) and len(trace.x) > 0:
                            self.logger.error(f"First X element type: {type(trace.x[0])}")
                            self.logger.error(f"First X element value: {trace.x[0]}")
            self.logger.exception("Full stack trace:")
            raise

    def _save_plot(self, plot_name: str, fig: Union[plt.Figure, go.Figure] = None, auto_close: bool = False):
        """Helper function to save plots with timestamp and metadata"""

        self.logger.info(f"=== _save_plot: Saved Plotly - {plot_name} - figure ===")

        try:
            # Fallback for dictionary input
            if isinstance(fig, dict) and 'figure' in fig:
                plotly_fig = fig['figure']
                # Save Plotly figure as HTML
                charts_dir = os.path.join(self.charts_dir, 'BasicStockModel')
                os.makedirs(charts_dir, exist_ok=True)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                html_path = os.path.join(charts_dir, f"{plot_name}_{timestamp}.html")
                plotly_fig.write_html(html_path)
                self.logger.info(f"Saved Plotly figure as HTML: {html_path}")

                time.sleep(Config.SLEEP)  # Give time for file to be written

                return {'plot': html_path}

            # Get current figure if not provided
            if fig is None:
                fig = plt.gcf()
            
            # Create BasicStockModel subdirectory in charts
            charts_dir = os.path.join(self.charts_dir, 'BasicStockModel')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Initialize base metadata
            metadata = {
                'plot_type': plot_name,
                'timestamp': timestamp,
                'ticker': self.config.TICKER_NAME,
                'script_version': self.config.SCRIPT_VERSION,
                'model_config': {
                    'hidden_layer_size': self.hidden_layer_size,
                    'dropout_prob': self.dropout_prob,
                    'use_attention': self.use_attention,
                    'time_step': self.time_step
                }
            }
            
            # Handle different figure types
            if isinstance(fig, go.Figure):
                # Save Plotly figure
                html_path = os.path.join(charts_dir, f"{plot_name}_{timestamp}.html")
                fig.write_html(html_path)
                filepath = html_path
                
                try:
                    png_path = os.path.join(charts_dir, f"{plot_name}_{timestamp}.png")
                    fig.write_image(png_path)

                    time.sleep(Config.SLEEP)  # Give time for file to be written

                    self.logger.info(f"Saved Plotly figure as PNG: {png_path}")
                    filepath = png_path
                except Exception as e:
                    self.logger.warning(f"Could not save PNG (kaleido might not be installed): {str(e)}")
                
                # Add Plotly-specific metadata (safely handling template)
                layout_info = {
                    'width': fig.layout.width,
                    'height': fig.layout.height
                }
                
                # Safely get template information
                if hasattr(fig.layout, 'template'):
                    if hasattr(fig.layout.template, 'layout') and hasattr(fig.layout.template.layout, 'name'):
                        layout_info['template_name'] = fig.layout.template.layout.name
                    else:
                        layout_info['template_name'] = str(fig.layout.template)
                
                metadata.update({
                    'figure_type': 'plotly',
                    'layout': layout_info
                })
                
            else:
                # Save matplotlib figure
                filepath = os.path.join(charts_dir, f"{plot_name}_{timestamp}.png")
                fig.savefig(filepath, 
                        bbox_inches='tight', 
                        dpi=300,
                        facecolor='white',
                        edgecolor='none',
                        pad_inches=0.1,
                        metadata={
                            'Creator': 'BasicStockModel',
                            'Version': self.config.SCRIPT_VERSION,
                            'Timestamp': timestamp,
                            'PlotType': plot_name,
                            'Ticker': self.config.TICKER_NAME
                        })
                
                # Add matplotlib-specific metadata
                metadata.update({
                    'figure_type': 'matplotlib',
                    'figure_size': fig.get_size_inches().tolist(),
                    'dpi': fig.dpi,
                    'num_axes': len(fig.axes)
                })
            
            # Add plot-specific metadata
            if plot_name == "default_visualization":
                metadata.update({
                    'train_size': self.train_size,
                    'test_size': len(self.test_index),
                    'metrics': self.evaluate()
                })
            elif plot_name == "candlestick":
                metadata.update({
                    'prediction_length': len(self._generate_synthetic_data()),
                    'moving_averages': [3, 6, 9]
                })
            elif plot_name == "residuals" or plot_name == "residuals_diagnostics":
                train_residuals = self.train_prices_rescaled - self.train_predictions_rescaled
                test_residuals = self.test_prices_rescaled - self.test_predictions_rescaled
                metadata.update({
                    'residuals_stats': {
                        'train': {
                            'mean': float(np.mean(train_residuals)),
                            'std': float(np.std(train_residuals)),
                            'skew': float(pd.Series(train_residuals).skew()),
                            'kurtosis': float(pd.Series(train_residuals).kurtosis())
                        },
                        'test': {
                            'mean': float(np.mean(test_residuals)),
                            'std': float(np.std(test_residuals)),
                            'skew': float(pd.Series(test_residuals).skew()),
                            'kurtosis': float(pd.Series(test_residuals).kurtosis())
                        }
                    }
                })
            
            # Save metadata to JSON
            metadata_filename = f"{plot_name}_metadata_{timestamp}.json"
            metadata_filepath = os.path.join(charts_dir, metadata_filename)
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            time.sleep(Config.SLEEP)  # Give time for file to be written

            self.logger.info(f"Saved plot: {filepath}")
            self.logger.info(f"Saved plot metadata: {metadata_filepath}")
            
            return {
                'plot': filepath,
                'metadata': metadata_filepath
            }
            
        except Exception as e:
            self.logger.error(f"Error saving plot '{plot_name}': {str(e)}")
            self.logger.error(f"Attempted save location: {filepath if 'filepath' in locals() else 'Not created'}")
            self.logger.exception("Stack trace:")
            return None
        finally:
            if auto_close:
                if isinstance(fig, go.Figure):
                    # No explicit close method for Plotly figures
                    pass
                else:
                    try:
                        plt.close()
                    except:
                        pass
        
    def _add_metrics_box(self, fig: go.Figure, x: float = 1.02, y: float = 0.98) -> None:
        """Add metrics box to the plotly figure"""
        try:
            metrics = self.evaluate()
            if not metrics:  # If metrics is empty
                self.logger.warning("No metrics available to display")
                return
                
            metrics_text = (
                f"<b>Model Metrics</b><br>"
                f"MAE: {metrics['test_mae']:.6f}<br>"
                f"MSE: {metrics['test_mse']:.6f}<br>"
                f"RMSE: {metrics['test_rmse']:.6f}<br>"
                f"R²: {metrics['test_r2']:.4f}<br>"
                f"Direction: {metrics['test_direction_accuracy']:.2f}%<br>"
                f"Error Std: {metrics['test_error_std']:.6f}<br>"
                f"Max Error: {metrics['test_error_max']:.6f}"
            )
            
            fig.add_annotation(
                text=metrics_text,
                xref="paper", yref="paper",
                x=x, y=y,
                showarrow=False,
                font=dict(family="monospace", size=12, color="white"),
                align="left",
                bgcolor="rgba(50,50,50,0.7)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )
                        
        except Exception as e:
            self.logger.error(f"Error adding metrics box: {str(e)}")
            self.logger.warning("Continuing without metrics box")

    def _add_metrics_box_horizontal_v2(self, fig: go.Figure, metrics_type: str = 'default', 
                                predictions = None, residuals_train = None, residuals_test = None,
                                x: float = 0.5, y: float = 1.07) -> None:
        """
        Add metrics box with interpretations to the plotly figure
        
        Args:
            fig (go.Figure): Plotly figure to add metrics to
            metrics_type
            predictions
            residuals_train
            residuals_test
            x (float): X position of metrics box (0-1)
            y (float): Y position of metrics box
        """
        try:
            metrics = self.evaluate()
            if not metrics:
                self.logger.warning("No metrics available to display")
                return
                
            # Debug logging
            self.logger.info("\nAvailable metrics:")
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value}")
            self.logger.info("\n")

            overall_performance, metrics_text = None, None

            if metrics_type == 'default':

                # Debug before R² interpretation
                self.logger.info(f"\ntest_r2 value: {metrics.get('test_r2')}\n")

                # Interpret R² score
                r2_interpretation = (
                    "Excellent" if metrics['test_r2'] > 0.9 else
                    "Good" if metrics['test_r2'] > 0.8 else
                    "Fair" if metrics['test_r2'] > 0.7 else
                    "Poor"
                )
                
                # Debug after R² interpretation
                self.logger.info(f"R² interpretation: {r2_interpretation}")
                
                # Debug direction accuracy
                self.logger.info(f"test_direction_accuracy value: {metrics.get('test_direction_accuracy')}")

                # Interpret Direction Accuracy
                direction_interpretation = (
                    "Excellent" if metrics['test_direction_accuracy'] > 75 else
                    "Good" if metrics['test_direction_accuracy'] > 65 else
                    "Fair" if metrics['test_direction_accuracy'] > 55 else
                    "Poor"
                )
                
                # Debug MAE percentage
                self.logger.info(f"test_mae_percentage value: {metrics.get('test_mae_percentage')}")

                # Interpret MAE Percentage
                mae_pct = float(metrics.get('test_mae_percentage', 0))
                mae_interpretation = (
                    "Excellent" if mae_pct < 1 else
                    "Good" if mae_pct < 3 else
                    "Fair" if mae_pct < 5 else
                    "Poor"
                )
                
                # Debug before metrics text formatting
                self.logger.info("About to format metrics text")

                # Format metrics text with interpretations
                metrics_text = (
                    f"R²: {metrics['test_r2']:.4f} ({r2_interpretation}) • "
                    f"Direction: {metrics['test_direction_accuracy']:.2f}% ({direction_interpretation}) • "
                    f"MAE%: {mae_pct:.2f}% ({mae_interpretation}) • "
                    f"RMSE: {metrics['test_rmse']:.6f} • "
                    f"Error Std: {metrics['test_error_std']:.6f}"
                )
                
                # Debug metrics text
                self.logger.info(f"Formatted metrics text: {metrics_text}\n")

                # Add model performance indicator
                overall_score = (
                    metrics['test_r2'] * 0.4 +  # 40% weight to R²
                    (metrics['test_direction_accuracy'] / 100) * 0.4 +  # 40% weight to direction
                    (1 - min(mae_pct / 10, 1)) * 0.2  # 20% weight to MAE%, capped at 10%
                )
                
                overall_performance = (
                    "🟢 Excellent" if overall_score > 0.85 else
                    "🟡 Good" if overall_score > 0.75 else
                    "🟠 Fair" if overall_score > 0.65 else
                    "🔴 Poor"
                )
                
            if metrics_type == 'residuals':

                # Calculate interpretations
                r2_interpretation = (
                    "Excellent" if metrics['test_r2'] > 0.9 else
                    "Good" if metrics['test_r2'] > 0.8 else
                    "Fair" if metrics['test_r2'] > 0.7 else
                    "Poor"
                )
                
                direction_interpretation = (
                    "Excellent" if metrics['test_direction_accuracy'] > 75 else
                    "Good" if metrics['test_direction_accuracy'] > 65 else
                    "Fair" if metrics['test_direction_accuracy'] > 55 else
                    "Poor"
                )
                
                mae_pct = float(metrics.get('test_mae_percentage', 0))
                mae_interpretation = (
                    "Excellent" if mae_pct < 1 else
                    "Good" if mae_pct < 3 else
                    "Fair" if mae_pct < 5 else
                    "Poor"
                )
                
                # Calculate overall score
                overall_score = (
                    metrics['test_r2'] * 0.4 +
                    (metrics['test_direction_accuracy'] / 100) * 0.4 +
                    (1 - min(mae_pct / 10, 1)) * 0.2
                )
                
                overall_performance = (
                    "🟢 Excellent" if overall_score > 0.85 else
                    "🟡 Good" if overall_score > 0.75 else
                    "🟠 Fair" if overall_score > 0.65 else
                    "🔴 Poor"
                )
                
                # Format metrics text
                metrics_text = (
                    f"R²: {metrics['test_r2']:.4f} ({r2_interpretation}) • "
                    f"Direction: {metrics['test_direction_accuracy']:.2f}% ({direction_interpretation}) • "
                    f"MAE%: {mae_pct:.2f}% ({mae_interpretation}) • "
                    f"RMSE: {metrics['test_rmse']:.6f} • "
                    f"Error Std: {metrics['test_error_std']:.6f}"
                )
            
            if metrics_type == 'residuals_diagnostics':
                # Calculate statistics
                train_stats = pd.Series(residuals_train).describe()
                test_stats = pd.Series(residuals_test).describe()
                
                # Calculate additional metrics
                test_skew = float(pd.Series(residuals_test).skew())
                test_kurtosis = float(pd.Series(residuals_test).kurtosis())
                
                # Interpret metrics
                skew_interpretation = (
                    "Excellent" if abs(test_skew) < 0.5 else
                    "Good" if abs(test_skew) < 1.0 else
                    "Fair" if abs(test_skew) < 1.5 else
                    "Poor"
                )
                
                kurtosis_interpretation = (
                    "Excellent" if abs(test_kurtosis - 3) < 1 else
                    "Good" if abs(test_kurtosis - 3) < 2 else
                    "Fair" if abs(test_kurtosis - 3) < 3 else
                    "Poor"
                )
                
                std_interpretation = (
                    "Excellent" if test_stats['std'] < 0.1 else
                    "Good" if test_stats['std'] < 0.2 else
                    "Fair" if test_stats['std'] < 0.3 else
                    "Poor"
                )
                
                # Format metrics text
                metrics_text = (
                    f"Skewness: {test_skew:.3f} ({skew_interpretation}) • "
                    f"Kurtosis: {test_kurtosis:.3f} ({kurtosis_interpretation}) • "
                    f"Std Dev: {test_stats['std']:.3f} ({std_interpretation})"
                )
                
                # Calculate overall score
                overall_score = (
                    (1 - min(abs(test_skew), 2)/2) * 0.4 +  # 40% weight to skewness
                    (1 - min(abs(test_kurtosis - 3)/6, 1)) * 0.3 +  # 30% weight to kurtosis
                    (1 - min(test_stats['std'], 0.4)/0.4) * 0.3  # 30% weight to std
                )
                
                overall_performance = (
                    "🟢 Excellent" if overall_score > 0.85 else
                    "🟡 Good" if overall_score > 0.75 else
                    "🟠 Fair" if overall_score > 0.65 else
                    "🔴 Poor"
                )

            if metrics_type == 'detailed_analysis':
                metrics = predictions['metrics']
                
                # Get metrics with safe defaults
                test_r2 = metrics.get('test_r2', 0.0)
                test_direction = metrics.get('test_direction_accuracy', 0.0)
                test_mae = metrics.get('test_mae', 0.0)
                test_mape = metrics.get('test_mape', metrics.get('test_mae_percentage', 0.0))
                
                # Interpret R² score
                r2_interpretation = (
                    "Excellent" if test_r2 > 0.9 else
                    "Good" if test_r2 > 0.8 else
                    "Fair" if test_r2 > 0.7 else
                    "Poor"
                )
                
                # Interpret Direction Accuracy
                direction_interpretation = (
                    "Excellent" if test_direction > 75 else
                    "Good" if test_direction > 65 else
                    "Fair" if test_direction > 55 else
                    "Poor"
                )
                
                # Interpret MAPE
                mape_interpretation = (
                    "Excellent" if test_mape < 1 else
                    "Good" if test_mape < 3 else
                    "Fair" if test_mape < 5 else
                    "Poor"
                )
                
                # Calculate overall score
                overall_score = (
                    test_r2 * 0.4 +  # 40% weight to R²
                    (test_direction / 100) * 0.4 +  # 40% weight to direction
                    (1 - min(test_mape / 10, 1)) * 0.2  # 20% weight to MAPE
                )
                
                overall_performance = (
                    "🟢 Excellent" if overall_score > 0.85 else
                    "🟡 Good" if overall_score > 0.75 else
                    "🟠 Fair" if overall_score > 0.65 else
                    "🔴 Poor"
                )
                
                # Format metrics text
                metrics_text = (
                    f"R²: {test_r2:.4f} ({r2_interpretation}) • "
                    f"Direction: {test_direction:.1f}% ({direction_interpretation}) • "
                    f"Error: {test_mape:.2f}% ({mape_interpretation}) • "
                    f"MAE: {test_mae:.6f}"
                )

            # Combine performance indicator with metrics
            display_text = f"{overall_performance} | {metrics_text}"
            
            fig.add_annotation(
                text=display_text,
                xref="paper", yref="paper",
                x=0.5, y=y,
                showarrow=False,
                font=dict(family="monospace", size=11, color="white"),
                align="center",
                bgcolor="rgba(50,50,50,0.7)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4,
                xanchor='center',
                yanchor='bottom',
                width=1600
            )

        except Exception as e:
            self.logger.error(f"Error adding metrics box: {str(e)}")
            self.logger.warning("Continuing without metrics box")

    def _validate_prepared_data(self) -> None:
        """Validate the prepared data"""
        try:
            # Check for NaN values
            if self.df.isna().any().any():
                raise ValueError("Dataset contains NaN values after preparation")
                
            # Validate sequence shapes
            expected_feature_count = len(self.features)
            if self.X_train.shape[-1] != expected_feature_count:
                raise ValueError(f"Training data feature count mismatch. Expected: {expected_feature_count}, Got: {self.X_train.shape[-1]}")
            
            # Validate sequence lengths
            if len(self.X_train) != len(self.y_train):
                raise ValueError("Training data and labels length mismatch")
            if len(self.X_test) != len(self.y_test):
                raise ValueError("Test data and labels length mismatch")
            
            # Validate data ranges
            if not (0 <= self.scaled_data.min() <= self.scaled_data.max() <= 1):
                raise ValueError("Scaled data outside expected range [0, 1]")
                
            self.logger.info("Data validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
        
    def _update_features(self) -> None:
        """Update features list based on data type and available columns"""
        try:
            # Start with base features
            self.features = self.BASE_FEATURES.copy()
            
            if self.is_crypto:
                # Add crypto-specific features if they exist in the dataframe
                crypto_features = [f for f in self.CRYPTO_FEATURES if f in self.df.columns]
                self.features.extend(crypto_features)
                self.logger.info(f"Added {len(crypto_features)} crypto-specific features")
                
            # Update input size
            self.input_size = len(self.features)
            self.logger.info(f"Updated feature list. Total features: {self.input_size}")
            self.logger.debug(f"Features list: {', '.join(self.features)}")
            
        except Exception as e:
            self.logger.error(f"Error updating features: {str(e)}")
            raise
        
    def _validate_data_format(self, df: pd.DataFrame) -> None:
        """
        Validate the data format for both stock and crypto data
        
        Args:
            df (pd.DataFrame): DataFrame to validate
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("DataFrame is empty")
                
            # Check index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be DatetimeIndex")
                
            # Required columns for both types
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add data type specific required columns
            if self.is_crypto:
                if 'Market_Cap' in df.columns:
                    required_columns.append('Market_Cap')
            else:  # Stock
                if 'Adj_Close' in df.columns:
                    required_columns.append('Adj_Close')
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check for null values in required columns
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Null values found in columns:\n{null_counts[null_counts > 0]}")
                
            # Validate data types
            numeric_columns = required_columns
            non_numeric = [col for col in numeric_columns if not np.issubdtype(df[col].dtype, np.number)]
            if non_numeric:
                raise ValueError(f"Non-numeric data found in columns: {non_numeric}")
                
            # Validate price relationships
            invalid_prices = (
                (df['High'] < df['Low']) |
                (df['High'] < df['Open']) |
                (df['High'] < df['Close']) |
                (df['Low'] > df['Open']) |
                (df['Low'] > df['Close'])
            )
            if invalid_prices.any():
                self.logger.warning(f"Found {invalid_prices.sum()} invalid OHLC relationships")
                
            # Validate volume
            if (df['Volume'] < 0).any():
                raise ValueError("Negative volume values found")
                
            # Crypto-specific validations
            if self.is_crypto and 'Market_Cap' in df.columns:
                if (df['Market_Cap'] < 0).any():
                    raise ValueError("Negative market cap values found")
                    
            self.logger.info("Data format validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error validating data format: {str(e)}")
            raise
        
    def _get_model_type(self) -> str:
        """Get the type of model (Stock or Crypto)"""
        try:
            model_type = "Crypto" if self.is_crypto else "Stock"
            model_name = f"Basic{model_type}Model"
            
            # Add attention info if applicable
            if hasattr(self, 'config') and hasattr(self.config, 'USE_ATTENTION'):
                model_name += " with Attention" if self.config.USE_ATTENTION else " without Attention"
                
            return model_name
            
        except Exception as e:
            self.logger.error(f"Error getting model type: {str(e)}")
            return "BasicStockModel"  # Default fallback
        
    def _prepare_plot_data(self, future_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """
        Prepare data for plotting by merging and generating indicators
        
        Args:
            future_days: Optional number of days for future predictions
            
        Returns:
            Tuple containing (combined_df, df_with_indicators, prediction_start_date)
        """
        try:
            # Get merged data
            combined_df, prediction_start_date = self._merge_real_and_predicted_data(future_days)
            
            # Generate technical indicators
            df_with_indicators = self._generate_indicators(combined_df)
            
            return combined_df, df_with_indicators, prediction_start_date
            
        except Exception as e:
            error_msg = f"Failed to prepare plot data: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Stack trace:")
            raise RuntimeError(error_msg)
        
    def _create_plot_base(self, plot_type: str) -> go.Figure:
        """
        Create base plot setup common to both advanced and comparison plots
        
        Args:
            plot_type: Type of plot to create ('advanced' or 'comparison')
        Returns:
            Plotly figure object
        """
        try:
            # Create figure
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=[
                    f"{self.config.TICKER_NAME}: {'Real & Predicted' if plot_type == 'comparison' else ''} Stock Data + Indicators",
                    "MACD Plot",
                    "Stochastic Oscillator"
                ]
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Failed to create plot base: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _get_processed_data(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.Timestamp, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get or generate processed data for plotting"""
        try:
            if not self._data_processed or force_refresh:
                # Prepare data
                self._combined_df, self._df_with_indicators, self._prediction_start_date = self._prepare_plot_data(Config.FUTURE_STEPS)
                
                # Build segmented data
                (self._training_df, self._test_real_df, 
                self._test_pred_df, self._future_df, _) = self._build_segmented_data(self._df_with_indicators)
                
                self._data_processed = True
                
            return (
                self._df_with_indicators,
                self._prediction_start_date,
                self._training_df,
                self._test_real_df,
                self._test_pred_df,
                self._future_df
            )
            
        except Exception as e:
            self.logger.error(f"Error getting processed data: {str(e)}")
            self.logger.exception("Stack trace:")
            raise
        
    def parse_model_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse model parameters from filename and update config
        
        Args:
            filename: Name of the model file to parse
            
        Returns:
            Dict containing display and configuration parameters
        """
        try:
            # Remove file extension and split by underscores
            parts = os.path.splitext(filename)[0].split('_')
            
            # Extract parameters with both display and actual values
            params = {
                'display': {},  # For display purposes
                'config': {}    # For model configuration
            }
            
            # Parse asset and type info
            for part in parts:
                if part in ['crypto', 'stock']:
                    params['display']['Data Type'] = part.title()
                    params['config']['is_crypto'] = (part == 'crypto')
                elif part in ['raw', 'full']:
                    params['display']['Feature Set'] = 'Raw Features Only' if part == 'raw' else 'Full Feature Set'
                    params['config']['use_raw_features_only'] = (part == 'raw')
            
            # Parse version if present
            for part in parts:
                if part.startswith('v.'):
                    version = part.split('.', 1)[1]
                    params['display']['Version'] = version
                    params['config']['script_version'] = version

            # Parse parameters from filename
            for part in parts:
                if '.' not in part:
                    continue
                    
                identifier, value = part.split('.', 1)
                if identifier == 'h':  # Hidden size
                    params['display']['Hidden Size'] = value
                    params['config']['hidden_layer_size'] = int(value)
                elif identifier == 'd':  # Dropout
                    params['display']['Dropout'] = f"0.{value}"
                    params['config']['dropout_prob'] = float(f"0.{value}")
                elif identifier == 'att':  # Attention
                    params['display']['Attention Layer'] = 'Enabled' if value == '1' else 'Disabled'
                    params['config']['use_attention'] = bool(int(value))
                elif identifier == 't':  # Time step
                    params['display']['Time Step'] = value
                    params['config']['time_step'] = int(value)
                elif identifier == 'f':  # Feature count
                    params['display']['Features'] = value
                    params['config']['input_size'] = int(value)
            
            # Set defaults for any missing parameters
            default_config = {
                'hidden_layer_size': self.config.HIDDEN_LAYER_SIZE,
                'dropout_prob': self.config.DROPOUT_PROB,
                'use_attention': self.config.USE_ATTENTION,
                'time_step': self.config.TIME_STEP,
                'input_size': self.get_input_size(),
                'use_raw_features_only': self.config.USE_RAW_FEATURES_ONLY,
                'script_version': self.config.SCRIPT_VERSION
            }
            
            # Update config with defaults for missing values
            for key, default_value in default_config.items():
                if key not in params['config']:
                    params['config'][key] = default_value
                    params['display'][key.title().replace('_', ' ')] = str(default_value)
            
            # Update self.config with parsed values
            self.config.HIDDEN_LAYER_SIZE = params['config']['hidden_layer_size']
            self.config.DROPOUT_PROB = params['config']['dropout_prob']
            self.config.USE_ATTENTION = params['config']['use_attention']
            self.config.TIME_STEP = params['config']['time_step']
            self.config.INPUT_SIZE = params['config']['input_size']
            self.config.USE_RAW_FEATURES_ONLY = params['config']['use_raw_features_only']
            
            # Log configuration updates
            self.logger.info("\nUpdated configuration from model filename:")
            self.logger.info(f"Data Type: {'Cryptocurrency' if params['config'].get('is_crypto') else 'Stock'}")
            self.logger.info(f"Feature Set: {'Raw Features Only' if self.config.USE_RAW_FEATURES_ONLY else 'Full Feature Set'}")
            self.logger.info(f"Hidden Layer Size: {self.config.HIDDEN_LAYER_SIZE}")
            self.logger.info(f"Dropout Probability: {self.config.DROPOUT_PROB}")
            self.logger.info(f"Use Attention: {self.config.USE_ATTENTION}")
            self.logger.info(f"Time Step: {self.config.TIME_STEP}")
            self.logger.info(f"Input Size: {self.config.INPUT_SIZE}")
            self.logger.info(f"Version: {params['config'].get('script_version', 'Unknown')}\n")
            
            # Print updates for user
            print("\nUpdated model configuration:")
            print(f"Data Type: {'Cryptocurrency' if params['config'].get('is_crypto') else 'Stock'}")
            print(f"Feature Set: {'Raw Features Only' if self.config.USE_RAW_FEATURES_ONLY else 'Full Feature Set'}")
            print(f"Hidden Layer Size: {self.config.HIDDEN_LAYER_SIZE}")
            print(f"Dropout Probability: {self.config.DROPOUT_PROB}")
            print(f"Use Attention: {self.config.USE_ATTENTION}")
            print(f"Time Step: {self.config.TIME_STEP}")
            print(f"Input Size: {self.config.INPUT_SIZE}")
            print(f"Version: {params['config'].get('script_version', 'Unknown')}\n")
            
            return params
                
        except Exception as e:
            self.logger.error(f"Error parsing filename {filename}: {str(e)}")
            return {'display': {}, 'config': {}}

    def save_model(self, path: str = None, save_training_data: bool = True) -> None:
        """
        Save model to disk
        
        Args:
            path (str, optional): Custom path to save the model. Defaults to None.
        """
        try:
            # Debug Feature Configuration
            print("\nFeature Configuration:")
            print(f"Raw Features Only: {self.use_raw_features_only}")
            print(f"Base Raw Features: {self.config.USE_RAW_FEATURES_ONLY}\n")
            print(f"Current Features List: {self.FEATURES}")
            print(f"Number of Features: {len(self.FEATURES)}")
            print(f"Current *features List: {self.features}")
            print(f"Number of *features: {len(self.features)}")
            
            # Debug Data Shapes
            print("\nData Shapes:")
            print(f"Features Shape: {self.features.shape if hasattr(self, 'features') else 'Not Set'}")
            print(f"X_train Shape: {self.X_train.shape if hasattr(self, 'X_train') else 'Not Set'}")
            print(f"X_test Shape: {self.X_test.shape if hasattr(self, 'X_test') else 'Not Set'}")

            # Verify model parameters before saving
            print("\nModel Parameters:")
            print(f"Config INPUT_SIZE: {self.config.INPUT_SIZE}")
            print(f"Model input_size: {self.input_size}")
            print(f"Actual LSTM input size: {next(self.model.parameters()).shape[1]}")
            print(f"Config HIDDEN_LAYER_SIZE: {self.config.HIDDEN_LAYER_SIZE}")
            print(f"Model hidden_layer_size: {self.hidden_layer_size}")
            print(f"Actual LSTM hidden size: {self.model.lstm.hidden_size}")
            
            # Debug Model Architecture
            print("\nModel Architecture:")
            print(self.model)

            # Verify state dict shapes
            print("\nState Dictionary Shapes:")
            for name, param in self.model.state_dict().items():
                print(f"{name}: {param.shape}")
                if torch.isnan(param).any():
                    print(f"WARNING: NaN values detected in {name}")

            # Debug Feature Processing
            print("\nFeature Processing State:")
            print(f"Scaler Type: {type(self.scaler).__name__}")
            print(f"Is Crypto: {self.is_crypto}")
            if hasattr(self, 'training_history'):
                print(f"Training History Length: {len(self.training_history)}")

            # Use provided path or generate default
            if path is None:
                path = self._get_model_path()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
                    
            # Create a dictionary of essential config values
            config_dict = {
                'HIDDEN_LAYER_SIZE': self.config.HIDDEN_LAYER_SIZE,
                'DROPOUT_PROB': self.config.DROPOUT_PROB,
                'USE_ATTENTION': self.config.USE_ATTENTION,
                'TIME_STEP': self.config.TIME_STEP,
                'BATCH_SIZE': self.config.BATCH_SIZE,
                'LEARNING_RATE': self.config.LEARNING_RATE,
                'INPUT_SIZE': self.config.INPUT_SIZE,
                'SCRIPT_VERSION': self.config.SCRIPT_VERSION,
                'TICKER_NAME': self.config.TICKER_NAME,
                'USE_RAW_FEATURES_ONLY': self.use_raw_features_only,
                'BASE_RAW_FEATURES': self.BASE_RAW_FEATURES,
                'features_used': self.features
            }
            
            # Add data type specific information
            model_info = {
                'model_type': 'BasicStockModel',
                'data_type': 'Cryptocurrency' if self.is_crypto else 'Stock',
                'features': self.FEATURES,
                'USE_RAW_FEATURES_ONLY': self.use_raw_features_only,  # Save raw features flag
                'is_crypto': self.is_crypto,
                'input_size': self.input_size,
                'hidden_layer_size': self.hidden_layer_size,
                'dropout_prob': self.dropout_prob,
                'use_attention': self.use_attention,
                'time_step': self.time_step,
                'batch_size': self.batch_size,
                'learning_rate': self.lr
            }
            
            # Add crypto-specific information if applicable
            if self.is_crypto:
                model_info.update({
                    'crypto_features': [f for f in self.CRYPTO_FEATURES if f in self.FEATURES],
                    'market_cap_included': 'Market_Cap' in self.FEATURES
                })
                    
            # Create checkpoint with all necessary information
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': config_dict,
                'model_info': model_info,
                'scaler': self.scaler,
                'features': self.FEATURES,
                'last_metrics': getattr(self, 'last_metrics', None),
                'training_history': getattr(self, 'training_history', None),
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add training data if requested
            if save_training_data:
                checkpoint['training_data'] = {
                    'X_train': self.X_train,
                    'y_train': self.y_train,
                    'X_test': self.X_test,
                    'y_test': self.y_test,
                    'scaler': self.scaler,
                    'feature_scalers': self.feature_scalers if hasattr(self, 'feature_scalers') else None,
                    'df': self.df,
                    'features': self.features,

                    # Raw data and configuration
                    'raw_data': {
                        'file_path': self.config.FILE_PATH,
                        'original_df': pd.read_csv(self.config.FILE_PATH),  # Store original CSV data
                        'data_config': {
                            'is_crypto': self.is_crypto,
                            'use_raw_features': self.config.USE_RAW_FEATURES_ONLY,
                            'base_raw_features': self.BASE_RAW_FEATURES,
                            'crypto_features': self.CRYPTO_FEATURES if self.is_crypto else None,
                            'stock_features': self.STOCK_RAW_FEATURES if not self.is_crypto else None
                        }
                    }
                }

            # Save the model
            torch.save(checkpoint, path)
            
            # Log saving information
            self.logger.info(f"Model saved successfully to: {path}")
            self.logger.info(f"Model type: {model_info['data_type']}")
            self.logger.info(f"Features count: {len(self.FEATURES)}")
            if self.is_crypto:
                self.logger.info(f"Crypto-specific features: {model_info['crypto_features']}")
            
            # Print basic information
            print(f"Model saved to: {path}")

            if save_training_data:
                # Log data storage information
                self.logger.info("Stored training data in checkpoint:")
                self.logger.info(f"- Processed data shapes: X_train {self.X_train.shape}, X_test {self.X_test.shape}")
                self.logger.info(f"- Raw data shape: {checkpoint['training_data']['raw_data']['original_df'].shape}")
                print("\nStored both processed and raw data in checkpoint")
            
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Stack trace:")
            raise        
    
    def get_checkpoint_training_data(self, checkpoint) -> bool:
        """
        Load training data from checkpoint.
        Returns True if data was loaded successfully, False if fresh data should be used.
        
        Args:
            checkpoint: The loaded model checkpoint containing training data
            
        Returns:
            bool: True if data was loaded successfully, False if fresh data should be used
        """
        try:
            self.logger.info("Found stored training data in checkpoint")
            print("\nThis model contains saved data:")
            print("1. Use stored processed training data")
            print("2. Use stored raw data and reprocess")
            print("3. Use fresh data")
            
            while True:
                try:
                    choice = input("\nSelect option (1-3): ").strip()
                    if choice in ['1', '2', '3']:
                        self.logger.info(f"User selected option: {choice}")
                        break
                    print("Please enter 1, 2, or 3")
                except ValueError:
                    print("Please enter a valid number")
                    self.logger.warning("Invalid input received for data loading choice")

            if choice == '1':
                try:
                    print("Loading stored processed training data...")
                    self.logger.info("Attempting to load stored processed training data")
                    
                    training_data = checkpoint['training_data']
                    self.X_train = training_data['X_train']
                    self.y_train = training_data['y_train']
                    self.X_test = training_data['X_test']
                    self.y_test = training_data['y_test']
                    self.scaler = training_data['scaler']
                    self.df = training_data['df']
                    self.features = training_data['features']
                    
                    # Set train_size based on X_train shape
                    self.train_size = len(self.X_train)
                    self.logger.info(f"Set train_size to: {self.train_size}")

                    if 'feature_scalers' in training_data:
                        self.feature_scalers = training_data['feature_scalers']
                        self.logger.info("Feature scalers loaded successfully")
                    
                    self.logger.info(f"Loaded training data shapes: X_train {self.X_train.shape}, X_test {self.X_test.shape}")
                    self.logger.info(f"Loaded features count: {len(self.features)}")
                    print("Processed training data loaded successfully")
                    return True

                except KeyError as e:
                    self.logger.error(f"Missing key in training data: {str(e)}")
                    print(f"Error loading processed data: {str(e)}")
                    print("Falling back to fresh data...")
                    return False
                except Exception as e:
                    self.logger.error(f"Unexpected error loading processed data: {str(e)}")
                    print(f"Error: {str(e)}")
                    print("Falling back to fresh data...")
                    return False
                    
            elif choice == '2':
                try:
                    print("Loading raw data...")
                    self.logger.info("Attempting to load raw data")
                    
                    raw_data = checkpoint['training_data']['raw_data']
                    
                    # Store original configuration
                    original_file_path = self.config.FILE_PATH
                    self.logger.info(f"Original file path: {original_file_path}")
                    
                    # Temporarily set the stored DataFrame
                    self.df = raw_data['original_df']
                    self.is_crypto = raw_data['data_config']['is_crypto']
                    self.config.USE_RAW_FEATURES_ONLY = raw_data['data_config']['use_raw_features']
                    
                    self.logger.info(f"Loaded raw data shape: {self.df.shape}")
                    self.logger.info(f"Data type: {'Crypto' if self.is_crypto else 'Stock'}")
                    self.logger.info(f"Using raw features: {self.config.USE_RAW_FEATURES_ONLY}")
                    
                    self.prepare_data()  # Creates all necessary training data from raw DataFrame

                    # Restore original file path
                    self.config.FILE_PATH = original_file_path
                    
                    print("Raw data loaded successfully")
                    self.logger.info("Raw data loading completed successfully")
                    return True
                        
                except KeyError as e:
                    self.logger.error(f"Missing key in raw data: {str(e)}")
                    print(f"Error loading raw data: {str(e)}")
                    print("Falling back to fresh data...")
                    return False
                except Exception as e:
                    self.logger.error(f"Unexpected error processing raw data: {str(e)}")
                    print(f"Error: {str(e)}")
                    print("Falling back to fresh data...")
                    return False
                    
            else:  # choice == '3'
                print("Will use fresh data")
                self.logger.info("User chose to use fresh data")
                return False  # Return False to trigger prepare_data()
                
        except Exception as e:
            self.logger.error(f"Error in data loading process: {str(e)}")
            print(f"Error handling training data: {str(e)}")
            print("Falling back to fresh data...")
            return False  # Return False to trigger prepare_data()

    def load_model(self, path: str = None) -> None:
        """Load model from disk. If path is None, shows available models for selection."""
        try:
            # If no path provided, show model selection menu
            if path is None:
                models_dir = os.path.join(
                    get_script_based_dir(),  # Get root dir
                    'models',                # Use models directory
                    'BasicStockModel'        # BasicStockModel subdirectory
                )

                if not os.path.exists(models_dir):
                    print(f"No models directory found at {models_dir}")
                    return
                
                # Get available model files
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
                if not model_files:
                    print(f"No pre-trained models found in {models_dir}")
                    
                    # Temp solution
                    sys.exit(0)

                    # Ask user if they want to train a new model
                    print("\nNo pre-trained model found. Would you like to train a new model?")
                    print("1. Yes, train a new model")
                    print("2. No, exit")
                    
                    while True:
                        try:
                            choice = int(input("Select option (1-2): "))
                            if choice == 1:
                                self.logger.info("Starting model training...")
                                self.train()
                                return
                            elif choice == 2:
                                self.logger.info("Exiting without training")
                                return
                            else:
                                print("Please enter 1 or 2")
                        except ValueError:
                            print("Please enter a valid number")
                    return

                # Display available models
                print("\nAvailable models:")
                for i, file in enumerate(model_files, 1):
                    model_path = os.path.join(models_dir, file)
                    mod_time = pd.Timestamp.fromtimestamp(os.path.getmtime(model_path))
                    file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
                    
                    # Load checkpoint to get actual configuration
                    try:
                        checkpoint = torch.load(model_path)
                        config_dict = checkpoint.get('config', {})
                        model_info = checkpoint.get('model_info', {})
                        
                        print(f"{i}. {file}")
                        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Size: {file_size:.2f} MB")
                        print("   Parameters:")
                        
                        # Display model configuration
                        if config_dict:
                            print(f"      Hidden Size: {config_dict.get('HIDDEN_LAYER_SIZE', 'N/A')}")
                            print(f"      Dropout: {config_dict.get('DROPOUT_PROB', 'N/A')}")
                            print(f"      Attention Layer: {'Enabled' if config_dict.get('USE_ATTENTION', False) else 'Disabled'}")
                            print(f"      Time Step: {config_dict.get('TIME_STEP', 'N/A')}")
                        
                        # Display feature information
                        if model_info:
                            features = model_info.get('features', [])
                            print(f"      Features: {len(features)}")
                            raw_features = config_dict.get('USE_RAW_FEATURES_ONLY', model_info.get('USE_RAW_FEATURES_ONLY', False))
                            print(f"      Raw Features Only: {'Yes' if raw_features else 'No'}")
                            
                            if model_info.get('is_crypto', False):
                                crypto_features = model_info.get('crypto_features', [])
                                print(f"      Crypto Features: {len(crypto_features)}")
                        
                        print("")
                        
                    except Exception as e:
                        print(f"   Error loading model info: {str(e)}")
                        print("")
                        continue

                # Get user choice
                while True:
                    try:
                        choice = int(input(f"Select model (1-{len(model_files)}) or 0 to cancel: "))
                        if 0 <= choice <= len(model_files):
                            break
                        print(f"Please enter a number between 0 and {len(model_files)}")
                    except ValueError:
                        print("Please enter a valid number")

                if choice == 0:
                    return

                selected_file = model_files[choice - 1]
                path = os.path.join(models_dir, selected_file)
                
                # Parse parameters from filename
                params = self.parse_model_filename(selected_file)

                print("\nDebug: params from the file name:")
                print(params)
                print("\n")
                    
            print(f"\nLoading model from: {path}")
                
            if not os.path.exists(path):
                raise FileNotFoundError(f"No model file found at: {path}")
                
            print("\nDebug: Initial config values:")
            print(f"Hidden Layer Size: {self.config.HIDDEN_LAYER_SIZE}")
            print(f"Input Size: {self.config.INPUT_SIZE}")
            
            checkpoint = torch.load(path)
            
            # Handle training data loading
            has_training_data = 'training_data' in checkpoint
            use_loaded_data = False
            
            if has_training_data:
                use_loaded_data = self.get_checkpoint_training_data(checkpoint)

            # Debug print checkpoint contents
            print("\nCheckpoint contents:")
            for key in checkpoint.keys():
                print(f"Found key: {key}")
            
            # Update config and model attributes from checkpoint
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                print("\nUpdating configuration from checkpoint:")
                
                print("\nLoaded Config Dictionary Contents:")
                for key, value in config_dict.items():
                    print(f"{key}: {value}")

                print("\nModel Info Dictionary Contents:")
                model_info = checkpoint['model_info']
                for key, value in model_info.items():
                    print(f"{key}: {value}")
                print("\n:")

                # Update Config values
                self.config.HIDDEN_LAYER_SIZE = config_dict.get('HIDDEN_LAYER_SIZE', self.config.HIDDEN_LAYER_SIZE)
                self.config.DROPOUT_PROB = config_dict.get('DROPOUT_PROB', self.config.DROPOUT_PROB)
                self.config.USE_ATTENTION = config_dict.get('USE_ATTENTION', self.config.USE_ATTENTION)
                self.config.TIME_STEP = config_dict.get('TIME_STEP', self.config.TIME_STEP)
                self.config.BATCH_SIZE = config_dict.get('BATCH_SIZE', self.config.BATCH_SIZE)
                self.config.LEARNING_RATE = config_dict.get('LEARNING_RATE', self.config.LEARNING_RATE)
                self.config.INPUT_SIZE = config_dict.get('INPUT_SIZE', self.config.INPUT_SIZE)
                
                # Load feature configuration
                self.use_raw_features_only = config_dict.get('USE_RAW_FEATURES_ONLY', self.use_raw_features_only)
                self.config.USE_RAW_FEATURES_ONLY = self.use_raw_features_only

                self.logger.info(f"Loaded raw features setting self.use_raw_features_only: {self.use_raw_features_only}")
                self.logger.info(f"Loaded raw features setting self.config.USE_RAW_FEATURES_ONLY: {self.config.USE_RAW_FEATURES_ONLY}")
                
                # temp
                #sys.exit(0)

                # Load features from checkpoint root level
                loaded_features = checkpoint.get('features', [])
                if loaded_features:
                    print(f"Found features in checkpoint: {len(loaded_features)}")
                    self.FEATURES = loaded_features
                    self.config.INPUT_SIZE = len(self.FEATURES)
                    self.input_size = self.config.INPUT_SIZE
                else:
                    print("Warning: No features found in checkpoint")
                    # Try to get features from model_info as fallback
                    model_info = checkpoint.get('model_info', {})
                    if 'features' in model_info:
                        print("Found features in model_info")
                        self.FEATURES = model_info['features']
                        self.config.INPUT_SIZE = len(self.FEATURES)
                        self.input_size = self.config.INPUT_SIZE
                
                print(f"\nUpdated configuration:")
                print(f"Input Size: {self.config.INPUT_SIZE}")
                print(f"Number of features: {len(self.FEATURES)}")
                print(f"Raw features only: {self.use_raw_features_only}")

                # Update model attributes from config
                self.hidden_layer_size = self.config.HIDDEN_LAYER_SIZE
                self.dropout_prob = self.config.DROPOUT_PROB
                self.use_attention = self.config.USE_ATTENTION
                self.time_step = self.config.TIME_STEP
                self.batch_size = self.config.BATCH_SIZE
                self.lr = self.config.LEARNING_RATE
                
                # Prepare data if needed
                if not use_loaded_data:
                    print("\nPreparing fresh data...")
                    self.prepare_data()

                if hasattr(self, 'features'):
                    print(f"Features array shape: {self.features.shape}")

                # Print updated values
                print(f"Hidden Layer Size: {self.config.HIDDEN_LAYER_SIZE}")
                print(f"Dropout Probability: {self.config.DROPOUT_PROB}")
                print(f"Use Attention: {self.config.USE_ATTENTION}")
                print(f"Time Step: {self.config.TIME_STEP}")
                print(f"Batch Size: {self.config.BATCH_SIZE}")
                print(f"Learning Rate: {self.config.LEARNING_RATE}")
                print(f"Input Size: {self.config.INPUT_SIZE}")

                # temp
                #sys.exit(0)

                # Initialize model with correct parameters
                self.model = None
                print("Debug: Removing existing model")

            # Get state dict from checkpoint
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
            
            # Try to initialize model if needed
            if self.model is None:
                print("Debug: initialize a new model")
                self.initialize_model()
            
            # Try to load state dict
            try:
                # Filter state dict based on attention configuration
                if not self.model.use_attention:
                    state_dict = {k: v for k, v in state_dict.items() 
                                if not k.startswith('attention')}

                # Load filtered state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    state_dict, strict=False
                )
                
                if missing_keys:
                    print(f"Warning: Missing keys in state dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
                    
                print("Model state loaded successfully")
            except Exception as e:
                print(f"Error loading state dict: {str(e)}")
                raise
                
            # Try to load optimizer if available
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Optimizer state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {str(e)}")
            
            # Try to load scaler if available
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
                print("Scaler loaded successfully")
            
            # Set model to evaluation mode
            self.model.eval()

            # Verify model parameters
            print("\nVerifying loaded model parameters:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"Parameter {name}: shape {param.shape}, "
                        f"mean {param.data.mean():.6f}, "
                        f"std {param.data.std():.6f}")
            
            print("\nModel loaded successfully!")
            self.logger.info(f"Model loaded successfully from: {path}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nFalling back to default initialization...")
            if self.model is None:
                self.initialize_model()
            return
          

    def prepare_data(self) -> None:
        """Prepare data for training"""
        try:
            # Get root directory and file path
            root_dir = get_script_based_dir()
            self.logger.info(f"Loading data from: {self.config.FILE_PATH}")
            
            # Detect data format
            self.is_crypto = self._detect_data_format(self.config.FILE_PATH)
            self.logger.info(f"Detected data format: {'Cryptocurrency' if self.is_crypto else 'Stock'}")
            
            # Now set BASE_RAW_FEATURES based on detected type
            self.BASE_RAW_FEATURES = (
                self.CRYPTO_RAW_FEATURES if self.is_crypto 
                else self.STOCK_RAW_FEATURES
            )
            self.logger.info(f"Using {'crypto' if self.is_crypto else 'stock'} raw features: {self.BASE_RAW_FEATURES}")

            # Initialize features based on configuration
            if self.config.USE_RAW_FEATURES_ONLY:
                self.features = self.BASE_RAW_FEATURES.copy()  # Start with raw features only
                self.logger.info("Initialized with raw features only")
            else:
                # Initialize with both raw and technical features
                self.features = self.BASE_RAW_FEATURES.copy()
                self.features.extend(self.BASE_FEATURES)
                self.logger.info("Initialized with full feature set")

            # Get name and script name for model weights path
            asset_name = self.config.STOCK_DATA_FILENAME.split('_')[0]
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            
            # Create BasicStockModel directory in models folder
            models_dir = os.path.join(root_dir, 'models', 'BasicStockModel')
            os.makedirs(models_dir, exist_ok=True)
            
            # Create model weights path under BasicStockModel directory
            self.model_weights_path = os.path.join(
                models_dir,
                f"{script_name}_{asset_name}_"
                f"{'crypto' if self.is_crypto else 'stock'}_"
                f"h{self.hidden_layer_size}_"
                f"d{str(self.dropout_prob).replace('.', '')}.pth"
            )
            self.logger.info(f"Model weights path: {self.model_weights_path}")

            # Load data
            try:
                # Try reading with semicolon delimiter first for crypto data
                try:
                    self.df = pd.read_csv(self.config.FILE_PATH, delimiter=';')
                    if len(self.df.columns) == 1:  # If only one column, delimiter might be wrong
                        self.df = pd.read_csv(self.config.FILE_PATH)
                except:
                    self.df = pd.read_csv(self.config.FILE_PATH)
                
                self.logger.info(f"\nOriginal columns in DataFrame: {list(self.df.columns)}\n")

                # Standardize column names to Title Case immediately after loading
                column_mapping = {
                    'marketcap': 'Marketcap',
                    'marketCap': 'Marketcap',
                    'market_cap': 'Marketcap',
                    'volume': 'Volume',
                    'close': 'Close',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'adj close': 'Adj Close',
                    'adj_close': 'Adj Close',
                    'adjclose': 'Adj Close',
                    'timestamp': 'Timestamp',
                    'timeopen': 'TimeOpen',
                    'timeclose': 'TimeClose',
                    'timehigh': 'TimeHigh',
                    'timelow': 'TimeLow',
                    'name': 'Name'
                }

                # Convert all columns to lowercase first for consistent mapping
                self.df.columns = self.df.columns.str.lower()
                
                # Apply mapping to standardize column names
                self.df.columns = [column_mapping.get(col, col.title()) for col in self.df.columns]
                
                self.logger.info(f"\nStandardized columns in DataFrame: {list(self.df.columns)}\n")

                # Find date column before converting to lowercase
                if self.is_crypto:
                    crypto_date_cols = ['TimeOpen', 'TimeClose', 'TimeHigh', 'TimeLow', 'TimesTamp']
                    date_col = next((col for col in crypto_date_cols if col in self.df.columns), None)
                    
                    if date_col is None:
                        self.logger.warning("No standard crypto date column found")
                        self.logger.info(f"Available columns: {list(self.df.columns)}")
                        raise ValueError("No valid date column found in crypto data")
                else:
                    date_col = 'Date'  # Standard for stock data
                    if date_col not in self.df.columns:
                        self.logger.warning(f"'{date_col}' column not found")
                        self.logger.info(f"Available columns: {list(self.df.columns)}")
                        raise ValueError("No valid date column found in stock data")

                self.logger.info(f"Using date column: {date_col}")

                # Convert to datetime and set index
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                self.df.set_index(date_col, inplace=True)
                
            except Exception as e:
                raise ValueError(f"Error loading data file: {str(e)}")

            # Sort index and handle missing values
            self.df.sort_index(inplace=True)
            self.df.ffill(inplace=True)
            
            # Standardize column names
            #self.df.columns = [col.title() for col in self.df.columns]
            
            # Validate data format
            self._validate_data_format(self.df)
            
            # Select and prepare features
            self._select_features()

            # Add detailed feature debugging
            print("\n=== Feature Selection Debug Information ===")
            print(f"Raw features only: {self.use_raw_features_only}")
            print(f"Total features selected: {len(self.FEATURES)}")
            print("\nSelected features list:")
            for i, feature in enumerate(self.FEATURES, 1):
                print(f"{i}. {feature}")
                
            print("\nFeature array information:")
            print(f"Shape: {self.features.shape}")
            print(f"Non-null count: {np.count_nonzero(~np.isnan(self.features))}")
            
            print("\nFirst few rows of features array:")
            print(self.features[:5])
            
            print("\nFeature statistics:")
            feature_stats = pd.DataFrame(self.features, columns=self.FEATURES).describe()
            print(feature_stats)
            
            print("\nDataFrame columns after feature selection:")
            print(f"Available columns: {list(self.df.columns)}")
            
            print("\nSequence information:")
            if hasattr(self, 'X_train'):
                print(f"X_train shape: {self.X_train.shape}")
                print(f"y_train shape: {self.y_train.shape}")
                print(f"X_test shape: {self.X_test.shape}")
                print(f"y_test shape: {self.y_test.shape}")

            self.initialize_model()

            self.logger.info("Data preparation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            self.logger.exception("Stack trace:")
            raise
        
    def initialize_model(self) -> None:
        """Initialize the LSTM model"""
        try:
            self.model = self.LSTMModel(
                input_size=self.input_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=1,
                dropout_prob=self.dropout_prob,
                use_attention=self.use_attention
            )
            
            self.loss_function = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            self.logger.info(f"Model initialized with input size: {self.input_size}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
        
    def train(self) -> None:
        """Train the model"""
        model_path = self._get_model_path()
        
        # Check if we should load pre-trained weights
        #if os.path.exists(model_path) and input("Load pre-trained weights? (y/n): ").lower() == 'y':
        #    self.load_model(model_path)
        #    print("Loaded pre-trained weights")
        #    return
            
        print("Training from scratch...")

        # Training loop
        for epoch in tqdm(range(self.epochs), desc='Training Progress'):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = self.loss_function(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{self.epochs} - Loss: {epoch_loss / len(self.train_loader)}')

        # Save weights after training
        print("Saving model weights...")
        self.save_model()
        print("Model weights saved!")
        
    def predict(self) -> None:
        """Make predictions using the trained model"""
        try:
            self.logger.info("Starting prediction process...")
            self.model.eval()
            with torch.no_grad():
                self.logger.info(f"X_train shape: {self.X_train.shape}")
                self.train_predictions = self.model(self.X_train).detach().numpy()
                self.logger.info(f"Train predictions shape: {self.train_predictions.shape}")
                
                self.logger.info(f"X_test shape: {self.X_test.shape}")
                self.test_predictions = self.model(self.X_test).detach().numpy()
                self.logger.info(f"Test predictions shape: {self.test_predictions.shape}")
            
            self.logger.info("Predictions generated successfully.")
            
            # Inverse transform predictions
            self._inverse_transform_predictions()
            
            # Set test index for plotting
            self.test_index = self.df.index[self.train_size + self.time_step:]
            self.logger.info(f"Test index set. Length: {len(self.test_index)}")
            
            # Save predictions automatically after making them
            self.save_predictions()
            self.logger.info("Predictions saved successfully.")
            
        except Exception as e:
            self.logger.error(f"Error in prediction process: {str(e)}")
            self.logger.exception("Stack trace:")
            raise
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance and return metrics for both stock and crypto data"""
        try:
            # Calculate basic metrics
            metrics = {
                # Train metrics
                'train_mae': float(np.mean(np.abs(self.train_prices_rescaled - self.train_predictions_rescaled))),
                'train_mse': float(np.mean((self.train_prices_rescaled - self.train_predictions_rescaled) ** 2)),
                'train_rmse': float(np.sqrt(np.mean((self.train_prices_rescaled - self.train_predictions_rescaled) ** 2))),
                'train_r2': float(r2_score(self.train_prices_rescaled, self.train_predictions_rescaled)),
                # Test metrics
                'test_mae': float(np.mean(np.abs(self.test_prices_rescaled - self.test_predictions_rescaled))),
                'test_mse': float(np.mean((self.test_prices_rescaled - self.test_predictions_rescaled) ** 2)),
                'test_rmse': float(np.sqrt(np.mean((self.test_prices_rescaled - self.test_predictions_rescaled) ** 2))),
                'test_r2': float(r2_score(self.test_prices_rescaled, self.test_predictions_rescaled))
            }
            
            # Calculate percentage-based metrics
            metrics.update({
                'train_mae_percentage': float(np.mean(np.abs(
                    (self.train_prices_rescaled - self.train_predictions_rescaled) / self.train_prices_rescaled)) * 100),
                'test_mae_percentage': float(np.mean(np.abs(
                    (self.test_prices_rescaled - self.test_predictions_rescaled) / self.test_prices_rescaled)) * 100),
                # Add MAPE metrics
                'train_mape': float(np.mean(np.abs(
                    (self.train_prices_rescaled - self.train_predictions_rescaled) / self.train_prices_rescaled)) * 100),
                'test_mape': float(np.mean(np.abs(
                    (self.test_prices_rescaled - self.test_predictions_rescaled) / self.test_prices_rescaled)) * 100)
            })
            
            # Calculate directional accuracy
            train_actual_direction = np.diff(self.train_prices_rescaled) > 0
            train_pred_direction = np.diff(self.train_predictions_rescaled) > 0
            test_actual_direction = np.diff(self.test_prices_rescaled) > 0
            test_pred_direction = np.diff(self.test_predictions_rescaled) > 0
            metrics.update({
                'train_direction_accuracy': float(np.mean(train_actual_direction == train_pred_direction) * 100),
                'test_direction_accuracy': float(np.mean(test_actual_direction == test_pred_direction) * 100)
            })
            
            # Calculate error statistics
            train_errors = self.train_prices_rescaled - self.train_predictions_rescaled
            test_errors = self.test_prices_rescaled - self.test_predictions_rescaled
            metrics.update({
                'train_error_std': float(np.std(train_errors)),
                'train_error_max': float(np.max(np.abs(train_errors))),
                'test_error_std': float(np.std(test_errors)),
                'test_error_max': float(np.max(np.abs(test_errors)))
            })
            
            # Add crypto-specific metrics if applicable
            if self.is_crypto and 'Market_Cap' in self.df.columns:
                try:
                    # Calculate price-market cap correlation
                    price_mc_corr = np.corrcoef(
                        self.df['Close'].values[-len(self.test_predictions_rescaled):],
                        self.df['Market_Cap'].values[-len(self.test_predictions_rescaled):]
                    )[0, 1]
                    
                    # Calculate market cap prediction error
                    mc_error = np.mean(np.abs(
                        self.df['Market_Cap'].pct_change().values[-len(self.test_predictions_rescaled):]
                    ))
                    
                    # Add crypto metrics
                    metrics.update({
                        'test_price_mc_correlation': float(price_mc_corr),
                        'test_mc_error': float(mc_error),
                        'test_mc_volatility': float(np.std(
                            self.df['Market_Cap'].pct_change().values[-len(self.test_predictions_rescaled):]
                        ))
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating crypto-specific metrics: {str(e)}")
            
            # Log all metrics
            #self.logger.info(f"\nEvaluation Metrics for {self._get_model_type()} Model:")
            #for metric, value in metrics.items():
            #    self.logger.info(f"{metric}: {value:.4f}")
            #self.logger.info(f"\n")
            
            # Store metrics for later use
            self.last_metrics = metrics
            
            return metrics
            
        except Exception as e:
            error_msg = f"Error calculating metrics: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Stack trace:")
            return {}  # Return empty dict on error
    
    def log_model_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log model metrics in a formatted way
        
        Args:
            metrics: Dictionary containing model metrics
        """
        try:
            border = "="*50
            self.logger.info(f"\n{border}")
            self.logger.info(f"{' Model Performance Metrics ':=^50}")
            self.logger.info(f"{border}")
            
            # Performance metrics
            self.logger.info("\nMain Metrics:")
            self.logger.info(f"{'MAE:':<25} {metrics.get('test_mae', 0.0):>10.6f}")
            self.logger.info(f"{'MSE:':<25} {metrics.get('test_mse', 0.0):>10.6f}")
            self.logger.info(f"{'RMSE:':<25} {metrics.get('test_rmse', 0.0):>10.6f}")
            self.logger.info(f"{'R²:':<25} {metrics.get('test_r2', 0.0):>10.6f}")
            
            # Additional metrics
            self.logger.info("\nAdditional Metrics:")
            self.logger.info(f"{'MAE Percentage:':<25} {metrics.get('test_mae_percentage', '0.00%'):>10}")
            self.logger.info(f"{'Direction Accuracy:':<25} {metrics.get('test_direction_accuracy', 0.0):>9.2f}%")
            
            # Error statistics
            self.logger.info("\nError Statistics:")
            self.logger.info(f"{'Error Std:':<25} {metrics.get('test_error_std', 0.0):>10.6f}")
            self.logger.info(f"{'Max Error:':<25} {metrics.get('test_error_max', 0.0):>10.6f}")
            
            self.logger.info(f"\n{border}\n")
            
        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")
      
    # ============== Extra - START ============== #
        
    def resample_data(self, timeframe: str = 'D') -> None:
        """
        Resample data to different timeframes for both stock and crypto data
        
        Args:
            timeframe (str): Timeframe to resample to:
                - 'D': Daily
                - 'W': Weekly
                - 'M': Monthly
                - 'H': Hourly (crypto only)
                - '15T': 15 minutes (crypto only)
        """
        try:
            self.logger.info(f"Resampling data to {timeframe} timeframe")
            
            # Store original frequency for logging
            original_freq = pd.infer_freq(self.df.index)
            
            # Validate timeframe for data type
            valid_stock_timeframes = ['D', 'W', 'M']
            valid_crypto_timeframes = ['D', 'W', 'M', 'H', '15T', '30T', '1H', '4H']
            
            if not self.is_crypto and timeframe not in valid_stock_timeframes:
                raise ValueError(f"Invalid timeframe for stock data. Valid options: {valid_stock_timeframes}")
            elif self.is_crypto and timeframe not in valid_crypto_timeframes:
                raise ValueError(f"Invalid timeframe for crypto data. Valid options: {valid_crypto_timeframes}")
            
            # Define base resampling rules
            ohlc_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # Add data type specific columns
            if self.is_crypto:
                if 'Market_Cap' in self.df.columns:
                    ohlc_dict['Market_Cap'] = 'last'
                if 'timeOpen' in self.df.columns:
                    ohlc_dict['timeOpen'] = 'first'
                if 'timeClose' in self.df.columns:
                    ohlc_dict['timeClose'] = 'last'
            else:  # Stock specific
                if 'Adj_Close' in self.df.columns:
                    ohlc_dict['Adj_Close'] = 'last'
                if 'Dividends' in self.df.columns:
                    ohlc_dict['Dividends'] = 'sum'
                if 'Stock_Splits' in self.df.columns:
                    ohlc_dict['Stock_Splits'] = 'sum'
            
            # Resample data
            self.df = self.df.resample(timeframe).agg(ohlc_dict)
            
            # Handle missing values with appropriate method
            if timeframe in ['D', 'H', '15T', '30T', '1H', '4H']:
                # For shorter timeframes, forward fill is appropriate
                self.df.ffill(inplace=True)
            else:
                # For longer timeframes (weekly, monthly), interpolate
                self.df = self.df.interpolate(method='time')
                # Forward fill any remaining NaN at edges
                self.df.ffill(inplace=True)
                self.df.bfill(inplace=True)
            
            # Validate resampled data
            self._validate_resampled_data()
            
            # Log resampling results
            self.logger.info(f"Data resampled from {original_freq} to {timeframe}")
            self.logger.info(f"Original shape: {self.df.shape}")
            self.logger.info(f"New shape: {self.df.shape}")
            
            # Recalculate technical indicators if needed
            #self._add_technical_indicators()

            # Recalculate features based on configuration
            self._select_features()
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {str(e)}")
            raise

    def _validate_resampled_data(self) -> None:
        """Validate resampled data for both stock and crypto"""
        try:
            # Check for missing values
            missing_values = self.df.isnull().sum()
            if missing_values.any():
                self.logger.warning(f"Missing values after resampling:\n{missing_values[missing_values > 0]}")
            
            # Validate price continuity
            price_gaps = (self.df['Close'] - self.df['Close'].shift(1)).abs()
            large_gaps = price_gaps[price_gaps > price_gaps.mean() + 2 * price_gaps.std()]
            if not large_gaps.empty:
                self.logger.warning(f"Found {len(large_gaps)} large price gaps after resampling")
            
            # Validate volume
            if (self.df['Volume'] < 0).any():
                raise ValueError("Negative volume values found after resampling")
            
            # Validate OHLC relationship
            invalid_ohlc = (
                (self.df['High'] < self.df['Low']) |
                (self.df['High'] < self.df['Open']) |
                (self.df['High'] < self.df['Close']) |
                (self.df['Low'] > self.df['Open']) |
                (self.df['Low'] > self.df['Close'])
            )
            if invalid_ohlc.any():
                self.logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships after resampling")
            
            # Data type specific validations
            if self.is_crypto and 'Market_Cap' in self.df.columns:
                if (self.df['Market_Cap'] < 0).any():
                    raise ValueError("Negative market cap values found after resampling")
            
            self.logger.info("Resampled data validation completed")
            
        except Exception as e:
            self.logger.error(f"Error validating resampled data: {str(e)}")
            raise

    # ============== Extra - END ============== #
    
    # ============== Advanced Predicts - START ============== #
        
    def _process_predictions(self, data: Union[np.ndarray, torch.Tensor], name: str) -> np.ndarray:
        """
        Process and transform predictions with unified error handling
        
        Args:
            data: Input data to process
            name: Name of the prediction type for error messages
        Returns:
            np.ndarray: Processed predictions in original scale
        """
        try:
            if not isinstance(data, (np.ndarray, torch.Tensor)):
                raise ValueError(f"Invalid data type for {name}")
                
            # Convert to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.numpy()
                
            # Reshape if needed
            if len(data.shape) != 2:
                data = data.reshape(-1, 1)
                
            # Pad and inverse transform
            padded = np.concatenate((
                data, 
                np.zeros((data.shape[0], self.features.shape[1] - 1))
            ), axis=1)
            
            return self.scaler.inverse_transform(padded)[:, 0]
            
        except Exception as e:
            error_msg = f"Error processing {name}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_plot_base(self, plot_type: str) -> Tuple[go.Figure, pd.DataFrame, pd.Timestamp]:
        """
        Create base plot setup common to both advanced and comparison plots
        
        Args:
            plot_type: Type of plot to create ('advanced' or 'comparison')
        Returns:
            Tuple containing figure, processed dataframe, and prediction start date
        """
        try:
            # Create figure
            fig = make_subplots(
                rows=3, cols=1,  # Changed from 2 to 3 rows
                shared_xaxes=True,
                vertical_spacing=0.1,  # Increased spacing between subplots
                row_heights=[0.6, 0.2, 0.2],  # Adjusted heights for three plots
                subplot_titles=[
                    f"{self.config.TICKER_NAME}: {'Real & Predicted' if plot_type == 'comparison' else ''} Stock Data + Indicators",
                    "MACD Plot",
                    "Stochastic Oscillator"  # Added new subplot title
                ]
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Failed to create {plot_type} plot base: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _update_plot_layout(self, fig: go.Figure, plot_type: str = 'advanced') -> None:
        """
        Update plot layout with unified configuration
        
        Args:
            fig: Plotly figure object to update
            plot_type: Type of plot ('advanced' or 'comparison')
        """
        try:
            # Basic layout configuration
            layout = {
                'template': 'plotly_dark',
                'height': 1200 if plot_type == 'comparison' else 1000,
                'margin': dict(l=50, r=150, t=150, b=50),  # Increased right margin for metrics
                'hovermode': 'x unified',
                'dragmode': 'zoom',
                'showlegend': True,
                'legend': self._create_legend_dict(),
                'plot_bgcolor': 'rgb(15,15,15)',
                'paper_bgcolor': 'rgb(15,15,15)',
                'font': dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="white"
                )
            }

            # Common axes style for both main and MACD plots
            common_axes_style = dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.1)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255,255,255,0.2)',
                title_font=dict(size=12, color='white'),
                tickfont=dict(size=10, color='white'),
                showticklabels=True,
                mirror=True,
                ticks='outside',
                tickwidth=1,
                tickcolor='rgba(255,255,255,0.2)'
            )

            # Update main price chart axes
            fig.update_xaxes(
                title_text="Date",
                row=1, col=1,
                **common_axes_style
            )
            
            fig.update_yaxes(
                title_text="Price",
                row=1, col=1,
                **common_axes_style
            )

            # Update Stochastic subplot axes
            fig.update_xaxes(
                title_text="Date",
                row=3, col=1,
                **common_axes_style
            )
            
            fig.update_yaxes(
                title_text="Stochastic",
                row=3, col=1,
                **common_axes_style,
                range=[0, 100]  # Fixed range for Stochastic
            )

            # Update MACD subplot axes
            fig.update_xaxes(
                title_text="Date",
                row=2, col=1,
                **common_axes_style,
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)',
                zerolinewidth=1
            )
            
            fig.update_yaxes(
                title_text="MACD",
                row=2, col=1,
                **common_axes_style,
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.2)',
                zerolinewidth=1
            )

            # Add timeframe buttons
            timeframe_buttons = self._add_timeframe_buttons(fig)

            # Update hover templates
            for trace in fig.data:
                if isinstance(trace, go.Candlestick):
                    trace.update(
                        hoverlabel=dict(
                            bgcolor='rgb(15,15,15)',
                            bordercolor='rgba(255,255,255,0.2)',
                            font=dict(size=12, color='white')
                        ),
                        hoverinfo='all'
                    )
                elif isinstance(trace, go.Scatter):
                    trace.update(
                        hoverlabel=dict(
                            bgcolor='rgb(15,15,15)',
                            bordercolor='rgba(255,255,255,0.2)',
                            font=dict(size=12, color='white')
                        ),
                        hovertemplate="<b>%{text}</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Value: %{y:.6f}<br>" +
                                    "<extra></extra>"
                    )
                elif isinstance(trace, go.Bar):
                    trace.update(
                        hoverlabel=dict(
                            bgcolor='rgb(15,15,15)',
                            bordercolor='rgba(255,255,255,0.2)',
                            font=dict(size=12, color='white')
                        ),
                        hovertemplate="<b>MACD</b><br>" +
                                    "Date: %{x}<br>" +
                                    "Value: %{y:.6f}<br>" +
                                    "<extra></extra>"
                    )

            # Update layout based on plot type
            if plot_type == 'comparison':
                layout.update({
                    'updatemenus': [
                        self._create_comparison_buttons(fig),
                        *timeframe_buttons
                    ],
                    'title': self._create_title_dict()
                })
            else:
                layout.update({
                    'updatemenus': timeframe_buttons,
                    'title': self._create_title_dict()
                })

            # Update main layout
            fig.update_layout(layout)

            # Disable rangeslider
            fig.update_xaxes(rangeslider_visible=False)

            # Add modebar buttons
            fig.update_layout(
                modebar=dict(
                    bgcolor='rgba(15,15,15,0)',
                    color='rgba(255,255,255,0.8)',
                    activecolor='cyan'
                ),
                modebar_add=[
                    'drawline',
                    'drawopenpath',
                    'drawclosedpath',
                    'drawcircle',
                    'drawrect',
                    'eraseshape'
                ]
            )

        except Exception as e:
            error_msg = f"Error updating plot layout: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug("Stack trace:", exc_info=True)
            # Try to apply minimal layout instead of failing completely
            try:
                fig.update_layout(
                    template='plotly_dark',
                    height=1000,
                    margin=dict(l=50, r=50, t=50, b=50),
                    showlegend=True
                )
            except Exception as fallback_error:
                self.logger.error(f"Failed to apply fallback layout: {str(fallback_error)}")

    def _add_prediction_markers(self, fig: go.Figure, prediction_start_date: pd.Timestamp, 
                            df_with_indicators: pd.DataFrame) -> None:
        """
        Add prediction start line and annotation to the figure
        
        Args:
            fig: Plotly figure object
            prediction_start_date: Timestamp marking prediction start
            df_with_indicators: DataFrame containing indicator data
        """
        try:
            # Add vertical line at prediction start
            fig.add_shape(
                dict(
                    type="line",
                    x0=prediction_start_date,
                    y0=0,
                    x1=prediction_start_date,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(
                        color="cyan",
                        width=2,
                        dash="dot"
                    )
                )
            )
            
            # Add annotation for prediction start
            fig.add_annotation(
                x=prediction_start_date,
                y=df_with_indicators['High'].max(),
                text="Prediction Start",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="cyan",
                font=dict(
                    size=12,
                    color="cyan"
                ),
                align="center",
                ax=0,
                ay=-40
            )
            
        except Exception as e:
            error_msg = f"Error adding prediction markers: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _merge_real_and_predicted_data(self, future_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Merge real stock data with predicted data into a continuous DataFrame
        
        Args:
            future_days (int, optional): Number of days to include in predictions.
                                    If None, uses all available predictions.
        
        Returns:
            Tuple[pd.DataFrame, pd.Timestamp]: Combined DataFrame and prediction start date
        """
        try:
            print("Merging real and predicted data...")
            
            if self.df is None:
                raise ValueError("No real data available")
                
            # Get prediction start date
            prediction_start_date = self.df.index[-1] + pd.Timedelta(days=1)
            
            # Determine which predictions to use
            try:
                # First check if we have test predictions
                if not hasattr(self, 'test_predictions_rescaled'):
                    raise ValueError("No predictions available. Run predict() first.")
                    
                # Default to test predictions
                predictions_to_use = self.test_predictions_rescaled
                print("Using test predictions")
                
                # Override with future predictions if available
                if hasattr(self, 'future_predictions') and self.future_predictions is not None:
                    predictions_to_use = self.future_predictions_rescaled
                    print("Using future predictions")
                    
                # Limit to specified days if requested
                if future_days is not None:
                    if future_days <= 0:
                        raise ValueError("future_days must be positive")
                    predictions_to_use = predictions_to_use[:future_days]
                    print(f"Using {future_days} days of predictions")
                    
            except Exception as e:
                error_msg = f"Error selecting predictions: {str(e)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            try:
                # Create predicted DataFrame
                predicted_dates = pd.date_range(
                    start=prediction_start_date, 
                    periods=len(predictions_to_use), 
                    freq='D'
                )
                
                predicted_df = pd.DataFrame({
                    'Date': predicted_dates,
                    'Close': predictions_to_use,
                    'Type': 'Predicted'
                })
                
                # Generate synthetic OHLV data
                predicted_df['Open'] = predicted_df['Close'] * (1 + np.random.uniform(-0.02, 0.02, size=len(predicted_df)))
                predicted_df['High'] = predicted_df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, size=len(predicted_df)))
                predicted_df['Low'] = predicted_df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, size=len(predicted_df)))
                
                # Add volume if available
                if 'Volume' in self.df.columns:
                    avg_volume = self.df['Volume'].mean()
                    
                    # Handle large volume numbers safely
                    try:
                        if avg_volume > 1e9:  # If volume is very large
                            # Scale down calculations to avoid int32 overflow
                            scale_factor = avg_volume / 1e6
                            scaled_avg = avg_volume / scale_factor
                            volume_noise = np.random.uniform(-0.2, 0.2, size=len(predicted_df)) * scaled_avg
                            volume_noise = volume_noise * scale_factor
                        else:
                            # Use uniform distribution instead of randint for smaller numbers
                            volume_noise = np.random.uniform(-0.2, 0.2, size=len(predicted_df)) * avg_volume
                        
                        # Ensure volumes are non-negative
                        predicted_df['Volume'] = np.maximum(0, avg_volume + volume_noise)
                        
                    except Exception as e:
                        self.logger.warning(f"Error generating volume noise: {str(e)}. Using simplified calculation.")
                        # Fallback to simple percentage-based variation
                        variation = np.random.uniform(self.traine_test_split, 1.2, size=len(predicted_df))
                        predicted_df['Volume'] = avg_volume * variation
                else:
                    predicted_df['Volume'] = 0  # Default value if no volume data
                
                # Label real data
                real_df = self.df.copy()
                real_df['Type'] = 'Real'
                real_df['Date'] = real_df.index
                
                # Combine and set index
                combined_df = pd.concat([real_df, predicted_df])
                combined_df.set_index('Date', inplace=True)
                
                print(f"Successfully merged data with {len(predicted_df)} predicted days")
                return combined_df, prediction_start_date
                
            except Exception as e:
                error_msg = f"Error creating and merging DataFrames: {str(e)}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to merge real and predicted data: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Stack trace:")
            raise

    def _generate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicators for the dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            print("Generating technical indicators...")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            # Create a copy to avoid modifying original
            df = df.copy()
            
            # Calculate Multiple EMAs
            ema_periods = [9, 20, 40, 50, 100, 200]
            for period in ema_periods:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            # Calculate Multiple SMAs
            sma_periods = [10, 20, 50, 100, 200]
            for period in sma_periods:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            # Calculate MACD with multiple settings
            macd_settings = [
                (12, 26, 9),   # Traditional
                (5, 35, 5),    # Fast
                (8, 21, 5)     # Custom
            ]
            
            for fast, slow, signal in macd_settings:
                prefix = f'MACD_{fast}_{slow}_{signal}'
                df[f'{prefix}_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
                df[f'{prefix}_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
                df[f'{prefix}'] = df[f'{prefix}_fast'] - df[f'{prefix}_slow']
                df[f'{prefix}_signal'] = df[f'{prefix}'].ewm(span=signal, adjust=False).mean()
                df[f'{prefix}_hist'] = df[f'{prefix}'] - df[f'{prefix}_signal']
            
            # Keep original MACD for main display
            df['MACD'] = df['MACD_12_26_9']
            df['Signal_Line'] = df['MACD_12_26_9_signal']
            df['MACD_Histogram'] = df['MACD_12_26_9_hist']
            
            # Calculate Stochastic Oscillator with multiple periods
            stoch_periods = [14, 21]
            for period in stoch_periods:
                low_n = df['Low'].rolling(window=period).min()
                high_n = df['High'].rolling(window=period).max()
                df[f'%K_{period}'] = 100 * ((df['Close'] - low_n) / (high_n - low_n))
                df[f'%D_{period}'] = df[f'%K_{period}'].rolling(window=3).mean()
            
            # Keep original Stochastic for main display
            df['%K'] = df['%K_14']
            df['%D'] = df['%D_14']
            
            # Calculate RSI with multiple periods
            rsi_periods = [14, 21]
            for period in rsi_periods:
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # Keep original RSI for main display
            df['RSI'] = df['RSI_14']
            
            # Calculate Bollinger Bands with multiple standard deviations
            bb_periods = [20]
            bb_stds = [2, 3]
            for period in bb_periods:
                sma = df['Close'].rolling(window=period).mean()
                std = df['Close'].rolling(window=period).std()
                for n_std in bb_stds:
                    df[f'BB_Upper_{period}_{n_std}'] = sma + (std * n_std)
                    df[f'BB_Lower_{period}_{n_std}'] = sma - (std * n_std)
                    df[f'BB_SMA_{period}'] = sma
            
            # Keep original Bollinger Bands for main display
            df['BB_Upper'] = df['BB_Upper_20_2']
            df['BB_Lower'] = df['BB_Lower_20_2']
            df['SMA_20'] = df['BB_SMA_20']
            
            # Volume indicators (if volume data is available)
            if 'Volume' in df.columns:
                df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_SMA_50'] = df['Volume'].rolling(window=50).mean()
                # Volume Rate of Change
                df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
            
            # Momentum indicators
            # Rate of Change
            df['ROC'] = df['Close'].pct_change(periods=10) * 100
            # Momentum
            df['Momentum'] = df['Close'].diff(10)
            # Williams %R
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            df['Williams_%R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
            
            # Optional: Calculate STL decomposition if enough data
            if len(df) > 365:  # Only if we have more than a year of data
                try:
                    stl = STL(df['Close'], period=365)
                    result = stl.fit()
                    df['Trend'] = result.trend
                    df['Seasonal'] = result.seasonal
                    df['Residual'] = result.resid
                except Exception as e:
                    print(f"Warning: STL decomposition failed: {str(e)}")
                    # Fallback to simple trend
                    df['Trend'] = df['Close'].rolling(window=20).mean()
                    df['Seasonal'] = 0
                    df['Residual'] = df['Close'] - df['Trend']
            
            # Fill NaN values with forward fill, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            print("Technical indicators generated successfully")
            return df
            
        except Exception as e:
            error_msg = f"Failed to generate technical indicators: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Stack trace:")
            raise RuntimeError(error_msg)

    def _add_advanced_traces(self, fig: go.Figure, df: pd.DataFrame) -> None:
        """
        Add traces for advanced plot visualization
        
        Args:
            fig: Plotly figure object
            df: DataFrame with indicators
        """
        try:
            # Split data into real and predicted if Type column exists
            if 'Type' in df.columns:
                real_data = df[df['Type'] == 'Real']
                predicted_data = df[df['Type'] == 'Predicted']
                
                # Add real data candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=real_data.index,
                        open=real_data['Open'],
                        high=real_data['High'],
                        low=real_data['Low'],
                        close=real_data['Close'],
                        name='Real OHLC',
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        text=[f"Date: {d}<br>"
                            f"Open: {o:.6f}<br>"
                            f"High: {h:.6f}<br>"
                            f"Low: {l:.6f}<br>"
                            f"Close: {c:.6f}"
                            for d, o, h, l, c in zip(real_data.index,
                                                    real_data['Open'],
                                                    real_data['High'],
                                                    real_data['Low'],
                                                    real_data['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
                
                # Add predicted data candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=predicted_data.index,
                        open=predicted_data['Open'],
                        high=predicted_data['High'],
                        low=predicted_data['Low'],
                        close=predicted_data['Close'],
                        name='Predicted OHLC',
                        increasing_line_color='cyan',
                        decreasing_line_color='magenta',
                        text=[f"Date: {d}<br>"
                            f"Open: {o:.6f}<br>"
                            f"High: {h:.6f}<br>"
                            f"Low: {l:.6f}<br>"
                            f"Close: {c:.6f}"
                            for d, o, h, l, c in zip(predicted_data.index,
                                                    predicted_data['Open'],
                                                    predicted_data['High'],
                                                    predicted_data['Low'],
                                                    predicted_data['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
                
                # Add close price lines
                fig.add_trace(
                    go.Scatter(
                        x=real_data.index,
                        y=real_data['Close'],
                        mode='lines',
                        name='Real Close',
                        line=dict(color='lime', width=1.5),
                        visible='legendonly',
                        text=[f"Date: {d}<br>Close: {c:.6f}"
                            for d, c in zip(real_data.index, real_data['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=predicted_data.index,
                        y=predicted_data['Close'],
                        mode='lines',
                        name='Predicted Close',
                        line=dict(color='cyan', width=1.5),
                        visible='legendonly',
                        text=[f"Date: {d}<br>Close: {c:.6f}"
                            for d, c in zip(predicted_data.index, predicted_data['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
            else:
                # Add single candlestick chart for non-split data
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='OHLC',
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        text=[f"Date: {d}<br>"
                            f"Open: {o:.6f}<br>"
                            f"High: {h:.6f}<br>"
                            f"Low: {l:.6f}<br>"
                            f"Close: {c:.6f}"
                            for d, o, h, l, c in zip(df.index,
                                                    df['Open'],
                                                    df['High'],
                                                    df['Low'],
                                                    df['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
            
            # Add EMAs
            for ema, color in [('EMA_20', 'orange'), ('EMA_40', 'red')]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ema],
                        mode='lines',
                        name=ema,
                        line=dict(color=color, width=1),
                        visible='legendonly',
                        text=[f"Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df[ema])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
            
            # Add Bollinger Bands
            for band, color in [('BB_Upper', 'rgba(173, 204, 255, 0.3)'), 
                            ('BB_Lower', 'rgba(173, 204, 255, 0.3)')]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[band],
                        mode='lines',
                        name=band,
                        line=dict(color=color, width=1),
                        visible='legendonly',
                        text=[f"Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df[band])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
            
            # Add MACD components based on data type
            if 'Type' in df.columns:
                for data_type, data_subset, color_suffix in [
                    ('Real', real_data, ''), 
                    ('Predicted', predicted_data, ' (Pred)')
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=data_subset.index,
                            y=data_subset['MACD'],
                            mode='lines',
                            name=f'MACD{color_suffix}',
                            line=dict(
                                color='blue' if data_type == 'Real' else 'cyan',
                                width=1.5
                            ),
                            text=[f"Date: {d}<br>Value: {v:.6f}"
                                for d, v in zip(data_subset.index, data_subset['MACD'])],
                            hoverinfo='text'
                        ), row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data_subset.index,
                            y=data_subset['Signal_Line'],
                            mode='lines',
                            name=f'Signal{color_suffix}',
                            line=dict(
                                color='orange' if data_type == 'Real' else 'magenta',
                                width=1.5
                            ),
                            text=[f"Date: {d}<br>Value: {v:.6f}"
                                for d, v in zip(data_subset.index, data_subset['Signal_Line'])],
                            hoverinfo='text'
                        ), row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=data_subset.index,
                            y=data_subset['MACD_Histogram'],
                            name=f'Histogram{color_suffix}',
                            marker_color=np.where(
                                data_subset['MACD_Histogram'] >= 0,
                                'green' if data_type == 'Real' else 'cyan',
                                'red' if data_type == 'Real' else 'magenta'
                            ),
                            text=[f"Date: {d}<br>Value: {v:.6f}"
                                for d, v in zip(data_subset.index, data_subset['MACD_Histogram'])],
                            hoverinfo='text'
                        ), row=2, col=1
                    )
            else:
                # Add MACD components for non-split data
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=1.5),
                        text=[f"Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df['MACD'])],
                        hoverinfo='text'
                    ), row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Signal_Line'],
                        mode='lines',
                        name='Signal Line',
                        line=dict(color='orange', width=1.5),
                        text=[f"Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df['Signal_Line'])],
                        hoverinfo='text'
                    ), row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=np.where(df['MACD_Histogram'] >= 0, 'green', 'red'),
                        text=[f"Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df['MACD_Histogram'])],
                        hoverinfo='text'
                    ), row=2, col=1
                )
                
        except Exception as e:
            error_msg = f"Error adding advanced traces: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _add_comparison_traces(self, fig: go.Figure, df: pd.DataFrame) -> None:
        """
        Add traces for comparison plot visualization
        
        Args:
            fig: Plotly figure object
            df: DataFrame with indicators
        """
        try:
            # Split data into real and predicted
            real_data = df[df['Type'] == 'Real']
            predicted_data = df[df['Type'] == 'Predicted']
            
            # Add real data candlestick
            fig.add_trace(
                go.Candlestick(
                    x=real_data.index,
                    open=real_data['Open'],
                    high=real_data['High'],
                    low=real_data['Low'],
                    close=real_data['Close'],
                    name='Real OHLC',
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    text=[f"Real OHLC<br>Date: {d}<br>"
                        f"Open: {o:.6f}<br>"
                        f"High: {h:.6f}<br>"
                        f"Low: {l:.6f}<br>"
                        f"Close: {c:.6f}"
                        for d, o, h, l, c in zip(real_data.index,
                                                real_data['Open'],
                                                real_data['High'],
                                                real_data['Low'],
                                                real_data['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add predicted data candlestick with different colors
            fig.add_trace(
                go.Candlestick(
                    x=predicted_data.index,
                    open=predicted_data['Open'],
                    high=predicted_data['High'],
                    low=predicted_data['Low'],
                    close=predicted_data['Close'],
                    name='Predicted OHLC',
                    increasing_line_color='cyan',
                    decreasing_line_color='magenta',
                    text=[f"Predicted OHLC<br>Date: {d}<br>"
                        f"Open: {o:.6f}<br>"
                        f"High: {h:.6f}<br>"
                        f"Low: {l:.6f}<br>"
                        f"Close: {c:.6f}"
                        for d, o, h, l, c in zip(predicted_data.index,
                                                predicted_data['Open'],
                                                predicted_data['High'],
                                                predicted_data['Low'],
                                                predicted_data['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add real close price line
            fig.add_trace(
                go.Scatter(
                    x=real_data.index,
                    y=real_data['Close'],
                    mode='lines',
                    name='Real Close',
                    line=dict(color='lime', width=1.5),
                    visible='legendonly',
                    text=[f"Real Close<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(real_data.index, real_data['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add predicted close price line
            fig.add_trace(
                go.Scatter(
                    x=predicted_data.index,
                    y=predicted_data['Close'],
                    mode='lines',
                    name='Predicted Close',
                    line=dict(color='cyan', width=1.5),
                    visible='legendonly',
                    text=[f"Predicted Close<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(predicted_data.index, predicted_data['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add MACD components to subplot
            for data_type, data_subset, color_suffix in [
                ('Real', real_data, ''), 
                ('Predicted', predicted_data, ' (Pred)')
            ]:
                # MACD Line
                fig.add_trace(
                    go.Scatter(
                        x=data_subset.index,
                        y=data_subset['MACD'],
                        mode='lines',
                        name=f'MACD{color_suffix}',
                        line=dict(
                            color='blue' if data_type == 'Real' else 'cyan',
                            width=1.5
                        ),
                        text=[f"MACD{color_suffix}<br>Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(data_subset.index, data_subset['MACD'])],
                        hoverinfo='text'
                    ), row=2, col=1
                )
                
                # Signal Line
                fig.add_trace(
                    go.Scatter(
                        x=data_subset.index,
                        y=data_subset['Signal_Line'],
                        mode='lines',
                        name=f'Signal{color_suffix}',
                        line=dict(
                            color='orange' if data_type == 'Real' else 'magenta',
                            width=1.5
                        ),
                        text=[f"Signal{color_suffix}<br>Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(data_subset.index, data_subset['Signal_Line'])],
                        hoverinfo='text'
                    ), row=2, col=1
                )
                
                # MACD Histogram
                fig.add_trace(
                    go.Bar(
                        x=data_subset.index,
                        y=data_subset['MACD_Histogram'],
                        name=f'Histogram{color_suffix}',
                        marker_color=np.where(
                            data_subset['MACD_Histogram'] >= 0,
                            'green' if data_type == 'Real' else 'cyan',
                            'red' if data_type == 'Real' else 'magenta'
                        ),
                        text=[f"Histogram{color_suffix}<br>Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(data_subset.index, data_subset['MACD_Histogram'])],
                        hoverinfo='text'
                    ), row=2, col=1
                )
            
        except Exception as e:
            error_msg = f"Error adding comparison traces: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _add_technical_indicator_traces(self, fig: go.Figure, df: pd.DataFrame) -> None:
        """
        Add technical indicators to the plot
        
        Args:
            fig: Plotly figure object
            df: DataFrame with technical indicators
        """
        try:
            # Calculate daily and weekly trends
            df['Daily_Trend'] = df['Close'].rolling(window=1).mean()
            df['Weekly_Trend'] = df['Close'].rolling(window=5).mean()
            
            # Add RSI
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1),
                    visible='legendonly',
                    text=[f"RSI<br>Date: {d}<br>Value: {v:.6f}"
                        for d, v in zip(df.index, df['RSI'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add Stochastic Oscillator
            for indicator, color in [('%K', 'blue'), ('%D', 'red')]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        mode='lines',
                        name=f'Stochastic {indicator}',
                        line=dict(color=color, width=1),
                        visible='legendonly',
                        text=[f"Stochastic {indicator}<br>Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df[indicator])],
                        hoverinfo='text'
                    ), row=3, col=1
                )
            
            # Add horizontal lines for overbought/oversold levels
            for level in [20, 80]:
                fig.add_hline(
                    y=level,
                    line=dict(color='rgba(200,200,200,0.2)', width=1, dash='dash'),
                    row=3, col=1
                )
            
            # Add Bollinger Bands
            for band, color in [
                ('BB_Upper', 'rgba(173, 204, 255, 0.3)'),
                ('SMA_20', 'rgba(173, 204, 255, 0.8)'),
                ('BB_Lower', 'rgba(173, 204, 255, 0.3)')
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[band],
                        mode='lines',
                        name=f'Bollinger {band}',
                        line=dict(color=color, width=1),
                        visible='legendonly',
                        text=[f"Bollinger {band}<br>Date: {d}<br>Value: {v:.6f}"
                            for d, v in zip(df.index, df[band])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
            
            # Add trend components if available
            if 'Trend' in df.columns:
                for component, color in [
                    ('Trend', 'yellow'),
                    ('Seasonal', 'cyan'),
                    ('Residual', 'gray'),
                    ('Daily_Trend', 'rgba(255, 0, 255, 0.7)'),
                    ('Weekly_Trend', 'rgba(0, 255, 255, 0.7)')
                ]:
                    component_name = f'{"STL " if component in ["Trend", "Seasonal", "Residual"] else ""}{component}'
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[component],
                            mode='lines',
                            name=component_name,
                            line=dict(
                                color=color,
                                width=1,
                                dash='dot' if component in ['Daily_Trend', 'Weekly_Trend'] else 'solid'
                            ),
                            visible='legendonly',
                            text=[f"{component_name}<br>Date: {d}<br>Value: {v:.6f}"
                                for d, v in zip(df.index, df[component])],
                            hoverinfo='text'
                        ), row=1, col=1
                    )
            else:
                # Add daily and weekly trends even if STL components aren't available
                for trend, color in [
                    ('Daily_Trend', 'rgba(255, 0, 255, 0.7)'),
                    ('Weekly_Trend', 'rgba(0, 255, 255, 0.7)')
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[trend],
                            mode='lines',
                            name=trend,
                            line=dict(
                                color=color,
                                width=1,
                                dash='dot'
                            ),
                            visible='legendonly',
                            text=[f"{trend}<br>Date: {d}<br>Value: {v:.6f}"
                                for d, v in zip(df.index, df[trend])],
                            hoverinfo='text'
                        ), row=1, col=1
                    )
            
        except Exception as e:
            error_msg = f"Error adding technical indicators: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_legend_dict(self) -> dict:
        """
        Create dictionary for plot legend configuration
        
        Returns:
            dict: Legend configuration
        """
        return {
            'x': 1.05,
            'y': 1.0,
            'xanchor': 'left',
            'yanchor': 'top',
            'bgcolor': 'rgba(50, 50, 50, 0.8)',
            'bordercolor': 'rgba(255, 255, 255, 0.3)',
            'borderwidth': 1,
            'font': dict(
                size=10,
                color='white'
            ),
            'orientation': 'v',
            'itemsizing': 'constant',
            'itemwidth': 30,
            'itemclick': 'toggle',
            'itemdoubleclick': 'toggleothers'
        }

    def _create_title_dict(self) -> dict:
        """Create standardized title configuration"""
        return dict(
            text=f"{self.config.TICKER_NAME} Price Prediction Analysis",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(
                family="Arial, sans-serif",
                size=24,
                color='white'
            )
        )

    def _create_comparison_buttons(self, fig: go.Figure) -> dict:
        """
        Create buttons for comparison plot
        
        Args:
            fig: Plotly figure object
            
        Returns:
            dict: Button configuration
        """
        return [{
            'buttons': [
                {
                    'label': 'All Data',
                    'method': 'update',
                    'args': [{'visible': [True] * len(fig.data)}]
                },
                {
                    'label': 'Real Only',
                    'method': 'update',
                    'args': [{'visible': [i < len(fig.data)//2 for i in range(len(fig.data))]}]
                },
                {
                    'label': 'Predicted Only',
                    'method': 'update',
                    'args': [{'visible': [i >= len(fig.data)//2 for i in range(len(fig.data))]}]
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.1,
            'xanchor': 'left',
            'yanchor': 'top',
            'bgcolor': 'rgba(50, 50, 50, 0.8)',
            'font': dict(
                size=12,
                color='white'
            )
        }]

    def _add_timeframe_buttons(self, fig: go.Figure) -> list:
        """Add timeframe selection buttons to the figure
        
        Args:
            fig: Plotly figure object to update with timeframe selection buttons
            
        Returns:
            list: Timeframe buttons configuration
        """
        try:
            # Add timeframe selector buttons
            timeframe_buttons = [{
                'buttons': [
                    dict(
                        label="1D",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="1W",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(weeks=1)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="1M",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="3M",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="6M",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="1Y",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="2Y",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="3Y",
                        method="relayout",
                        args=[{"xaxis.range": [
                            (pd.Timestamp.now() - pd.Timedelta(days=1095)).strftime('%Y-%m-%d'),
                            pd.Timestamp.now().strftime('%Y-%m-%d')
                        ]}]
                    ),
                    dict(
                        label="MAX",
                        method="relayout",
                        args=[{"xaxis.autorange": True}]
                    )
                ],
                'direction': 'right',
                'pad': {"r": 10, "t": 10},
                'showactive': True,
                'x': 0.0,
                'xanchor': 'left',
                'y': 1.15,
                'yanchor': 'top',
                'type': 'buttons',
                'bgcolor': 'rgba(50,50,50,0.8)',
                'font': dict(color='white')
            }]

            return timeframe_buttons

        except Exception as e:
            error_msg = f"Error adding timeframe buttons: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _build_segmented_data(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build segmented data from dataframe
        
        Args:
            df: DataFrame with either Type column or Segment/Prediction_Type columns
                
        Returns:
            Tuple containing (training_df, test_real_df, test_pred_df, future_df)
        """
        try:
            combined_df, prediction_start_date  = None, None
            
            if df is None:
                # Original logic for generating segments from scratch
                combined_df, prediction_start_date = self._merge_real_and_predicted_data(Config.FUTURE_STEPS)
                df = self._generate_indicators(combined_df)
                
                # Use Type column filtering
                training_df = df[
                    (df.index < self.test_index[0]) & 
                    (df['Type'] == 'Real')
                ].copy()
                
                test_real_df = df[
                    (df.index >= self.test_index[0]) & 
                    (df['Type'] == 'Real')
                ].copy()
                
                test_pred_df = df[
                    (df.index >= self.test_index[0]) & 
                    (df.index <= self.test_index[-1]) & 
                    (df['Type'] == 'Predicted')
                ].copy()
                
                future_df = df[
                    (df.index > self.test_index[-1]) & 
                    (df['Type'] == 'Predicted')
                ].copy()
                
            else:
                # Check which columns are available for segmentation
                if 'Segment' in df.columns and 'Prediction_Type' in df.columns:
                    # Use Segment and Prediction_Type columns
                    training_df = df[df['Segment'] == 'Training'].copy()
                    
                    test_real_df = df[
                        (df['Segment'] == 'Testing') & 
                        (df['Prediction_Type'] == 'Real')
                    ].copy()
                    
                    test_pred_df = df[
                        (df['Segment'] == 'Testing') & 
                        (df['Prediction_Type'] == 'Predicted')
                    ].copy()
                    
                    future_df = df[df['Segment'] == 'Future'].copy()
                    
                elif 'Type' in df.columns:
                    # Use Type column (original logic)
                    training_df = df[
                        (df.index < self.test_index[0]) & 
                        (df['Type'] == 'Real')
                    ].copy()
                    
                    test_real_df = df[
                        (df.index >= self.test_index[0]) & 
                        (df['Type'] == 'Real')
                    ].copy()
                    
                    test_pred_df = df[
                        (df.index >= self.test_index[0]) & 
                        (df.index <= self.test_index[-1]) & 
                        (df['Type'] == 'Predicted')
                    ].copy()
                    
                    future_df = df[
                        (df.index > self.test_index[-1]) & 
                        (df['Type'] == 'Predicted')
                    ].copy()
                else:
                    raise ValueError("DataFrame must have either 'Type' or 'Segment'/'Prediction_Type' columns")
            
            return training_df, test_real_df, test_pred_df, future_df, prediction_start_date
            
        except Exception as e:
            self.logger.error(f"Error building segmented data: {str(e)}")
            self.logger.exception("Stack trace:")
            raise RuntimeError(f"Failed to build segmented data: {str(e)}")

    def _add_segmented_comparison_traces(self, fig: go.Figure, 
                                    training_df: pd.DataFrame, 
                                    test_real_df: pd.DataFrame, 
                                    test_pred_df: pd.DataFrame,
                                    future_df: pd.DataFrame) -> None:
        """
        Add segmented comparison traces to the plot
        
        Args:
            fig: Plotly figure object
            training_df: Training data segment
            test_real_df: Real test data segment
            test_pred_df: Predicted test data segment
            future_df: Future predictions segment
        """
        try:
            # Add training data
            fig.add_trace(
                go.Scatter(
                    x=training_df.index,
                    y=training_df['Close'],
                    name='Training Data',
                    mode='lines',
                    line=dict(color='blue', width=1.5),
                    text=[f"Training<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(training_df.index, training_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            # Add training actual line
            fig.add_trace(
                go.Scatter(
                    x=training_df.index,
                    y=training_df['Close'],
                    name='Training (Actual)',
                    mode='lines',
                    line=dict(color='yellow', width=1),
                    visible='legendonly',
                    text=[f"Training (Actual)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(training_df.index, training_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add test data (real)
            fig.add_trace(
                go.Scatter(
                    x=test_real_df.index,
                    y=test_real_df['Close'],
                    name='Test Data (Real)',
                    mode='lines',
                    line=dict(color='green', width=1.5),
                    text=[f"Test (Real)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(test_real_df.index, test_real_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            # Add test actual line
            fig.add_trace(
                go.Scatter(
                    x=test_real_df.index,
                    y=test_real_df['Close'],
                    name='Test Real (Actual)',
                    mode='lines',
                    line=dict(color='yellow', width=1),
                    visible='legendonly',
                    text=[f"Test (Real Actual)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(test_real_df.index, test_real_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add test predictions
            fig.add_trace(
                go.Scatter(
                    x=test_pred_df.index,
                    y=test_pred_df['Close'],
                    name='Test Predictions',
                    mode='lines',
                    line=dict(color='orange', dash='dot', width=1.5),
                    opacity=0.7,
                    text=[f"Test (Predicted)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(test_pred_df.index, test_pred_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add future predictions if available
            if not future_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=future_df.index,
                        y=future_df['Close'],
                        name='Future Predictions',
                        mode='lines',
                        line=dict(color='lime', dash='dot', width=1.5),
                        opacity=0.7,
                        text=[f"Future Prediction<br>Date: {d}<br>Close: {c:.6f}"
                            for d, c in zip(future_df.index, future_df['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
                # Add future actual line
                fig.add_trace(
                    go.Scatter(
                        x=future_df.index,
                        y=future_df['Close'],
                        name='Future (Actual)',
                        mode='lines',
                        line=dict(color='yellow', width=1),
                        visible='legendonly',
                        text=[f"Future (Actual)<br>Date: {d}<br>Close: {c:.6f}"
                            for d, c in zip(future_df.index, future_df['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
            
        except Exception as e:
            self.logger.error(f"Error adding segmented comparison traces: {str(e)}")
            self.logger.exception("Stack trace:")
            raise RuntimeError(f"Failed to add segmented comparison traces: {str(e)}")

    def _add_segmented_candlestick_traces(self, fig: go.Figure, 
                                        training_df: pd.DataFrame, 
                                        test_real_df: pd.DataFrame, 
                                        test_pred_df: pd.DataFrame,
                                        future_df: pd.DataFrame) -> None:
        """Add segmented candlestick traces to the plot"""
        try:
            # Create hover text templates
            def create_hover_text(df, segment_name):
                return [f"{segment_name}<br>" +
                    f"Date: {date}<br>" +
                    f"Open: {open:.6f}<br>" +
                    f"High: {high:.6f}<br>" +
                    f"Low: {low:.6f}<br>" +
                    f"Close: {close:.6f}"
                    for date, open, high, low, close in zip(df.index, 
                        df['Open'], df['High'], df['Low'], df['Close'])]

            # Add training data candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=training_df.index,
                    open=training_df['Open'],
                    high=training_df['High'],
                    low=training_df['Low'],
                    close=training_df['Close'],
                    name='Training Data',
                    text=create_hover_text(training_df, 'Training'),
                    hoverinfo='text',
                    increasing_line_color='rgba(0,255,0,0.7)',
                    decreasing_line_color='rgba(255,0,0,0.7)'
                ), row=1, col=1
            )
            # Add training actual line
            fig.add_trace(
                go.Scatter(
                    x=training_df.index,
                    y=training_df['Close'],
                    name='Training (Actual)',
                    mode='lines',
                    line=dict(color='yellow', width=1),
                    visible='legendonly',
                    text=[f"Training (Actual)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(training_df.index, training_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            # Add training comparison line
            fig.add_trace(
                go.Scatter(
                    x=training_df.index,
                    y=training_df['Close'],
                    name='Training (Line)',
                    mode='lines',
                    line=dict(color='blue', width=1),
                    visible='legendonly',
                    text=[f"Training (Line)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(training_df.index, training_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add test data (real) candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=test_real_df.index,
                    open=test_real_df['Open'],
                    high=test_real_df['High'],
                    low=test_real_df['Low'],
                    close=test_real_df['Close'],
                    name='Test Data (Real)',
                    text=create_hover_text(test_real_df, 'Test (Real)'),
                    hoverinfo='text',
                    increasing_line_color='rgba(0,255,0,0.7)',
                    decreasing_line_color='rgba(255,0,0,0.7)'
                ), row=1, col=1
            )
            # Add test actual line
            fig.add_trace(
                go.Scatter(
                    x=test_real_df.index,
                    y=test_real_df['Close'],
                    name='Test Real (Actual)',
                    mode='lines',
                    line=dict(color='yellow', width=1),
                    visible='legendonly',
                    text=[f"Test Real (Actual)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(test_real_df.index, test_real_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            # Add test comparison line
            fig.add_trace(
                go.Scatter(
                    x=test_real_df.index,
                    y=test_real_df['Close'],
                    name='Test Real (Line)',
                    mode='lines',
                    line=dict(color='green', width=1),
                    visible='legendonly',
                    text=[f"Test Real (Line)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(test_real_df.index, test_real_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add test data (predicted) candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=test_pred_df.index,
                    open=test_pred_df['Open'],
                    high=test_pred_df['High'],
                    low=test_pred_df['Low'],
                    close=test_pred_df['Close'],
                    name='Test Data (Pred)',
                    text=create_hover_text(test_pred_df, 'Test (Predicted)'),
                    hoverinfo='text',
                    increasing_line_color='rgba(0,255,0,0.4)',
                    decreasing_line_color='rgba(255,0,0,0.4)'
                ), row=1, col=1
            )
            # Add test predictions line
            fig.add_trace(
                go.Scatter(
                    x=test_pred_df.index,
                    y=test_pred_df['Close'],
                    name='Test Pred (Line)',
                    mode='lines',
                    line=dict(color='orange', width=1, dash='dot'),
                    visible='legendonly',
                    text=[f"Test Pred (Line)<br>Date: {d}<br>Close: {c:.6f}"
                        for d, c in zip(test_pred_df.index, test_pred_df['Close'])],
                    hoverinfo='text'
                ), row=1, col=1
            )
            
            # Add future predictions if available
            if not future_df.empty:
                fig.add_trace(
                    go.Candlestick(
                        x=future_df.index,
                        open=future_df['Open'],
                        high=future_df['High'],
                        low=future_df['Low'],
                        close=future_df['Close'],
                        name='Future Predictions',
                        text=create_hover_text(future_df, 'Future Prediction'),
                        hoverinfo='text',
                        increasing_line_color='rgba(0,255,0,0.7)',
                        decreasing_line_color='rgba(255,0,0,0.7)'
                    ), row=1, col=1
                )
                # Add future actual line
                fig.add_trace(
                    go.Scatter(
                        x=future_df.index,
                        y=future_df['Close'],
                        name='Future (Actual)',
                        mode='lines',
                        line=dict(color='yellow', width=1),
                        visible='legendonly',
                        text=[f"Future (Actual)<br>Date: {d}<br>Close: {c:.6f}"
                            for d, c in zip(future_df.index, future_df['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
                # Add future predictions line
                fig.add_trace(
                    go.Scatter(
                        x=future_df.index,
                        y=future_df['Close'],
                        name='Future (Line)',
                        mode='lines',
                        line=dict(color='lime', width=1, dash='dot'),
                        visible='legendonly',
                        text=[f"Future (Line)<br>Date: {d}<br>Close: {c:.6f}"
                            for d, c in zip(future_df.index, future_df['Close'])],
                        hoverinfo='text'
                    ), row=1, col=1
                )
                
        except Exception as e:
            self.logger.error(f"Error adding segmented candlestick traces: {str(e)}")
            self.logger.exception("Stack trace:")
            raise RuntimeError(f"Failed to add segmented candlestick traces: {str(e)}")
    
    def _available_columns(self, df_with_indicators, prediction_start_date, training_df, test_real_df, test_pred_df, future_df):
        """Debug helper to log available columns and data samples"""
        # Debug logging for available columns
        self.logger.info("\n=== Debug: Available Columns ===")
        self.logger.info("self.df columns:")
        self.logger.info(list(self.df.columns))
        self.logger.info("df_with_indicators columns:")
        self.logger.info(list(df_with_indicators.columns))
        self.logger.info(f"prediction_start_date: {prediction_start_date}")  # Fixed logging format
        self.logger.info("training_df columns:")
        self.logger.info(list(training_df.columns))
        self.logger.info("test_real_df columns:")
        self.logger.info(list(test_real_df.columns))
        self.logger.info("test_pred_df columns:")
        self.logger.info(list(test_pred_df.columns))
        self.logger.info("future_df columns:")
        self.logger.info(list(future_df.columns))

        # Debug sample data
        self.logger.info("=== Debug: Sample Data ===")
        if 'Marketcap' in self.df.columns:  # Changed from Market_Cap to Marketcap
            self.logger.info("self.df Marketcap sample:")
            self.logger.info(self.df['Marketcap'].head())
        if 'Marketcap' in df_with_indicators.columns:  # Changed from Market_Cap to Marketcap
            self.logger.info("df_with_indicators Marketcap sample:")
            self.logger.info(df_with_indicators['Marketcap'].head())
        self.logger.info("\n")

    def plot_predictions(self, plot_type: str = 'advanced', future_days: Optional[int] = None) -> None:
        """
        Create interactive plot with indicators and toggles
        
        Args:
            plot_type: Type of plot ('advanced' or 'comparison')
            future_days: Number of days to predict into the future
        """

        try:
            print(f"Creating {plot_type} prediction plot...")
            
            if future_days is None:
                future_days = self.config.FUTURE_STEPS

            df_with_indicators, prediction_start_date, training_df, test_real_df, test_pred_df, future_df = self._get_processed_data()
            
            # Log the prediction periods
            self.logger.info(f"Test set size: {len(self.test_predictions)} days")
            self.logger.info(f"Future predictions: {future_days} days")

            self._available_columns(df_with_indicators, prediction_start_date, training_df, test_real_df, test_pred_df, future_df)

            # Create base plot
            fig = self._create_plot_base(plot_type)
                        
            # Build segmented data using the already processed df_with_indicators
            #training_df, test_real_df, test_pred_df, future_df, prediction_sd = self._build_segmented_data(df_with_indicators)
            
            # Add traces based on plot type
            if plot_type == 'comparison':
                self._add_comparison_traces(fig, df_with_indicators)
                self._add_segmented_comparison_traces(fig, training_df, test_real_df, test_pred_df, future_df)
            else:
                self._add_advanced_traces(fig, df_with_indicators)
                self._add_segmented_candlestick_traces(fig, training_df, test_real_df, test_pred_df, future_df)
            
            # Add common elements
            self._add_technical_indicator_traces(fig, df_with_indicators)
            self._add_prediction_markers(fig, prediction_start_date, df_with_indicators)
            
            # Add metrics box
            # Defalt: x = 1.02, y = 0.98
            #self._add_metrics_box(fig, 1.145, 0.2)
            self._add_metrics_box_horizontal_v2(fig, metrics_type='default', y=1.05)
            
            self._update_plot_layout(fig, plot_type)
            
            # Save and display
            try:
                self.logger.exception("Try saving prediction_plot as figure:")
                saved_plot = self._save_plot(f'{plot_type}_prediction_plot', {
                    'figure': fig,
                    'data': df_with_indicators.to_dict(),
                    'metadata': {
                        'prediction_start': prediction_start_date.strftime('%Y-%m-%d'),
                        'ticker': self.config.TICKER_NAME,
                        'future_days': future_days
                }
                })
            
                if saved_plot and 'plot' in saved_plot:
                    print(f"\nPlot saved successfully at: {saved_plot['plot']}")
                else:
                    print("\nWarning: Plot may not have saved correctly")
            
            except Exception as save_error:
                self.logger.error(f"plot_predictions: Failed to save save_plot: {str(save_error)}")
                self.logger.exception("Save error details:")
            
            try:
                self._save_plot_data("combined_predictions", {
                    'df_with_indicators': df_with_indicators,
                    'training_df': training_df,
                    'test_real_df': test_real_df,
                    'test_pred_df': test_pred_df,
                    'future_df': future_df,
                    'prediction_start_date': prediction_start_date,
                    'metadata': {
                        'prediction_start': prediction_start_date.strftime('%Y-%m-%d'),
                        'ticker': self.config.TICKER_NAME,
                        'future_days': future_days
                    }
                })
            except Exception as save_error:
                self.logger.error(f"plot_predictions: Failed to save combined_predictions: {str(save_error)}")
                self.logger.exception("Save error details:")
            
            print(f"Displaying interactive {plot_type} plot...")

            # Try to show the plot, but handle failure gracefully
            try:
                self.logger.info("Attempting to display interactive plot...")
                fig.show()
                self.logger.info("Plot displayed successfully")
            except Exception as show_error:
                self.logger.warning(f"Could not display interactive plot (browser connection issue): {str(show_error)}")
            
        except Exception as e:
            error_msg = f"Failed to create {plot_type} prediction plot: {str(e)}"
            self.logger.error(error_msg)
            self.logger.exception("Stack trace:")
            raise RuntimeError(error_msg)
        
    def save_predictions(self, filename: Optional[str] = None) -> None:
        """Save predictions and combined plot data to files
        
        Args:
            filename: Optional custom filename. If None, generates default name.
        """
        try:
            # Get paths with BasicStockModel subdirectory
            root_dir = get_script_based_dir()
            predictions_dir = os.path.join(root_dir, 'data', 'BasicStockModel', 'predictions')
            os.makedirs(predictions_dir, exist_ok=True)

            # Generate filename if not provided
            if filename is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.config.TICKER_NAME}_predictions_{timestamp}"

            base_filepath = os.path.join(predictions_dir, filename)

            # Log initial array lengths for debugging
            self.logger.debug("Initial array lengths:")
            self.logger.debug(f"Train size: {self.train_size}")
            self.logger.debug(f"Train prices rescaled: {len(self.train_prices_rescaled)}")
            self.logger.debug(f"Train predictions rescaled: {len(self.train_predictions_rescaled)}")
            self.logger.debug(f"Test prices rescaled: {len(self.test_prices_rescaled)}")
            self.logger.debug(f"Test predictions rescaled: {len(self.test_predictions_rescaled)}")

            # 1. Save Basic Predictions Data with length validation
            train_dates = self.df.index[:self.train_size]
            test_dates = self.test_index if self.test_index is not None else self.df.index[self.train_size:]

            # Validate and adjust lengths for train data
            min_train_length = min(
                len(train_dates),
                len(self.train_prices_rescaled),
                len(self.train_predictions_rescaled)
            )
            train_dates = train_dates[:min_train_length]
            train_prices = self.train_prices_rescaled[:min_train_length]
            train_preds = self.train_predictions_rescaled[:min_train_length]

            # Validate and adjust lengths for test data
            min_test_length = min(
                len(test_dates),
                len(self.test_prices_rescaled),
                len(self.test_predictions_rescaled)
            )
            test_dates = test_dates[:min_test_length]
            test_prices = self.test_prices_rescaled[:min_test_length]
            test_preds = self.test_predictions_rescaled[:min_test_length]

            # Create DataFrames with validated lengths
            train_data = pd.DataFrame({
                'date': train_dates,
                'actual': train_prices,
                'predicted': train_preds,
                'segment': ['train'] * min_train_length
            })

            test_data = pd.DataFrame({
                'date': test_dates,
                'actual': test_prices,
                'predicted': test_preds,
                'segment': ['test'] * min_test_length
            })

            # Combine basic predictions data
            predictions_df = pd.concat([train_data, test_data])
            
            # Add metadata columns
            predictions_df['model_version'] = self.config.SCRIPT_VERSION
            predictions_df['ticker'] = self.config.TICKER_NAME
            predictions_df['timestamp'] = pd.Timestamp.now()

            # 2. Get Combined Plot Data (if available)
            try:
                # Get raw combined data first
                combined_df_raw, prediction_start_date = self._merge_real_and_predicted_data(Config.FUTURE_STEPS)
                
                # Create copy for indicators
                combined_df_with_indicators = self._generate_indicators(combined_df_raw.copy())
                
            except Exception as e:
                self.logger.warning(f"Could not generate combined plot data: {str(e)}")
                combined_df_raw = None
                combined_df_with_indicators = None
                prediction_start_date = None

            # 3. Save All Data Files
            
            # Save basic predictions
            predictions_filepath = f"{base_filepath}_basic.csv"
            predictions_df.to_csv(predictions_filepath, index=False)
            self.logger.info(f"Saved basic predictions to: {predictions_filepath}")

            # Save basic predictions pickle
            pickle_path = predictions_filepath.replace('.csv', '.pkl')
            predictions_df.to_pickle(pickle_path)
            self.logger.info(f"Saved basic predictions pickle to: {pickle_path}")

            # Save raw combined data if available
            if combined_df_raw is not None:
                raw_combined_filepath = f"{base_filepath}_raw_combined.csv"
                combined_df_raw.to_csv(raw_combined_filepath)
                raw_combined_pickle_path = raw_combined_filepath.replace('.csv', '.pkl')
                combined_df_raw.to_pickle(raw_combined_pickle_path)
                self.logger.info(f"Saved raw combined data to: {raw_combined_filepath}")

            # Save combined data with indicators if available
            if combined_df_with_indicators is not None:
                combined_filepath = f"{base_filepath}_combined.csv"
                combined_df_with_indicators.to_csv(combined_filepath)
                combined_pickle_path = combined_filepath.replace('.csv', '.pkl')
                combined_df_with_indicators.to_pickle(combined_pickle_path)
                self.logger.info(f"Saved combined data with indicators to: {combined_filepath}")

            # 4. Save Configuration and Metadata
            config_path = f"{base_filepath}_config.json"
            config_data = {
                'model_config': {
                    'hidden_layer_size': self.hidden_layer_size,
                    'dropout_prob': self.dropout_prob,
                    'use_attention': self.use_attention,
                    'time_step': self.time_step,
                    'batch_size': self.batch_size,
                    'learning_rate': self.lr
                },
                'metrics': self.evaluate(),
                'data_info': {
                    'basic_predictions': {
                        'train_size': len(train_data),
                        'test_size': len(test_data),
                        'date_range': f"{train_dates.min()} to {test_dates.max()}",
                        'original_lengths': {
                            'train': {
                                'dates': len(self.df.index[:self.train_size]),
                                'prices': len(self.train_prices_rescaled),
                                'predictions': len(self.train_predictions_rescaled)
                            },
                            'test': {
                                'dates': len(test_dates),
                                'prices': len(self.test_prices_rescaled),
                                'predictions': len(self.test_predictions_rescaled)
                            }
                        },
                        'adjusted_lengths': {
                            'train': min_train_length,
                            'test': min_test_length
                        }
                    },
                    'raw_combined_data': {
                        'available': combined_df_raw is not None,
                        'size': len(combined_df_raw) if combined_df_raw is not None else 0
                    },
                    'combined_data_with_indicators': {
                        'available': combined_df_with_indicators is not None,
                        'prediction_start': prediction_start_date.strftime('%Y-%m-%d') if prediction_start_date else None,
                        'size': len(combined_df_with_indicators) if combined_df_with_indicators is not None else 0
                    }
                },
                'files': {
                    'basic_predictions': os.path.basename(predictions_filepath),
                    'raw_combined': os.path.basename(raw_combined_filepath) if combined_df_raw is not None else None,
                    'combined_with_indicators': os.path.basename(combined_filepath) if combined_df_with_indicators is not None else None
                },
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_version': self.config.SCRIPT_VERSION,
                'ticker': self.config.TICKER_NAME
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.logger.info(f"Saved configuration to: {config_path}")

            # Log summary of saved files
            self.logger.info("\nSaved files summary:")
            self.logger.info(f"- Basic predictions: {os.path.basename(predictions_filepath)}")
            if combined_df_raw is not None:
                self.logger.info(f"- Raw combined data: {os.path.basename(raw_combined_filepath)}")
            if combined_df_with_indicators is not None:
                self.logger.info(f"- Combined data with indicators: {os.path.basename(combined_filepath)}")
            self.logger.info(f"- Configuration: {os.path.basename(config_path)}")

        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            self.logger.exception("Stack trace:")
            raise
        
    def load_predictions(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load saved predictions from file
        
        Args:
            filename: Optional filename to load. If None, shows available files for selection.
                Can be either .csv or .pkl file.
                
        Returns:
            pd.DataFrame: Loaded predictions with associated configuration
        """
        try:
            # Get predictions directory path
            root_dir = get_script_based_dir()
            predictions_dir = os.path.join(root_dir, 'data', 'BasicStockModel', 'predictions')
            
            if not os.path.exists(predictions_dir):
                raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
                
            # If no filename provided, show selection menu
            if filename is None:
                prediction_files = [f for f in os.listdir(predictions_dir) 
                                if f.endswith(('.csv', '.pkl'))]
                
                if not prediction_files:
                    raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
                    
                print("\nAvailable prediction files:")
                for i, file in enumerate(prediction_files, 1):
                    # Get file metadata
                    filepath = os.path.join(predictions_dir, file)
                    mod_time = pd.Timestamp.fromtimestamp(os.path.getmtime(filepath))
                    file_size = os.path.getsize(filepath) / 1024  # Size in KB
                    
                    # Check if config file exists
                    config_path = filepath.replace('.csv', '_config.json').replace('.pkl', '_config.json')
                    has_config = os.path.exists(config_path)
                    
                    print(f"\n{i}. {file}")
                    print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Size: {file_size:.2f} KB")
                    print(f"   Has config: {'Yes' if has_config else 'No'}")
                    
                while True:
                    try:
                        choice = int(input(f"\nSelect file (1-{len(prediction_files)}) or 0 to cancel: "))
                        if 0 <= choice <= len(prediction_files):
                            break
                        print(f"Please enter a number between 0 and {len(prediction_files)}")
                    except ValueError:
                        print("Please enter a valid number")
                        
                if choice == 0:
                    return None
                    
                filename = prediction_files[choice - 1]
                
            # Construct full filepath
            filepath = os.path.join(predictions_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
                
            # Load predictions
            if filepath.endswith('.pkl'):
                predictions_df = pd.read_pickle(filepath)
            else:
                predictions_df = pd.read_csv(filepath)
                predictions_df['date'] = pd.to_datetime(predictions_df['date'])
                
            # Try to load associated configuration
            config_path = filepath.replace('.csv', '_config.json').replace('.pkl', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                self.logger.info("Loaded associated configuration file")
                
                # Print key metrics if available
                if 'metrics' in config_data:
                    print("\nModel Metrics:")
                    for metric, value in config_data['metrics'].items():
                        print(f"{metric}: {value:.6f}")
                        
                # Print model configuration
                if 'model_config' in config_data:
                    print("\nModel Configuration:")
                    for param, value in config_data['model_config'].items():
                        print(f"{param}: {value}")
            
            self.logger.info(f"Loaded predictions from: {filepath}")
            print(f"\nLoaded {len(predictions_df)} predictions")
            print(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
            
            return predictions_df

        except Exception as e:
            self.logger.error(f"Error loading predictions: {str(e)}")
            raise
        
    # ============== Advanced Predicts - END ============== #
    
    # ============== Extra for Crypto - START ============== #
    
    def validate_crypto_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Validate and adjust crypto predictions based on market cap constraints
        
        Args:
            predictions (np.ndarray): Raw model predictions
            
        Returns:
            np.ndarray: Validated predictions
        """
        try:
            if not self.is_crypto:
                return predictions
                
            validated_predictions = predictions.copy()
            
            if 'Market_Cap' in self.df.columns:
                # Get latest market cap
                latest_mc = self.df['Market_Cap'].iloc[-1]
                
                # Calculate reasonable price bounds based on market cap
                max_price_change = 0.20  # 20% max change per prediction
                min_price = self.df['Close'].iloc[-1] * (1 - max_price_change)
                max_price = self.df['Close'].iloc[-1] * (1 + max_price_change)
                
                # Adjust predictions outside bounds
                validated_predictions = np.clip(validated_predictions, min_price, max_price)
                
                # Log adjustments
                adjustments_made = (predictions != validated_predictions).sum()
                if adjustments_made > 0:
                    self.logger.warning(f"Adjusted {adjustments_made} predictions to maintain realistic bounds")
            
            return validated_predictions
            
        except Exception as e:
            self.logger.error(f"Error validating crypto predictions: {str(e)}")
            return predictions

    def handle_missing_market_cap(self) -> None:
        """Handle missing market cap data for crypto assets"""
        try:
            if not self.is_crypto or 'Market_Cap' not in self.df.columns:
                return
                
            # Check for missing values
            missing_mc = self.df['Market_Cap'].isnull().sum()
            if missing_mc > 0:
                self.logger.warning(f"Found {missing_mc} missing market cap values")
                
                # Interpolate missing values
                self.df['Market_Cap'] = self.df['Market_Cap'].interpolate(method='time')
                
                # For any remaining missing values at the start/end
                self.df['Market_Cap'] = self.df['Market_Cap'].ffill().bfill()
                
                self.logger.info("Missing market cap values have been interpolated")
                
        except Exception as e:
            self.logger.error(f"Error handling missing market cap data: {str(e)}")
            raise

    def plot_crypto_analysis(self, save: bool = True) -> None:
        """
        Generate crypto-specific visualizations
        
        Args:
            save (bool): Whether to save the plots to disk
        """
        try:
            if not self.is_crypto:
                return
                    
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Price vs Predictions', 'Market Cap Analysis', 'Price-MC Correlation'),
                vertical_spacing=0.1,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Price predictions plot
            fig.add_trace(
                go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                        y=self.test_prices_rescaled,
                        name='Actual Price',
                        line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                        y=self.test_predictions_rescaled,
                        name='Predicted Price',
                        line=dict(color='red')),
                row=1, col=1
            )
            
            if 'Marketcap' in self.df.columns:  # Changed from Market_Cap to Marketcap
                # Convert Market Cap to billions for better readability
                market_cap_billions = self.df['Marketcap'].astype(float) / 1e9
                
                # Market Cap analysis
                fig.add_trace(
                    go.Bar(x=self.df.index,
                            y=market_cap_billions,
                            name='Market Cap',
                            marker_color='rgba(0, 150, 255, 0.6)'),
                    row=2, col=1
                )
                
                # Convert dates to numerical values for coloring
                date_nums = np.arange(len(self.df.index))
                
                # Price-MC correlation with horizontal colorbar at top
                fig.add_trace(
                    go.Scatter(x=market_cap_billions,
                            y=self.df['Close'],
                            mode='markers',
                            name='Price-MC Correlation',
                            marker=dict(
                                size=8,
                                color=date_nums,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title='Time',
                                    titleside='top',
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.1,
                                    x=0.5,
                                    xanchor='center',
                                    len=0.5,
                                    ticktext=[self.df.index[0].strftime('%Y-%m-%d'), 
                                            self.df.index[-1].strftime('%Y-%m-%d')],
                                    tickvals=[date_nums[0], date_nums[-1]]
                                )
                            )),
                    row=3, col=1
                )
            
            # Add Bollinger Bands to price plot
            if 'Bollinger_Upper' in self.df.columns and 'Bollinger_Lower' in self.df.columns:
                fig.add_trace(
                    go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                            y=self.df['Bollinger_Upper'][-len(self.test_predictions_rescaled):],
                            name='Upper Bollinger',
                            line=dict(color='gray', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                            y=self.df['Bollinger_Lower'][-len(self.test_predictions_rescaled):],
                            name='Lower Bollinger',
                            line=dict(color='gray', dash='dash')),
                    row=1, col=1
                )
            
            # Update layout with more space at top for metrics
            fig.update_layout(
                height=1200,
                margin=dict(t=150),
                title=dict(
                    text=f"Crypto Analysis for {self.config.TICKER_NAME}",
                    y=0.98,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                showlegend=True,
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Market Cap", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=3, col=1)
            fig.update_xaxes(title_text="Market Cap", row=3, col=1)
            
            # Add horizontal metrics box at the top
            self._add_metrics_box_horizontal_v2(fig, metrics_type='default', y=1.05)
            
            # Save if requested
            if save:
                filename = f"crypto_analysis_{self.config.TICKER_NAME}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
                filepath = os.path.join(self.charts_dir, filename)
                fig.write_html(filepath)
                self.logger.info(f"Saved crypto analysis plot to: {filepath}")
            
            # Show plot
            fig.show()
            
        except Exception as e:
            self.logger.error(f"Error generating crypto analysis plots: {str(e)}")
            raise

    def plot_stock_analysis(self, save: bool = True) -> None:
        """
        Generate stock-specific visualizations
        
        Args:
            save (bool): Whether to save the plots to disk
        """
        try:
            if self.is_crypto:
                return
                
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Price vs Predictions', 'Volume Analysis', 'Price-Volume Correlation'),
                vertical_spacing=0.1,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Price predictions plot
            fig.add_trace(
                go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                        y=self.test_prices_rescaled,
                        name='Actual Price',
                        line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                        y=self.test_predictions_rescaled,
                        name='Predicted Price',
                        line=dict(color='red')),
                row=1, col=1
            )
            
            if 'Volume' in self.df.columns:
                # Volume analysis
                fig.add_trace(
                    go.Bar(x=self.df.index,
                            y=self.df['Volume'],
                            name='Trading Volume',
                            marker_color='rgba(0, 150, 255, 0.6)'),
                    row=2, col=1
                )
                
                # Convert dates to numerical values for coloring
                date_nums = np.arange(len(self.df.index))
                
                # Price-Volume correlation with horizontal colorbar at top
                fig.add_trace(
                    go.Scatter(x=self.df['Volume'],
                            y=self.df['Close'],
                            mode='markers',
                            name='Price-Volume Correlation',
                            marker=dict(
                                size=8,
                                color=date_nums,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title='Time',
                                    titleside='top',
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.1,
                                    x=0.5,
                                    xanchor='center',
                                    len=0.5,
                                    ticktext=[self.df.index[0].strftime('%Y-%m-%d'), 
                                            self.df.index[-1].strftime('%Y-%m-%d')],
                                    tickvals=[date_nums[0], date_nums[-1]]
                                )
                            )),
                    row=3, col=1
                )
            
            # Add Bollinger Bands to price plot
            if 'Bollinger_Upper' in self.df.columns and 'Bollinger_Lower' in self.df.columns:
                fig.add_trace(
                    go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                            y=self.df['Bollinger_Upper'][-len(self.test_predictions_rescaled):],
                            name='Upper Bollinger',
                            line=dict(color='gray', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=self.df.index[-len(self.test_predictions_rescaled):],
                            y=self.df['Bollinger_Lower'][-len(self.test_predictions_rescaled):],
                            name='Lower Bollinger',
                            line=dict(color='gray', dash='dash')),
                    row=1, col=1
                )
            
            # Update layout with more space at top for metrics
            fig.update_layout(
                height=1200,
                margin=dict(t=150),  # Increase top margin for metrics box
                title=dict(
                    text=f"Stock Analysis for {self.config.TICKER_NAME}",
                    y=0.98,
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                showlegend=True,
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=3, col=1)
            fig.update_xaxes(title_text="Volume", row=3, col=1)
            
            # Add horizontal metrics box at the top
            self._add_metrics_box_horizontal_v2(fig, metrics_type='default', y=1.05)
            
            # Save if requested
            if save:
                filename = f"stock_analysis_{self.config.TICKER_NAME}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
                filepath = os.path.join(self.charts_dir, filename)
                fig.write_html(filepath)
                self.logger.info(f"Saved stock analysis plot to: {filepath}")
            
            # Show plot
            fig.show()
            
        except Exception as e:
            self.logger.error(f"Error generating stock analysis plots: {str(e)}")
            raise
    
    # ============== Extra for Crypto - END ============== #
    
    # ============== Extra Plotting Metrix - START ============== #
    
    def plot_residuals(self,
                       train_prices_rescaled: Optional[np.ndarray] = None,
                       train_predictions_rescaled: Optional[np.ndarray] = None,
                       test_prices_rescaled: Optional[np.ndarray] = None,
                       test_predictions_rescaled: Optional[np.ndarray] = None) -> None:
        """Plot residuals analysis using Plotly

        Args:
            df_with_indicators (pd.DataFrame): Full dataset with indicators
            prediction_start_date (pd.Timestamp): Start date of predictions
            training_df (pd.DataFrame): Training data with predictions
            test_real_df (pd.DataFrame): Test data with actual values
            test_pred_df (pd.DataFrame): Test data with predictions
            future_df (pd.DataFrame): Future predictions
        """

        plot_type = "plot_residuals"

        try:
            print("Creating interactive residuals plot...")
            
            # Calculate residuals
            try:
                residuals_train = train_prices_rescaled - train_predictions_rescaled
                residuals_test = test_prices_rescaled - test_predictions_rescaled

                print("Residuals calculated successfully")
            except Exception as e:
                print(f"Error calculating residuals: {str(e)}")
                raise

            # Create interactive plot
            try:
                # Create continuous date range for both train and test
                train_dates = pd.date_range(
                    start=self.df.index[0],
                    periods=len(residuals_train),
                    freq=self.df.index.freq or pd.infer_freq(self.df.index)
                )
                
                test_dates = pd.date_range(
                    start=train_dates[-1] + pd.Timedelta(days=1),
                    periods=len(residuals_test),
                    freq=self.df.index.freq or pd.infer_freq(self.df.index)
                )

                # Create figure with secondary y-axis
                fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=('Residuals Over Time', 'Residuals Distribution'),
                                vertical_spacing=0.2,
                                row_heights=[0.7, 0.3])

                # Add residuals traces
                fig.add_trace(
                    go.Scatter(
                        x=train_dates,
                        y=residuals_train,
                        name='Train Residuals',
                        line=dict(color='purple', width=1.5),
                        hovertemplate="Date: %{x}<br>" +
                                    "Residual: %{y:.6f}<extra></extra>"
                    ), row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=test_dates,
                        y=residuals_test,
                        name='Test Residuals',
                        line=dict(color='orange', width=1.5),
                        hovertemplate="Date: %{x}<br>" +
                                    "Residual: %{y:.6f}<extra></extra>"
                    ), row=1, col=1
                )

                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)

                # Add distribution plots
                fig.add_trace(
                    go.Histogram(
                        x=residuals_train,
                        name='Train Distribution',
                        nbinsx=30,
                        marker_color='purple',
                        opacity=0.7,
                        hovertemplate="Value: %{x:.6f}<br>" +
                                    "Count: %{y}<extra></extra>"
                    ), row=2, col=1
                )

                fig.add_trace(
                    go.Histogram(
                        x=residuals_test,
                        name='Test Distribution',
                        nbinsx=30,
                        marker_color='orange',
                        opacity=0.7,
                        hovertemplate="Value: %{x:.6f}<br>" +
                                    "Count: %{y}<extra></extra>"
                    ), row=2, col=1
                )

                # Calculate statistics for annotation
                stats = {
                    'train_mean': float(np.mean(residuals_train)),
                    'train_std': float(np.std(residuals_train)),
                    'test_mean': float(np.mean(residuals_test)),
                    'test_std': float(np.std(residuals_test)),
                    'train_skew': float(pd.Series(residuals_train).skew()),
                    'test_skew': float(pd.Series(residuals_test).skew()),
                    'train_kurtosis': float(pd.Series(residuals_train).kurtosis()),
                    'test_kurtosis': float(pd.Series(residuals_test).kurtosis())
                }

                # Add statistics annotation
                stats_text = (
                    f"Train Statistics:<br>" +
                    f"Mean: {stats['train_mean']:.6f}<br>" +
                    f"Std: {stats['train_std']:.6f}<br>" +
                    f"Skew: {stats['train_skew']:.6f}<br>" +
                    f"Kurtosis: {stats['train_kurtosis']:.6f}<br><br>" +
                    f"Test Statistics:<br>" +
                    f"Mean: {stats['test_mean']:.6f}<br>" +
                    f"Std: {stats['test_std']:.6f}<br>" +
                    f"Skew: {stats['test_skew']:.6f}<br>" +
                    f"Kurtosis: {stats['test_kurtosis']:.6f}"
                )

                # Add horizontal metrics box
                try:
                    self._add_metrics_box_horizontal_v2(fig, metrics_type='residuals', y=1.03)
                except Exception as e:
                    print(f"Warning: Error adding metrics box: {str(e)}")

                # Add statistics annotation
                fig.add_annotation(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=1.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )

                # Update layout
                fig.update_layout(
                    title=f"{Config.TICKER_NAME} Residuals Analysis",
                    template='plotly_dark',
                    showlegend=True,
                    height=800,
                    width=1200,
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )

                # Save and display
                try:
                    self._save_plot(f'{plot_type}', {
                        'figure': fig
                    })
                except Exception as save_error:
                    self.logger.error(f"plot_residuals: Failed to save save_plot: {str(save_error)}")
                    self.logger.exception("Save error details:")

                # Save plot data
                plot_data = {
                    'train_dates': self.df.index[:self.train_size].tolist(),
                    'test_dates': self.test_index.tolist(),
                    'train_residuals': residuals_train.tolist(),
                    'test_residuals': residuals_test.tolist(),
                    'metadata': stats
                }
                self._save_plot_data('residuals', plot_data)
                print("Plot data saved successfully")

                # Show the plot
                fig.show()
                print("Interactive plot displayed successfully")

            except Exception as e:
                print(f"Error creating plot: {str(e)}")
                raise

        except Exception as e:
            print(f"Error in plot_residuals: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_residuals_diagnostics(self,
                            train_prices_rescaled: Optional[np.ndarray] = None,
                            train_predictions_rescaled: Optional[np.ndarray] = None,
                            test_prices_rescaled: Optional[np.ndarray] = None,
                            test_predictions_rescaled: Optional[np.ndarray] = None) -> None:
        """Create interactive residuals diagnostics plots using Plotly
        
        Args:
            train_prices_rescaled: Optional array of rescaled training prices
            train_predictions_rescaled: Optional array of rescaled training predictions
            test_prices_rescaled: Optional array of rescaled test prices
            test_predictions_rescaled: Optional array of rescaled test predictions
        """
        try:
            print("Creating residuals diagnostics plot...")
            
            # Use provided arrays or fall back to class attributes
            train_prices = train_prices_rescaled if train_prices_rescaled is not None else self.train_prices_rescaled
            train_preds = train_predictions_rescaled if train_predictions_rescaled is not None else self.train_predictions_rescaled
            test_prices = test_prices_rescaled if test_prices_rescaled is not None else self.test_prices_rescaled
            test_preds = test_predictions_rescaled if test_predictions_rescaled is not None else self.test_predictions_rescaled

            # Ensure arrays are 1D numpy arrays
            train_prices = np.asarray(train_prices).flatten()
            train_preds = np.asarray(train_preds).flatten()
            test_prices = np.asarray(test_prices).flatten()
            test_preds = np.asarray(test_preds).flatten()

            # Calculate residuals
            residuals_train = train_prices - train_preds
            residuals_test = test_prices - test_preds

            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Train Residuals Distribution', 'Test Residuals Distribution',
                    'Train Residuals ACF', 'Test Residuals ACF',
                    'Train Residuals Over Time', 'Test Residuals Over Time'
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # Add metrics box at the top
            try:
                self._add_metrics_box_horizontal_v2(
                    fig,
                    metrics_type='residuals_diagnostics',
                    residuals_train=residuals_train,
                    residuals_test=residuals_test,
                    y=1.02
                )
            except Exception as e:
                logger.warning(f"plot_residuals_diagnostics: Could not add metrics box: {str(e)}")

            # Add distribution plots
            fig.add_trace(
                go.Histogram(
                    x=residuals_train,
                    name='Train Distribution',
                    nbinsx=30,
                    marker_color='purple',
                    opacity=0.7,
                    hovertemplate="Value: %{x:.6f}<br>Count: %{y}<extra></extra>"
                ), row=1, col=1
            )

            fig.add_trace(
                go.Histogram(
                    x=residuals_test,
                    name='Test Distribution',
                    nbinsx=30,
                    marker_color='orange',
                    opacity=0.7,
                    hovertemplate="Value: %{x:.6f}<br>Count: %{y}<extra></extra>"
                ), row=1, col=2
            )

            # Calculate and add ACF plots
            def add_acf_trace(residuals: np.ndarray, row: int, col: int, name: str, color: str):
                # Calculate ACF manually
                n = len(residuals)
                mean = np.mean(residuals)
                var = np.var(residuals)
                acf_values = []
                
                for lag in range(31):  # 0 to 30 lags
                    if lag == 0:
                        acf_values.append(1.0)
                    else:
                        sum_product = 0
                        for i in range(n - lag):
                            sum_product += (residuals[i] - mean) * (residuals[i + lag] - mean)
                        acf_values.append(sum_product / ((n - lag) * var))
                
                lags = list(range(len(acf_values)))
                
                # Add confidence intervals
                ci = 1.96/np.sqrt(n)
                
                # Add ACF bars
                fig.add_trace(
                    go.Bar(
                        x=lags,
                        y=acf_values,
                        name=f'{name} ACF',
                        marker_color=color,
                        opacity=0.7,
                        hovertemplate="Lag: %{x}<br>ACF: %{y:.3f}<extra></extra>"
                    ), row=row, col=col
                )
                
                # Add confidence interval lines
                fig.add_hline(y=ci, line_dash="dash", line_color="red", row=row, col=col)
                fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=row, col=col)

            add_acf_trace(residuals_train, 2, 1, 'Train', 'purple')
            add_acf_trace(residuals_test, 2, 2, 'Test', 'orange')

            # Add time series plots
            fig.add_trace(
                go.Scatter(
                    y=residuals_train,
                    mode='lines',
                    name='Train Residuals',
                    line=dict(color='purple'),
                    hovertemplate="Index: %{x}<br>Residual: %{y:.6f}<extra></extra>"
                ), row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    y=residuals_test,
                    mode='lines',
                    name='Test Residuals',
                    line=dict(color='orange'),
                    hovertemplate="Index: %{x}<br>Residual: %{y:.6f}<extra></extra>"
                ), row=3, col=2
            )

            # Calculate statistics
            train_stats = pd.Series(residuals_train).describe()
            test_stats = pd.Series(residuals_test).describe()
            
            stats_text = (
                f"Train Statistics:<br>" +
                f"Mean: {train_stats['mean']:.6f}<br>" +
                f"Std: {train_stats['std']:.6f}<br>" +
                f"Skew: {pd.Series(residuals_train).skew():.6f}<br>" +
                f"Kurtosis: {pd.Series(residuals_train).kurtosis():.6f}<br><br>" +
                f"Test Statistics:<br>" +
                f"Mean: {test_stats['mean']:.6f}<br>" +
                f"Std: {test_stats['std']:.6f}<br>" +
                f"Skew: {pd.Series(residuals_test).skew():.6f}<br>" +
                f"Kurtosis: {pd.Series(residuals_test).kurtosis():.6f}"
            )

            # Add statistics annotation
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=1.02, y=0.98,
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="white",
                borderwidth=1
            )

            # Update layout
            fig.update_layout(
                title=f"{self.config.TICKER_NAME} Residuals Diagnostics",
                template='plotly_dark',
                showlegend=True,
                height=1200,
                width=1600,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.1)"
                )
            )

            # Update axes labels
            fig.update_xaxes(title_text="Residual Value", row=1, col=1)
            fig.update_xaxes(title_text="Residual Value", row=1, col=2)
            fig.update_xaxes(title_text="Lag", row=2, col=1)
            fig.update_xaxes(title_text="Lag", row=2, col=2)
            fig.update_xaxes(title_text="Observation", row=3, col=1)
            fig.update_xaxes(title_text="Observation", row=3, col=2)
            
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_yaxes(title_text="ACF", row=2, col=1)
            fig.update_yaxes(title_text="ACF", row=2, col=2)
            fig.update_yaxes(title_text="Residual", row=3, col=1)
            fig.update_yaxes(title_text="Residual", row=3, col=2)

            # Save and display
            try:
                self._save_plot(f'residuals_diagnostics', {
                    'figure': fig
                })
            except Exception as save_error:
                self.logger.error(f"plot_residuals: Failed to save plot_residuals_diagnostics: {str(save_error)}")
                self.logger.exception("Save error details:")

            # Save diagnostic data
            diagnostic_data = {
                'train_residuals': residuals_train.tolist(),
                'test_residuals': residuals_test.tolist(),
                'statistics': {
                    'train': train_stats.to_dict(),
                    'test': test_stats.to_dict(),
                    'train_skew': float(pd.Series(residuals_train).skew()),
                    'test_skew': float(pd.Series(residuals_test).skew()),
                    'train_kurtosis': float(pd.Series(residuals_train).kurtosis()),
                    'test_kurtosis': float(pd.Series(residuals_test).kurtosis())
                }
            }
            self._save_plot_data('residuals_diagnostics', diagnostic_data)
            #self._save_plot('residuals_diagnostics', fig=fig)

            # Show the plot
            print("Displaying residuals diagnostics plot...")
            fig.show()
            print("Residuals diagnostics plot generated successfully")

        except Exception as e:
            logger.error(f"Error in residuals diagnostics: {str(e)}")
            logger.exception("Stack trace:")
            raise
    
    def plot_detailed_analysis(self,
                        df_with_indicators: pd.DataFrame,
                        train_prices_rescaled: np.ndarray,
                        train_predictions_rescaled: np.ndarray,
                        test_prices_rescaled: np.ndarray,
                        test_predictions_rescaled: np.ndarray,
                        train_index: pd.DatetimeIndex,
                        test_index: pd.DatetimeIndex,
                        predictions: Dict[str, Any]) -> None:
        """Create interactive detailed analysis dashboard using Plotly"""
        try:
            
            # Ensure we have a valid DataFrame
            if not isinstance(df_with_indicators, pd.DataFrame):
                raise ValueError("Invalid DataFrame provided for df_with_indicators")
                
            # Log DataFrame columns
            self.logger.info("\nAvailable columns in df_with_indicators:")
            self.logger.info(f"Columns: {list(df_with_indicators.columns)}\n")

            # Check for volume data
            has_volume = 'Volume' in df_with_indicators.columns and not (df_with_indicators['Volume'] == 0).all()
            
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=(
                    'Price Prediction Analysis',
                    'MACD Indicator',
                    'Stochastic Oscillator',
                    'Volume Analysis'
                ),
                specs=[[{"secondary_y": True}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}],
                    [{"secondary_y": False}]]
            )

            # Add horizontal metrics box at the top
            try:
                self._add_metrics_box_horizontal_v2(
                    fig,
                    metrics_type='detailed_analysis',
                    predictions=predictions,
                    y=1.015
                )
            except Exception as e:
                logger.warning(f"plot_detailed_analysis: Could not add metrics box: {str(e)}")

            # 1. Price Prediction Plot
            # Find the overlap point between train and test
            overlap_mask = (train_index[-1] <= df_with_indicators.index) & (df_with_indicators.index <= test_index[0])
            overlap_index = df_with_indicators.index[overlap_mask]

            # Create continuous arrays for training data
            train_dates = pd.concat([pd.Series(train_index), pd.Series(overlap_index)])
            train_real_values = np.concatenate([train_prices_rescaled, [test_prices_rescaled[0]]])
            train_pred_values = np.concatenate([train_predictions_rescaled, [test_predictions_rescaled[0]]])

            # Create continuous arrays for test data
            test_dates = pd.concat([pd.Series(overlap_index), pd.Series(test_index)])
            test_real_values = np.concatenate([[train_prices_rescaled[-1]], test_prices_rescaled])
            test_pred_values = np.concatenate([[train_predictions_rescaled[-1]], test_predictions_rescaled])

            # Plot training data
            fig.add_trace(
                go.Scatter(
                    x=train_dates,
                    y=train_real_values,
                    name='Real Train',
                    line=dict(color='blue', width=1.5),
                    hovertemplate="Date: %{x}<br>Price: %{y:.6f}<extra></extra>"
                ), row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=train_dates,
                    y=train_pred_values,
                    name='Predicted Train',
                    line=dict(color='orange', width=1.5, dash='dash'),
                    hovertemplate="Date: %{x}<br>Price: %{y:.6f}<extra></extra>"
                ), row=1, col=1
            )

            # Plot test data
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=test_real_values,
                    name='Real Test',
                    line=dict(color='green', width=1.5),
                    hovertemplate="Date: %{x}<br>Price: %{y:.6f}<extra></extra>"
                ), row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=test_pred_values,
                    name='Predicted Test',
                    line=dict(color='red', width=1.5, dash='dash'),
                    hovertemplate="Date: %{x}<br>Price: %{y:.6f}<extra></extra>"
                ), row=1, col=1
            )

            # Add error bands for test predictions
            mae_error_band = np.abs(test_real_values - test_pred_values)
            fig.add_trace(
                go.Scatter(
                    x=test_dates,  # Using test_dates instead of test_index
                    y=test_pred_values + mae_error_band,  # Using test_pred_values instead of test_predictions_rescaled
                    fill=None,
                    mode='lines',
                    line_color='rgba(68, 68, 68, 0)',
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=test_dates,  # Using test_dates instead of test_index
                    y=test_pred_values - mae_error_band,  # Using test_pred_values instead of test_predictions_rescaled
                    fill='tonexty',
                    mode='lines',
                    name='Error Band',
                    line_color='rgba(68, 68, 68, 0)',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    hoverinfo='skip'
                ), row=1, col=1
            )

            # 2. MACD Plot
            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=1.5),
                    hovertemplate="Date: %{x}<br>MACD: %{y:.6f}<extra></extra>"
                ), row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=df_with_indicators['Signal_Line'],
                    name='Signal Line',
                    line=dict(color='orange', width=1.5),
                    hovertemplate="Date: %{x}<br>Signal: %{y:.6f}<extra></extra>"
                ), row=2, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=df_with_indicators.index,
                    y=df_with_indicators['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=np.where(df_with_indicators['MACD_Histogram'] >= 0, 'green', 'red'),
                    hovertemplate="Date: %{x}<br>Histogram: %{y:.6f}<extra></extra>"
                ), row=2, col=1
            )

            # 3. Stochastic Oscillator
            if '%K' in df_with_indicators.columns and '%D' in df_with_indicators.columns:
                k_percent = df_with_indicators['%K']
                d_percent = df_with_indicators['%D']
            else:
                high_14 = df_with_indicators['high'].rolling(window=14).max()
                low_14 = df_with_indicators['low'].rolling(window=14).min()
                k_percent = 100 * ((df_with_indicators['close'] - low_14) / (high_14 - low_14))
                d_percent = k_percent.rolling(window=3).mean()

            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=k_percent,
                    name='%K',
                    line=dict(color='blue', width=1.5),
                    hovertemplate="Date: %{x}<br>%K: %{y:.2f}<extra></extra>"
                ), row=3, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df_with_indicators.index,
                    y=d_percent,
                    name='%D',
                    line=dict(color='orange', width=1.5),
                    hovertemplate="Date: %{x}<br>%D: %{y:.2f}<extra></extra>"
                ), row=3, col=1
            )

            # Add overbought/oversold lines
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

            # 4. Volume Plot
            fig.add_trace(
                go.Bar(
                    x=df_with_indicators.index,
                    y=df_with_indicators['Volume'],
                    name='Volume',
                    marker_color='rgba(128,128,128,0.5)',
                    hovertemplate="Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>"
                ), row=4, col=1
            )

            # Add metrics box
            metrics = predictions['metrics']
            metrics_text = (
                f"Train Metrics:<br>" +
                f"MAE: {metrics.get('train_mae', 0):.4f}<br>" +
                f"RMSE: {metrics.get('train_rmse', 0):.4f}<br>" +
                f"R²: {metrics.get('train_r2', 0):.4f}<br>" +
                f"Error %: {metrics.get('train_mae_percentage', 0):.2f}%<br>" +
                f"Direction: {metrics.get('train_direction_accuracy', 0):.1f}%<br>" +
                f"<br>Test Metrics:<br>" +
                f"MAE: {metrics.get('test_mae', 0):.4f}<br>" +
                f"RMSE: {metrics.get('test_rmse', 0):.4f}<br>" +
                f"R²: {metrics.get('test_r2', 0):.4f}<br>" +
                f"Error %: {metrics.get('test_mae_percentage', 0):.2f}%<br>" +
                f"Direction: {metrics.get('test_direction_accuracy', 0):.1f}%"
            )

            # not in use
            fig.add_annotation(
                text=metrics_text,
                xref="paper", yref="paper",
                x=1.02, y=0.98,
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="white",
                borderwidth=1
            )
            
            # Only add volume plot if volume data exists
            if has_volume:
                fig.add_trace(
                    go.Bar(
                        x=df_with_indicators.index,
                        y=df_with_indicators['Volume'],
                        name='Volume',
                        marker_color='rgba(128,128,128,0.5)',
                        hovertemplate="Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>"
                    ), row=4, col=1
                )

            # Update layout and axes
            fig.update_layout(
                title=f"{Config.TICKER_NAME} Detailed Analysis",
                template='plotly_dark',
                height=1600,
                width=1800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.1)"
                ),
                hovermode='x unified',
                xaxis_showticklabels=True,
                xaxis2_showticklabels=True,
                xaxis3_showticklabels=True,
                xaxis4_showticklabels=True
            )

            # Update all x-axes and y-axes properties
            fig.update_xaxes(
                rangeslider_visible=False,
                showticklabels=True,
                tickformat="%Y-%m-%d",
                tickangle=45
            )

            # Update y-axes titles
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.update_yaxes(title_text="%", row=3, col=1)
            fig.update_yaxes(title_text="Volume", row=4, col=1)

            # Save and display
            try:
                self._save_plot(f'detailed_analysis', {
                    'figure': fig
                })
            except Exception as save_error:
                self.logger.error(f"plot_residuals: Failed to save plot_detailed_analysis: {str(save_error)}")
                self.logger.exception("Save error details:")

            # Save the analysis data
            analysis_data = {
                'prices': df_with_indicators['Close'].to_dict(),
                'volume': df_with_indicators['Volume'].to_dict(),
                'macd': df_with_indicators['MACD'].to_dict(),
                'signal_line': df_with_indicators['Signal_Line'].to_dict(),
                'macd_histogram': df_with_indicators['MACD_Histogram'].to_dict(),
                'stochastic_k': k_percent.to_dict(),
                'stochastic_d': d_percent.to_dict(),
                'metrics': metrics
            }
            self._save_plot_data('detailed_analysis', analysis_data)

            # Show the plot
            fig.show()

        except Exception as e:
            logger.error(f"Error in detailed analysis plot: {str(e)}")
            logger.exception("Stack trace:")
    
    def plot_extra_metrix(self) -> None:
        """Plot all extra metrics and analysis visualizations"""
        try:
            logger.info("Generating extra metrics plots...")
            
            # Validate that we have all required data
            required_attrs = [
                'df', 'train_size', 'time_step', 'test_index',
                'train_prices_rescaled', 'train_predictions_rescaled',
                'test_prices_rescaled', 'test_predictions_rescaled'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                if getattr(self, attr) is None:
                    raise ValueError(f"Required attribute is None: {attr}")

            # Get processed data
            df_with_indicators, prediction_start_date, training_df, test_real_df, test_pred_df, future_df = self._get_processed_data()
            
            # Get metrics using evaluate method
            metrics = self.evaluate()
            if not metrics:
                logger.warning("No metrics available to display")
                return
                
            # Create predictions dictionary with metrics
            predictions = {
                'metrics': {
                    'train_mae': metrics['train_mae'],
                    'train_mae_formatted': f"{metrics['train_mae']:.6f}",
                    'train_mse': metrics['train_mse'],
                    'train_rmse': metrics['train_rmse'],
                    'train_r2': metrics['train_r2'],
                    'train_direction_accuracy': metrics['train_direction_accuracy'],
                    'train_error_std': metrics['train_error_std'],
                    'train_error_max': metrics['train_error_max'],
                    'test_mae': metrics['test_mae'],
                    'test_mae_formatted': f"{metrics['test_mae']:.6f}",
                    'test_mse': metrics['test_mse'],
                    'test_rmse': metrics['test_rmse'],
                    'test_r2': metrics['test_r2'],
                    'test_direction_accuracy': metrics['test_direction_accuracy'],
                    'test_error_std': metrics['test_error_std'],
                    'test_error_max': metrics['test_error_max']
                }
            }

            # Generate plots in sequence
            try:
                # 1. Plot residuals analysis
                self.plot_residuals(
                    train_prices_rescaled=self.train_prices_rescaled,
                    train_predictions_rescaled=self.train_predictions_rescaled,
                    test_prices_rescaled=self.test_prices_rescaled,
                    test_predictions_rescaled=self.test_predictions_rescaled
                )
                logger.info("Residuals plot generated successfully")
                
                # 2. Plot residuals diagnostics
                self.plot_residuals_diagnostics(
                    train_prices_rescaled=self.train_prices_rescaled,
                    train_predictions_rescaled=self.train_predictions_rescaled,
                    test_prices_rescaled=self.test_prices_rescaled,
                    test_predictions_rescaled=self.test_predictions_rescaled
                )
                logger.info("Residuals diagnostics plot generated successfully")
                
                # 3. Plot detailed analysis
                self.plot_detailed_analysis(
                    df_with_indicators=df_with_indicators,
                    train_prices_rescaled=self.train_prices_rescaled,
                    train_predictions_rescaled=self.train_predictions_rescaled,
                    test_prices_rescaled=self.test_prices_rescaled,
                    test_predictions_rescaled=self.test_predictions_rescaled,
                    train_index=self.df.index[:self.train_size],
                    test_index=self.test_index,
                    predictions=predictions
                )
                logger.info("Detailed analysis plot generated successfully")
                
                if self.is_crypto:
                    self.plot_crypto_analysis()
                else:
                    self.plot_stock_analysis()

            except Exception as e:
                logger.error(f"Error generating plots: {str(e)}")
                raise
            
            logger.info("\nAll extra metrics plots generated successfully\n")
            
        except Exception as e:
            logger.error(f"Error in plot_extra_metrix: {str(e)}")
            logger.exception("Stack trace:")
            raise
    
    # ============== Extra Plotting Metrix   - END ============== #

# ========================= Utility Functions ========================= #

def set_global_seeds(seed: int = Config.RANDOM_SEED):
    """Set all random seeds for complete reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set global random seed to: {seed}")

def setup_basic_stock_model() -> bool:
    """Set up and run BasicStockModel with support for both stock and crypto data"""
    try:
        print("\nSetting up BasicStockModel...")
        
        # Load existing model prompt
        print("\nModel Loading Options:")
        print("1. Load existing model")
        print("2. Create new model")
        
        load_choice = None
        while load_choice is None:
            try:
                choice = input("Select option (1-2): ").strip()
                if not choice:
                    continue
                choice = int(choice)
                if choice in [1, 2]:
                    load_choice = choice
                else:
                    print("Please enter 1 or 2")
            except ValueError:
                print("Please enter a valid number")

        # Initialize model parameters only for new model
        if load_choice == 2:
            # Attention layer prompt
            print("\nBase Raw Features Configuration:")
            print("1. Enable Base Raw Features")
            print("2. Disable Base Raw Features")
            
            while True:
                try:
                    attention_choice = input("Select option (1-2): ")
                    if not attention_choice:  # Handle empty input
                        continue
                    attention_choice = int(attention_choice)
                    if attention_choice in [1, 2]:
                        Config.USE_RAW_FEATURES_ONLY = (attention_choice == 1)
                        break
                    print("Please enter 1 or 2")
                except ValueError:
                    print("Please enter a valid number")


            print("\nEnter model parameters (press Enter for default values):")

            # Model architecture parameters
            Config.HIDDEN_LAYER_SIZE = int(input(f"Hidden layer size (default {Config.HIDDEN_LAYER_SIZE}): ") or Config.HIDDEN_LAYER_SIZE)

            # Ensure dropout is between 0 and 1
            while True:
                dropout = input(f"Dropout probability (default {Config.DROPOUT_PROB}): ") or Config.DROPOUT_PROB
                try:
                    dropout = float(dropout)
                    if 0 <= dropout <= 1:
                        Config.DROPOUT_PROB = dropout
                        break
                    print("Dropout must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")

            Config.TIME_STEP = int(input(f"Time step [7, 14, 30, 60] (default {Config.TIME_STEP}): ") or Config.TIME_STEP)
            Config.FUTURE_STEPS = int(input(f"Future steps (default {Config.FUTURE_STEPS}): ") or Config.FUTURE_STEPS)

            # Ensure train-test split is between 0 and 1
            while True:
                split = input(f"Train-test split (default {Config.TRAIN_TEST_SPLIT}): ") or Config.TRAIN_TEST_SPLIT
                try:
                    split = float(split)
                    if 0 < split < 1:
                        Config.TRAIN_TEST_SPLIT = split
                        break
                    print("Train-test split must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")
            
            Config.RANDOM_SEED = int(input(f"Random seed (default {Config.RANDOM_SEED}): ") or Config.RANDOM_SEED)

            epochs = int(input("\nEnter number of epochs (default 150): ") or 150)
            batch_size = int(input("Enter batch size (default 64): ") or 64)
            learning_rate = float(input("Enter learning rate (default 0.001): ") or 0.001)
        else:
            # Use default values when loading existing model
            epochs, batch_size, learning_rate = Config.EPOCHS, Config.BATCH_SIZE, Config.LEARNING_RATE


        if load_choice == 1:
            try:
                Config.FUTURE_STEPS = int(input(f"Future steps (default {Config.FUTURE_STEPS}): ") or Config.FUTURE_STEPS)

                # Initialize BasicStockModel
                model = BasicStockModel(Config, epochs, batch_size, learning_rate)
                
                # Prepare data first
                #model.prepare_data()

                model.load_model()
                print("\nGenerating predictions and plots...")
                predictions = model.predict()
                
                if model.is_crypto:
                    predictions = model.validate_crypto_predictions(predictions)
                
                metrics = model.evaluate()
                model.log_model_metrics(metrics)
                
                # Generate future predictions
                print(f"\nGenerating future predictions for {Config.FUTURE_STEPS} days...")
                model.plot_predictions('advanced', future_days=Config.FUTURE_STEPS)
                
                model.plot_extra_metrix()
                
                input("\nPress Enter after reviewing the plots...")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
            
        else:
            
            # Attention layer prompt
            print("\nAttention Layer Configuration:")
            print("1. Enable attention layer")
            print("2. Disable attention layer")
            
            while True:
                try:
                    attention_choice = input("Select option (1-2): ")
                    if not attention_choice:  # Handle empty input
                        continue
                    attention_choice = int(attention_choice)
                    if attention_choice in [1, 2]:
                        Config.USE_ATTENTION = (attention_choice == 1)
                        break
                    print("Please enter 1 or 2")
                except ValueError:
                    print("Please enter a valid number")

            # Initialize BasicStockModel
            model = BasicStockModel(Config, epochs, batch_size, learning_rate)
            
            # Prepare data first
            model.prepare_data()

            # Currently not in use - in development
            # ===================================================== #
            USE_HANDLE_MISSING_MARKET_CAP_DATA = False
            USE_DATA_RESAMPLING_TIME_FRAME = False

            if USE_HANDLE_MISSING_MARKET_CAP_DATA is True:
                # Handle data preprocessing
                if model.is_crypto:
                    print("\nCrypto Data Preprocessing Options:")
                    if input("Handle missing market cap data? (y/n): ").lower() == 'y':
                        model.handle_missing_market_cap()
            
            if USE_DATA_RESAMPLING_TIME_FRAME is True:
                # Resampling option
                print("\nData Resampling Options:")
                print("1. Keep original timeframe")
                print("2. Resample to different timeframe")
                
                if int(input("Select option (1-2): ") or 1) == 2:
                    timeframe_options = ['D', 'W', 'M']
                    if model.is_crypto:
                        timeframe_options.extend(['H', '15T', '30T', '1H', '4H'])
                    
                    print("\nAvailable timeframes:", ', '.join(timeframe_options))
                    timeframe = input(f"Enter timeframe ({timeframe_options[0]}): ") or timeframe_options[0]
                    model.resample_data(timeframe)
            # ===================================================== #

            try:
                # Train model
                model.train()
                print("\nGenerating predictions and plots...")
                predictions = model.predict()
                
                if model.is_crypto:
                    predictions = model.validate_crypto_predictions(predictions)
                
                metrics = model.evaluate()
                model.log_model_metrics(metrics)
                
                # Generate future predictions
                print(f"\nGenerating future predictions for {Config.FUTURE_STEPS} days...")
                model.plot_predictions('advanced', future_days=Config.FUTURE_STEPS)
                          
                model.plot_extra_metrix()
                                
                input("\nPress Enter after reviewing the plots...")
                
            except Exception as e:
                print(f"Error training model: {str(e)}")
                return False
            
        print("BasicStockModel setup completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in BasicStockModel setup: {str(e)}")
        return False

# ========================= Main Execution ========================= #
def main():
    """Main execution function"""
    try:
        # Initial seed setting
        set_global_seeds()
        
        # Select data type and update configuration
        Config.select_data_type()
              
        # Load initial data to determine available columns
        data_processor = DataProcessor(os.path.join(Config.DATA_PATH, Config.STOCK_DATA_FILENAME))
        initial_df = data_processor.load_data()
        data_processor.validate_dataset(initial_df)

        # Setup model with enhanced path generation
        setup_basic_stock_model()
        
        Config.save_config()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(f"Stack trace:", exc_info=True)  # Added for better error tracking
        raise
    
if __name__ == "__main__":
    main()