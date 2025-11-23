# src/utils.py
from src.data_processing import load_data, clean_data, create_rfm

DEFAULT_PATH = "C:/Users/USER/OneDrive/Desktop/learningUtsav/data/online_retail_II.xlsx"

def load_data_raw(path=DEFAULT_PATH):
    """Load raw Excel data."""
    return load_data(path)

def load_clean_rfm(path=DEFAULT_PATH):
    """Load, clean, and create RFM table."""
    df = load_data(path)
    df = clean_data(df)
    rfm = create_rfm(df)
    return df, rfm
