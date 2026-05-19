import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def load_customers(path: str) -> pd.DataFrame:
    """
    Load and preprocess customer transaction data from Online Retail dataset.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Cleaned DataFrame with valid transactions
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No data found at {path}")
            
        df = pd.read_csv(path, parse_dates=['InvoiceDate'], encoding='ISO-8859-1')
        logger.info(f"Loaded {len(df)} rows from {path}")
        
        # Keep only rows with a valid customer id
        df = df.dropna(subset=['CustomerID']).copy()
        
        # Remove cancelled invoices when invoice numbers are available
        if 'InvoiceNo' in df.columns:
            invoice_no = df['InvoiceNo'].astype(str)
            df = df[~invoice_no.str.startswith('C', na=False)].copy()
        
        # Remove rows with invalid quantities (negative or zero)
        df = df[df['Quantity'] > 0].copy()
        
        # Remove rows with invalid unit prices (negative or zero)
        df = df[df['UnitPrice'] > 0].copy()
        
        # Calculate transaction value
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Ensure CustomerID is integer
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        logger.info(f"Final cleaned dataset: {len(df)} rows")
        return df
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading customers: {str(e)}")
        raise