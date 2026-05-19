from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

def prepare_features(df, scaler=None):
    try:
        working_df = df.copy()
        required_cols = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']
        missing_cols = [col for col in required_cols if col not in working_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for feature preparation: {', '.join(missing_cols)}")

        working_df['CustomerID'] = pd.to_numeric(working_df['CustomerID'], errors='coerce')
        working_df['InvoiceDate'] = pd.to_datetime(working_df['InvoiceDate'], errors='coerce')
        working_df['Quantity'] = pd.to_numeric(working_df['Quantity'], errors='coerce')
        working_df['UnitPrice'] = pd.to_numeric(working_df['UnitPrice'], errors='coerce')
        working_df = working_df.dropna(subset=['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice']).copy()
        working_df = working_df[(working_df['Quantity'] > 0) & (working_df['UnitPrice'] > 0)].copy()
        working_df['CustomerID'] = working_df['CustomerID'].astype(int)
        working_df['TotalPrice'] = working_df['Quantity'] * working_df['UnitPrice']
        
        snapshot_date = working_df['InvoiceDate'].max() + timedelta(days=1)
        frequency_column = 'InvoiceNo' if 'InvoiceNo' in working_df.columns else 'CustomerID'
        
        rfm_df = working_df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda date: (snapshot_date - date.max()).days),
            Frequency=(frequency_column, 'nunique' if frequency_column == 'InvoiceNo' else 'size'),
            Monetary=('TotalPrice', 'sum')
        ).reset_index()
        
        X_rfm = rfm_df[['Recency', 'Frequency', 'Monetary']].values
        
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_rfm)
        else:
            X_scaled = scaler.transform(X_rfm)
        
        return X_scaled, scaler, rfm_df['CustomerID']
    
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise