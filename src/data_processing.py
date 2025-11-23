import pandas as pd

def load_data(path: str, sheet_name="Year 2010-2011"):
    """Load the Online Retail II Excel file."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    return df


def clean_data(df: pd.DataFrame):
    """Clean dataset for RFM analysis."""

    # Remove missing customers
    df = df[df["Customer ID"].notnull()]

    # Remove cancellations
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Remove invalid values
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # Create Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Fix date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df


def create_rfm(df: pd.DataFrame, snapshot_date=None):
    """Generate RFM metrics per customer."""

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "Revenue": "sum"
    })

    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "Invoice": "Frequency",
        "Revenue": "Monetary"
    }, inplace=True)

    return rfm
