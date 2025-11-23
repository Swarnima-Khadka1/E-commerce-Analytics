import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def prepare_ltv_data(df):
    """Prepare LTV features using RFM metrics."""
    customer_df = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (x.max() - x.min()).days + 1,  # Customer age in days
        "Quantity": "sum",
        "Price": "mean",
        "Revenue": "sum",
        "Invoice": "nunique"
    }).reset_index()

    customer_df.columns = [
        "CustomerID", "Customer_Age_Days",
        "Total_Quantity", "Avg_Price",
        "Total_Spend", "Num_Orders"
    ]

    # Log-transform target to stabilize variance
    customer_df["LTV_log"] = np.log1p(customer_df["Total_Spend"])
    return customer_df

def train_ltv_model(ltv_df):
    """Train LTV model using RandomForest + RFM features."""
    features = ["Customer_Age_Days", "Total_Quantity", "Avg_Price", "Num_Orders"]
    X = ltv_df[features].values
    y = ltv_df["LTV_log"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    # Return exponentiated predictions if needed
    return model, score, scaler, features
