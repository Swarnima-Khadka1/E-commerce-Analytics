import streamlit as st
import numpy as np
from src.utils import load_clean_rfm
from src.ltv_model import prepare_ltv_data, train_ltv_model
import plotly.express as px

st.title("💰 Customer LTV Prediction")

# Load cleaned data
df, rfm = load_clean_rfm("data/online_retail_II.xlsx")

st.subheader(" Preparing LTV Dataset")
ltv = prepare_ltv_data(df)
st.dataframe(ltv.head())

st.subheader("Training Model")
model, score, scaler, features = train_ltv_model(ltv)

st.write(f"### Model R² Score: **{score:.2f}**")
st.success("Model trained successfully!")

# Optional: Predict LTV for top 5 customers
st.subheader("🔮 Sample Predictions")
sample_X = scaler.transform(ltv[features].head())
preds = np.expm1(model.predict(sample_X))
st.dataframe(ltv[["CustomerID"]].head().assign(Predicted_LTV=preds))


st.subheader("📊 Residual Plot")
# Predict on the test set
from sklearn.model_selection import train_test_split
X_scaled = scaler.transform(ltv[features].values)
y = np.log1p(ltv["Total_Spend"].values)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# Residuals
residuals = np.expm1(y_test) - np.expm1(y_pred)

fig = px.scatter(
    x=np.expm1(y_pred),
    y=residuals,
    labels={"x": "Predicted LTV", "y": "Residuals"},
    title="Residual Plot"
)
st.plotly_chart(fig, use_container_width=True)