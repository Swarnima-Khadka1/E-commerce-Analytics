import streamlit as st
from src.utils import load_data_raw
from src.data_processing import clean_data

st.title("📦 Data Cleaning & Overview")

df = load_data_raw("C:/Users/USER/OneDrive/Desktop/learningUtsav/data/online_retail_II.xlsx")

st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head())

# Clean
clean_df = clean_data(df)

st.subheader("🧹 Cleaned Data Preview")
st.dataframe(clean_df.head())

st.subheader("📊 Summary Statistics")
st.write(clean_df.describe())

st.success("Data cleaned successfully!")
