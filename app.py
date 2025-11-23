import streamlit as st

st.set_page_config(page_title="E-Commerce Analytics", layout="wide")

st.title("🛒 E-Commerce Customer Intelligence Platform")
st.write("""
Welcome!  
This app provides a **complete customer analytics pipeline**:

### ✔️ Data Cleaning  
### ✔️ RFM Segmentation  
### ✔️ LTV Prediction (Machine Learning)  
### ✔️ Visual Insights

Use the left navigation menu to explore each module.
""")

st.info("Dataset used: *Online Retail II — UCI Machine Learning Repository*")
