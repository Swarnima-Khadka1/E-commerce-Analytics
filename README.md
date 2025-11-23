# E-Commerce Customer Segmentation & LTV Prediction

This project performs **customer segmentation** using RFM analysis and predicts **Customer Lifetime Value (LTV)** using machine learning on the [Online Retail II dataset](https://archive.ics.uci.edu/ml/datasets/online+retail+ii). It includes an interactive **Streamlit app** for visualization and LTV predictions.

**Features:**
- RFM segmentation with KMeans clustering
- LTV prediction using RandomForest and RFM features
- Interactive Streamlit interface
- Residual plots and predicted LTV display

**Demo:**  
![Demo Video](https://youtu.be/5ajCyfCsvAc)

**Setup:**
```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv ecommerce
pip install -r requirements.txt
streamlit run app.py
