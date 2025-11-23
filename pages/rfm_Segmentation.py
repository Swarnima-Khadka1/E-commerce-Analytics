# pages/rfm_segmentation.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os
# ensure src importable when Streamlit runs pages
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils import load_clean_rfm
from src.segmentation import (
    scale_rfm, kmeans_segment, find_elbow, find_silhouette,
    pca_reduce, profile_clusters
)

st.set_page_config(page_title="RFM Segmentation", layout="wide")
st.title("📊 RFM Segmentation & Cluster Profiling")

# ---- load data ----
DATA_PATH = "C:/Users/USER/OneDrive/Desktop/learningUtsav/data/online_retail_II.xlsx"
with st.spinner("Loading and preparing RFM..."):
    df, rfm = load_clean_rfm(DATA_PATH)

st.sidebar.header("Segmentation Controls")
k_default = 4
k = st.sidebar.slider("Number of clusters (K)", 2, 10, k_default)
run_opt = st.sidebar.button("Run Optimal-K diagnostics")

# show RFM head
st.subheader("RFM (sample)")
st.dataframe(rfm.head())

# ---- diagnostics: elbow + silhouette ----
if run_opt:
    st.subheader("Optimal K diagnostics")
    ks, inertias = find_elbow(*scale_rfm(rfm)[:1])
    ks_s, sils = find_silhouette(*scale_rfm(rfm)[:1])
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(ks), y=inertias, mode="lines+markers"))
    fig_elbow.update_layout(title="Elbow: Inertia vs K", xaxis_title="K", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)

    fig_sil = go.Figure()
    fig_sil.add_trace(go.Bar(x=list(ks_s), y=sils))
    fig_sil.update_layout(title="Silhouette Score vs K", xaxis_title="K", yaxis_title="Silhouette")
    st.plotly_chart(fig_sil, use_container_width=True)

# ---- clustering ----
scaled, scaler, cols = scale_rfm(rfm)
labels, km_model = kmeans_segment(scaled, n_clusters=k)
rfm_clusters = rfm.copy()
rfm_clusters["Cluster"] = labels

st.subheader("Cluster Summary Table")
cluster_profile = profile_clusters(rfm, labels)
st.dataframe(cluster_profile)

# ---- radar chart (avg RFM per cluster) ----
st.subheader("Cluster RFM Radar Chart")
radar_df = rfm_clusters.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
# normalize each column for plotting clarity
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
radar_norm = radar_norm.reset_index()

fig = go.Figure()
categories = ["Recency", "Frequency", "Monetary"]
for idx, row in radar_norm.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row[c] for c in categories] + [row[categories[0]]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f"Cluster {int(row['Cluster'])}"
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ---- PCA scatter for cluster separation ----
st.subheader("PCA 2D Scatter of Customers (by cluster)")
reduced, pca = pca_reduce(scaled, n_components=2)
pca_df = pd.DataFrame(reduced, columns=["PC1", "PC2"], index=rfm.index)
pca_df["Cluster"] = labels
pca_df_sample = pca_df.sample(min(5000, len(pca_df)), random_state=42)  # sample for speed

fig2 = px.scatter(
    pca_df_sample.reset_index(),
    x="PC1", y="PC2", color="Cluster",
    hover_data=["Customer ID"] if "Customer ID" in pca_df_sample.index.names else None,
    title="PCA-reduced view (sample)"
)
st.plotly_chart(fig2, use_container_width=True)

# ---- per-cluster quick visuals ----
st.subheader("Per-Cluster Distributions")
selected_cluster = st.selectbox("Select cluster to inspect", sorted(rfm_clusters["Cluster"].unique().tolist()))
cluster_customers = rfm_clusters[rfm_clusters["Cluster"] == selected_cluster]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Customers in cluster", int(cluster_customers.shape[0]))
    st.write("Median Monetary", round(cluster_customers["Monetary"].median(), 2))
with col2:
    st.write("Median Frequency", round(cluster_customers["Frequency"].median(), 2))
with col3:
    st.write("Median Recency", round(cluster_customers["Recency"].median(), 2))

# histograms
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=cluster_customers["Monetary"], nbinsx=40, name="Monetary"))
fig_hist.add_trace(go.Histogram(x=cluster_customers["Frequency"], nbinsx=40, name="Frequency"))
fig_hist.update_layout(barmode='overlay', title=f"Monetary & Frequency distribution (Cluster {selected_cluster})")
fig_hist.update_traces(opacity=0.6)
st.plotly_chart(fig_hist, use_container_width=True)

# ---- recommendations (simple) ----
st.subheader("Suggested Marketing Action")
seg_row = cluster_profile[cluster_profile["Cluster"] == selected_cluster].iloc[0]
seg_name = seg_row["SegmentName"]
if seg_name == "Champions":
    st.success("Champions: Reward with loyalty program, VIP offers, exclusive discounts.")
elif seg_name == "Big Spenders":
    st.info("Big Spenders: Upsell premium products, personalized recommendations.")
elif seg_name == "Frequent Low Value":
    st.info("Frequent but low value: Introduce bundles or cross-sell to increase AOV.")
elif seg_name == "At Risk":
    st.warning("At-Risk: Re-engagement campaigns, win-back coupons.")
else:
    st.write("Potential: Target with personalized onboarding/offers to increase spend.")
