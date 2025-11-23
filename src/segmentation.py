# src/segmentation.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def scale_rfm(rfm_df):
    """
    Scale RFM columns and return (scaled_array, scaler, columns_order).
    Expects rfm_df to contain Recency, Frequency, Monetary columns.
    """
    cols = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    arr = rfm_df[cols].values.astype(float)
    scaled = scaler.fit_transform(arr)
    return scaled, scaler, cols

def kmeans_segment(scaled_rfm, n_clusters=4):
    """
    Fit KMeans and return (labels, model).
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(scaled_rfm)
    return labels, km

def find_elbow(scaled_rfm, kmin=2, kmax=10):
    """
    Returns list of (k, inertia) for plotting elbow.
    """
    inertias = []
    ks = list(range(kmin, kmax + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled_rfm)
        inertias.append(km.inertia_)
    return ks, inertias

def find_silhouette(scaled_rfm, kmin=2, kmax=10):
    """
    Returns list of (k, silhouette_score) for plotting silhouette curve.
    If silhouette cannot be computed for a k (e.g., single cluster), returns nan.
    """
    scores = []
    ks = list(range(kmin, kmax + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_rfm)
        try:
            s = silhouette_score(scaled_rfm, labels)
        except Exception:
            s = np.nan
        scores.append(s)
    return ks, scores

def pca_reduce(scaled_rfm, n_components=2):
    """
    Reduce to 2D/3D for plotting. Returns reduced array and PCA object.
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled_rfm)
    return reduced, pca

def profile_clusters(rfm_df, labels):
    """
    Return a profiling DataFrame for clusters with business-friendly metrics.
    rfm_df: dataframe indexed by Customer ID with Recency, Frequency, Monetary
    labels: array aligned with rfm_df rows
    """
    df = rfm_df.copy()
    df["Cluster"] = labels
    agg = df.groupby("Cluster").agg(
        Avg_Recency = ("Recency", "mean"),
        Avg_Frequency = ("Frequency", "mean"),
        Avg_Monetary = ("Monetary", "mean"),
        Total_Revenue = ("Monetary", "sum"),
        Customer_Count = ("Monetary", "count")
    ).reset_index()

    # Add median thresholds to derive simple segment labels
    med_rec = rfm_df["Recency"].median()
    med_freq = rfm_df["Frequency"].median()
    med_mon = rfm_df["Monetary"].median()

    def label_row(r):
        if r["Avg_Monetary"] >= med_mon and r["Avg_Frequency"] >= med_freq:
            return "Champions"
        if r["Avg_Monetary"] >= med_mon and r["Avg_Frequency"] < med_freq:
            return "Big Spenders"
        if r["Avg_Monetary"] < med_mon and r["Avg_Frequency"] >= med_freq:
            return "Frequent Low Value"
        if r["Avg_Recency"] > med_rec:
            return "At Risk"
        return "Potential"

    agg["SegmentName"] = agg.apply(label_row, axis=1)
    # Add relative share
    total_customers = agg["Customer_Count"].sum()
    agg["CustomerShare%"] = (agg["Customer_Count"] / total_customers * 100).round(2)

    return agg.sort_values("Avg_Monetary", ascending=False)
