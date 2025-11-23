import plotly.graph_objects as go
import pandas as pd

def plot_radar_clusters(cluster_df):
    categories = list(cluster_df.columns)

    fig = go.Figure()

    for idx, row in cluster_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row.values,
            theta=categories,
            fill='toself',
            name=f'Cluster {idx}'
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )

    return fig
