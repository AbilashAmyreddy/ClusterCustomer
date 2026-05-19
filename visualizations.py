import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns


COLOR_MAP = {
    '0': '#4f46e5',
    '1': '#ec4899',
    '2': '#10b981',
    '3': '#f59e0b',
    '4': '#06b6d4',
    '5': '#8b5cf6',
    '6': '#ef4444',
    '7': '#14b8a6',
}


def _normalize_cluster_labels(labels):
    return pd.Series(labels, dtype='object').astype(str)


def plot_rfm_3d_scatter(rfm_df):
    df = rfm_df.copy()
    df['cluster'] = _normalize_cluster_labels(df['cluster'])
    fig = px.scatter_3d(
        df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='cluster',
        hover_data=['CustomerID'],
        title='Customer Segments in 3D RFM Space',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.85))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_rfm_2d_scatter(rfm_df, x='Recency', y='Monetary'):
    df = rfm_df.copy()
    df['cluster'] = _normalize_cluster_labels(df['cluster'])
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color='cluster',
        hover_data=['CustomerID'],
        title=f'{x} vs {y} by Cluster',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_pca_clusters(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(X_scaled)
    df = pd.DataFrame(
        {
            'PC1': pca_components[:, 0],
            'PC2': pca_components[:, 1],
            'cluster': _normalize_cluster_labels(labels),
        }
    )
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='cluster',
        title='PCA Projection of Customer Segments',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_cluster_distribution(labels):
    df = pd.DataFrame({'cluster': _normalize_cluster_labels(labels)})
    counts = df['cluster'].value_counts().sort_index().reset_index()
    counts.columns = ['cluster', 'count']
    fig = px.bar(
        counts,
        x='cluster',
        y='count',
        text='count',
        title='Customer Count by Cluster',
        color='cluster',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_rfm_distributions(rfm_df):
    df = rfm_df.copy()
    df['cluster'] = _normalize_cluster_labels(df['cluster'])
    melted = df.melt(
        id_vars='cluster',
        value_vars=['Recency', 'Frequency', 'Monetary'],
        var_name='Metric',
        value_name='Value',
    )
    fig = px.box(
        melted,
        x='Metric',
        y='Value',
        color='cluster',
        points='outliers',
        title='RFM Distribution Across Segments',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def plot_cluster_heatmap(rfm_df):
    cluster_summary = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, max(4, len(cluster_summary) * 0.7)))
    sns.heatmap(
        cluster_summary,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title('Average RFM Profile by Cluster')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    return fig
