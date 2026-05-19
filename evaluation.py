import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_customers
from feature_engineering import prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_prepared_data():
    df = load_customers('data/OnlineRetail.csv')
    X_scaled, _, _ = prepare_features(df)
    return X_scaled


def plot_elbow_method(max_k=10):
    """
    Plot the Elbow Method to determine optimal number of clusters.
    
    Args:
        max_k: Maximum number of clusters to test
    """
    try:
        logger.info("Generating Elbow Method plot...")
        X_scaled = _load_prepared_data()
        
        wcss = []
        silhouette_scores = []
        
        for i in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        os.makedirs('assets', exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow Method plot
        ax1.plot(range(2, max_k + 1), wcss, marker='o', linestyle='--', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method For Optimal K', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette Score plot
        ax2.plot(range(2, max_k + 1), silhouette_scores, marker='s', linestyle='--', 
                linewidth=2, markersize=8, color='orange')
        ax2.set_title('Silhouette Score For Different K', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('assets/elbow_silhouette_plot.png', dpi=300, bbox_inches='tight')
        logger.info("Elbow and Silhouette plot saved to assets/elbow_silhouette_plot.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating elbow plot: {str(e)}")
        raise


def plot_davies_bouldin(max_k=10):
    """
    Plot Davies-Bouldin Index for cluster validation.
    
    Args:
        max_k: Maximum number of clusters to test
    """
    try:
        logger.info("Generating Davies-Bouldin Index plot...")
        X_scaled = _load_prepared_data()
        
        db_scores = []
        
        for i in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            db_scores.append(davies_bouldin_score(X_scaled, labels))
        
        os.makedirs('assets', exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_k + 1), db_scores, marker='D', linestyle='--', 
                linewidth=2, markersize=8, color='red')
        plt.title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Davies-Bouldin Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/davies_bouldin_plot.png', dpi=300, bbox_inches='tight')
        logger.info("Davies-Bouldin plot saved to assets/davies_bouldin_plot.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating Davies-Bouldin plot: {str(e)}")
        raise


if __name__ == '__main__':
    plot_elbow_method()
    plot_davies_bouldin()