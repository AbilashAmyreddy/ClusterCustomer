import joblib
import logging
from sklearn.cluster import KMeans
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_customers
from feature_engineering import prepare_features
from segmentation import save_model, evaluate_clustering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_optimal_k(X, max_k=10):
    """
    Find optimal number of clusters using Silhouette Score and Elbow Method.
    
    Args:
        X: Scaled feature matrix
        max_k: Maximum number of clusters to test
        
    Returns:
        Tuple of (best_k, results_dict)
    """
    logger.info("Finding optimal number of clusters...")
    
    best_silhouette = -1
    best_k = 2
    results = {}
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        metrics = evaluate_clustering(X, labels, kmeans)
        results[k] = metrics
        
        if metrics['silhouette_score'] > best_silhouette:
            best_silhouette = metrics['silhouette_score']
            best_k = k
        
        logger.info(f"k={k}: Silhouette={metrics['silhouette_score']:.4f}, "
                   f"Davies-Bouldin={metrics['davies_bouldin_score']:.4f}")
    
    logger.info(f"Optimal k={best_k} with Silhouette Score={best_silhouette:.4f}")
    return best_k, results


def train():
    """
    Main training pipeline: load data, compute RFM, find optimal clusters, train model.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Customer Segmentation Model Training")
        logger.info("=" * 60)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load and prepare data
        logger.info("Loading customer data...")
        df = load_customers('data/OnlineRetail.csv')
        logger.info(f"Loaded {len(df)} transactions for {df['CustomerID'].nunique()} unique customers")
        
        # Prepare RFM features
        logger.info("Calculating RFM metrics and scaling features...")
        X_scaled, scaler, _ = prepare_features(df)
        logger.info(f"Prepared {X_scaled.shape[0]} customer records with {X_scaled.shape[1]} features")
        
        # Find optimal number of clusters
        optimal_k, cluster_results = find_optimal_k(X_scaled, max_k=10)
        
        # Train final model with optimal k
        logger.info(f"Training KMeans model with k={optimal_k}...")
        model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
        model.fit(X_scaled)
        
        # Save model and scaler
        logger.info("Saving model and scaler...")
        save_model(model, 'models/kmeans_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(cluster_results, 'models/cluster_analysis.pkl')
        logger.info("Model training complete!")
        logger.info(f"Model saved to models/kmeans_model.pkl")
        logger.info(f"Scaler saved to models/scaler.pkl")
        logger.info(f"Cluster analysis saved to models/cluster_analysis.pkl")
        
        logger.info("=" * 60)
        logger.info(f"✓ Successfully trained model with {optimal_k} clusters")
        logger.info("=" * 60)
        
        return True
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error("Please ensure data/OnlineRetail.csv exists")
        return False
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False


if __name__ == '__main__':
    success = train()
    exit(0 if success else 1)