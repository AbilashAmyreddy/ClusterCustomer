import joblib
import os
import logging
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def save_model(model, model_path):
    try:
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def evaluate_clustering(X, labels, model):
    try:
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': -1.0,
                'davies_bouldin_score': float('inf'),
                'inertia': getattr(model, 'inertia_', None),
            }

        return {
            'silhouette_score': silhouette_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels),
            'inertia': getattr(model, 'inertia_', None),
        }
    except Exception as e:
        logger.error(f"Error evaluating clustering: {str(e)}")
        raise

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def predict_clusters(model, X):
    try:
        if model is None:
            logger.warning("Model is None, returning zero labels")
            return np.zeros(len(X), dtype=int)
        labels = model.predict(X)
        logger.info(f"Predicted clusters for {len(X)} samples")
        return labels
    except Exception as e:
        logger.error(f"Error predicting clusters: {str(e)}")
        raise