import os
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")


def generate_cluster_insights(rfm_df, original_df=None):
    """
    Generate business insights for each cluster.
    
    Args:
        rfm_df: DataFrame with RFM features and cluster assignments
        original_df: Optional original transaction DataFrame
        
    Returns:
        Dictionary with insights for each cluster
    """
    insights = {}
    
    try:
        clusters = rfm_df['cluster'].unique()
        
        for cluster in sorted(clusters):
            cluster_data = rfm_df[rfm_df['cluster'] == cluster]
            
            insights[int(cluster)] = {
                'size': len(cluster_data),
                'percentage': f"{(len(cluster_data) / len(rfm_df) * 100):.1f}%",
                'avg_recency': f"{cluster_data['Recency'].mean():.0f} days",
                'avg_frequency': f"{cluster_data['Frequency'].mean():.0f} purchases",
                'avg_monetary': f"${cluster_data['Monetary'].mean():.2f}",
                'total_monetary': f"${cluster_data['Monetary'].sum():.2f}",
                'recency_range': f"{cluster_data['Recency'].min():.0f} - {cluster_data['Recency'].max():.0f}",
                'frequency_range': f"{cluster_data['Frequency'].min():.0f} - {cluster_data['Frequency'].max():.0f}",
                'monetary_range': f"${cluster_data['Monetary'].min():.2f} - ${cluster_data['Monetary'].max():.2f}"
            }
            
            # Generate segment name based on RFM characteristics
            avg_recency = cluster_data['Recency'].mean()
            avg_frequency = cluster_data['Frequency'].mean()
            avg_monetary = cluster_data['Monetary'].mean()
            
            if avg_recency < rfm_df['Recency'].quantile(0.33) and avg_frequency > rfm_df['Frequency'].quantile(0.67):
                segment = "💎 Loyal Champions"
            elif avg_monetary > rfm_df['Monetary'].quantile(0.67):
                segment = "🌟 High Value Customers"
            elif avg_recency > rfm_df['Recency'].quantile(0.67) and avg_monetary < rfm_df['Monetary'].quantile(0.33):
                segment = "⚠️ At-Risk Customers"
            elif avg_recency > rfm_df['Recency'].quantile(0.67):
                segment = "😴 Dormant Customers"
            elif avg_frequency < rfm_df['Frequency'].quantile(0.33):
                segment = "🌱 New Customers"
            else:
                segment = "📊 Standard Customers"
            
            insights[int(cluster)]['segment_name'] = segment
        
        logger.info(f"Generated insights for {len(insights)} clusters")
        return insights
    
    except Exception as e:
        logger.error(f"Error generating cluster insights: {str(e)}")
        return {}


def get_cluster_recommendations(cluster_insights):
    """
    Generate business recommendations based on cluster characteristics.
    
    Args:
        cluster_insights: Dictionary of cluster insights
        
    Returns:
        Dictionary with recommendations for each cluster
    """
    recommendations = {}
    
    try:
        for cluster_id, insight in cluster_insights.items():
            segment = insight.get('segment_name', '')
            
            if 'Loyal Champions' in segment:
                recommendations[cluster_id] = [
                    "🎁 Offer exclusive loyalty rewards",
                    "📧 Send VIP member communications",
                    "🔔 Request referral reviews",
                    "💝 Provide early access to new products"
                ]
            elif 'High Value' in segment:
                recommendations[cluster_id] = [
                    "👑 Provide premium customer service",
                    "💰 Offer premium/luxury products",
                    "📱 Personalized shopping experiences",
                    "🎯 Target with high-margin offerings"
                ]
            elif 'At-Risk' in segment:
                recommendations[cluster_id] = [
                    "🚨 Launch re-engagement campaign",
                    "🎁 Offer special discounts",
                    "📞 Direct customer outreach",
                    "💌 Personalized win-back offers"
                ]
            elif 'Dormant' in segment:
                recommendations[cluster_id] = [
                    "⏰ Launch dormancy recovery campaign",
                    "🔥 Flash sales and urgency offers",
                    "📬 Multi-channel re-activation",
                    "🎉 Nostalgia-based marketing"
                ]
            elif 'New' in segment:
                recommendations[cluster_id] = [
                    "👋 Welcome series emails",
                    "🎓 Educational content",
                    "🏃 Encourage repeat purchase",
                    "💳 First-time buyer incentives"
                ]
            else:
                recommendations[cluster_id] = [
                    "📈 Standard marketing campaigns",
                    "🎯 Upsell relevant products",
                    "📊 A/B test personalization",
                    "💡 Gradual engagement increase"
                ]
        
        logger.info(f"Generated recommendations for {len(recommendations)} clusters")
        return recommendations
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {}


def export_segmentation_results(rfm_df, output_path='data/segmentation_results.csv'):
    """
    Export segmentation results to CSV.
    
    Args:
        rfm_df: DataFrame with RFM features and cluster assignments
        output_path: Path to save the results
        
    Returns:
        Path to exported file
    """
    try:
        ensure_dir(os.path.dirname(output_path))
        rfm_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rfm_df)} records to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise


def format_currency(value):
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value, total):
    """Format as percentage."""
    return f"{(value / total * 100):.1f}%"


def get_summary_statistics(rfm_df):
    """
    Get summary statistics for RFM features.
    
    Args:
        rfm_df: DataFrame with RFM features
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_customers': len(rfm_df),
        'avg_recency': rfm_df['Recency'].mean(),
        'avg_frequency': rfm_df['Frequency'].mean(),
        'avg_monetary': rfm_df['Monetary'].mean(),
        'total_revenue': rfm_df['Monetary'].sum(),
        'median_recency': rfm_df['Recency'].median(),
        'median_frequency': rfm_df['Frequency'].median(),
        'median_monetary': rfm_df['Monetary'].median()
    }
