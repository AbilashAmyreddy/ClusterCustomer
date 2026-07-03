import streamlit as st
import pandas as pd
import joblib
import logging
import sys
import io
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
BASE_DIR = Path(__file__).resolve().parent

try:
    from data_loader import load_customers
    from feature_engineering import prepare_features
    from segmentation import load_model, predict_clusters
    from utils import (
        generate_cluster_insights, get_cluster_recommendations,
        get_summary_statistics
    )
except ImportError as e:
    # Fallback/mock functions in case the user's src folder is missing or modifying local setup
    logging.warning(f"Failed to import core local modules. Ensure the 'src' directory exists. Error: {e}")

    def load_customers(*args, **kwargs):
        raise ImportError("data_loader module is unavailable")

    def prepare_features(*args, **kwargs):
        raise ImportError("feature_engineering module is unavailable")

    def load_model(*args, **kwargs):
        raise ImportError("segmentation module is unavailable")

    def predict_clusters(*args, **kwargs):
        raise ImportError("segmentation module is unavailable")

    def generate_cluster_insights(*args, **kwargs):
        return {}

    def get_cluster_recommendations(*args, **kwargs):
        return {}

    def get_summary_statistics(rfm_df):
        return {
            'total_customers': len(rfm_df),
            'avg_recency': 0,
            'avg_frequency': 0,
            'avg_monetary': 0,
            'total_revenue': 0,
            'median_recency': 0,
            'median_frequency': 0,
            'median_monetary': 0,
        }

try:
    from visualizations import (
        plot_rfm_3d_scatter, plot_rfm_2d_scatter, plot_pca_clusters,
        plot_cluster_distribution, plot_rfm_distributions, plot_cluster_heatmap
    )
except ImportError as e:
    logging.warning(f"Failed to import visualization modules. Error: {e}")

    def _placeholder_figure(title):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_annotation(text=f"{title} is unavailable", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
        )
        return fig

    def plot_rfm_3d_scatter(*args, **kwargs):
        return _placeholder_figure("3D RFM Scatter")

    def plot_rfm_2d_scatter(*args, **kwargs):
        return _placeholder_figure("2D RFM Scatter")

    def plot_pca_clusters(*args, **kwargs):
        return _placeholder_figure("PCA Projection")

    def plot_cluster_distribution(*args, **kwargs):
        return _placeholder_figure("Cluster Distribution")

    def plot_rfm_distributions(*args, **kwargs):
        return _placeholder_figure("RFM Distributions")

    def plot_cluster_heatmap(*args, **kwargs):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Heatmap unavailable", ha="center", va="center")
        ax.axis("off")
        return fig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern, Smooth, and Adaptive Design
CUSTOM_CSS = """
<style>
    /* Global Styling & Animations */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Header Section - Stunning Animated Gradient */
    .header-section {
        background: linear-gradient(-45deg, #0f172a, #1d4ed8, #0ea5a4, #2563eb);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite, fadeSlideUp 0.8s ease-out;
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(124, 58, 237, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .header-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+PGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMSIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIi8+PC9zdmc+') repeat;
        opacity: 0.5;
    }

    .header-section h1 {
        font-size: 3.2em !important;
        font-weight: 800 !important;
        margin: 0 0 0.5rem 0 !important;
        letter-spacing: -1.5px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .header-section p {
        font-size: 1.25em !important;
        margin: 0 !important;
        opacity: 0.9;
        font-weight: 300;
        letter-spacing: 0.5px;
        position: relative;
        z-index: 1;
    }

    /* Modern Metric Cards */
    .metric-card {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeSlideUp 0.6s ease-out both;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1);
        border-color: #7c3aed;
    }

    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1d4ed8, #0ea5a4);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover::after {
        opacity: 1;
    }

    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 10px 20px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(124, 58, 237, 0.1) !important;
        border-bottom: 3px solid #7c3aed !important;
    }

    /* Insight & Alert Boxes */
    .insight-box, .success-box, .info-box {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        animation: fadeSlideUp 0.5s ease-out both;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.05), rgba(124, 58, 237, 0.05));
        border-left: 4px solid #7c3aed;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(5, 150, 105, 0.05));
        border-left: 4px solid #10b981;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(37, 99, 235, 0.05));
        border-left: 4px solid #3b82f6;
    }

    /* Enhance Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(128,128,128,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        border-color: #7c3aed !important;
        color: #7c3aed !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.15);
    }
    
    /* Primary Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
        transform: translateY(-2px);
    }

    /* Subtitles and Typography */
    h2, h3 {
        letter-spacing: -0.5px;
    }
    
    /* Smooth Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, 0.3); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(128, 128, 128, 0.5); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_default_dataset():
    return load_customers(str(BASE_DIR / 'data' / 'OnlineRetail.csv'))


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes, file_name):
    raw_df = pd.read_csv(io.BytesIO(file_bytes), encoding='ISO-8859-1')
    required_cols = ['InvoiceDate', 'CustomerID', 'Quantity', 'UnitPrice']
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in uploaded file: {', '.join(missing_cols)}")

    raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'], errors='coerce')
    raw_df = raw_df.dropna(subset=['CustomerID']).copy()
    if 'InvoiceNo' in raw_df.columns:
        invoice_no = raw_df['InvoiceNo'].astype(str)
        raw_df = raw_df[~invoice_no.str.startswith('C', na=False)].copy()
    raw_df = raw_df[raw_df['Quantity'] > 0].copy()
    raw_df = raw_df[raw_df['UnitPrice'] > 0].copy()
    raw_df['TotalPrice'] = raw_df['Quantity'] * raw_df['UnitPrice']
    raw_df['CustomerID'] = raw_df['CustomerID'].astype(int)
    return raw_df


@st.cache_resource(show_spinner=False)
def load_trained_artifacts():
    scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
    model = load_model(str(BASE_DIR / 'models' / 'kmeans_model.pkl'))
    if model is None:
        raise FileNotFoundError("Model failed to load")
    return model, scaler


def initialize_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'use_custom_data' not in st.session_state:
        st.session_state.use_custom_data = False


def load_data_and_model():
    try:
        # Determine which data to load
        if st.session_state.use_custom_data and st.session_state.uploaded_file is not None:
            logger.info("Loading custom uploaded data")
            raw_df = load_uploaded_dataset(st.session_state.uploaded_file.getvalue(), st.session_state.uploaded_file.name)
            
            st.session_state.data_loaded = True
            logger.info(f"Custom data loaded: {len(raw_df)} transactions")
        else:
            # Load default data
            raw_df = load_default_dataset()
            st.session_state.data_loaded = True
        
        # Load pre-trained models
        model, scaler = load_trained_artifacts()
        
        X_scaled, _, customer_ids = prepare_features(raw_df.copy(), scaler=scaler)
        
        if customer_ids.empty or X_scaled.shape[0] == 0:
            st.error("No valid customer data found after processing.")
            return None, None, None, None, None, None, False
        
        labels = predict_clusters(model, X_scaled)
        original_rfm_values = scaler.inverse_transform(X_scaled)
        
        clustered_rfm_df = pd.DataFrame({
            'CustomerID': customer_ids.values,
            'Recency': original_rfm_values[:, 0],
            'Frequency': original_rfm_values[:, 1],
            'Monetary': original_rfm_values[:, 2],
            'cluster': labels.astype(str)
        })
        
        st.session_state.model_loaded = True
        return raw_df, clustered_rfm_df, model, scaler, X_scaled, labels, True
    
    except FileNotFoundError:
        st.error("⚠️ Model or Scaler not found. Please ensure 'train_model.py' has been run to generate 'models/kmeans_model.pkl' and 'models/scaler.pkl'.")
        return None, None, None, None, None, None, False
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        return None, None, None, None, None, None, False
    except Exception as e:
        st.error(f"⚠️ Error processing data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None, False


def display_header():
    st.markdown("""
    <div class="header-section">
        <h1>👥 Customer Segmentation Intelligence</h1>
        <p>🚀 Advanced RFM Analysis & K-Means Clustering for Strategic Business Insights</p>
    </div>
    """, unsafe_allow_html=True)


def display_data_source_banner():
    """Display current data source information cleanly."""
    if st.session_state.use_custom_data and st.session_state.uploaded_file is not None:
        st.markdown(f"""
        <div class="success-box">
            <h4 style="margin: 0 0 8px 0; display: flex; align-items: center; gap: 8px;">
                <span style="font-size:1.2em;">✅</span> Custom Data Loaded
            </h4>
            <p style="margin: 0; opacity: 0.9;">
                📁 Analyzing: <strong>{st.session_state.uploaded_file.name}</strong><br>
                Switch to "Default Data" in the sidebar to view the sample OnlineRetail analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0 0 8px 0; display: flex; align-items: center; gap: 8px;">
                <span style="font-size:1.2em;">📊</span> Default Dataset Active
            </h4>
            <p style="margin: 0; opacity: 0.9;">
                Currently viewing the <strong>OnlineRetail.csv</strong> sample dataset. Upload your own CSV file in the sidebar to analyze your data!
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_summary_metrics(rfm_df):
    st.subheader("📊 Key Performance Indicators", anchor=False)
    stats = get_summary_statistics(rfm_df)
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.1s;">
            <div style="color: #6366f1; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                👥 Total Customers
            </div>
            <div style="font-size: 2.2em; font-weight: 800; margin-top: 0.5rem; color: var(--text-color);">
                {stats['total_customers']:,}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.2s;">
            <div style="color: #10b981; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                💰 Total Revenue
            </div>
            <div style="font-size: 2.2em; font-weight: 800; margin-top: 0.5rem; color: var(--text-color);">
                ${stats['total_revenue']/1e6:.1f}M
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.3s;">
            <div style="color: #ec4899; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                🕐 Avg Recency
            </div>
            <div style="font-size: 2.2em; font-weight: 800; margin-top: 0.5rem; color: var(--text-color);">
                {stats['avg_recency']:.0f} <span style="font-size: 0.4em; color: gray; font-weight: 400;">days</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.4s;">
            <div style="color: #f59e0b; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">
                🛍️ Avg Frequency
            </div>
            <div style="font-size: 2.2em; font-weight: 800; margin-top: 0.5rem; color: var(--text-color);">
                {stats['avg_frequency']:.0f} <span style="font-size: 0.4em; color: gray; font-weight: 400;">buys</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_cluster_insights(clustered_rfm_df):
    st.subheader("🎯 Detailed Cluster Analysis", anchor=False)
    
    insights = generate_cluster_insights(clustered_rfm_df)
    recommendations = get_cluster_recommendations(insights)
    
    clusters = sorted(clustered_rfm_df['cluster'].unique())
    
    # Create tabs for each cluster
    tab_names = [f"Segment {c}" for c in clusters]
    tabs = st.tabs(tab_names)
    
    for idx, cluster in enumerate(clusters):
        with tabs[idx]:
            if int(cluster) in insights:
                insight = insights[int(cluster)]
                segment_name = insight.get('segment_name', f'Cluster {cluster}')
                
                st.markdown(f"<h3 style='margin-top: 1rem;'>✨ {segment_name}</h3>", unsafe_allow_html=True)
                
                # Key metrics layout
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    st.markdown(f"""
                    <div class="insight-box" style="border-left-color: #6366f1;">
                        <div style="color: gray; font-size: 0.9em; font-weight: 600;">👥 Segment Size</div>
                        <div style="font-size: 1.8em; font-weight: 800; color: #6366f1;">{insight['size']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f"""
                    <div class="insight-box" style="border-left-color: #8b5cf6;">
                        <div style="color: gray; font-size: 0.9em; font-weight: 600;">📈 Market Share</div>
                        <div style="font-size: 1.8em; font-weight: 800; color: #8b5cf6;">{insight['percentage']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c3:
                    st.markdown(f"""
                    <div class="insight-box" style="border-left-color: #ec4899;">
                        <div style="color: gray; font-size: 0.9em; font-weight: 600;">💸 Avg Value</div>
                        <div style="font-size: 1.8em; font-weight: 800; color: #ec4899;">{insight['avg_monetary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with c4:
                    st.markdown(f"""
                    <div class="insight-box" style="border-left-color: #10b981;">
                        <div style="color: gray; font-size: 0.9em; font-weight: 600;">💰 Total Revenue</div>
                        <div style="font-size: 1.8em; font-weight: 800; color: #10b981;">{insight['total_monetary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed RFM Expander
                with st.expander("📊 Detailed RFM Profile", expanded=True):
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        st.metric("🕐 Recency (Days)", insight['avg_recency'], delta=insight['recency_range'], delta_color="off")
                    with rc2:
                        st.metric("🛍️ Frequency", insight['avg_frequency'], delta=insight['frequency_range'], delta_color="off")
                    with rc3:
                        st.metric("💰 Monetary ($)", insight['avg_monetary'], delta=insight['monetary_range'], delta_color="off")

                # Recommendations
                if int(cluster) in recommendations:
                    st.markdown(f"""
                    <div class="info-box" style="margin-top: 1.5rem;">
                        <h4 style="margin: 0 0 1rem 0; color: var(--text-color);">💡 Strategic Recommendations</h4>
                        <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                            {''.join([f"<li>{rec}</li>" for rec in recommendations[int(cluster)]])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)


def display_visualizations(clustered_rfm_df, X_scaled, labels):
    st.subheader("📈 Advanced Visual Analytics", anchor=False)
    
    viz_tabs = st.tabs([
        "🎯 3D Space", 
        "📊 2D Comparison", 
        "🔵 PCA Projection",
        "📈 Distribution",
        "📦 Box Plots",
        "🔥 Heatmap"
    ])
    
    with viz_tabs[0]:
        st.markdown("**Interactive 3D visualization of customer clusters in RFM space**")
        st.plotly_chart(plot_rfm_3d_scatter(clustered_rfm_df), use_container_width=True)
    
    with viz_tabs[1]:
        st.markdown("**Compare customer segments across different RFM dimensions**")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_rfm_2d_scatter(clustered_rfm_df, x='Recency', y='Monetary'), use_container_width=True)
        with col2:
            st.plotly_chart(plot_rfm_2d_scatter(clustered_rfm_df, x='Frequency', y='Monetary'), use_container_width=True)
    
    with viz_tabs[2]:
        st.markdown("**Principal Component Analysis - Reduced dimensional view**")
        st.plotly_chart(plot_pca_clusters(X_scaled, labels), use_container_width=True)
    
    with viz_tabs[3]:
        st.plotly_chart(plot_cluster_distribution(labels), use_container_width=True)
    
    with viz_tabs[4]:
        st.plotly_chart(plot_rfm_distributions(clustered_rfm_df), use_container_width=True)
    
    with viz_tabs[5]:
        st.markdown("**Cluster profile comparison - Average RFM values heatmap**")
        fig = plot_cluster_heatmap(clustered_rfm_df)
        st.pyplot(fig, use_container_width=True)


def display_data_tables(clustered_rfm_df):
    st.subheader("📋 Data Explorer & Export", anchor=False)
    
    table_tabs = st.tabs(["🔍 Explore & Filter", "📊 Statistical Summary"])
    
    with table_tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            cluster_filter = st.multiselect(
                "Filter by Segment",
                options=sorted(clustered_rfm_df['cluster'].unique()),
                default=sorted(clustered_rfm_df['cluster'].unique())
            )
        with col2:
            sort_by = st.selectbox("Sort Data By", options=['Monetary', 'Frequency', 'Recency'])
        with col3:
            sort_order = st.selectbox("Order", options=['Descending', 'Ascending'])
            
        # Slider filters embedded in expander
        with st.expander("🎛️ Advanced Range Filters"):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                r_min, r_max = int(clustered_rfm_df['Recency'].min()), int(clustered_rfm_df['Recency'].max())
                recency_range = st.slider("Recency Range (Days)", r_min, r_max, (r_min, r_max))
            with r_col2:
                m_min, m_max = int(clustered_rfm_df['Monetary'].min()), int(clustered_rfm_df['Monetary'].max())
                monetary_range = st.slider("Monetary Range ($)", m_min, m_max, (m_min, m_max))
        
        # Apply filters
        filtered_df = clustered_rfm_df[
            (clustered_rfm_df['cluster'].isin(cluster_filter)) &
            (clustered_rfm_df['Recency'].between(recency_range[0], recency_range[1])) &
            (clustered_rfm_df['Monetary'].between(monetary_range[0], monetary_range[1]))
        ].sort_values(by=sort_by, ascending=(sort_order == 'Ascending'))
        
        st.markdown(f"**Showing {len(filtered_df):,} matching records**")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results (CSV)",
            data=csv,
            file_name="segmented_customers.csv",
            mime="text/csv"
        )
        
    with table_tabs[1]:
        st.markdown("**Aggregate Statistics by Segment**")
        cluster_stats = clustered_rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].agg(
            ['count', 'mean', 'median', 'max']
        ).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)


def main():
    initialize_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2950/2950993.png", width=60)
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown("---")
        
        st.markdown("### 📁 Data Source")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Default Data", use_container_width=True):
                st.session_state.use_custom_data = False
                st.session_state.data_loaded = False
                st.rerun()
        with col2:
            if st.button("📤 Custom CSV", use_container_width=True):
                st.session_state.use_custom_data = True
                
        current_source = "Custom Upload" if st.session_state.use_custom_data else "OnlineRetail.csv"
        st.caption(f"📍 Active: **{current_source}**")
        
        if st.session_state.use_custom_data:
            st.markdown("---")
            uploaded_file = st.file_uploader(
                "Upload Transaction CSV", 
                type="csv",
                help="Needs: InvoiceDate, CustomerID, Quantity, UnitPrice"
            )
            
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.success("✅ File ready to process")
                if st.button("🔄 Process File", use_container_width=True):
                    st.session_state.data_loaded = False
                    st.rerun()
        
        st.markdown("---")
        if st.button("🔄 Refresh Application", use_container_width=True, type="secondary"):
            st.session_state.data_loaded = False
            st.rerun()
            
        with st.expander("📖 Guide & Terminology"):
            st.markdown("""
            **RFM Framework:**
            - **Recency**: Days since last purchase.
            - **Frequency**: Total transaction count.
            - **Monetary**: Total lifetime value.
            """)

    # Main Canvas
    display_header()
    
    with st.spinner("⏳ Synthesizing models and fetching data..."):
        raw_df, clustered_rfm_df, model, scaler, X_scaled, labels, success = load_data_and_model()
    
    if not success or clustered_rfm_df is None:
        st.info("👋 Please upload a valid CSV file or switch to the Default Dataset in the sidebar to begin.")
        st.stop()
    
    display_data_source_banner()
    st.markdown("<br>", unsafe_allow_html=True)
    
    display_summary_metrics(clustered_rfm_df)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Core Dashboard Tabs
    tab_viz, tab_insights, tab_data, tab_about = st.tabs([
        "🎨 Visual Analytics", 
        "💡 Segment Insights", 
        "🗃️ Data Explorer", 
        "ℹ️ Platform Info"
    ])
    
    with tab_viz:
        display_visualizations(clustered_rfm_df, X_scaled, labels)
    with tab_insights:
        display_cluster_insights(clustered_rfm_df)
    with tab_data:
        display_data_tables(clustered_rfm_df)
    with tab_about:
        ac1, ac2 = st.columns(2, gap="large")
        with ac1:
            st.markdown("""
            ### 🚀 About This Platform
            **ClusterCustomer Intelligence** transforms raw transaction data into actionable behavioral segments using Machine Learning.
            
            **Methodology:**
            1. Data preprocessing and anomaly removal.
            2. RFM (Recency, Frequency, Monetary) metric engineering.
            3. Feature standardization via `StandardScaler`.
            4. Unsupervised clustering using **K-Means**.
            5. Automated insight generation and visualization.
            """)
        with ac2:
            st.markdown("""
            ### 🛠️ Technical Stack
            - **Core**: Python, Pandas, NumPy
            - **Machine Learning**: Scikit-Learn
            - **Visualization**: Plotly, Matplotlib, Seaborn
            - **Frontend**: Streamlit
            """)
            st.markdown(f"""
            <div class="success-box" style="margin-top:1rem;">
                <h4 style="margin:0 0 10px 0;">✅ Model Status Active</h4>
                <div style="font-size: 0.9em; opacity: 0.9;">
                    Clusters Discovered: {len(clustered_rfm_df['cluster'].unique())}<br>
                    Records Analyzed: {len(clustered_rfm_df):,}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 1.25rem 0 0.5rem 0; color: rgba(128, 128, 128, 0.85); font-size: 0.95rem;">
        Built by Abilash Amyreddy
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()