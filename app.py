# =====================================================================
# DATA INSIGHTS HUB - IMPORTS
# =====================================================================
# Professional time-series data exploration and visualization platform

# Standard Library
from __future__ import annotations
import os
import io
import math
import time
import hashlib
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime

# Third-party: Scientific & Data Processing
import numpy as np
import pandas as pd
from scipy.stats import chi2

# Third-party: Visualization
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib as mpl

# Third-party: Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Third-party: Web Framework
import streamlit as st

# =====================================================================
# ENVIRONMENT SETUP
# =====================================================================

os.environ["OMP_NUM_THREADS"] = "1"

# =====================================================================
# CONSTANTS & CONFIGURATION
# =====================================================================

# Plotly Theme Configuration
TEMPLATE = "plotly_white"
PAPER_BG = "white"
PLOT_BG = "white"
FONT_COLOR = "#0E1117"
GRID_COLOR = "#E5ECF6"
FONT_PRIMARY = "#0E1117"
FONT_SIZE_TICK = 11
STANDARD_MARGIN = dict(l=60, r=40, t=50, b=60)
GRID_WIDTH = 1

# =====================================================================
# CONDITIONAL IMPORTS (with Fallbacks)
# =====================================================================

# Scikit-learn (Optional)
SKLEARN_AVAILABLE = False
try:
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.cluster import MiniBatchKMeans as SkMiniBatchKMeans
    from sklearn.metrics import silhouette_score
    PCA = SkPCA
    KMeans = SkKMeans
    MiniBatchKMeans = SkMiniBatchKMeans
    SKLEARN_AVAILABLE = True
except Exception:
    MiniBatchKMeans = None
    
    class PCA:
        """Fallback PCA implementation when scikit-learn unavailable"""
        def __init__(self, n_components=2, random_state=None):
            self.n_components = n_components
            self.random_state = random_state
        def fit(self, X):
            X = np.asarray(X, dtype=float)
            self.mean_ = X.mean(axis=0)
            Xc = X - self.mean_
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.components_ = Vt[: self.n_components]
            return self
        def fit_transform(self, X):
            self.fit(X)
            Xc = np.asarray(X, dtype=float) - self.mean_
            return np.dot(Xc, self.components_.T)

    class KMeans:
        """Fallback KMeans implementation when scikit-learn unavailable"""
        def __init__(self, n_clusters=8, n_init=10, random_state=None, max_iter=300):
            self.n_init = int(n_init)
            self.n_clusters = int(n_clusters)
            self.random_state = random_state
            self.max_iter = int(max_iter)
        def fit_predict(self, X):
            X = np.asarray(X, dtype=float)
            n_samples = X.shape[0]
            rng = np.random.RandomState(self.random_state)
            centers = X[rng.choice(n_samples, min(self.n_clusters, n_samples), replace=False)]
            for _ in range(self.max_iter):
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(self.n_clusters)])
                if np.allclose(new_centers, centers):
                    break
                centers = new_centers
            self.labels_ = labels
            return labels

    silhouette_score = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

def _initialize_session_state():
    """Initialize all session state variables at app startup"""
    if "events" not in st.session_state:
        st.session_state["events"] = []
    if "user_prefs" not in st.session_state:
        st.session_state["user_prefs"] = {
            "default_chart": "Time Series Trend",
            "show_stats": False,
        }
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        st.session_state['prev_datetime_col'] = None
        st.session_state['prev_dayfirst'] = None
        st.session_state['prev_file_hash'] = None
        st.session_state['num_cols'] = []
        st.session_state['cat_cols'] = []
        st.session_state['prev_df_id_for_types'] = None
    if 'df_filtered' not in st.session_state:
        st.session_state['df_filtered'] = None
        st.session_state['prev_start_date'] = None
        st.session_state['prev_end_date'] = None
        st.session_state['prev_missing_strategy'] = None
        st.session_state['prev_filters_tuple'] = None
        st.session_state['prev_resample_freq'] = None
        st.session_state['prev_df_id'] = None
    if 'ss_k' not in st.session_state: st.session_state['ss_k'] = 3
    if 'ss_window' not in st.session_state: st.session_state['ss_window'] = "1H"
    if 'ss_threshold' not in st.session_state: st.session_state['ss_threshold'] = 1.0
    if 'ss_duration' not in st.session_state: st.session_state['ss_duration'] = 60
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

_initialize_session_state()

# =====================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================

# Page Config
st.set_page_config(
    page_title="Data Insights Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# AUTHENTICATION
# =====================================================================

def authenticate_user():
    """Simple front-end password gate"""
    if "password_ok" not in st.session_state:
        st.session_state["password_ok"] = False

    if not st.session_state["password_ok"]:
        st.markdown("## 🔐 Enter Password to Access Application")
        entered_password = st.text_input(
            "Password", 
            type="password", 
            placeholder="Enter password to continue"
        )
        if st.button("Login"):
            try:
                correct_password = st.secrets.get("PASSWORD", "")
                if entered_password == correct_password:
                    st.session_state["password_ok"] = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            except Exception as e:
                st.error(f"❌ Authentication error: {str(e)}")
        st.stop()

authenticate_user()

# Enhanced CSS with industry-standard styling
st.markdown("""
<style>
    /* Main Layout */
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1F2A44;
        text-align: center;
        padding: 1rem 0 1rem 0;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 14px;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    /* Section Headers */
    .section-header {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
        font-size: 20px;
        font-weight: 600;
        color: #1F2A44;
    }
    
    /* Improved Tiles */
    .tile {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .tile:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .tile h3 {
        font-size: 28px;
        margin: 0 0 8px 0;
        color: #1F2A44;
        font-weight: 700;
    }
    
    .tile p {
        font-size: 14px;
        color: #6B7280;
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .tile .icon {
        font-size: 24px;
        margin-bottom: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background-color: #2196F3;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1976D2;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* ============================================= */
    /* CUSTOM NAVIGATION TABS (Radio Button Styling) */
    /* ============================================= */
    
    div[role="radiogroup"] {
        background-color: #f1f3f5;
        padding: 6px;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 24px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);
    }
    
    div[role="radiogroup"] label {
        flex-grow: 1;
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 10px 16px;
        text-align: center;
        transition: all 0.2s;
        font-weight: 600;
        color: #4B5563;
        cursor: pointer;
        position: relative;
    }
    
    /* Hides the actual circle button of radio */
    div[role="radiogroup"] label[data-testid*="stRadioButton"] > div:first-child {
        display: none;
    }
    
    /* 🌟 SELECTED STATE - 3D Light Blue Gradient 🌟 */
    /* Uses :has() to target the label containing the checked input */
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(to bottom, #E3F2FD 0%, #90CAF9 100%);
        color: white !important;
        border: 1px solid #0277BD;
        /* Inner highlight for 3D effect + Drop shadow */
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.4),
            inset 0 -2px 0 rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }

    /* Hover state for non-selected items */
    div[role="radiogroup"] label:not(:has(input:checked)):hover {
        background-color: rgba(255,255,255,0.6);
        color: #0288d1;
    }
    
    /* ============================================= */
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background-color: #f9fafb;
        border-radius: 12px;
        border: 2px dashed #d1d5db;
        margin: 2rem 0;
    }
    
    .empty-state-icon {
        font-size: 64px;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .empty-state-title {
        font-size: 20px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-desc {
        color: #6B7280;
        font-size: 14px;
    }
    
    /* Alert Improvements */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 12px 16px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 15px;
        color: #1F2A44;
    }
    
    /* Input Fields */
    .stSelectbox, .stMultiselect, .stNumberInput, .stTextInput {
        margin-bottom: 1rem;
    }
    
    /* Loading State */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    
    /* Keyboard Shortcuts Hint */
    .shortcut-hint {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(31, 42, 68, 0.9);
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 11px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .shortcut-hint kbd {
        background: rgba(255,255,255,0.2);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Custom CSS for Sidebar Border ====================
st.markdown(
    """
    <style>
    /* Target the sidebar element by its data-testid */
    section[data-testid="stSidebar"] {
        border-right: 3px solid #ADD8E6; /* Blue border on the right (NEW) */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==================== Helper Functions ====================

def show_empty_state(icon: str, title: str, description: str):
    """Display an attractive empty state"""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def handle_error(error: Exception, context: str = "operation"):
    """Provide helpful error messages"""
    error_messages = {
        "ValueError": "Invalid data format. Please check your input values.",
        "KeyError": "Column not found. The data structure may have changed.",
        "MemoryError": "Dataset too large. Try filtering or resampling first.",
        "FileNotFoundError": "File not found. Please check the file path.",
        "PermissionError": "Permission denied. Check file access rights.",
    }
    
    error_type = type(error).__name__
    message = error_messages.get(error_type, "An unexpected error occurred")
    
    with st.expander(f"❌ {context.title()} Error - Click for details", expanded=False):
        st.error(f"**{message}**")
        st.code(f"{error_type}: {str(error)}", language="python")
        st.info("💡 **Tip**: Try reducing the date range, number of parameters, or apply filters to reduce data size.")

def show_success(message: str, duration: int = 3):
    """Show success message that auto-dismisses"""
    success_placeholder = st.empty()
    success_placeholder.success(f"✅ {message}")
    time.sleep(duration)
    success_placeholder.empty()

@st.cache_data(show_spinner=False)
def _file_bytes(file) -> Optional[bytes]:
    if not file:
        return None
    return file.getvalue()

def fix_invalid_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns containing mixed numeric + text values into clean numeric columns.
    Invalid numeric strings become NaN.
    """
    df = df.copy()
    for col in df.columns:
        try:
            # Try converting the entire column to numeric (coerce invalid → NaN)
            converted = pd.to_numeric(df[col], errors="coerce")
            # If at least some values converted to numbers, treat column as numeric
            if converted.notna().sum() > 0:
                df[col] = converted
        except Exception:
            # If any unexpected error, skip conversion for this column
            continue
    return df

@st.cache_data(show_spinner=False)
def load_dataframe(file_bytes: Optional[bytes], filename: Optional[str]) -> Optional[pd.DataFrame]:
    if not file_bytes or not filename:
        return None
    name = filename.lower()
    bio = io.BytesIO(file_bytes)
    try:
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(bio, low_memory=False)
            except UnicodeDecodeError:
                bio.seek(0)
                df = pd.read_csv(bio, low_memory=False, encoding="latin1")
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(bio)
        elif name.endswith(".parquet"):
            try:
                df = pd.read_parquet(bio, engine="pyarrow")
            except Exception:
                df = pd.read_parquet(bio)
        else:
            st.error("❌ Unsupported file type. Use CSV, Excel, or Parquet.")
            return None
            
        if df is not None and not df.empty:
            df = fix_invalid_numeric_strings(df)
            
    except Exception as e:
        handle_error(e, "File Loading")
        return None
    return df

def coerce_datetime(df: pd.DataFrame, dt_col: str, dayfirst: bool) -> pd.DataFrame:
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce", dayfirst=dayfirst)
    out = out.dropna(subset=[dt_col])
    out = out.set_index(dt_col).sort_index()
    return out

def compress_types(df: pd.DataFrame, cat_threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        try:
            nunique = out[c].nunique(dropna=True)
            total = len(out[c])
            if total > 0 and (nunique / total) <= cat_threshold:
                out[c] = out[c].astype("category")
        except Exception:
            continue
    return out

def numeric_df(df_in: pd.DataFrame) -> pd.DataFrame:
    return df_in.apply(pd.to_numeric, errors="coerce")

def zscore(df_in: pd.DataFrame) -> pd.DataFrame:
    return (df_in - df_in.mean()) / (df_in.std(ddof=0) + 1e-12)

def thin_index(idx: pd.DatetimeIndex, max_points: int) -> np.ndarray:
    n = len(idx)
    if n <= max_points:
        return np.arange(n)
    step = max(1, math.ceil(n / max_points))
    return np.arange(0, n, step)

def apply_missing(df_in: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df_out = df_in.copy()
    if strategy == "Drop missing rows":
        df_out.dropna(inplace=True)
    elif strategy == "Forward fill":
        df_out.ffill(inplace=True)
    elif strategy == "Backward fill":
        df_out.bfill(inplace=True)
    elif strategy == "Interpolate (time)":
        try:
            df_out = df_out.interpolate(method="time")
        except Exception:
            df_out = df_out.interpolate()
    return df_out

def apply_filters(df_in: pd.DataFrame, filters: Tuple[Tuple[str, str, Any], ...]) -> pd.DataFrame:
    df_out = df_in
    for (column, operator, value) in filters:
        col_vals = df_out[column]
        converted = pd.to_numeric(col_vals, errors="coerce")
        if converted.notna().any():
            series = converted
            if operator == ">":
                df_out = df_out[series > float(value)]
            elif operator == "<":
                df_out = df_out[series < float(value)]
            elif operator == "=":
                df_out = df_out[series == float(value)]
            elif operator == "≠":
                df_out = df_out[series != float(value)]
            elif operator == ">=":
                df_out = df_out[series >= float(value)]
            elif operator == "<":
                df_out = df_out[series <= float(value)]
        else:
            s = col_vals.astype(str)
            if operator == "=":
                df_out = df_out[s == str(value)]
            elif operator == "≠":
                df_out = df_out[s != str(value)]
    return df_out

# --- SMART CONFIGURATION HELPER ---
def calculate_smart_defaults(df: pd.DataFrame, col: str) -> Dict:
    """
    Optimized version with sampling for large datasets
    Handles non-monotonic indexes properly
    """
    series = df[col].dropna()
    if series.empty:
        return {}

    # CRITICAL: Ensure monotonic index before any operations
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    
    # Remove duplicate timestamps
    series = series[~series.index.duplicated(keep='first')]
    
    # Check if we still have data after cleaning
    if series.empty or len(series) < 10:
        return {
            "k": 3,
            "window": "1H",
            "threshold": 1.0,
            "duration": 60
        }

    # Sample for large datasets using systematic sampling (preserves time order)
    if len(series) > 10000:
        step = max(1, len(series) // 10000)
        series = series.iloc[::step]

    # 1. Detect Sampling Frequency
    try:
        if len(series) > 1:
            diffs = series.index.to_series().diff().dropna()
            if len(diffs) > 0:
                freq_seconds = diffs.median().total_seconds()
            else:
                freq_seconds = 60
        else:
            freq_seconds = 60
    except:
        freq_seconds = 60

    # 2. Determine Window Size
    target_samples = 5
    window_seconds = max(freq_seconds * target_samples, 60)
    
    if window_seconds < 60:
        window_size = "1min"
        window_minutes = 1
    elif window_seconds < 3600:
        mins = int(window_seconds / 60)
        window_size = f"{mins}min"
        window_minutes = mins
    else:
        hours = int(window_seconds / 3600)
        if hours == 0: 
            hours = 1
        window_size = f"{hours}H"
        window_minutes = hours * 60

    # 3. Determine Std Threshold (Optimized)
    try:
        rolling_std = series.rolling(window=window_size, min_periods=1).std().dropna()
        
        if not rolling_std.empty:
            baseline_noise = rolling_std.quantile(0.20)
            if baseline_noise == 0 or pd.isna(baseline_noise):
                baseline_noise = series.std() * 0.01 
            std_threshold = float(baseline_noise * 3.0)
        else:
            std_threshold = series.std() * 0.1
    except Exception as e:
        st.warning(f"Could not calculate rolling std: {e}")
        std_threshold = series.std() * 0.1

    # Handle edge case where std is 0 or NaN
    if pd.isna(std_threshold) or std_threshold == 0:
        std_threshold = 1.0

    # 4. Determine K using faster MiniBatch KMeans
    best_k = 3
    if SKLEARN_AVAILABLE and len(series) > 50:
        try:
            sample_size = min(len(series), 1000)
            # Systematic sampling to preserve distribution
            step = max(1, len(series) // sample_size)
            X = series.iloc[::step].values.reshape(-1, 1)
            
            best_score = -1
            for k in range(2, 6):
                try:
                    if MiniBatchKMeans is not None:
                        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100, n_init=3)
                    else:
                        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    labels = km.fit_predict(X)
                    
                    if len(np.unique(labels)) > 1 and silhouette_score is not None:
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                except:
                    continue
        except:
            pass

    min_duration = max(window_minutes, 10)

    return {
        "k": int(best_k),
        "window": window_size,
        "threshold": float(std_threshold),
        "duration": int(min_duration)
    }


def run_kmeans_clustering(series: pd.Series, num_clusters: int, 
                          progress_callback=None) -> tuple:
    """
    Run K-Means clustering with optimization for large datasets
    Handles non-monotonic indexes properly
    
    Returns:
        (cluster_dataframe, cluster_means_sorted)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("Scikit-learn is required")
    
    # CRITICAL: Ensure monotonic index
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    series = series[~series.index.duplicated(keep='first')]
    
    if series.empty or len(series) < num_clusters:
        raise ValueError(f"Insufficient data points ({len(series)}) for {num_clusters} clusters")
    
    # Use sampling for very large datasets
    max_samples = 50000
    if len(series) > max_samples:
        if progress_callback:
            progress_callback(0.1, f"Sampling {max_samples} points from {len(series):,}...")
        
        # Systematic sampling to preserve distribution and time order
        sample_indices = np.linspace(0, len(series) - 1, max_samples, dtype=int)
        series_sample = series.iloc[sample_indices]
        full_series = series.copy()
        use_prediction = True
    else:
        series_sample = series
        full_series = series
        use_prediction = False
    
    if progress_callback:
        progress_callback(0.3, "Running K-Means clustering...")
    
    # Prepare data
    data = series_sample.values.reshape(-1, 1)
    
    # Use MiniBatchKMeans for large datasets
    if MiniBatchKMeans is not None and len(series_sample) > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters, 
            random_state=42, 
            batch_size=min(1000, len(series_sample) // 10),
            n_init=3
        )
    else:
        kmeans = KMeans(
            n_clusters=num_clusters, 
            random_state=42, 
            n_init='auto' if SKLEARN_AVAILABLE else 10
        )
    
    # Fit the model
    if use_prediction:
        kmeans.fit(data)
        if progress_callback:
            progress_callback(0.7, "Predicting clusters for full dataset...")
        # Predict on full dataset
        full_data = full_series.values.reshape(-1, 1)
        clusters = kmeans.predict(full_data)
        series_to_use = full_series
    else:
        clusters = kmeans.fit_predict(data)
        series_to_use = series_sample
    
    if progress_callback:
        progress_callback(0.9, "Calculating cluster statistics...")
    
    # Create dataframe with results
    df_temp = pd.DataFrame({
        'Value': series_to_use.values, 
        'Cluster': clusters
    }, index=series_to_use.index)
    
    # Calculate cluster means
    cluster_means = df_temp.groupby('Cluster')['Value'].mean().sort_values(ascending=False)
    
    if progress_callback:
        progress_callback(1.0, "Complete!")
    
    return df_temp, cluster_means

def automated_steady_state_detection(df: pd.DataFrame, cols: Any, 
                                   min_duration_minutes: int, 
                                   sensitivity: float = 1.0) -> Tuple[pd.DataFrame, Any]:
    """
    Detects steady states automatically. Supports both Univariate and Multivariate.
    
    Args:
        df: Dataframe with datetime index
        cols: Column name (str) or list of column names (list)
        min_duration_minutes: Minimum duration to count as steady
        sensitivity: 0.1 (Strict) to 5.0 (Loose). Default 1.0.
    """
    # Determine mode
    is_multivariate = isinstance(cols, list)
    target_cols = cols if is_multivariate else [cols]
    
    # 1. OPTIMIZATION: Work with a lightweight copy
    ts = df[target_cols].dropna().sort_index()
    if ts.empty: return pd.DataFrame(), None

    original_len = len(ts)
    if original_len > 50000:
        total_duration = ts.index[-1] - ts.index[0]
        if total_duration > pd.Timedelta(days=30): rule = '1H'
        elif total_duration > pd.Timedelta(days=7): rule = '15min'
        elif total_duration > pd.Timedelta(days=1): rule = '5min'
        else: rule = '1min'
        
        # Resample: Mean for value, we will recalc std later
        ts_resampled = ts.resample(rule).mean().dropna()
    else:
        ts_resampled = ts

    # 2. CALCULATE STABILITY (Vectorized)
    window_points = max(3, int(min_duration_minutes / 2))
    
    if is_multivariate:
        # Normalize columns (Z-score) so variance is comparable across different units
        # (e.g. 3000 RPM vs 0.5 Bar)
        df_norm = (ts_resampled - ts_resampled.mean()) / (ts_resampled.std() + 1e-12)
        
        # Calculate rolling std on NORMALIZED data
        rolling_std_all = df_norm.rolling(window=window_points, min_periods=2).std()
        
        # Composite Instability: Take the MAX instability across all parameters at each timestamp.
        # This implies "Steady State" means ALL parameters must be steady simultaneously.
        rolling_std = rolling_std_all.max(axis=1)
    else:
        rolling_std = ts_resampled[cols].rolling(window=window_points, min_periods=2).std()

    # 3. AUTO-THRESHOLDING
    base_noise_level = rolling_std.quantile(0.15) 
    
    if base_noise_level == 0:
        base_noise_level = 0.001 if is_multivariate else ts_resampled.std().iloc[0] * 0.001
        
    dynamic_threshold = base_noise_level * (2.0 * sensitivity)

    # 4. IDENTIFY STEADY ZONES
    is_steady = rolling_std <= dynamic_threshold
    
    # 5. MERGE & FILTER SEGMENTS
    steady_int = is_steady.astype(int)
    diff = steady_int.diff()
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if steady_int.iloc[0] == 1: starts = np.insert(starts, 0, 0)
    if steady_int.iloc[-1] == 1: ends = np.append(ends, len(steady_int))
        
    if len(starts) > len(ends): starts = starts[:len(ends)]
    if len(ends) > len(starts): ends = ends[:len(starts)]
    
    results = []
    
    for start_idx, end_idx in zip(starts, ends):
        t_start = ts_resampled.index[start_idx]
        t_end = ts_resampled.index[end_idx - 1]
        
        duration = t_end - t_start
        
        if duration >= pd.Timedelta(minutes=min_duration_minutes):
            # OPTIMIZATION 1: Use 'ts_resampled' instead of 'ts'
            # This drastically reduces the number of data points to process
            segment_data = ts_resampled[t_start:t_end]
            
            if not segment_data.empty:
                # Basic info
                row_data = {
                    'Start': t_start,
                    'End': t_end,
                    'Duration': duration,
                    'Duration_Min': duration.total_seconds() / 60
                }
                
                # OPTIMIZATION 2: Vectorized aggregation
                # Calculate all stats for all target columns in one optimized Pandas call
                # This is significantly faster than looping through columns and calculating individually
                stats = segment_data[target_cols].agg(['mean', 'std', 'min', 'max'])
                
                # Flatten the stats into the row_data dictionary
                for c in target_cols:
                    row_data[f'Mean_{c}'] = stats.at['mean', c]
                    row_data[f'Std_{c}'] = stats.at['std', c]
                    row_data[f'Min_{c}'] = stats.at['min', c]
                    row_data[f'Max_{c}'] = stats.at['max', c]
                
                # If univariate, add generic keys for backward compatibility
                if not is_multivariate:
                    # 'cols' is a string in univariate mode
                    row_data['Mean'] = row_data[f'Mean_{cols}']
                    row_data['Std'] = row_data[f'Std_{cols}']
                
                results.append(row_data)

    df_results = pd.DataFrame(results)
    
    # 6. AUTO-CLUSTERING (Regime Detection)
    if not df_results.empty and len(df_results) >= 2:
        n_segments = len(df_results)
        k = max(2, min(5, int(np.sqrt(n_segments))))
        
        if SKLEARN_AVAILABLE:
            if is_multivariate:
                # Cluster based on Means of ALL parameters (Multi-dimensional clustering)
                feature_cols = [f'Mean_{c}' for c in target_cols]
                # Normalize features before clustering so high-value cols don't dominate clustering
                X = df_results[feature_cols]
                X_norm = (X - X.mean()) / (X.std() + 1e-12)
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                df_results['Regime_Cluster'] = kmeans.fit_predict(X_norm)
                
                # Sort clusters logic is harder for multivariate, just map by frequency or PC1
                # Here we simply map randomly 1-K, or sort by the first column's mean
                cluster_map = df_results.groupby('Regime_Cluster')[feature_cols[0]].mean().sort_values().index
            else:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                df_results['Regime_Cluster'] = kmeans.fit_predict(df_results[['Mean']])
                cluster_map = df_results.groupby('Regime_Cluster')['Mean'].mean().sort_values().index
            
            remap = {old: new for new, old in enumerate(cluster_map)}
            df_results['Regime'] = df_results['Regime_Cluster'].map(remap)
            df_results = df_results.drop(columns=['Regime_Cluster'])
        else:
            df_results['Regime'] = 0 
            
    return df_results, dynamic_threshold

# --- END NEW FUNCTION ---

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = []
    for c in df.columns:
        try:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() > 0:
                num_cols.append(c)
        except:
            pass
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def calculate_chart_height(num_series: int, base_height: int = 300, height_per_series: int = 150) -> int:
    """Calculate responsive chart height"""
    return min(base_height + (num_series - 1) * height_per_series, 800)

# =====================================================================
# SESSION STATE INITIALIZATION (Early initialization for all features)
# =====================================================================

def _initialize_session_state():
    """Initialize all session state variables"""
    
    # Events tracking
    if "events" not in st.session_state:
        st.session_state["events"] = []

    # User preferences
    if "user_prefs" not in st.session_state:
        st.session_state["user_prefs"] = {
            "default_chart": "Time Series Trend",
            "show_stats": False,
        }

    # Raw dataframe state
    if 'df' not in st.session_state:
        st.session_state['df'] = None
        st.session_state['prev_datetime_col'] = None
        st.session_state['prev_dayfirst'] = None
        st.session_state['prev_file_hash'] = None
        st.session_state['num_cols'] = []
        st.session_state['cat_cols'] = []
        st.session_state['prev_df_id_for_types'] = None

    # Filtered dataframe state
    if 'df_filtered' not in st.session_state:
        st.session_state['df_filtered'] = None
        st.session_state['prev_start_date'] = None
        st.session_state['prev_end_date'] = None
        st.session_state['prev_missing_strategy'] = None
        st.session_state['prev_filters_tuple'] = None
        st.session_state['prev_resample_freq'] = None
        st.session_state['prev_df_id'] = None

    # Steady State parameters
    if 'ss_k' not in st.session_state: 
        st.session_state['ss_k'] = 3
    if 'ss_window' not in st.session_state: 
        st.session_state['ss_window'] = "1H"
    if 'ss_threshold' not in st.session_state: 
        st.session_state['ss_threshold'] = 1.0
    if 'ss_duration' not in st.session_state: 
        st.session_state['ss_duration'] = 60

    # Upload file state
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    # Sidebar state
    if "sidebar_state_set" not in st.session_state:
        st.session_state.sidebar_state_set = True
        st.markdown("""
            <script>
            window.parent.document.querySelector("section[data-testid='stSidebar']").style.display = 'none';
            </script>
        """, unsafe_allow_html=True)

# Initialize session state early
_initialize_session_state()

# =====================================================================
# STREAMLIT PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="Data Insights Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# AUTHENTICATION FUNCTION
# =====================================================================

def authenticate_user():
    
    expand_upload = st.session_state['uploaded_file'] is None
    
    with st.expander("📂 Upload File", expanded=False):
        st.info("👆 Click 'Browse files' below to upload your data")
        
        uploaded_file_sidebar = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Supported formats: CSV, Excel, Parquet. Max size: 200 MB",
            key="file_uploader"
        )
        
        if uploaded_file_sidebar is not None:
            st.session_state['uploaded_file'] = uploaded_file_sidebar
            file_size_mb = len(uploaded_file_sidebar.getvalue()) / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum: 200 MB")
                st.stop()
            else:
                st.success(f"✅ {uploaded_file_sidebar.name}")
                st.caption(f"📦 Size: {file_size_mb:.2f} MB")
        elif st.session_state['uploaded_file'] is not None:
             st.caption(f"✅ Using: {st.session_state['uploaded_file'].name}")
        else:
            st.caption("💡 Supported: CSV, Excel (.xlsx, .xls), Parquet")

# Load raw dataframe
df_raw = None
uploaded_file = None

if 'uploaded_file' in st.session_state and st.session_state['uploaded_file'] is not None:
    uploaded_file = st.session_state['uploaded_file']

try:
    _bytes = _file_bytes(uploaded_file)
    file_name = uploaded_file.name if uploaded_file else None
    
    if _bytes and file_name:
        with st.spinner('📂 Loading data...'):
            df_raw = load_dataframe(_bytes, file_name)
except Exception as e:
    if uploaded_file is not None:
        handle_error(e, "Data Loading")

if df_raw is None or df_raw.empty:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1rem;
        border-radius: 24px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        text-align: center;
    ">
        <div style="font-size: 80px; margin-bottom: 1rem; animation: float 3s ease-in-out infinite;">📊</div>
        <h1 style="
            font-size: 48px;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        ">Welcome to Data Insights Hub</h1>
        <p style="
            color: rgba(255,255,255,0.95);
            font-size: 20px;
            margin-bottom: 2rem;
            font-weight: 400;
        ">
            Transform your time-series data into actionable insights with powerful visualizations and AI-driven analytics
        </p>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 2rem;">
            <span style="
                background: rgba(255,255,255,0.2);
                padding: 8px 20px;
                border-radius: 20px;
                color: white;
                font-size: 14px;
                backdrop-filter: blur(10px);
            ">✓ No coding required</span>
            <span style="
                background: rgba(255,255,255,0.2);
                padding: 8px 20px;
                border-radius: 20px;
                color: white;
                font-size: 14px;
                backdrop-filter: blur(10px);
            ">✓ Real-time processing</span>
            <span style="
                background: rgba(255,255,255,0.2);
                padding: 8px 20px;
                border-radius: 20px;
                color: white;
                font-size: 14px;
                backdrop-filter: blur(10px);
            ">✓ Enterprise-ready</span>
        </div>
    </div>
    
    <style>
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Upload Section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:      
                
        uploaded_file_landing = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Supported formats: CSV, Excel, Parquet. Max size: 200 MB",
            key="file_uploader_landing",
            label_visibility="collapsed"
        )
        
        if uploaded_file_landing is not None:
            file_size_mb = len(uploaded_file_landing.getvalue()) / (1024 * 1024)
            if file_size_mb > 200:
                st.error(f"⚠️ File too large ({file_size_mb:.1f} MB). Maximum: 200 MB")
            else:
                st.session_state['uploaded_file'] = uploaded_file_landing
                st.success(f"✅ File uploaded: {uploaded_file_landing.name}")
                with st.spinner("🔄 Loading data..."):
                    time.sleep(0.5)
                st.rerun()
    
    # Enhanced Key Features Section
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0 2rem 0;">
        <h2 style="color: #1F2A44; font-size: 36px; font-weight: 700;">
            🎯 Powerful Features
        </h2>
        <p style="color: #6B7280; font-size: 16px; margin-top: 0.5rem;">
            Everything you need for comprehensive data analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)

    # ---------- TILE 1: TIME SERIES ----------
    with col1:
        st.markdown("""
    <div style="background:white;border:1px solid #e0e0e0;border-radius:16px;padding:2.5rem 1.5rem;
    text-align:center;transition:all 0.3s ease;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.08);
    position:relative;overflow:hidden;">

    <div style="position:absolute;top:0;left:0;right:0;height:4px;
    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);"></div>

    <div style="font-size:48px;margin-bottom:1rem;">📈</div>

    <h3 style="font-size:20px;margin:1rem 0 .75rem;color:#1F2A44;font-weight:700;">
    Time Series
    </h3>

    <p style="font-size:13px;color:#6B7280;line-height:1.5;">
    Interactive visualizations with multiple display modes and customizable parameters
    </p>

    <div style="margin-top:1.5rem;padding:6px 14px;background:linear-gradient(135deg,#667eea,#764ba2);
    color:white;border-radius:20px;font-size:11px;font-weight:600;display:inline-block;text-transform:uppercase;">
    Popular
    </div>

    </div>
    """, unsafe_allow_html=True)


    # ---------- TILE 2: MULTIVARIATE INSIGHTS ----------
    with col2:
        st.markdown("""
    <div style="background:white;border:1px solid #e0e0e0;border-radius:16px;padding:2.5rem 1.5rem;
    text-align:center;transition:all 0.3s ease;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.08);
    position:relative;overflow:hidden;">

    <div style="position:absolute;top:0;left:0;right:0;height:4px;
    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);"></div>

    <div style="font-size:48px;margin-bottom:1rem;">🔗</div>

    <h3 style="font-size:20px;margin:1rem 0 .75rem;color:#1F2A44;font-weight:700;">
    Multivariate Insights
    </h3>

    <p style="font-size:13px;color:#6B7280;line-height:1.5;">
    Correlation heatmaps, PCA, clustering, dimension reduction, and pairwise parameter exploration
    </p>

    </div>
    """, unsafe_allow_html=True)


    # ---------- TILE 3: STEADY STATE DETECTION ----------
    with col3:
        st.markdown("""
    <div style="background:white;border:1px solid #e0e0e0;border-radius:16px;padding:2.5rem 1.5rem;
    text-align:center;transition:all 0.3s ease;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.08);
    position:relative;overflow:hidden;">

    <div style="position:absolute;top:0;left:0;right:0;height:4px;
    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);"></div>

    <div style="font-size:48px;margin-bottom:1rem;">🔬</div>

    <h3 style="font-size:20px;margin:1rem 0 .75rem;color:#1F2A44;font-weight:700;">
    Steady State Detection
    </h3>

    <p style="font-size:13px;color:#6B7280;line-height:1.5;">
    Automated identification of stable operating regimes using adaptive multivariate thresholds
    </p>

    <div style="margin-top:1.5rem;padding:6px 14px;background:#3b82f6;color:white;border-radius:20px;
    font-size:11px;font-weight:600;display:inline-block;text-transform:uppercase;">
    New
    </div>

    </div>
    """, unsafe_allow_html=True)


    # ---------- TILE 4: ANOMALY DETECTION ----------
    with col4:
        st.markdown("""
    <div style="background:white;border:1px solid #e0e0e0;border-radius:16px;padding:2.5rem 1.5rem;
    text-align:center;transition:all 0.3s ease;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.08);
    position:relative;overflow:hidden;">

    <div style="position:absolute;top:0;left:0;right:0;height:4px;
    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);"></div>

    <div style="font-size:48px;margin-bottom:1rem;">🚨</div>

    <h3 style="font-size:20px;margin:1rem 0 .75rem;color:#1F2A44;font-weight:700;">
    Anomaly Detection
    </h3>

    <p style="font-size:13px;color:#6B7280;line-height:1.5;">
    Detect anomalies automatically using Multivariate Models
    </p>

    <div style="margin-top:1.5rem;padding:6px 14px;background:#10B981;color:white;border-radius:20px;
    font-size:11px;font-weight:600;display:inline-block;text-transform:uppercase;">
    AI-Powered
    </div>

    </div>
    """, unsafe_allow_html=True)


    # ---------- TILE 5: EXPORT ----------
    with col5:
        st.markdown("""
    <div style="background:white;border:1px solid #e0e0e0;border-radius:16px;padding:2.5rem 1.5rem;
    text-align:center;transition:all 0.3s ease;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.08);
    position:relative;overflow:hidden;">

    <div style="position:absolute;top:0;left:0;right:0;height:4px;
    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);"></div>

    <div style="font-size:48px;margin-bottom:1rem;">📤</div>

    <h3 style="font-size:20px;margin:1rem 0 .75rem;color:#1F2A44;font-weight:700;">
    Export
    </h3>

    <p style="font-size:13px;color:#6B7280;line-height:1.5;">
    Export filtered data and visualizations in CSV, Excel, Parquet, or JSON formats
    </p>

    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    

    
    # File Requirements Section with Better Design
    with st.expander("📋 File Requirements & Supported Formats", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Supported Formats:**
            - 📄 **CSV** (.csv) - Comma-separated values
            - 📊 **Excel** (.xlsx, .xls) - Microsoft Excel
            - 🚀 **Parquet** (.parquet) - Recommended for large files
            
            **📏 File Requirements:**
            - Maximum file size: **200 MB**
            - First column should contain date/time values
            - At least one numeric column for visualizations
            """)
        
        with col2:
            st.markdown("""
            **💡 Tips for Best Results:**
            - Use **Parquet format** for files > 50 MB
            - Ensure dates are in consistent format
            - Remove any file passwords or protection
            - Clean column headers (no special characters)
            
            **📊 Recommended Data Structure:**
            ```
            Date/Time    Parameter1    Parameter2    Parameter3
            2024-01-01   100          200           300
            2024-01-02   110          210           310
            ```
            """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Call to Action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="
            text-align: center;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2.5rem;
            border-radius: 16px;
            margin: 2rem 0;
        ">
            <h3 style="color: #1F2A44; margin-bottom: 1rem; font-size: 24px;">
                Ready to get started?
            </h3>
            <p style="color: #6B7280; margin-bottom: 1.5rem;">
                Upload your first dataset or access the uploader from the sidebar
            </p>
            <div style="
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 14px 36px;
                border-radius: 50px;
                font-size: 16px;
                font-weight: 600;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
                cursor: pointer;
            ">
                ⬆️ Upload Data Above
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

file_hash = hashlib.sha256(_bytes).hexdigest() if _bytes else None

# ==================== Settings & Configuration ====================

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1f77b4; font-weight: bold;'>📊 Data Insights Hub</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("⚙️ General Settings", expanded=False):
        config_tabs = st.tabs(["Data Config", "Display", "Advanced"])
        
        with config_tabs[0]:
            st.markdown("**Date/Time Configuration**")
            datetime_col_choice = st.selectbox(
                "Datetime column",
                options=[None] + list(df_raw.columns),
                index=0,
                help="Select the column to use as the time index"
            )
            
            dayfirst_choice = st.checkbox(
                "Day-first format (DD/MM/YYYY)",
                value=True,
                help="Enable if your dates use day-first format"
            )
        
        with config_tabs[1]:
            normalize = st.checkbox("Normalize (Z-score)", help="Apply z-score normalization to data")
            log_scale = st.checkbox("Log Scale Y-axis", help="Use logarithmic scale for Y-axis")
        
        with config_tabs[2]:
            if st.button("🔄 Reset All Settings"):
                st.session_state.clear()
                st.rerun()
            
            st.caption("💾 Settings are automatically saved")

filters_container = st.sidebar.container()

# Process dataframe
recompute_df = False
if (st.session_state['prev_datetime_col'] != datetime_col_choice or
    st.session_state['prev_dayfirst'] != dayfirst_choice or
    st.session_state['prev_file_hash'] != file_hash):
    recompute_df = True

if recompute_df:
    with st.spinner('🔄 Processing data...'):
        dt_col = datetime_col_choice if datetime_col_choice is not None else df_raw.columns[0]
        try:
            df = coerce_datetime(df_raw, dt_col, dayfirst_choice)
            df = compress_types(df)
            st.session_state['df'] = df
            st.session_state['prev_datetime_col'] = datetime_col_choice
            st.session_state['prev_dayfirst'] = dayfirst_choice
            st.session_state['prev_file_hash'] = file_hash
            num_cols, cat_cols = detect_column_types(df)
            st.session_state['num_cols'] = num_cols
            st.session_state['cat_cols'] = cat_cols
            st.session_state['prev_df_id_for_types'] = id(df)
        except Exception as e:
            handle_error(e, "Data Processing")
            st.stop()
else:
    df = st.session_state['df']
    if st.session_state['prev_df_id_for_types'] != id(df):
        num_cols, cat_cols = detect_column_types(df)
        st.session_state['num_cols'] = num_cols
        st.session_state['cat_cols'] = cat_cols
        st.session_state['prev_df_id_for_types'] = id(df)
    else:
        num_cols = st.session_state['num_cols']
        cat_cols = st.session_state['cat_cols']

# Apply theme (Light only)
TEMPLATE = "plotly_white"
PAPER_BG = "white"
PLOT_BG = "white"
FONT_COLOR = "#0E1117"
GRID_COLOR = "#E5ECF6"
FONT_PRIMARY = "#0E1117"
FONT_SIZE_TICK = 11
STANDARD_MARGIN = dict(l=60, r=40, t=50, b=60)
GRID_WIDTH = 1

# ==================== Sidebar Filters & Controls ====================

with filters_container:
    st.markdown("---")
    st.markdown("### 🎯 Data Filters")
    
    with st.expander("⏱️ Date-Time & Processing", expanded=False):
        min_date, max_date = df.index.min().date(), df.index.max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
            
        st.markdown("---")
        st.markdown("**🧹 Data Processing**")
        missing_strategy = st.selectbox(
            "Missing Values",
            ["No action", "Drop missing rows", "Forward fill", "Backward fill", "Interpolate (time)"],
            help="How to handle missing data points"
        )
        
        resample_options = {
            "None": None, "1 sec": "1s", "5 sec": "5s", "10 sec": "10s",
            "1 min": "1min", "5 min": "5min", "10 min": "10min", "30 min": "30min",
            "1 hour": "1H", "1 day": "1D", "1 week": "1W", "1 month": "1M"
        }
        resample_label = st.selectbox("Resample Frequency", list(resample_options.keys()))
        resample_freq = resample_options[resample_label]
    
        max_points_disabled = resample_freq is not None
        downsample_limit = st.number_input(
            "Max Plot Points",
            min_value=1000,
            max_value=200000,
            value=50000,
            step=5000,
            disabled=max_points_disabled,
            help="Limit number of points for performance"
        )
    
    with st.expander("🔍 Advanced Filters"):
        filter_count = st.number_input(
            "Number of filters",
            min_value=0,
            max_value=6,
            value=0,
            step=1
        )
        
        filters: List[Tuple[str, str, object]] = []
        for i in range(filter_count):
            st.markdown(f"**Filter {i+1}**")
            filter_param = st.selectbox(
                "Parameter",
                options=[None] + list(df.columns),
                key=f"param_{i}"
            )
            
            if filter_param:
                converted = pd.to_numeric(df[filter_param], errors="coerce")
                if converted.notna().any():
                    operator = st.selectbox("Condition", [">", "<", "=", "≠", ">=", "<="], key=f"op_{i}")
                    value = st.number_input("Value", value=float(converted.dropna().median()), key=f"val_{i}")
                else:
                    operator = st.selectbox("Condition", ["=", "≠"], key=f"op_{i}")
                    value = st.selectbox("Value", options=sorted(map(str, df[filter_param].dropna().unique())), key=f"val_{i}")
                
                if operator and value is not None:
                    filters.append((filter_param, operator, value))
    
    with st.expander("📍 Events & Annotations"):
        col1, col2 = st.columns(2)
        with col1:
            evt_date = st.date_input("Date", key="evt_date")
        with col2:
            evt_time = st.time_input("Time", value=pd.Timestamp.now().time(), key="evt_time")
        
        evt_label = st.text_input("Label", placeholder="Event name", key="evt_label")
        evt_color = st.color_picker("Color", value="#FF0000", key="evt_color")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Event", width="stretch"):
                try:
                    ts = pd.to_datetime(f"{evt_date} {evt_time}")
                    st.session_state.events.append({
                        "time": ts,
                        "label": evt_label or "Event",
                        "color": evt_color
                    })
                    st.success("✅ Added!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    handle_error(e, "Event Creation")
        
        with col2:
            if st.button("🗑️ Clear All", width="stretch"):
                st.session_state.events = []
                st.success("✅ Cleared!")
                time.sleep(1)
                st.rerun()
        
        if st.session_state.events:
            st.caption(f"📍 {len(st.session_state.events)} event(s) defined")
    
    st.markdown("---")
    filters_tuple = tuple(filters)
    if filters_tuple:
        st.caption("🔍 **Active Filters:**")
        for p, o, v in filters_tuple:
            st.caption(f"• {p} {o} {v}")
    else:
        st.caption("✨ No filters applied")

# ==================== Apply Filters & Create df_filtered ====================

current_df_id = id(df)
recompute_filtered = False

if (st.session_state['prev_start_date'] != start_date or
    st.session_state['prev_end_date'] != end_date or
    st.session_state['prev_missing_strategy'] != missing_strategy or
    st.session_state['prev_filters_tuple'] != filters_tuple or
    st.session_state['prev_resample_freq'] != resample_freq or
    st.session_state['prev_df_id'] != current_df_id):
    recompute_filtered = True

if recompute_filtered:
    with st.spinner('🔄 Applying filters...'):
        try:
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            df_filtered = df.loc[mask].copy()
            
            if missing_strategy != "No action":
                df_filtered = apply_missing(df_filtered, missing_strategy)
            
            if filters_tuple:
                df_filtered = apply_filters(df_filtered, filters_tuple)
            
            if resample_freq is not None:
                df_filtered = df_filtered[~df_filtered.index.duplicated(keep="first")]
                df_filtered = df_filtered.sort_index()
                num_df = df_filtered.select_dtypes(include=["number"])
                non_num_df = df_filtered.select_dtypes(exclude=["number"])
                num_resampled = num_df.resample(resample_freq).mean()
                if not non_num_df.empty:
                    non_num_resampled = non_num_df.resample(resample_freq).ffill()
                    df_filtered = pd.concat([num_resampled, non_num_resampled], axis=1)
                else:
                    df_filtered = num_resampled
                df_filtered = df_filtered.dropna(how="all")
            
            st.session_state['df_filtered'] = df_filtered
            st.session_state['prev_start_date'] = start_date
            st.session_state['prev_end_date'] = end_date
            st.session_state['prev_missing_strategy'] = missing_strategy
            st.session_state['prev_filters_tuple'] = filters_tuple
            st.session_state['prev_resample_freq'] = resample_freq
            st.session_state['prev_df_id'] = current_df_id
        except Exception as e:
            handle_error(e, "Filter Application")
            st.stop()
else:
    df_filtered = st.session_state['df_filtered']

if len(df_filtered) > 100000:
    st.warning(f"⚠️ Large dataset ({len(df_filtered):,} rows). Consider resampling or filtering for better performance.")

# ==================== Main Navigation (Replaces Tabs) ====================

nav_options = ["📋 Overview", "📊 Visualize", "🔬 Steady State", "🚨 Anomaly Detection", "📤 Export"]

selected_view = st.radio(
    "Navigation",
    options=nav_options,
    horizontal=True,
    label_visibility="collapsed",
    key="navigation_selection" 
)

st.markdown("---")

# ==================== TAB 1: Overview ====================

if selected_view == "📋 Overview":
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="tile">
            <div class="icon">📊</div>
            <h3>{len(df_filtered):,}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="tile">
            <div class="icon">📅</div>
            <h3>{df_filtered.index.min().strftime('%Y-%m-%d')}</h3>
            <p>Start Date</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="tile">
            <div class="icon">🏁</div>
            <h3>{df_filtered.index.max().strftime('%Y-%m-%d')}</h3>
            <p>End Date</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="tile">
            <div class="icon">🔢</div>
            <h3>{len(num_cols)} / {len(cat_cols)}</h3>
            <p>Numeric / Categorical</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📄 Data Preview")
    with col2:
        search_term = st.text_input("🔍 Search", placeholder="Filter table...", label_visibility="collapsed")
    
    if search_term:
        mask = df_filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        preview_df = df_filtered[mask].head(100)
        st.caption(f"Showing {len(preview_df)} matching rows")
    else:
        preview_df = df_filtered.head(100)
        st.caption(f"Showing first 100 of {len(df_filtered):,} rows")
    
    st.dataframe(preview_df, width="stretch", height=300)
    
    st.markdown("---")
    
    st.markdown("### 📊 Column Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔢 Numeric Columns**")
        if num_cols:
            for col in num_cols:
                non_null = df_filtered[col].notna().sum()
                pct = (non_null / len(df_filtered)) * 100
                st.write(f"• **{col}** - {non_null:,} values ({pct:.1f}%)")
        else:
            st.info("No numeric columns detected")
    
    with col2:
        st.markdown("**🏷️ Categorical Columns**")
        if cat_cols:
            for col in cat_cols:
                unique_count = df_filtered[col].nunique()
                st.write(f"• **{col}** - {unique_count} unique values")
        else:
            st.info("No categorical columns detected")
    
    if num_cols:
        st.markdown("---")
        st.markdown("### 📈 Quick Statistics")
        
        stats_df = numeric_df(df_filtered[num_cols]).describe().T
        stats_df = stats_df.round(2)
        
        st.dataframe(
            stats_df.style.background_gradient(cmap="Blues", axis=1),
            width="stretch"
        )

# ==================== TAB 2: Visualize ====================

elif selected_view == "📊 Visualize":
    st.markdown("### 📊 Select Parameters")
    selected_params = st.multiselect(
        "Choose parameters to visualize",
        options=list(df.columns),
        default=(list(num_cols[:3]) if len(num_cols) >= 3 else list(num_cols)),
        help="Select one or more parameters for visualization",
        key="viz_params"
    )
    
    st.markdown("---")
    
    if not selected_params:
        show_empty_state(
            "📊",
            "No Parameters Selected",
            "Select one or more parameters from the sidebar to begin visualization"
        )
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown('<p style="font-size: 24px; font-weight: 700; margin-bottom: 5px; color: #1F2A44;">📊 Visualization Type</p>', unsafe_allow_html=True)
            
            chart_choice = st.selectbox(
                "Visualization Type",
                [
                    "Time Series Trend",
                    "Correlation Heatmap",
                    "Histogram",
                    "Box Plot",
                    "Scatter + Clustering",
                    "PCA (2D)",
                    "Moving Average",
                    "Rolling Correlation",
                    "Cross-Correlation (Lag)",
                    "Decomposition",
                    "ACF Plot",
                    "PACF Plot"
                ],
                index=0,
                label_visibility="collapsed"
            )
        
        with col2:
            display_mode = "Stacked"  # Default value
            if chart_choice == "Time Series Trend":
                display_mode = st.radio(
                    "Display Mode",
                    ["Stacked", "Overlay"],
                    horizontal=True
                )
        
        with col3:
            show_stats = st.checkbox("📈 Descriptive Stats", value=False)
        
        st.markdown("---")
        
        if chart_choice == "Time Series Trend":
            if resample_freq is None:
                keep_idx = thin_index(df_filtered.index, downsample_limit)
                df_plot = df_filtered.iloc[keep_idx]
            else:
                df_plot = df_filtered
            
            df_plot = numeric_df(df_plot[selected_params])
            if normalize:
                df_plot = zscore(df_plot)
            
            color_cycle = px.colors.qualitative.Plotly
            
            if display_mode == "Stacked":
                rows = len(selected_params)
                fig = make_subplots(
                    rows=rows,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=selected_params
                )

                border_color = "#D1D5DB"
                
                for i, col in enumerate(selected_params, start=1):
                    color = color_cycle[(i - 1) % len(color_cycle)]
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot.index,
                            y=df_plot[col],
                            mode="lines",
                            name=col,
                            line=dict(color=color, width=2),
                            hovertemplate="%{y:.2f}"
                        ),
                        row=i, col=1
                    )
                    
                    fig.update_yaxes(
                        title_text=(f"log({col})" if log_scale else col),
                        gridcolor=GRID_COLOR,
                        tickfont=dict(color=color),
                        title_font=dict(color=color, size=12),
                        type=("log" if log_scale else "linear"),
                        row=i, col=1
                    )
                    
                    fig.update_xaxes(
                        showline=True,
                        linewidth=1,
                        linecolor=border_color, 
                        mirror=False,
                        showticklabels=(i == rows),
                        row=i, col=1
                    )
                
                for evt in st.session_state.events:
                    fig.add_vline(
                        x=evt["time"],
                        line=dict(color=evt["color"], width=2, dash="dash"),
                    )

                master_xaxis = 'x' if rows == 1 else f'x{rows}'
                fig.update_traces(xaxis=master_xaxis)

                fig.update_xaxes(
                    showspikes=True,
                    spikemode="across",
                    spikesnap="cursor",
                    showline=True,
                    spikecolor=FONT_COLOR,
                    spikethickness=1,
                    spikedash="dashdot",
                    row=rows, col=1
                )
                
                fig.update_layout(
                    template=TEMPLATE,
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=FONT_COLOR),
                    height=calculate_chart_height(rows, 250, 200),
                    margin=dict(l=60, r=40, t=50, b=60),
                    hovermode="x unified",
                    showlegend=False
                )
                
                st.plotly_chart(fig, width="stretch")
            

            
            else:
                fig = go.Figure()

                layout_updates = {}
                if len(selected_params) > 2:
                    layout_updates["xaxis"] = dict(domain=[0.1, 0.9])
                
                for i, col in enumerate(selected_params):
                    color = color_cycle[i % len(color_cycle)]
                    yaxis_name = "y" if i == 0 else f"y{i+1}"
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot.index,
                            y=df_plot[col],
                            mode="lines",
                            name=col,
                            line=dict(color=color, width=2),
                            yaxis=yaxis_name,
                            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:.2f}<extra>" + col + "</extra>"
                        )
                    )
                    
                    axis_config = dict(
                        title=dict(text=(f"log({col})" if log_scale else col), font=dict(color=color)),
                        tickfont=dict(color=color),
                        type=("log" if log_scale else "linear"),
                        showgrid=(True if i == 0 else False),
                    )
                    
                    if i > 0:
                        axis_config.update(dict(
                            overlaying="y",
                            anchor="free",
                            autoshift=True,
                        ))
                        axis_config["side"] = "right" if i % 2 != 0 else "left"
                        
                    key = "yaxis" if i == 0 else f"yaxis{i+1}"
                    layout_updates[key] = axis_config

                for evt in st.session_state.events:
                    fig.add_vline(
                        x=evt["time"],
                        line=dict(color=evt["color"], width=2, dash="dash"),
                        annotation_text=evt["label"],
                        annotation_position="top"
                    )
                
                fig.update_layout(
                    template=TEMPLATE,
                    paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG,
                    font=dict(color=FONT_COLOR),
                    title="Multi-Parameter Time Series (Independent Scales)",
                    height=600,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.05,
                        xanchor="center",
                        x=0.5
                    ),
                    **layout_updates
                )
                
                st.plotly_chart(fig, width="stretch")
        
        elif chart_choice == "Correlation Heatmap":
            if len(selected_params) < 2:
                st.info("ℹ️ Select at least 2 parameters for correlation analysis")
            else:
                corr_method = st.selectbox(
                    "Correlation Method",
                    ["Pearson", "Spearman", "Kendall"],
                    index=0
                )
                
                df_num = numeric_df(df_filtered[selected_params]).dropna(how="all", axis=1)
                if df_num.shape[1] < 2:
                    st.warning("⚠️ Insufficient numeric data for correlation")
                else:
                    corr = df_num.corr(method=corr_method.lower()).round(2)
                    
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title=f"Correlation Heatmap ({corr_method})",
                        template=TEMPLATE,
                        zmin=-1,
                        zmax=1
                    )
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, width="stretch")
                    
                    st.markdown("### 🔍 Correlation Insights")
                    
                    pairs = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                    
                    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
                    top_pos = [(a, b, v) for a, b, v in pairs_sorted if v > 0][:5]
                    top_neg = [(a, b, v) for a, b, v in pairs_sorted if v < 0][:5]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📈 Strongest Positive Correlations**")
                        if top_pos:
                            for a, b, v in top_pos:
                                st.metric(f"{a} ↔ {b}", f"{v:.3f}")
                        else:
                            st.caption("None found")
                    
                    with col2:
                        st.markdown("**📉 Strongest Negative Correlations**")
                        if top_neg:
                            for a, b, v in top_neg:
                                st.metric(f"{a} ↔ {b}", f"{v:.3f}")
                        else:
                            st.caption("None found")
        
        elif chart_choice == "Histogram":
            bins = st.slider("Number of bins", 10, 100, 50)
            
            for col in selected_params:
                series = pd.to_numeric(df_filtered[col], errors="coerce").dropna()
                if not series.empty:
                    fig = px.histogram(
                        series,
                        nbins=bins,
                        template=TEMPLATE,
                        title=f"Distribution: {col}",
                        histnorm="probability density"
                    )
                    
                    kde_fig = ff.create_distplot([series.values], [col], show_hist=False, show_rug=False)
                    fig.add_trace(kde_fig.data[0])
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, width="stretch")
        
        elif chart_choice == "Box Plot":
            for col in selected_params:
                series = pd.to_numeric(df_filtered[col], errors="coerce").dropna()
                if not series.empty:
                    fig = px.box(
                        series,
                        points="outliers",
                        template=TEMPLATE,
                        title=f"Box Plot: {col}"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width="stretch")
        
        elif chart_choice == "Scatter + Clustering":
            if len(selected_params) < 2:
                st.info("ℹ️ Select at least 2 parameters for scatter plot")
            else:
                dims = st.multiselect(
                    "Select X and Y",
                    options=selected_params,
                    default=selected_params[:2]
                )
                k = st.slider("Number of clusters", 2, 10, 3)
                
                if len(dims) == 2:
                    df_num = numeric_df(df_filtered[dims]).dropna()
                    if normalize:
                        df_num = zscore(df_num)
                    
                    km = KMeans(n_clusters=k, n_init=10 if not SKLEARN_AVAILABLE else "auto", random_state=42)
                    labels = km.fit_predict(df_num.values)
                    
                    fig = px.scatter(
                        df_num,
                        x=dims[0],
                        y=dims[1],
                        color=labels.astype(str),
                        title=f"K-Means Clustering (k={k})",
                        template=TEMPLATE,
                        labels={"color": "Cluster"}
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")
        
        elif chart_choice == "PCA (2D)":
            if len(selected_params) < 2:
                st.info("ℹ️ Select at least 2 parameters for PCA")
            else:
                df_num = numeric_df(df_filtered[selected_params]).dropna()
                if normalize:
                    df_num = zscore(df_num)
                
                pca = PCA(n_components=2, random_state=42)
                comps = pca.fit_transform(df_num.values)
                pc_df = pd.DataFrame(comps, index=df_num.index, columns=["PC1", "PC2"])
                
                fig = px.scatter(
                    pc_df,
                    x="PC1",
                    y="PC2",
                    title="PCA Projection (2D)",
                    template=TEMPLATE
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")
        
        elif chart_choice == "Moving Average":
            window_opts = ["30s", "1min", "5min", "15min", "1H", "6H", "12H", "1D", "7D"]
            window = st.selectbox("Rolling Window", options=window_opts, index=2)
            
            df_num = numeric_df(df_filtered[selected_params]).dropna(how="all", axis=1)
            
            rows = len(df_num.columns)
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=df_num.columns
            )

            color_cycle = px.colors.qualitative.Plotly
            
            for i, col in enumerate(df_num.columns, start=1):
                color = color_cycle[(i - 1) % len(color_cycle)]
                series = df_num[col]
                ma = series.rolling(window=window, min_periods=1).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series,
                        mode="lines",
                        name=col,
                        line=dict(color=color, width=1),
                        opacity=0.3
                    ),
                    row=i, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=ma.index,
                        y=ma,
                        mode="lines",
                        name=f"{col} (MA)",
                        line=dict(color=color, width=2)
                    ),
                    row=i, col=1
                )
                
                fig.update_yaxes(
                    title_text=col,
                    gridcolor=GRID_COLOR,
                    row=i, col=1
                )
            
            fig.update_layout(
                template=TEMPLATE,
                paper_bgcolor=PAPER_BG,
                plot_bgcolor=PLOT_BG,
                title=f"Moving Average (Window: {window})",
                height=calculate_chart_height(rows, 250, 180),
                showlegend=False
            )
            
            st.plotly_chart(fig, width="stretch")
        
        elif chart_choice == "Rolling Correlation":
            if len(selected_params) != 2:
                st.warning("Select exactly 2 parameters for rolling correlation.")
            else:
                window = st.slider("Rolling Window (samples)", 10, 500, 100)
                s1 = pd.to_numeric(df_filtered[selected_params[0]], errors='coerce')
                s2 = pd.to_numeric(df_filtered[selected_params[1]], errors='coerce')
                rc = s1.rolling(window).corr(s2)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rc.index, y=rc, mode="lines", name="Rolling Corr", line=dict(width=2)))
                fig.update_layout(title=f"Rolling Correlation ({selected_params[0]} vs {selected_params[1]})",
                                  template=TEMPLATE, height=400)
                st.plotly_chart(fig, width="stretch")

        elif chart_choice == "Cross-Correlation (Lag)":
            if len(selected_params) != 2:
                st.warning("Select exactly 2 parameters for cross-correlation.")
            else:
                max_lag = st.slider("Max Lag", 10, 500, 100)
                s1 = pd.to_numeric(df_filtered[selected_params[0]], errors='coerce').dropna()
                s2 = pd.to_numeric(df_filtered[selected_params[1]], errors='coerce').dropna()
                xcorr = [s1.corr(s2.shift(lag)) for lag in range(-max_lag, max_lag+1)]
                lags = list(range(-max_lag, max_lag+1))
                fig = go.Figure()
                fig.add_trace(go.Bar(x=lags, y=xcorr))
                fig.update_layout(title=f"Cross-Correlation: {selected_params[0]} vs {selected_params[1]}",
                                  template=TEMPLATE, xaxis_title="Lag", yaxis_title="Correlation", height=450)
                st.plotly_chart(fig, width="stretch")

        elif chart_choice == "Decomposition":
            if len(selected_params) != 1:
                st.warning("Select exactly 1 parameter for decomposition.")
            else:
                series = pd.to_numeric(df_filtered[selected_params[0]], errors="coerce").dropna()

                s = series.copy()
                s.index = pd.to_datetime(s.index)

                freq = s.index.inferred_freq
                if freq is None:
                    try:
                        freq = pd.infer_freq(s.index)
                    except:
                        st.error("Time index is irregular. Please resample to a fixed frequency (e.g., 1 min) before decomposition.")
                        st.stop()

                s = s.asfreq(freq)
                s = s.interpolate(method="time").bfill().ffill()

                try:
                    result = seasonal_decompose(s, model="additive", period=None)

                    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                    result.observed.plot(ax=ax[0], title="Observed")
                    result.trend.plot(ax=ax[1], title="Trend")
                    result.seasonal.plot(ax=ax[2], title="Seasonal")
                    result.resid.plot(ax=ax[3], title="Residual")

                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Decomposition failed: {e}")

        elif chart_choice == "ACF Plot":
            if len(selected_params) != 1:
                st.warning("Select only 1 parameter for ACF.")
            else:
                series = pd.to_numeric(df_filtered[selected_params[0]], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_acf(series, ax=ax, lags=50)
                st.pyplot(fig)

        elif chart_choice == "PACF Plot":
            if len(selected_params) != 1:
                st.warning("Select only 1 parameter for PACF.")
            else:
                series = pd.to_numeric(df_filtered[selected_params[0]], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_pacf(series, ax=ax, lags=50, method='ywm')
                st.pyplot(fig)

        if show_stats and selected_params:
            st.markdown("---")
            st.markdown("### 📊 Descriptive Statistics")
            
            stats_df = numeric_df(df_filtered[selected_params]).describe().T.round(2)
            st.dataframe(
                stats_df.style.background_gradient(cmap="Blues", axis=1),
                width="stretch"
            )

# ==================== TAB 3: Steady State Condition (UPDATED) ====================

elif selected_view == "🔬 Steady State":
    st.markdown('<div class="section-header">🔬 Automated Steady State Analysis</div>', unsafe_allow_html=True)
    
    # Initialize session state for this tab
    if 'ss_raw_results' not in st.session_state:
        st.session_state['ss_raw_results'] = None
    if 'ss_calc_threshold' not in st.session_state:
        st.session_state['ss_calc_threshold'] = 0.0
    if 'analysis_running' not in st.session_state:
        st.session_state['analysis_running'] = False

    # --- Mode Selector ---
    st.markdown("#### Configuration")
    ss_mode = st.radio("Analysis Mode", ["Single Parameter", "Multi-Parameter"], horizontal=True)

    # --- Input Section ---
    col_sel, col_conf = st.columns([1, 2])
    
    target_params = []
    
    with col_sel:
        if ss_mode == "Single Parameter":
            param = st.selectbox(
                "Select Parameter",
                options=st.session_state.get('num_cols', []),
                index=0 if st.session_state.get('num_cols') else None,
                help="Choose the process variable to analyze"
            )
            if param: target_params = param  # Pass as string
        else:
            target_params = st.multiselect(
                "Select Parameters (Asset/System)",
                options=st.session_state.get('num_cols', []),
                default=st.session_state.get('num_cols', [])[:3] if len(st.session_state.get('num_cols', [])) >=3 else None,
                help="Select all parameters that define the steady state."
            )
            if len(target_params) < 2:
                st.warning("⚠️ Select at least 2 parameters for Multi-Parameter mode.")

    with col_conf:
        c1, c2, c3 = st.columns(3)
        with c1:
            min_dur = st.number_input("Min Duration (min)", min_value=1, value=30, step=5)
        with c2:
            sensitivity = st.slider("Stability Tolerance", 0.1, 5.0, 5.0, 0.1, 
                                  help="Higher value = Looser tolerance.\nLower value = Stricter tolerance.")
    
        with c3:
            st.write("") # Spacer
            run_placeholder = st.empty()
            
            can_run = False
            if ss_mode == "Single Parameter" and target_params: can_run = True
            if ss_mode == "Multi-Parameter" and target_params and len(target_params) >= 2: can_run = True

            if st.session_state.get('analysis_running', False):
                if run_placeholder.button("🛑 Abort Analysis", type="secondary", width='stretch', key='abort_btn'):
                    st.session_state['analysis_running'] = False
                    st.session_state['ss_raw_results'] = None
                    st.toast("Analysis aborted.", icon="🛑")
                    st.rerun()
            else:
                if run_placeholder.button("🚀 Find Steady States", type="primary", width='stretch', key='run_btn', disabled=not can_run):
                    st.session_state['analysis_running'] = True
                    st.rerun()

    st.markdown("---")

    # --- Processing ---
    if st.session_state.get('analysis_running', False) and can_run:
        with st.spinner(f"🤖 Scanning data for stable periods..."):
            try:
                raw_df, thresh = automated_steady_state_detection(
                    st.session_state['df_filtered'], 
                    target_params, 
                    min_dur, 
                    sensitivity
                )
                
                # Initialize 'Selected' column for the new feature
                if not raw_df.empty:
                    raw_df.insert(0, 'Selected', False)
                    
                st.session_state['ss_raw_results'] = raw_df
                st.session_state['ss_calc_threshold'] = thresh
                st.session_state['analysis_running'] = False
                st.rerun()
            except Exception as e:
                st.session_state['analysis_running'] = False
                handle_error(e, "Steady State Calc")
    
    # --- Results Display ---
    if st.session_state['ss_raw_results'] is not None and not st.session_state['ss_raw_results'].empty:
        
        df_res = st.session_state['ss_raw_results']
        
        # Ensure columns exist
        if 'Selected' not in df_res.columns: df_res.insert(0, 'Selected', False)
        if 'Segment' not in df_res.columns: df_res['Segment'] = range(1, len(df_res) + 1)
        
        n_segments = len(df_res)
        suggested_k = max(2, min(5, int(np.sqrt(n_segments))))
        
        # 1. Info & Regime Controls Row
        c_info, c_regime, c_reset = st.columns([2, 2, 1])
        
        with c_info:
            st.success(f"✅ Found **{n_segments}** steady periods.")
            if isinstance(target_params, list):
                st.caption(f"Based on combined stability of: {', '.join(target_params)}")
            
        with c_regime:
            num_regimes = st.number_input(
                "Operating Regimes (Clusters)",
                min_value=1,
                max_value=max(1, n_segments),
                value=suggested_k,
                help=f"Group periods by similarity. System suggests {suggested_k}."
            )
            
        with c_reset:
            st.write("") 
            st.write("") 
            if st.button("🔄 Reset Analysis", width='stretch'):
                st.session_state['ss_raw_results'] = None
                st.session_state['ss_calc_threshold'] = 0.0
                st.rerun()

        # Re-Run Clustering logic
        if SKLEARN_AVAILABLE and n_segments >= num_regimes and num_regimes > 1:
            if isinstance(target_params, list):
                feature_cols = [f'Mean_{c}' for c in target_params]
                X = df_res[feature_cols]
                X_norm = (X - X.mean()) / (X.std() + 1e-12)
                kmeans = KMeans(n_clusters=num_regimes, n_init=10, random_state=42)
                df_res['Regime_Cluster'] = kmeans.fit_predict(X_norm)
                cluster_map = df_res.groupby('Regime_Cluster')[feature_cols[0]].mean().sort_values().index
            else:
                kmeans = KMeans(n_clusters=num_regimes, n_init=10, random_state=42)
                df_res['Regime_Cluster'] = kmeans.fit_predict(df_res[['Mean']])
                cluster_map = df_res.groupby('Regime_Cluster')['Mean'].mean().sort_values().index
                
            remap = {old: new+1 for new, old in enumerate(cluster_map)}
            df_res['Regime'] = df_res['Regime_Cluster'].map(remap)
        else:
            df_res['Regime'] = 1 

        st.markdown("---")

        # --- VIEW CONTROL ---
        view_param = None
        if isinstance(target_params, list):
            col_v1, col_v2 = st.columns([1, 3])
            with col_v1:
                st.markdown("##### 👁️ Visualization")
            with col_v2:
                view_param = st.selectbox("Select Parameter to Visualize", options=target_params)
            df_res['Mean'] = df_res[f'Mean_{view_param}']
        else:
            view_param = target_params


        # ==================== 2. REGIME SUMMARY (Always Visible) ====================
        st.markdown("#### 📊 Regime Summary")
        
        agg_dict = {'Duration_Min': ['sum', 'count']}
        mean_col_name = f'Mean_{view_param}' if isinstance(target_params, list) else 'Mean'
        agg_dict[mean_col_name] = ['mean', 'std']
        
        summary_stats = df_res.groupby('Regime').agg(agg_dict).reset_index()
        summary_stats['Duration_Min', 'sum'] = summary_stats['Duration_Min', 'sum'].fillna(0)
        
        # ==================== NEW REGIME STATISTICS PIE CHART ====================
        st.markdown("### 📊 Regime Duration Breakdown")

        # 1. Calculate Total Duration of the filtered time range
        df_filtered = st.session_state['df_filtered']
        # Handle case where index might be empty (e.g., if df_filtered is empty after resample/filter)
        if df_filtered.empty:
            st.warning("No data in the filtered range to calculate total time span.")
            total_minutes = 0
        else:
            total_time_span = df_filtered.index.max() - df_filtered.index.min()
            total_minutes = total_time_span.total_seconds() / 60
        
        total_days = total_minutes / (60 * 24)

        # 2. Prepare data for pie chart
        total_regime_minutes = summary_stats[('Duration_Min', 'sum')].sum()
        unassigned_minutes = max(0, total_minutes - total_regime_minutes)
        unassigned_days = unassigned_minutes / (60 * 24)

        pie_data = []

        # Add individual regime durations
        for _, row in summary_stats.iterrows():
            regime_id = int(row['Regime'])
            duration = row[('Duration_Min', 'sum')]
            if duration > 0: # Only include regimes with duration > 0
                pie_data.append({
                    'Category': f'Regime {regime_id}', 
                    'Duration_Min': duration,
                    'Duration_Days': duration / (60 * 24)
                })

        # Add unassigned duration if greater than 0
        if unassigned_minutes > 0:
            pie_data.append({
                'Category': 'Unassigned', 
                'Duration_Min': unassigned_minutes,
                'Duration_Days': unassigned_minutes / (60 * 24)
            })

        df_pie = pd.DataFrame(pie_data)
        
        # Display Total Metrics
        col_total_days, col_total_minutes, _ = st.columns(3)
        with col_total_days:
            st.metric("Total Time Span", f"{total_days:.2f} Days", help="Difference between start and end of filtered data.")
        with col_total_minutes:
            # Use total_minutes for the metric value for clearer context
            st.metric(
                "Unassigned Time", 
                f"{unassigned_days:.2f} Days ({unassigned_minutes:.0f} Mins)", 
                delta=f"Total Regimes: {total_regime_minutes:.0f} Mins", 
                delta_color="off", 
                help="Time that did not fall into any detected steady state regime."
            )

        # 3. Create Donut Chart
        if not df_pie.empty:
            try:
                # Use a custom color sequence for distinct regimes and a grey for 'Unassigned'
                colors = px.colors.qualitative.Plotly
                color_map = {f'Regime {i}': colors[i % len(colors)] for i in df_pie[df_pie['Category'] != 'Unassigned']['Category'].str.replace('Regime ', '').astype(int).unique()}
                if 'Unassigned' in df_pie['Category'].values:
                    color_map['Unassigned'] = '#bdbdbd' # Light gray for unassigned time

                fig_pie = px.pie(
                    df_pie, 
                    values='Duration_Min', 
                    names='Category', 
                    title='Proportion of Total Time Span (Regimes vs. Unassigned)',
                    hole=.5, # Donut chart
                    template=TEMPLATE,
                    color='Category',
                    color_discrete_map=color_map
                )
                
                # Customize hover text to show both Minutes and Days
                fig_pie.update_traces(
                    hovertemplate="<b>%{label}</b><br>Duration: %{value:.0f} Mins<br>(%{customdata[0]:.2f} Days)<br>Percentage: %{percent}<extra></extra>",
                    customdata=df_pie[['Duration_Days']].values
                )
                
                # Add annotation for total
                fig_pie.add_annotation(
                    text=f"Total: {total_minutes:.0f} Mins",
                    x=0.5, y=0.5, showarrow=False, 
                    font=dict(size=14, color="#1F2A44")
                )

                st.plotly_chart(fig_pie, width='stretch')

            except Exception as e:
                st.error(f"Could not generate Pie Chart: {e}")
        else:
            st.info("No regimes were detected, and the time span is zero.")

        # ==================== END NEW PIE CHART ====================

        # --- Color Scaling based on Duration ---
        max_duration = summary_stats['Duration_Min', 'sum'].max()
        # Generate a continuous color map (e.g., from light gray to bold blue)
        # Use a simpler scheme: scale opacity or background color
        regime_cols = st.columns(len(summary_stats))

        # --- Color Scaling based on Duration ---
        max_duration = summary_stats['Duration_Min', 'sum'].max()
        
        # Generate a continuous color map (e.g., from light gray to bold blue)
        # Use a simpler scheme: scale opacity or background color
        
        regime_cols = st.columns(len(summary_stats))
        
        for idx, row in summary_stats.iterrows():
            regime_id = int(row['Regime'].iloc[0]) if isinstance(row['Regime'], pd.Series) else int(row['Regime'])
            total_min = row[('Duration_Min', 'sum')]
            avg_val = row[(mean_col_name, 'mean')]
            count = int(row[('Duration_Min', 'count')].iloc[0]) if isinstance(row[('Duration_Min', 'count')], pd.Series) else int(row[('Duration_Min', 'count')])
            
            # Color calculation: Higher duration means higher intensity (more prominent)
            # Normalize duration from 0 to 1, then map to a color scale
            normalized_duration = total_min / max_duration if max_duration > 0 else 0
            
            # Simple color mapping: adjust hue or opacity based on duration (using Hex/CSS)
            # We'll use a color scale from a light blue to a primary blue/teal
            # Interpolating RGB values: Light Blue (e.g., #ADD8E6) to Primary Blue (#1f77b4)
            # Since Streamlit metrics don't accept complex HTML styling easily, we use a simple background gradient concept.
            # However, for a Streamlit `st.metric` visual color cue, the best we can do is use custom CSS or rely on the `delta` color. 
            # Sticking to the requirement for visual recognition, we will use a Markdown hack for the visual display:
            
            if total_min > 1440: dur_str = f"{total_min/1440:.1f} d"
            elif total_min > 60: dur_str = f"{total_min/60:.1f} h"
            else: dur_str = f"{total_min:.0f} m"
            
            # Determine color class (simplified visual intensity)
            if normalized_duration > 0.8:
                 color_code = '#A8DADC' # High dominance (light teal background)
            elif normalized_duration > 0.4:
                 color_code = '#F0F8FF' # Medium dominance (lightest blue)
            else:
                 color_code = '#F7F7F7' # Low dominance (light gray)

            
            metric_html = f"""
            <div style='background-color: {color_code}; padding: 10px; border-radius: 5px; border-left: 5px solid #1f77b4;'>
                <p style='margin: 0; font-size: 0.8rem; color: #555;'>Regime {regime_id}</p>
                <h3 style='margin: 0; color: #1f77b4;'>{avg_val:.2f}</h3>
                <p style='margin: 0; font-size: 0.8rem; color: #777;'>{count} segs | {dur_str}</p>
            </div>
            """

            with regime_cols[idx]:
                st.markdown(metric_html, unsafe_allow_html=True)

        st.markdown("---")


        # 3. Charts Section with Selection Sync
        st.markdown(f"#### 📈 Interactive Chart: {view_param}")
        
        # --- NEW CONSOLIDATED REGIME SELECTION TOOLS (Above Chart) ---   
        st.markdown("### 🗺️ **Regime Selection Tools**")
        
        # Use columns to place the controls together
        col_select_zone, col_confirm, col_clear_chart, _ = st.columns([1, 1, 1, 3])
        
        # 1. Dedicated "Select Zone" Button (Box Select Re-activation)
        with col_select_zone:
            # Clicking this button forces a Streamlit rerun, which re-applies the 
            # `chart_config` below, re-enabling Box Select even after Zoom/Pan.
            if st.button("🎯 Select Zone (Box Select)", key="activate_box_select_btn", type="primary", 
                        help="Re-activates the Box Select tool on the chart below. Click this after using Zoom or Pan."):
                st.toast("Box Select re-activated. Click and drag on the chart below.", icon="🎯")
                st.rerun() # Forces the new config to apply and re-draw the plot
            
        # 2. Confirm Selection 
        with col_confirm:
            if st.button("✅ Confirm Selection", key="confirm_chart_btn", help="Finalize the current chart selection."):
                st.toast("Selection confirmed in table.", icon="✅")
                
        # 3. Clear Selection
        with col_clear_chart:
            if st.button("🧹 Clear Selection", key="clear_chart_btn", help="Deselect all segments marked by the chart."):
                st.session_state['ss_raw_results']['Selected'] = False
                st.toast("All segments deselected.", icon="🧹")
                st.rerun() # Rerun to update the dataframe and chart visual instantly
        
        st.markdown("---")

        df_chart = st.session_state['df_filtered'][view_param]
        
        # Optimization for plotting large datasets
        if len(df_chart) > 20000:
            df_chart_plot = df_chart.iloc[thin_index(df_chart.index, 20000)]
        else:
            df_chart_plot = df_chart

        # Subplots with updated formatting
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.15, 
            subplot_titles=(
                f"Raw Parameter Trend: {view_param}", 
                f"Steady State Regimes (Threshold: {st.session_state['ss_calc_threshold']:.4f})"
            ),
            row_heights=[0.3, 0.7]
        )

        fig.add_trace(go.Scatter(
            x=df_chart_plot.index, y=df_chart_plot,
            mode='lines', name='Raw Signal',
            line=dict(color='steelblue', width=1),
            legendgroup="raw"
        ), row=1, col=1)

        # Trace 2: Background + Regimes
        fig.add_trace(go.Scatter(
            x=df_chart_plot.index, y=df_chart_plot,
            mode='lines', name='Background',
            line=dict(color='#777777', width=1), 
            hoverinfo='skip', showlegend=False
        ), row=2, col=1)

        colors = px.colors.qualitative.Bold
        regimes = sorted(df_res['Regime'].unique())
        
        for r in regimes:
            subset = df_res[df_res['Regime'] == r]
            color = colors[int(r-1) % len(colors)]
            
            for i, (_, row) in enumerate(subset.iterrows()):
                # Highlight if selected
                is_sel = row.get('Selected', False)
                opacity = 0.8 if is_sel else 0.2
                line_width = 4 if is_sel else 3
                
                fig.add_vrect(
                    x0=row['Start'], x1=row['End'],
                    fillcolor=color, opacity=opacity,
                    line_width=0, layer="below", row=2, col=1 
                )
                
                show_legend = (i == 0)
                mean_val = row[f'Mean_{view_param}'] if isinstance(target_params, list) else row['Mean']
                
                fig.add_trace(go.Scatter(
                    x=[row['Start'], row['End']],
                    y=[mean_val, mean_val],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name=f"Regime {r}", 
                    legendgroup=f"Regime {r}",
                    showlegend=show_legend,
                    hovertemplate=f"Regime {r}<br>Mean: %{{y:.2f}}<extra></extra>"
                ), row=2, col=1)

        fig.update_layout(
            template="plotly_white", 
            height=700, 
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified",
            dragmode='select',  # DEFAULT to BOX SELECT
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.1,
                xanchor="right", 
                x=1
            ),
            # --- 3. LEGEND PLACEMENT FIX ---
            # Reposition the Plotly modebar (tools) slightly lower or adjust legend position further
            # We'll adjust the modebar orientation and legend placement in Row 1 (Raw Trend)
            
        )
        # Update Row 1 Layout (Raw Parameter Trend) for Legend Fix
        fig.update_layout({
            'xaxis': {'anchor': 'y2'}, # Keep shared x-axis
            # Reposition Legend 1 (Raw Signal) below the title and away from controls
            'legend1': dict(
                orientation="h",
                yanchor="bottom",
                y=1.03, # Adjusted slightly higher, usually the top margin is 1
                xanchor="left",
                x=0,
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
        })
        
        # Ensure the modebar is positioned correctly (default often puts it top right)
        # Streamlit controls modebar visibility, but repositioning is tricky via fig object.
        # However, adjusting legend positioning usually solves the overlap.
        
        fig.update_annotations(font=dict(size=16, color="#1F2A44", family="Arial", weight="bold"))

        # --- GRAPHICAL SELECTION LOGIC ---
        chart_config = {
            'displayModeBar': True,
            # Remove and re-add 'select2d' to force the active tool state to reset.
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'modeBarButtonsToAdd': ['select2d']
        }

        chart_selection = st.plotly_chart(
            fig, 
            width='stretch', 
            on_select="rerun", 
            selection_mode="box",
            config=chart_config
        )
        
        # Process the selection
        if chart_selection and chart_selection.get("selection"):
            selected_boxes = chart_selection["selection"].get("box", [])
            
            ranges = []
            if selected_boxes:
                # Use only the x-axis range for time selection (assuming selection in Row 2)
                for box in selected_boxes:
                    if "x" in box:
                        ranges.append((pd.to_datetime(box["x"][0]), pd.to_datetime(box["x"][1])))

            if ranges:
                # Select any segment that overlaps with the selection box
                for idx, row in df_res.iterrows():
                    for (start_sel, end_sel) in ranges:
                        if (row['Start'] <= end_sel) and (row['End'] >= start_sel):
                            df_res.at[idx, 'Selected'] = True
                            
                st.session_state['ss_raw_results'] = df_res

        # ==================== COLLAPSIBLE ANALYSIS CHARTS ====================
        st.write("")
        with st.expander("📊 Regime Distribution Analysis", expanded=False):
            st.markdown("### 📊 Regime Distribution Analysis")
            st.caption(f"Comparing distribution of **{view_param}** across regimes.")
            
            # ... (Rest of Distribution Analysis Code remains the same)
            hist_data = []
            for _, row in df_res.iterrows():
                mask = (st.session_state['df_filtered'].index >= row['Start']) & \
                       (st.session_state['df_filtered'].index <= row['End'])
                segment_data = st.session_state['df_filtered'].loc[mask, view_param]
                temp_df = pd.DataFrame({'Value': pd.to_numeric(segment_data, errors='coerce'), 'Regime': f"Regime {int(row['Regime'])}"})
                hist_data.append(temp_df)

            if hist_data:
                df_hist = pd.concat(hist_data, ignore_index=True).dropna()
                df_hist = df_hist.sort_values('Regime')
                fig_hist = px.histogram(
                    df_hist, x="Value", color="Regime", 
                    barmode="overlay", marginal="box", opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    title=f"Value Distribution: {view_param}"
                )
                fig_hist.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_hist, width='stretch')


        with st.expander("🧬 Regime Consistency (Overlay)", expanded=False):
            st.markdown("### 🧬 Regime Consistency")
            st.caption(f"Visualizing consistency of **{view_param}** for a specific regime.")
            
            # ... (Rest of Overlay Code remains the same)
            selected_regime_ov = st.selectbox("Select Regime for Overlay", options=sorted(df_res['Regime'].unique()))
            subset_overlay = df_res[df_res['Regime'] == selected_regime_ov]
            
            fig_overlay = go.Figure()
            regime_mean_val = subset_overlay[mean_col_name].mean()

            for i, row in subset_overlay.iterrows():
                mask = (st.session_state['df_filtered'].index >= row['Start']) & \
                       (st.session_state['df_filtered'].index <= row['End'])
                segment_data = st.session_state['df_filtered'].loc[mask, view_param]
                
                if len(segment_data) > 0:
                    t_start = segment_data.index[0]
                    relative_time = (segment_data.index - t_start).total_seconds() / 60
                    fig_overlay.add_trace(go.Scatter(
                        x=relative_time, y=segment_data.values,
                        mode='lines', line=dict(width=1), opacity=0.5,
                        name=f"Seg {row['Segment']}"
                    ))

            fig_overlay.add_hline(y=regime_mean_val, line_dash="dash", line_color="black")
            fig_overlay.update_layout(
                template="plotly_white", height=450,
                title=f"Overlay: Regime {selected_regime_ov} ({view_param})",
                xaxis_title="Minutes from Start", yaxis_title=view_param
            )
            st.plotly_chart(fig_overlay, width='stretch')

        st.markdown("---")

        # ==================== DETAILED BREAKDOWN & SELECTION ====================
        st.markdown("### 📋 Detailed Breakdown & Selection")
        
        # --- FILTER BY REGIME ---
        unique_regimes = sorted(df_res['Regime'].unique())
        selected_regimes_filter = st.multiselect(
            "Filter by Regime",
            options=unique_regimes,
            default=unique_regimes,
            key="regime_multiselect"
        )
        
        df_filtered_view = df_res[df_res['Regime'].isin(selected_regimes_filter)].copy()

        # --- BULK ACTION BUTTONS ---
        col_btns1, col_btns2, _ = st.columns([1, 1, 4])
        
        with col_btns1:
            if st.button("☑️ Select All", help="Select all currently filtered segments"):
                st.session_state['ss_raw_results'].loc[df_filtered_view.index, 'Selected'] = True
                st.rerun()
        with col_btns2:
            if st.button("⬜ Deselect All", help="Deselect all currently filtered segments"):
                st.session_state['ss_raw_results'].loc[df_filtered_view.index, 'Selected'] = False
                st.rerun()

        # --- DATA EDITOR ---
        base_cols = ['Selected', 'Regime', 'Segment', 'Start', 'End', 'Duration_Min']
        if isinstance(target_params, list):
            param_cols = [f'Mean_{c}' for c in target_params]
            final_cols = base_cols + param_cols
        else:
            final_cols = base_cols + ['Mean', 'Std']

        display_df = df_filtered_view[final_cols].copy()
        display_df['Duration (Days)'] = display_df['Duration_Min'] / 1440
        
        column_configuration = {
            "Selected": st.column_config.CheckboxColumn("Select", default=False),
            "Regime": st.column_config.NumberColumn("Regime", format="Regime %d"),
            "Segment": st.column_config.NumberColumn("ID", width="small"),
            "Start": st.column_config.DatetimeColumn("Start", format="D MMM, HH:mm"),
            "End": st.column_config.DatetimeColumn("End", format="D MMM, HH:mm"),
            "Duration_Min": st.column_config.ProgressColumn(
                "Duration", format="%d min",
                min_value=0, max_value=int(display_df['Duration_Min'].max()) if not display_df.empty else 100
            ),
            "Duration (Days)": st.column_config.NumberColumn("Days", format="%.2f d")
        }
        
        if isinstance(target_params, list):
            for p in target_params:
                column_configuration[f'Mean_{p}'] = st.column_config.NumberColumn(f"Avg {p}", format="%.2f")
        else:
            column_configuration['Mean'] = st.column_config.NumberColumn("Average", format="%.2f")
            column_configuration['Std'] = st.column_config.NumberColumn("Std Dev", format="%.3f")

        # Display Data Editor
        edited_df = st.data_editor(
            display_df,
            column_config=column_configuration,
            column_order=['Selected', 'Regime', 'Segment', 'Start', 'End', 'Duration (Days)', 'Duration_Min'] + (param_cols if isinstance(target_params, list) else ['Mean', 'Std']),
            hide_index=True,
            width='stretch',
            height=400,
            key='steady_state_editor'
        )

        # Sync changes from Data Editor back to Session State
        if not edited_df.equals(display_df):
            st.session_state['ss_raw_results'].loc[edited_df.index, 'Selected'] = edited_df['Selected']
        
        # ==================== DOWNLOAD SECTION ====================
        st.write("")
        col_d1, col_d2 = st.columns([3, 1])
        
        full_df = st.session_state['ss_raw_results']
        count_selected = full_df['Selected'].sum()
        
        with col_d1:
            if count_selected > 0:
                st.info(f"📌 **{count_selected}** segments selected for export.")
            else:
                st.caption("ℹ️ No segments selected. Export will include ALL segments.")

        with col_d2:
            if count_selected > 0:
                df_to_download = full_df[full_df['Selected'] == True]
            else:
                df_to_download = full_df
            
            csv = df_to_download.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results",
                csv,
                "steady_states_data.csv",
                "text/csv",
                key='download-csv-table',
                width='stretch'
            )
        
elif selected_view == "🚨 Anomaly Detection":

    # Ensure session key exists (for PDF/report generation later)
    if "anomaly_results" not in st.session_state:
        st.session_state["anomaly_results"] = None

    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)

    # Set the only available method
    anomaly_mode = "Unsupervised Fault Detection"
    st.markdown(f"**Detection Method:** {anomaly_mode}")

    # ==================== UNSUPERVISED FAULT DETECTION ====================
    if anomaly_mode == "Unsupervised Fault Detection":
        st.markdown("### 🏥 System Health Monitor (Multivariate)")
        st.info("ℹ️ This module correlates selected sensors to generate a composite 'Health Score'. 100% indicates normal operation; 0% indicates critical deviation.")
        
        # --- CONSOLIDATED CONFIG PANEL ---
        st.markdown('<div class="section-header" style="margin-top: 1rem;">1. Model Configuration</div>', unsafe_allow_html=True)
        with st.container(border=True):

            # -------- Feature Selection --------
            st.markdown("#### Feature Selection")
            
            # [NEW] Clustering-Based Selection Mode Toggle
            use_clustering_mode = st.toggle("✨ Enable Clustering-Based Selection", 
                                            help="Select a primary parameter and choose correlated sensors.")

            if use_clustering_mode:
                        # [NEW] Clustering Workflow Container
                        with st.expander("🛠️ Configure Cluster Parameters", expanded=True):
                            st.caption("Select a primary parameter to find others that move with it.")
                            
                            # 1. Select Primary Parameter
                            numeric_cols = st.session_state.get('num_cols', [])
                            primary_param = st.selectbox("Select Primary Parameter", options=numeric_cols, index=0)
                            
                            if primary_param:
                                # 2. Calculate Correlations (Cached)
                                corr_key = f"corr_{primary_param}_{st.session_state.get('prev_df_id', 'init')}"
                                
                                if corr_key not in st.session_state:
                                    df_analysis = df_filtered[numeric_cols].dropna()
                                    if not df_analysis.empty:
                                        correlations = df_analysis.corrwith(df_analysis[primary_param]).drop(primary_param)
                                        st.session_state[corr_key] = correlations
                                    else:
                                        st.session_state[corr_key] = pd.Series()
                                
                                correlations = st.session_state[corr_key]

                                # Split into Positive and Negative
                                pos_corr = correlations[correlations > 0].sort_values(ascending=False)
                                neg_corr = correlations[correlations < 0].sort_values(ascending=True)

                                # Get all correlated parameters
                                all_corr_params = list(pos_corr.index) + list(neg_corr.index)

                                # Initialize Checkbox State (Default: False/Unchecked)
                                for p in all_corr_params:
                                    k = f"chk_{p}"
                                    if k not in st.session_state:
                                        st.session_state[k] = False 

                                # 3. Bulk Action Buttons
                                col_btn1, col_btn2, _ = st.columns([1, 1, 3])
                                with col_btn1:
                                    if st.button("☑️ Select All", key="btn_sel_all", help="Select all correlated parameters"):
                                        for p in all_corr_params:
                                            st.session_state[f"chk_{p}"] = True
                                        st.rerun()
                                        
                                with col_btn2:
                                    if st.button("⬜ Deselect All", key="btn_desel_all", help="Deselect all parameters"):
                                        for p in all_corr_params:
                                            st.session_state[f"chk_{p}"] = False
                                        st.rerun()

                                st.markdown("---")
                                
                                # 4. Display Checkbox Lists
                                col_pos, col_neg = st.columns(2)
                                
                                selected_from_cluster = []

                                with col_pos:
                                    st.markdown("##### 📈 Positive Correlation")
                                    if pos_corr.empty:
                                        st.caption("No positive correlations found.")
                                    else:
                                        for param, score in pos_corr.items():
                                            k = f"chk_{param}"
                                            is_checked = st.checkbox(f"{param} ({score:.2f})", key=k)
                                            if is_checked:
                                                selected_from_cluster.append(param)

                                with col_neg:
                                    st.markdown("##### 📉 Negative Correlation")
                                    if neg_corr.empty:
                                        st.caption("No negative correlations found.")
                                    else:
                                        for param, score in neg_corr.items():
                                            k = f"chk_{param}"
                                            is_checked = st.checkbox(f"{param} ({score:.2f})", key=k)
                                            if is_checked:
                                                selected_from_cluster.append(param)
                                
                                # --- UPDATED LOGIC HERE ---
                                # Final List Construction: Exclude primary_param
                                multi_params = selected_from_cluster
                                
                                if len(multi_params) >= 2:
                                    st.success(f"✅ Selected **{len(multi_params)}** parameters for health model (Primary Excluded).")
                                elif len(multi_params) == 1:
                                    st.warning("⚠️ Please select at least 2 correlated parameters to run Multivariate Analysis.")
                                else:
                                    st.info("ℹ️ Select correlated parameters above to proceed.")
            
            else:
                # [EXISTING] Manual Selection Mode
                col_sel, col_info = st.columns([2, 1])

                with col_sel:
                    multi_params = st.multiselect(
                        "Select Sensors (Correlated Parameters)",
                        options=st.session_state.get('num_cols', []),
                        default=st.session_state.get('num_cols', [])[:3]
                        if len(st.session_state.get('num_cols', [])) >= 3 else None,
                        help="Select variables that should move together during normal operation."
                    )

                with col_info:
                    st.write("")
                    st.caption("Select at least 2 parameters representing a single asset.")
            st.markdown("---")
            st.markdown("#### Detection Controls")

            # Detection controls (no Run button here)
            col_sens, col_clean, col_spacer2 = st.columns([2, 1, 1])
            with col_sens:
                sensitivity = st.slider(
                    "Anomaly Multiplier (Fault Severity)",
                    min_value=0.01, max_value=1.00,
                    value=0.10, step=0.10, format="%.2f",
                    key='anomaly_multiplier_slider'
                )
                st.markdown(
                    '<div style="display:flex; justify-content:space-between;">'
                    '<small><b>Min: 0.01</b></small>'
                    '<small><b>Max: 1.00</b></small>'
                    '</div>',
                    unsafe_allow_html=True
                )
            with col_clean:
                auto_clean = st.checkbox(
                    "Exclude Abnormal Events from Baseline",
                    value=True
                )

        # ---------- PRE-RUN VALIDATION (still show baseline UI) ----------
        if len(multi_params) < 2:
            st.warning("⚠️ Please select at least 2 parameters for Multivariate analysis.")
            st.stop()

        # Prepare working dataframe (used for baseline UI)
        df_working = df_filtered[multi_params].dropna()
        if df_working.empty:
            st.error("No data available for selected parameters.")
            st.stop()

        # Step 2: Baseline selection UI
        st.markdown("---")
        st.markdown("#### 2. Define Baseline (Normal Operation)")

        ss_results = st.session_state.get('ss_raw_results')
        has_steady_states = False
        if ss_results is not None and not ss_results.empty and 'Selected' in ss_results.columns:
            if ss_results['Selected'].any():
                has_steady_states = True

        use_steady_state = False
        training_ranges = []

        if has_steady_states:
            baseline_source = st.radio(
                "Baseline Source",
                ["Use Suggested Baseline (Selected Steady States)", "Manual Time Range"],
                help="Select 'Suggested' to use steady state periods."
            )

            if baseline_source == "Use Suggested Baseline (Selected Steady States)":
                use_steady_state = True
                selected_ss = ss_results[ss_results['Selected'] == True]
                st.success(f"✅ Using {len(selected_ss)} confirmed steady state segments as baseline model.")
                train_slices = []
                for _, row in selected_ss.iterrows():
                    s, e = row['Start'], row['End']
                    slice_df = df_working[(df_working.index >= s) & (df_working.index <= e)]
                    if not slice_df.empty:
                        train_slices.append(slice_df)
                        training_ranges.append((s, e))
                if train_slices:
                    df_train = pd.concat(train_slices)
                else:
                    st.error("Selected segments contain no data matching current filters.")
                    st.stop()

        # Manual baseline selection
        if not use_steady_state:
            min_t, max_t = df_working.index.min(), df_working.index.max()
            # create 4 columns for date/time inputs (no Run button here)
            c1, c2, c3, c4 = st.columns([1, 0.6, 1, 0.6])

            with c1:
                train_start = st.date_input("Start Date", value=min_t, min_value=min_t, max_value=max_t)
            with c2:
                t_start_time = st.time_input("Start Time", value=min_t.time())
            with c3:
                train_end = st.date_input("End Date", value=min_t + pd.Timedelta(days=30) if (min_t + pd.Timedelta(days=1)) <= max_t else max_t, min_value=min_t, max_value=max_t)
            with c4:
                t_end_time = st.time_input("End Time", value=max_t.time())

            full_train_start = pd.Timestamp(f"{train_start} {t_start_time}")
            full_train_end = pd.Timestamp(f"{train_end} {t_end_time}")

            if full_train_start >= full_train_end:
                st.error("⚠️ Start time must be before End time.")
                st.stop()

            df_train = df_working[(df_working.index >= full_train_start) & (df_working.index <= full_train_end)]
            training_ranges.append((full_train_start, full_train_end))

        # If user used suggested baseline but no training ranges available -> stop
        if use_steady_state and len(training_ranges) == 0:
            st.error("No training ranges available from suggested steady states.")
            st.stop()

        # ---------- Section 3 header and run button placement ----------     
        st.markdown("---")
        st.markdown("#### 3. Health Monitor")

        st.markdown("""
                    <style>

                        /* Remove the colored middle track between the two slider handles */
                        div[data-baseweb="slider"] div[role="slider"] + div > div {
                            background: transparent !important;
                        }

                        /* Alternative catch-all: remove any colored bar shorter than the handle */
                        div[data-baseweb="slider"] div {
                            min-height: 10px !important;   /* keep handles unchanged */
                        }

                        div[data-baseweb="slider"] div:not(:has(div)) {
                            background: #dee2e6 !important;    /* normal track color */
                        }

                    </style>
                    """, unsafe_allow_html=True)
        # Layout: slider (big) | metric | run button aligned right
        col_slider, col_metric, col_btn = st.columns([3, 0.8, 0.8])

        # Slider area (we still let user adjust thresholds before/after run)
        with col_slider:
            # Dual-handle slider returns a tuple (lower_bound, upper_bound)
            thresholds = st.slider(
                "Health Zones Configuration",
                min_value=0, 
                max_value=100, 
                value=(40, 70), # Default: Critical < 40, Warning 40-70, Healthy > 70
                step=1,
                help="Adjust handles to define zones:\n• 0 to Lower Handle: 🚨 Critical\n• Lower to Upper Handle: ⚠️ Warning\n• Upper Handle to 100: ✅ Healthy"
            )
            thresh_crit, thresh_warn = thresholds
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; font-size:12px; color:#666;'>"
                f"<span>🚨 <b>Critical</b> (0-{thresh_crit}%)</span>"
                f"<span>⚠️ <b>Warning</b> ({thresh_crit}-{thresh_warn}%)</span>"
                f"<span>✅ <b>Healthy</b> ({thresh_warn}-100%)</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Metric area: show latest health if model run; otherwise show placeholder
        with col_metric:
            if st.session_state.get("anomaly_results") is not None:
                # show stored latest health if present
                stored = st.session_state["anomaly_results"]
                latest_health = stored.get("latest_health", None)
                latest_delta = stored.get("latest_delta", None)
                if latest_health is not None:
                    st.metric("Latest Health Score", f"{latest_health:.1f}%", f"{latest_delta:+.2f}%")
                else:
                    st.metric("Latest Health Score", "N/A", "")
            else:
                st.metric("Latest Health Score", "N/A", "")

        # Run Model button on the right of the metric
        with col_btn:
            st.write("")  # visual alignment
            run_model = st.button("🚀 Run Model", type="primary", key="run_button_health")

        # =========================================================
        #  LOGIC PART A: MODEL EXECUTION (Only runs on click)
        # =========================================================
        if run_model:
            with st.spinner("Analyzing system health..."):
                try:
                    # Basic validation
                    if len(df_train) < len(multi_params) + 5:
                        st.error("❌ Not enough training data. Please select a wider range or more steady state segments.")
                        st.stop()

                    # --- MODEL FITTING ---
                    mu = df_train.mean().values
                    cov = np.cov(df_train.values.T)
                    inv_cov = np.linalg.pinv(cov)

                    # Auto-clean baseline
                    if auto_clean:
                        diff_train = df_train.values - mu
                        left_train = np.dot(diff_train, inv_cov)
                        mahal_sq_train = np.sum(left_train * diff_train, axis=1)
                        clean_limit = chi2.ppf(0.99, df=len(multi_params))
                        clean_mask = mahal_sq_train <= clean_limit
                        if clean_mask.sum() > len(multi_params) + 2:
                            df_train_clean = df_train[clean_mask]
                            mu = df_train_clean.mean().values
                            cov = np.cov(df_train_clean.values.T)
                            inv_cov = np.linalg.pinv(cov)
                            removed_count = len(df_train) - len(df_train_clean)
                            if removed_count > 0:
                                st.caption(f"✨ Auto-cleaned baseline: Removed {removed_count} outliers from training data.")

                    # --- SCORING ON FULL DATASET ---
                    # We work on a copy to avoid mutating the original repeatedly
                    df_calc = df_working.copy()
                    
                    diff = df_calc.values - mu
                    left = np.dot(diff, inv_cov)
                    contrib_raw = left * diff
                    mahal_sq = np.sum(contrib_raw, axis=1)
                    df_calc['Mahalanobis_Dist'] = np.sqrt(mahal_sq)

                    # Contribution %
                    contrib_abs = np.abs(contrib_raw)
                    contrib_sum = np.sum(contrib_abs, axis=1)
                    contrib_sum[contrib_sum == 0] = 1.0
                    contrib_pcts = (contrib_abs / contrib_sum[:, None]) * 100

                    hover_strs = pd.Series("", index=df_calc.index)
                    for idx, col_name in enumerate(multi_params):
                        hover_strs += f"<b>{col_name}:</b> " + pd.Series(contrib_pcts[:, idx], index=df_calc.index).map('{:.1f}%'.format) + "<br>"
                    df_calc['Hover_Info'] = hover_strs

                    # Health Score transform
                    degrees_of_freedom = len(multi_params)
                    base_limit = np.sqrt(chi2.ppf(0.999, df=degrees_of_freedom))
                    adjusted_limit = base_limit / sensitivity
                    target_health = 0.70
                    k = np.log(target_health) / (adjusted_limit**2)
                    df_calc['Health_Score'] = 100 * np.exp(k * (df_calc['Mahalanobis_Dist']**2))
                    df_calc['Health_Score'] = df_calc['Health_Score'].clip(0, 100)

                    # Update statuses based on thresholds user selected
                    conditions = [
                        (df_calc['Health_Score'] <= thresh_crit),
                        (df_calc['Health_Score'] <= thresh_warn) & (df_calc['Health_Score'] > thresh_crit),
                        (df_calc['Health_Score'] > thresh_warn)
                    ]
                    choices = ['Critical', 'Warning', 'Healthy']
                    df_calc['Status'] = np.select(conditions, choices, default='Healthy')

                    # ---------- VISUALIZATION (fig_health) ----------
                    fig_health = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.12,
                        subplot_titles=("System Health Score", "Raw Parameter Trends (Overlaid)")
                    )

                    # Zones
                    fig_health.add_hrect(y0=thresh_warn, y1=100, fillcolor="#EAFAF1", opacity=0.7, line_width=0, layer="below", row=1, col=1)
                    fig_health.add_hrect(y0=thresh_crit, y1=thresh_warn, fillcolor="#FEF9E7", opacity=0.7, line_width=0, layer="below", row=1, col=1)
                    fig_health.add_hrect(y0=0, y1=thresh_crit, fillcolor="#FDEDEC", opacity=0.7, line_width=0, layer="below", row=1, col=1)

                    # Health score trace
                    fig_health.add_trace(go.Scatter(
                        x=df_calc.index, y=df_calc['Health_Score'],
                        mode='lines', name='Health Score',
                        line=dict(color='#17202A', width=2.5),
                        customdata=df_calc['Hover_Info'],
                        hovertemplate='<b>Health: %{y:.1f}%</b><br><br>Contribution:<br>%{customdata}<extra></extra>'
                    ), row=1, col=1)

                    # time region shading (warning / critical)
                    health = df_calc['Health_Score']
                    warning_mask = (health <= thresh_warn) & (health > thresh_crit)
                    critical_mask = (health <= thresh_crit)

                    def add_time_regions(mask, fillcolor, opacity):
                        mask = mask.astype(int)
                        diff = mask.diff().fillna(0)
                        starts = list(mask.index[diff == 1])
                        ends = list(mask.index[diff == -1])
                        if mask.iloc[0] == 1:
                            starts.insert(0, mask.index[0])
                        if mask.iloc[-1] == 1:
                            ends.append(mask.index[-1])
                        for s, e in zip(starts, ends):
                            fig_health.add_vrect(x0=s, x1=e, fillcolor=fillcolor, opacity=opacity, line_width=0, layer="below", row=1, col=1)

                    add_time_regions(warning_mask, fillcolor="rgba(255, 215, 0, 0.60)", opacity=0.40)
                    add_time_regions(critical_mask, fillcolor="rgba(255, 140, 0, 0.50)", opacity=0.40)

                    # thresholds lines
                    fig_health.add_hline(y=thresh_warn, line_dash="dash", line_color="#F39C12", row=1, col=1)
                    fig_health.add_hline(y=thresh_crit, line_dash="dash", line_color="#E74C3C", row=1, col=1)

                    # highlight baseline training ranges
                    for (rng_start, rng_end) in training_ranges:
                        fig_health.add_vrect(x0=rng_start, x1=rng_end, line_width=0, fillcolor="#C1DBEF", opacity=0.6, annotation_text="Baseline", annotation_position="top right", row=1, col=1)

                    # Raw parameter lines (row 2)
                    colors = px.colors.qualitative.Bold
                    for idx, col in enumerate(multi_params):
                        fig_health.add_trace(go.Scatter(
                            x=df_calc.index, y=df_calc[col],
                            mode='lines', name=col,
                            opacity=0.8, line=dict(width=1.3, color=colors[idx % len(colors)]),
                            hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>"
                        ), row=2, col=1)

                    # layout
                    fig_health.update_yaxes(title_text="<b>Health %</b>", range=[0, 105], showgrid=True, gridcolor='#F2F4F7', row=1, col=1)
                    fig_health.update_yaxes(title_text="<b>Raw Value</b>", showgrid=True, gridcolor='#F2F4F7', row=2, col=1)
                    fig_health.update_xaxes(showgrid=True, gridcolor='#F2F4F7')
                    fig_health.update_layout(template="plotly_white", height=700, hovermode="x unified", margin=dict(t=60, b=50, l=60, r=40),
                                            title={'text': "<b>System Health Monitor (Multivariate)</b>", 'y': 0.96, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                                            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))

                    # ---------------- SAVE RESULTS TO SESSION STATE ----------------
                    health_chart_path = "/mnt/data/health_chart.png"
                    raw_chart_path = "/mnt/data/raw_chart.png"
                    
                    crit_count = (df_calc['Status'] == 'Critical').sum()
                    warn_count = (df_calc['Status'] == 'Warning').sum()
                    healthy_pct = (df_calc['Status'] == 'Healthy').mean() * 100

                    latest_health = df_calc['Health_Score'].iloc[-1]
                    latest_delta = (df_calc['Health_Score'].iloc[-1] - df_calc['Health_Score'].iloc[-2]) if len(df_calc) > 1 else 0.0

                    # Store everything we need
                    st.session_state["anomaly_df_working"] = df_calc
                    st.session_state["fig_health"] = fig_health
                    st.session_state["anomaly_multi_params"] = multi_params
                    st.session_state["anomaly_results"] = {
                        "multi_params": multi_params,
                        "training_ranges": training_ranges,
                        "sensitivity": sensitivity,
                        "auto_clean": auto_clean,
                        "thresh_crit": thresh_crit,
                        "thresh_warn": thresh_warn,
                        "crit_count": crit_count,
                        "warn_count": warn_count,
                        "healthy_pct": healthy_pct,
                        "health_chart_path": health_chart_path,
                        "raw_chart_path": raw_chart_path,
                        "latest_health": latest_health,
                        "latest_delta": latest_delta
                    }
                    
                    st.rerun() # Refresh to show results immediately

                except np.linalg.LinAlgError:
                    st.error("❌ Singular Matrix Error: Variables are perfectly correlated. Remove duplicate parameters.")
                except Exception as e:
                    handle_error(e, "Health Calc")

        # =========================================================
        #  LOGIC PART B: PERSISTENT DISPLAY (Runs if data exists)
        # =========================================================
        
        # Check if we have results in session state
        if st.session_state.get("anomaly_results") is not None and "fig_health" in st.session_state:
            
            st.markdown("### 📊 Health Analysis Results")
            st.caption("⚠️ Note: Raw trends are overlaid; differing units/scales may dominate the Y-axis.")
            
            # Display the persistent chart
            st.plotly_chart(
                st.session_state["fig_health"], 
                width="stretch", 
                config={'displayModeBar': True, 'displaylogo': False},
                key="health_monitor_chart_persistent"
            )

        # =========================================================
        #  LOGIC PART C: REPORT GENERATION (Appears after chart)
        # =========================================================

        st.markdown("---")
        st.markdown('<div class="section-header">📄 Anomaly Detection Report</div>', unsafe_allow_html=True)

        if st.session_state.get("anomaly_results") is not None:
            # Create report generation UI
            col_report1, col_report2 = st.columns([2, 1])

            with col_report1:
                st.markdown("### Generate Comprehensive Report")
                st.info("📋 Generate a detailed report including model configuration, statistics, visualizations, and anomaly details.")
                
            with col_report2:
                st.write("")
                st.write("")
                generate_report = st.button("📊 Generate Report", type="primary", key="generate_anomaly_report")

            if generate_report:
                with st.spinner("Generating comprehensive anomaly detection report..."):
                    try:
                        from datetime import datetime
                        import io
                        
                        # Retrieve stored results
                        results = st.session_state["anomaly_results"]
                        df_calc = st.session_state["anomaly_df_working"]
                        multi_params = st.session_state.get("anomaly_multi_params", results.get("multi_params", []))
                        
                        crit_count = results["crit_count"]
                        warn_count = results["warn_count"]
                        healthy_pct = results["healthy_pct"]
                        latest_health = results["latest_health"]
                        sensitivity = results["sensitivity"]
                        auto_clean = results["auto_clean"]
                        
                        # Prepare report data
                        report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 1. Summary Stats
                        summary_stats = pd.DataFrame({
                            "Metric": [
                                "Total Data Points", "Critical Events", "Warning Events", 
                                "Healthy Percentage", "Latest Health Score", "Average Health Score",
                                "Minimum Health Score", "Parameters Monitored", "Sensitivity Setting", "Auto-Clean Enabled"
                            ],
                            "Value": [
                                f"{len(df_calc):,}", f"{crit_count:,}", f"{warn_count:,}",
                                f"{healthy_pct:.2f}%", f"{latest_health:.2f}%",
                                f"{df_calc['Health_Score'].mean():.2f}%",
                                f"{df_calc['Health_Score'].min():.2f}%",
                                ", ".join(multi_params), f"{sensitivity:.2f}",
                                "Yes" if auto_clean else "No"
                            ]
                        })
                        
                        # 2. Anomaly Details
                        anomaly_df = df_calc[df_calc['Status'].isin(['Critical', 'Warning'])].copy()
                        if not anomaly_df.empty:
                            anomaly_df = anomaly_df.reset_index()
                            # Safe column selection
                            timestamp_col = df_calc.index.name or 'Timestamp'
                            desired_cols = [timestamp_col, 'Health_Score', 'Status'] + multi_params
                            avail_cols = [c for c in desired_cols if c in anomaly_df.columns]
                            anomaly_df = anomaly_df[avail_cols]
                        
                        # 3. Status Dist
                        status_distribution = df_calc['Status'].value_counts().reset_index()
                        status_distribution.columns = ['Status', 'Count']
                        status_distribution['Percentage'] = (status_distribution['Count'] / len(df_calc) * 100).round(2)
                        
                        # --- DISPLAY REPORT ---
                        st.success("✅ Report Generated Successfully!")
                        
                        # Executive Summary
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
                            <h2 style="margin: 0; color: white;">🚨 Anomaly Detection Report</h2>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Generated: {report_timestamp}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Points", f"{len(df_calc):,}")
                        c2.metric("Critical Events", f"{crit_count:,}", delta=f"{(crit_count/len(df_calc)*100):.1f}%", delta_color="inverse")
                        c3.metric("Warning Events", f"{warn_count:,}", delta=f"{(warn_count/len(df_calc)*100):.1f}%", delta_color="inverse")
                        c4.metric("Healthy %", f"{healthy_pct:.1f}%", delta=f"{healthy_pct-100:.1f}%", delta_color="normal")
                        
                        st.markdown("---")
                        
                        # Tables and Charts
                        st.markdown("### ⚙️ Model Configuration")
                        st.dataframe(summary_stats, width='stretch', hide_index=True)
                        
                        st.markdown("### 📉 Status Distribution")
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.dataframe(status_distribution, width='stretch', hide_index=True)
                        with c2:
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=status_distribution['Status'],
                                values=status_distribution['Count'],
                                marker=dict(colors=['#10B981', '#F59E0B', '#EF4444']),
                                hole=0.4
                            )])
                            fig_pie.update_layout(title="Health Status Distribution", height=300)
                            st.plotly_chart(fig_pie, width='stretch')
                        
                        st.markdown("### 🚨 Detected Anomaly Events")
                        if not anomaly_df.empty:
                            st.dataframe(anomaly_df.head(100), width='stretch')
                        else:
                            st.success("✅ No anomalies detected!")

                        # Export Buttons
                        st.markdown("### 💾 Export Report Data")
                        ec1, ec2, ec3 = st.columns(3)
                        
                        with ec1:
                            buffer_excel = io.BytesIO()
                            with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
                                summary_stats.to_excel(writer, sheet_name='Summary', index=False)
                                status_distribution.to_excel(writer, sheet_name='Distribution', index=False)
                                if not anomaly_df.empty: anomaly_df.to_excel(writer, sheet_name='Anomalies', index=False)
                                df_calc.to_excel(writer, sheet_name='Full Results', index=True)
                            
                            st.download_button("📊 Download Excel Report", buffer_excel.getvalue(), f"report_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                        with ec2:
                            if not anomaly_df.empty:
                                st.download_button("📄 Download Anomalies CSV", anomaly_df.to_csv(index=False).encode('utf-8'), "anomalies.csv", "text/csv")
                            else:
                                st.button("📄 Download Anomalies CSV", disabled=True)

                        with ec3:
                             st.download_button("📋 Download Full Results CSV", df_calc.to_csv().encode('utf-8'), "full_results.csv", "text/csv")

                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        # Debug info
                        # st.write(e) 

        else:
            st.info("ℹ️ Run the anomaly detection model first to generate a report.")  
# ==================== TAB 4: Export ====================

elif selected_view == "📤 Export":
    st.markdown('<div class="section-header">Export Options</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Data Export")
        
        export_format = st.radio(
            "Format",
            ["CSV", "Excel", "Parquet", "JSON"],
            horizontal=True
        )
        
        include_metadata = st.checkbox("Include filter summary", value=True)
        
        if st.button("📥 Download Data", type="primary", width="stretch"):
            with st.spinner("Preparing export..."):
                try:
                    if export_format == "CSV":
                        data = df_filtered.to_csv().encode("utf-8")
                        mime = "text/csv"
                        filename = "filtered_data.csv"
                    
                    elif export_format == "Excel":
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            df_filtered.to_excel(writer, sheet_name="Data")
                            if include_metadata:
                                metadata = pd.DataFrame({
                                    "Setting": ["Start Date", "End Date", "Rows", "Columns"],
                                    "Value": [
                                        str(start_date),
                                        str(end_date),
                                        len(df_filtered),
                                        len(df_filtered.columns)
                                    ]
                                })
                                metadata.to_excel(writer, sheet_name="Metadata", index=False)
                        data = buffer.getvalue()
                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        filename = "filtered_data.xlsx"
                    
                    elif export_format == "Parquet":
                        buffer = io.BytesIO()
                        df_filtered.to_parquet(buffer)
                        data = buffer.getvalue()
                        mime = "application/octet-stream"
                        filename = "filtered_data.parquet"
                    
                    else:  # JSON
                        data = df_filtered.to_json(orient="records", date_format="iso").encode("utf-8")
                        mime = "application/json"
                        filename = "filtered_data.json"
                    
                    st.download_button(
                        label=f"⬇️ Download {export_format}",
                        data=data,
                        file_name=filename,
                        mime=mime,
                        width="stretch"
                    )
                    
                    st.success(f"✅ {export_format} file ready for download!")
                
                except Exception as e:
                    handle_error(e, "Data Export")
    
    with col2:
        st.markdown("### 📈 Chart Export")
        
        st.info("💡 **Tip**: Use the Plotly toolbar on any chart to export as PNG or SVG")
        
        st.markdown("""
        **Export Options:**
        - 📸 **PNG**: Click camera icon on chart
        - 🎨 **SVG**: Use download menu
        - 📋 **Data**: Export underlying data
        
        **For programmatic export:**
        ```python
        # Install kaleido
        pip install plotly[kaleido]
        ```
        """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Export Summary")
    
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        st.metric("Rows to Export", f"{len(df_filtered):,}")
    
    with summary_cols[1]:
        st.metric("Columns", len(df_filtered.columns))
    
    with summary_cols[2]:
        size_mb = df_filtered.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Est. Size", f"{size_mb:.2f} MB")
    
    with summary_cols[3]:
        st.metric("Date Range", f"{(df_filtered.index.max() - df_filtered.index.min()).days} days")
    
    if filters_tuple or resample_freq or missing_strategy != "No action":
        st.markdown("---")
        st.markdown("### 🔍 Applied Processing")
        
        with st.expander("View Details", expanded=False):
            if filters_tuple:
                st.markdown("**Filters:**")
                for p, o, v in filters_tuple:
                    st.write(f"- {p} {o} {v}")
            
            if resample_freq:
                st.markdown(f"**Resampling:** {resample_label}")
            
            if missing_strategy != "No action":
                st.markdown(f"**Missing Values:** {missing_strategy}")
            
            st.markdown(f"**Date Range:** {start_date} to {end_date}")

# ==================== Footer & Shortcuts ====================

st.markdown("""
<div class="shortcut-hint">
    💡 Press <kbd>R</kbd> to refresh
</div>
""", unsafe_allow_html=True)

# ==================== Performance Metrics (Optional) ====================

with st.sidebar:
    with st.expander("📊 Performance Info", expanded=False):
        st.caption(f"**Original Rows:** {len(df):,}")
        st.caption(f"**Filtered Rows:** {len(df_filtered):,}")
        
        if resample_freq:
            reduction = (1 - len(df_filtered)/len(df)) * 100
            st.caption(f"**Reduction:** {reduction:.1f}%")
        
        memory_mb = df_filtered.memory_usage(deep=True).sum() / (1024 * 1024)
        st.caption(f"**Memory Usage:** {memory_mb:.2f} MB")
        
        if len(df_filtered) > 50000:
            st.warning("⚠️ Consider resampling for better performance")

    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            **Purpose:** An interactive, high-performance platform for time-series data exploration, visualization, and anomaly detection.
            
            """
        )
