# app.py - Enhanced Data Discovery Dashboard
# Industry-standard UI/UX improvements with better organization, feedback, and user experience
from __future__ import annotations
import io
import math
import time
import hashlib
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import matplotlib as mpl

if "sidebar_state_set" not in st.session_state:
    st.session_state.sidebar_state_set = True
    st.markdown("""
        <script>
        // Collapse sidebar
        window.parent.document.querySelector("section[data-testid='stSidebar']").style.display = 'none';
        // Expand icon stays clickable
        </script>
    """, unsafe_allow_html=True)
    
# Optional imports (graceful fallbacks)
SKLEARN_AVAILABLE = False
try:
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.cluster import KMeans as SkKMeans
    from sklearn.ensemble import IsolationForest as SkIsolationForest
    PCA = SkPCA
    KMeans = SkKMeans
    IsolationForest = SkIsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    class PCA:
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

    IsolationForest = None

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

# Page Config
st.set_page_config(
    page_title="Data Insights Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    /* Dark Theme */
    .dark-theme .main-title {
        color: #E5E7EB;
    }
    
    .dark-theme .section-header {
        color: #E5E7EB;
        border-bottom-color: #4B5563;
    }
    
    .dark-theme .tile {
        background: linear-gradient(135deg, #2B2F36 0%, #1F2A44 100%);
        border-color: #4B5563;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .dark-theme .tile h3 {
        color: #E5E7EB;
    }
    
    .dark-theme .tile p {
        color: #9CA3AF;
    }
    
    .dark-theme div[role="radiogroup"] {
        background-color: #2B2F36;
    }
    
    .dark-theme div[role="radiogroup"] label {
        color: #9CA3AF;
    }
    
    .dark-theme div[role="radiogroup"] label:hover {
        background-color: rgba(255,255,255,0.1);
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
# =====================================================================
# ==================== Helper Functions ====================

LIGHT_TEMPLATE = "plotly_white"
DARK_TEMPLATE = "plotly_dark"

def theme_parts(dark: bool = False):
    if dark:
        return DARK_TEMPLATE, "#111418", "#111418", "#E5E7EB", "#2B2F36"
    return LIGHT_TEMPLATE, "white", "white", "#0E1117", "#E5ECF6"

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
            elif operator == "<=":
                df_out = df_out[series <= float(value)]
        else:
            s = col_vals.astype(str)
            if operator == "=":
                df_out = df_out[s == str(value)]
            elif operator == "≠":
                df_out = df_out[s != str(value)]
    return df_out

# --- NEW: Steady State Detection Function ---
def detect_steady_state(df_in: pd.DataFrame, column: str, window_size: str, std_threshold: float, min_duration_minutes: int) -> Dict[str, List[Dict]]:
    """
    Detects steady state periods for a single, specified numeric column (required by 'column' arg).
    
    A steady state is defined as a period where the rolling standard deviation
    is below a defined threshold for a minimum continuous duration.
    """
    
    # Initialization
    results = {}
    col = column
    
    # Ensure the column exists and is numeric, then extract and clean the series
    if col not in df_in.columns or not pd.api.types.is_numeric_dtype(df_in[col]):
        return {col: []}
        
    series = df_in[col].dropna()
    
    if series.empty:
        results[col] = []
        return results

    # Convert minimum duration from minutes to a Timedelta object
    min_duration = pd.Timedelta(minutes=min_duration_minutes)

    # 1. Calculate Rolling Standard Deviation (time-based window)
    rolling_std = series.rolling(window=window_size, min_periods=1).std().fillna(series.std())
    
    # 2. Identify periods where std is below threshold
    is_steady = rolling_std < std_threshold

    # 3. Group consecutive True values (steady state periods)
    if is_steady.any():
        # Run-length encoding: create groups of consecutive True/False values
        # The logic below efficiently finds continuous blocks of 'True'
        groups = is_steady.astype(int).diff().fillna(is_steady.astype(int)).abs().cumsum()
        steady_groups = is_steady.groupby(groups).filter(lambda x: x.all())
        
        periods = []
        
        for name, group in steady_groups.groupby(groups):
            start = group.index.min()
            end = group.index.max()
            
            # Check minimum duration
            if (end - start) >= min_duration:
                # Calculate metrics for the identified steady period
                mean_val = series.loc[start:end].mean()
                std_val = series.loc[start:end].std()
                
                periods.append({
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'mean': mean_val,
                    'std': std_val
                })
        
        results[col] = periods
    else:
        results[col] = []

    return results

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

# ==================== Session State Initialization ====================

if "events" not in st.session_state:
    st.session_state["events"] = []

if "user_prefs" not in st.session_state:
    st.session_state["user_prefs"] = {
        "dark_theme": False,
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

# ==================== Main Header ====================

if st.session_state.get('uploaded_file') is not None:
    st.markdown('<div class="main-title">📊 Data Insights Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Professional time-series data exploration and visualization platform</div>', unsafe_allow_html=True)
else:
    pass

# ==================== Sidebar ====================

with st.sidebar:
    st.markdown("### 📁 Data Source")
    
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    
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
            if df_raw is not None:
                st.sidebar.success(f"✅ Loaded {len(df_raw):,} rows")
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
    
    # How It Works Section
    st.markdown("""
    <div style="text-align: center; margin: 4rem 0 3rem 0;">
        <h2 style="color: #1F2A44; font-size: 36px; font-weight: 700; margin-bottom: 0.5rem;">
            How It Works
        </h2>
        <p style="color: #6B7280; font-size: 16px;">
            Get started in three simple steps
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    steps_col1, steps_col2, steps_col3 = st.columns(3)
    
    with steps_col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                font-weight: 700;
                margin: 0 auto 1.5rem auto;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            ">1</div>
            <h4 style="color: #1F2A44; font-size: 20px; font-weight: 600; margin-bottom: 0.75rem;">
                Upload Your Data
            </h4>
            <p style="color: #6B7280; font-size: 14px; line-height: 1.6;">
                Import CSV, Excel, or Parquet files containing your time-series data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                font-weight: 700;
                margin: 0 auto 1.5rem auto;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            ">2</div>
            <h4 style="color: #1F2A44; font-size: 20px; font-weight: 600; margin-bottom: 0.75rem;">
                Explore & Visualize
            </h4>
            <p style="color: #6B7280; font-size: 14px; line-height: 1.6;">
                Interact with powerful charts, correlations, and statistical analysis tools
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps_col3:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 32px;
                font-weight: 700;
                margin: 0 auto 1.5rem auto;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
            ">3</div>
            <h4 style="color: #1F2A44; font-size: 20px; font-weight: 600; margin-bottom: 0.75rem;">
                Export Insights
            </h4>
            <p style="color: #6B7280; font-size: 14px; line-height: 1.6;">
                Download filtered data, charts, and reports in multiple formats
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
        " class="feature-card">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            "></div>
            <div style="font-size: 48px; margin-bottom: 1rem;">📈</div>
            <h3 style="font-size: 20px; margin: 1rem 0 0.75rem 0; color: #1F2A44; font-weight: 700;">
                Time Series
            </h3>
            <p style="font-size: 13px; color: #6B7280; line-height: 1.5;">
                Interactive visualizations with multiple display modes and customizable parameters
            </p>
            <div style="
                margin-top: 1.5rem;
                padding: 6px 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                display: inline-block;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            ">Popular</div>
        </div>
        
        <style>
            .feature-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 12px 24px rgba(0,0,0,0.15);
                border-color: #2196F3;
            }
        </style>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
        " class="feature-card">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            "></div>
            <div style="font-size: 48px; margin-bottom: 1rem;">🔗</div>
            <h3 style="font-size: 20px; margin: 1rem 0 0.75rem 0; color: #1F2A44; font-weight: 700;">
                Correlation
            </h3>
            <p style="font-size: 13px; color: #6B7280; line-height: 1.5;">
                Discover hidden relationships between parameters with heatmaps and insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
        " class="feature-card">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            "></div>
            <div style="font-size: 48px; margin-bottom: 1rem;">🚨</div>
            <h3 style="font-size: 20px; margin: 1rem 0 0.75rem 0; color: #1F2A44; font-weight: 700;">
                Anomaly Detection
            </h3>
            <p style="font-size: 13px; color: #6B7280; line-height: 1.5;">
                Identify outliers automatically using Z-score and Isolation Forest algorithms
            </p>
            <div style="
                margin-top: 1.5rem;
                padding: 6px 14px;
                background: #10B981;
                color: white;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                display: inline-block;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            ">AI-Powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 16px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
        " class="feature-card">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            "></div>
            <div style="font-size: 48px; margin-bottom: 1rem;">📤</div>
            <h3 style="font-size: 20px; margin: 1rem 0 0.75rem 0; color: #1F2A44; font-weight: 700;">
                Export
            </h3>
            <p style="font-size: 13px; color: #6B7280; line-height: 1.5;">
                Save filtered data and visualizations in CSV, Excel, Parquet, or JSON format
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # File Requirements Section with Better Design
    with st.expander("📋 File Requirements & Supported Formats", expanded=False):
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

filters_container = st.sidebar.container()

with st.sidebar:
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
            use_dark = st.checkbox(
                "🌙 Dark Theme",
                value=st.session_state["user_prefs"]["dark_theme"]
            )
            st.session_state["user_prefs"]["dark_theme"] = use_dark
            
            normalize = st.checkbox("Normalize (Z-score)", help="Apply z-score normalization to data")
            log_scale = st.checkbox("Log Scale Y-axis", help="Use logarithmic scale for Y-axis")
        
        with config_tabs[2]:
            if st.button("🔄 Reset All Settings"):
                st.session_state.clear()
                st.rerun()
            
            st.caption("💾 Settings are automatically saved")

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

# Apply theme
TEMPLATE, PAPER_BG, PLOT_BG, FONT_COLOR, GRID_COLOR = theme_parts(use_dark)
if use_dark:
    st.markdown('<div class="dark-theme">', unsafe_allow_html=True)

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
            stats_df.style.background_gradient(cmap=("Greys" if use_dark else "Blues"), axis=1),
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
                
                border_color = "#4B5563" if use_dark else "#D1D5DB"
                
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
                stats_df.style.background_gradient(cmap=("Greys" if use_dark else "Blues"), axis=1),
                width="stretch"
            )

# ==================== TAB 3: Steady State Condition ====================
elif selected_view == "🔬 Steady State":
    st.markdown('<div class="section-header">🔬 Steady State Condition Analysis</div>', unsafe_allow_html=True)
    
    num_cols = st.session_state.get('num_cols', [])
    df_filtered = st.session_state['df_filtered']
    
    if df_filtered is None or df_filtered.empty:
        show_empty_state("⚠️", "No Data Loaded", "Please load a dataset and configure the date/time column.")
        st.stop()

    if not num_cols:
        st.warning("⚠️ No numeric columns available for steady state analysis.")
        st.stop()
        
    # --- Steady State Detection UI Controls ---
    st.markdown("### ⚙️ Detection Parameters")
    
    # 1. UI: Operating Regime Configuration (Fixed Mode)
    # We maintain the column layout from the screenshot but fix the mode
    col_alg, col_k = st.columns([2, 1])

    with col_alg:
        # Static display instead of radio button to lock the mode
        st.markdown("**Detection Mode**")
        st.info("Operating Regime")
        detection_mode = "Operating Regime"

    # Initialize num_clusters
    num_clusters = 0
    with col_k:
        if not SKLEARN_AVAILABLE:
            st.warning("Scikit-learn is required for K-Means.")
            num_clusters = 0
        else:
            num_clusters = st.number_input(
                "Number of Regimes (K)",
                min_value=2,
                max_value=10,
                value=3, # Typically 2 or 3 for Low/High/Transition
                step=1,
                key="steady_state_num_clusters",
                help="The number of distinct operating levels to search for."
            )
            
    # 2. Parameter Selection (Full width for visibility)
    steady_state_params = st.multiselect(
        "Select parameter(s) for analysis",
        options=num_cols,
        default=num_cols[:1],
        key="steady_state_params"
    )
    
    if not steady_state_params:
        show_empty_state("🔬", "Select Parameters", "Choose one or more parameters to begin steady state detection.")
        st.stop()
        
    # Calculate a sensible default threshold based on the first selected param's std dev
    default_std = df_filtered[steady_state_params[0]].std(skipna=True) * 0.1 if not df_filtered[steady_state_params[0]].empty else 0.1
    default_std = max(default_std, 1e-6) # Ensure non-zero minimum
    
    # 3. Rolling Window, Std Threshold, and Duration (Horizontal Columns)
    col_win, col_std, col_dur = st.columns(3)
    
    with col_win:
        window_opts = ["1min", "5min", "10min", "30min", "1H", "6H", "1D"]
        window_size = st.selectbox("Rolling Window Size (Std Dev)", options=window_opts, index=2, 
                                   help="Time window for calculating local standard deviation.")
        
    with col_std:
        std_threshold = st.number_input(
            "Std Dev Threshold (Max variance)",
            min_value=1e-12,
            value=default_std,
            format="%.6f",
            help="Maximum standard deviation allowed within the window to be considered steady."
        )
        
    with col_dur:
        min_duration_minutes = st.number_input(
            "Minimum Duration (Minutes)",
            min_value=1,
            value=60,
            step=1,
            help="Minimum continuous time (in minutes) a period must last to be considered a steady state."
        )

    st.markdown("---")
    
    # --- Run Detection ---
    steady_state_results = {}
    
    with st.spinner(f"⏳ Detecting steady state periods using {detection_mode}..."):
        
        if not SKLEARN_AVAILABLE:
            st.error("Scikit-learn is not installed. Please install sklearn to use Operating Regime detection.")
            st.stop()
        if num_clusters < 2:
            st.error("K-Means requires at least 2 clusters.")
            st.stop()
            
        for col in steady_state_params:
            series = df_filtered[col].dropna()
            if series.empty:
                steady_state_results[col] = []
                continue
                
            # 1. K-Means Clustering
            data = series.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(data)
            
            # Create a temporary DataFrame for filtering
            df_temp_cluster = pd.DataFrame(data=clusters, index=series.index, columns=[f'{col}_Cluster'])
            df_temp_cluster[col] = series
            
            # Sort clusters by mean value to easily identify the highest regime
            cluster_means = df_temp_cluster.groupby(f'{col}_Cluster')[col].mean().sort_values(ascending=False)
            
            st.markdown(f"**{col} Regimes Detected (Highest $\\rightarrow$ Lowest):**")
            
            # Create a selection box for the user to pick the desired steady state regime
            cluster_options = [f"Cluster {i} (Mean: {mean:.2f})" for i, mean in cluster_means.items()]
            selected_cluster_option = st.selectbox(
                f"Select the Operating Regime for '{col}'",
                options=cluster_options,
                index=0, # Default to the highest regime
                key=f"cluster_select_{col}"
            )
            
            # Extract the actual cluster index (e.g., 0, 1, 2) from the string
            selected_cluster_index = int(selected_cluster_option.split(' ')[1])
            
            # Filter the data to only include points in the selected cluster
            df_cluster = df_temp_cluster[df_temp_cluster[f'{col}_Cluster'] == selected_cluster_index]
            
            # 2. Apply Rolling SD stability check only to the clustered data
            col_result = detect_steady_state(
                df_cluster[[col]], # df_in
                column=col, # column
                window_size=window_size, 
                std_threshold=std_threshold, 
                min_duration_minutes=min_duration_minutes
            )
            
            # Extract the list of periods from the returned dictionary
            steady_state_results[col] = col_result.get(col, [])
            
    # --- Visualization ---
    st.markdown("### 📈 Steady State Trend View")
    
    # Prepare data for plotting (using existing logic)
    downsample_limit = 50000 
    resample_freq = st.session_state.get('prev_resample_freq')
    
    if resample_freq is None and len(df_filtered) > downsample_limit:
        keep_idx = thin_index(df_filtered.index, downsample_limit)
        df_plot = df_filtered.iloc[keep_idx]
    else:
        df_plot = df_filtered
        
    df_plot = numeric_df(df_plot[steady_state_params])
    
    # Get theme configuration
    TEMPLATE, PAPER_BG, PLOT_BG, TEXT_COLOR, GRID_COLOR = theme_parts(st.session_state["user_prefs"]["dark_theme"])

    rows = len(steady_state_params)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
        subplot_titles=steady_state_params
    )
    color_cycle = px.colors.qualitative.Plotly
    
    all_periods = []
    
    for i, col in enumerate(steady_state_params, start=1):
        color = color_cycle[(i - 1) % len(color_cycle)]
        series = df_plot[col]
        periods = steady_state_results.get(col, [])
        
        # Add the time series trace
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series, mode="lines", name=col, 
                line=dict(color=color, width=2), 
                hovertemplate="%{y:.2f}", showlegend=False
            ), 
            row=i, col=1
        )
        
        # Add steady state regions (visual marking)
        for p in periods:
            # Add a colored rectangle to mark the steady state zone
            fig.add_vrect(
                x0=p['start'], x1=p['end'],
                fillcolor="#A5D6A7", # Light green for steady state
                opacity=0.4,
                layer="below",
                line_width=0,
                row=i, col=1,
                name=f"Steady State: {col}"
            )
            
            # Store period data for the table
            all_periods.append({
                'Parameter': col,
                'Start Time': p['start'],
                'End Time': p['end'],
                'Duration': p['duration'],
                'Mean Value': p['mean'],
                'Std Dev': p['std']
            })

        fig.update_yaxes(title_text=col, gridcolor=GRID_COLOR, row=i, col=1)
        fig.update_xaxes(row=i, col=1)

    # Global layout settings
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        title=f"Steady State Periods ({detection_mode}) | Window: {window_size}, Std Threshold: {std_threshold:.6f}, Min Duration: {min_duration_minutes} min",
        height=calculate_chart_height(rows, 250, 180),
        margin=dict(t=80),
    )
    
    st.plotly_chart(fig, width='stretch')

    # --- Tabular View ---
    st.markdown("### 📋 Detected Steady State Periods")
    
    if all_periods:
        # Create a DataFrame for the table
        df_periods = pd.DataFrame(all_periods)
        
        # Format columns for display
        df_periods['Start Time'] = df_periods['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_periods['End Time'] = df_periods['End Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert Duration to a readable string (e.g., '1 days 05:00:00')
        df_periods['Duration'] = df_periods['Duration'].astype(str).str.replace('0 days ', '').str.replace('NaT', 'N/A')
        
        df_periods['Mean Value'] = df_periods['Mean Value'].round(4)
        df_periods['Std Dev'] = df_periods['Std Dev'].round(6)
        
        # Rename columns for display
        df_periods.columns = ['Parameter', 'Start Time', 'End Time', 'Duration', 'Mean Value', 'Std Dev']

        st.dataframe(
            df_periods.style.background_gradient(cmap='YlGn', subset=['Mean Value', 'Std Dev']), 
            width='stretch'
        )
        
        st.download_button(
            label="⬇️ Download Steady State Table (CSV)",
            data=df_periods.to_csv(index=False).encode('utf-8'),
            file_name='steady_state_periods.csv',
            mime='text/csv',
        )
    else:
        st.info(f"No steady state periods found using {detection_mode} with the current configuration. Try adjusting the detection mode, number of regimes (K), Std Dev Threshold, or Minimum Duration.")

# ==================== TAB 4: Anomaly Detection ====================

elif selected_view == "🚨 Anomaly Detection":
    st.markdown('<div class="section-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    selected_params_anomaly = st.multiselect(
        "Choose parameters for anomaly detection",
        options=list(df.columns),
        default=(list(num_cols[:2]) if len(num_cols) >= 2 else list(num_cols)),
        help="Select parameters to detect anomalies",
        key="anomaly_params"
    )
    
    if not selected_params_anomaly:
        show_empty_state(
            "🚨",
            "No Parameters Selected",
            "Select parameters from the sidebar to detect anomalies"
        )
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            anomaly_mode = st.selectbox(
                "Detection Method",
                ["Z-score", "Isolation Forest"],
                help="Choose anomaly detection algorithm"
            )
        
        with col2:
            if anomaly_mode == "Z-score":
                z_thresh = st.slider("Z-score threshold", 2.0, 6.0, 3.0, 0.1)
        
        st.markdown("---")
        
        if resample_freq is None:
            keep_idx = thin_index(df_filtered.index, downsample_limit)
            df_anomaly = df_filtered.iloc[keep_idx]
        else:
            df_anomaly = df_filtered
        
        anomaly_found = False
        
        for col in selected_params_anomaly:
            s = pd.to_numeric(df_anomaly[col], errors="coerce").dropna()
            if s.empty:
                continue
            
            if anomaly_mode == "Z-score":
                z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
                mask_anom = z.abs() >= z_thresh
            else:
                if IsolationForest is None:
                    st.error("❌ Isolation Forest requires scikit-learn")
                    break
                iso = IsolationForest(contamination="auto", random_state=42)
                X = s.values.reshape(-1, 1)
                pred = iso.fit_predict(X)
                mask_anom = pred == -1
            
            if mask_anom.any():
                anomaly_found = True
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        mode="lines",
                        name=col,
                        line=dict(color="steelblue", width=2)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=s.index[mask_anom],
                        y=s.values[mask_anom],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(color="red", size=10, symbol="x")
                    )
                )
                
                fig.update_layout(
                    template=TEMPLATE,
                    title=f"Anomaly Detection: {col}",
                    height=400,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, width="stretch")
                
                num_anomalies = mask_anom.sum()
                pct_anomalies = (num_anomalies / len(s)) * 100
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Anomalies Detected", f"{num_anomalies:,}")
                col_b.metric("Percentage", f"{pct_anomalies:.2f}%")
                col_c.metric("Method", anomaly_mode)
                
                st.markdown("---")
            else:
                st.success(f"✅ No anomalies detected in **{col}** using {anomaly_mode}")
        
        if not anomaly_found and anomaly_mode != "Isolation Forest":
            st.info("💡 Try adjusting the threshold or using a different detection method")

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

if use_dark:
    st.markdown('</div>', unsafe_allow_html=True)

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
            
            **Built by - Jeevan A. Jadhav**
            """
        )
