"""
Streamlit Web Application
=========================
Interactive fraud detection dashboard with API integration, SHAP explanations,
threshold tuning, and model performance visualization.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import FEATURE_COLS, SCALE_COLS, MODELS_DIR, FIGURES_DIR, FRONTEND_API_URL, FRONTEND_FALLBACK
from src.models import load_metrics


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* ===== GLOBAL TEXT VISIBILITY FIX ===== */
    .stApp, .stApp * { color: #eaeaea; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #f0f0f5 !important; }
    .stApp p, .stApp span, .stApp li, .stApp label, .stApp div { color: #e0e0e8; }
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li { color: #e0e0e8 !important; }
    .stApp .stCaption, .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stCaptionContainer"] span { color: #a0a8c0 !important; }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); border-right: 1px solid #e94560; }
    section[data-testid="stSidebar"] * { color: #dcdce5 !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #f5f5fa !important; }
    section[data-testid="stSidebar"] a { color: #64b5f6 !important; text-decoration: underline; }

    /* ===== FORM LABELS ===== */
    .stApp label, .stApp .stFormLabel,
    .stApp [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] span,
    .stApp [data-testid="stWidgetLabel"] { color: #e0e0e8 !important; }

    /* ===== SELECT / DROPDOWN ===== */
    .stSelectbox label, .stSelectbox span { color: #e0e0e8 !important; }
    .stSelectbox [data-baseweb="select"] { background: #16213e; border-color: #0f3460; }
    .stSelectbox [data-baseweb="select"] * { color: #eaeaea !important; }

    /* ===== NUMBER INPUT ===== */
    .stNumberInput label, .stNumberInput span { color: #e0e0e8 !important; }
    .stNumberInput > div > div > input { background: #16213e !important; border: 1px solid #0f3460 !important; color: #eaeaea !important; }

    /* ===== FILE UPLOADER ===== */
    .stFileUploader label, .stFileUploader span, .stFileUploader p { color: #e0e0e8 !important; }
    [data-testid="stFileUploaderDropzone"] { background: #16213e !important; border-color: #0f3460 !important; }
    [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] p { color: #e0e0e8 !important; }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader, .streamlit-expanderHeader p,
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary span { color: #e0e0e8 !important; }
    [data-testid="stExpander"] { border-color: #0f3460 !important; }

    /* ===== METRIC WIDGETS ===== */
    [data-testid="stMetric"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"], [data-testid="stMetric"] label,
    [data-testid="stMetric"] p, [data-testid="stMetric"] div { color: #eaeaea !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #b0b8d0 !important; }

    /* ===== CARDS ===== */
    .metric-card { background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%); border: 1px solid #0f3460; border-radius: 16px; padding: 24px; margin: 8px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .metric-card p, .metric-card strong, .metric-card span { color: #e0e0e8 !important; }

    .fraud-card { background: linear-gradient(135deg, #e74c3c22 0%, #c0392b22 100%); border: 2px solid #e74c3c; border-radius: 16px; padding: 32px; text-align: center; animation: pulse-red 2s infinite; }
    .legit-card { background: linear-gradient(135deg, #2ecc7122 0%, #27ae6022 100%); border: 2px solid #2ecc71; border-radius: 16px; padding: 32px; text-align: center; animation: pulse-green 2s infinite; }

    @keyframes pulse-red { 0%, 100% { box-shadow: 0 0 20px rgba(231,76,60,0.3); } 50% { box-shadow: 0 0 40px rgba(231,76,60,0.6); } }
    @keyframes pulse-green { 0%, 100% { box-shadow: 0 0 20px rgba(46,204,113,0.3); } 50% { box-shadow: 0 0 40px rgba(46,204,113,0.6); } }

    /* ===== HEADER ===== */
    .main-header { text-align: center; padding: 20px 0; background: linear-gradient(90deg, #e94560, #0f3460); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 900; letter-spacing: 2px; }
    .sub-header { text-align: center; color: #c0c0d0 !important; font-size: 1.1rem; margin-bottom: 30px; }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: #16213e; border-radius: 8px; border: 1px solid #0f3460; color: #eaeaea !important; padding: 10px 20px; }
    .stTabs [data-baseweb="tab"] * { color: #eaeaea !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #e94560, #0f3460) !important; border: none; }
    .stTabs [aria-selected="true"] * { color: #ffffff !important; }
    .stTabs [data-baseweb="tab-panel"] * { color: #e0e0e8; }
    .stTabs [data-baseweb="tab-panel"] h1, .stTabs [data-baseweb="tab-panel"] h2,
    .stTabs [data-baseweb="tab-panel"] h3, .stTabs [data-baseweb="tab-panel"] h4 { color: #f0f0f5 !important; }

    /* ===== BUTTONS ===== */
    .stButton > button { background: linear-gradient(135deg, #e94560, #c0392b) !important; color: white !important; border: none; border-radius: 8px; padding: 10px 24px; font-weight: bold; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(233,69,96,0.4); }
    .stButton > button span { color: white !important; }
    .stFormSubmitButton > button { background: linear-gradient(135deg, #e94560, #c0392b) !important; color: white !important; }
    .stFormSubmitButton > button span { color: white !important; }
    .stDownloadButton > button { background: linear-gradient(135deg, #3498db, #2980b9) !important; color: white !important; }
    .stDownloadButton > button span { color: white !important; }

    /* ===== MISC ===== */
    .stAlert p, [data-testid="stAlert"] p { color: #e0e0e8 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #0f3460; border-radius: 8px; }
    .stApp a { color: #64b5f6 !important; }
    [data-testid="stForm"] { background: rgba(22,33,62,0.4); border: 1px solid #0f3460; border-radius: 12px; padding: 20px; }
    .stSlider label, .stSlider span, .stSlider p { color: #e0e0e8 !important; }

    /* ===== API STATUS ===== */
    .api-status-online { background: #2ecc7133; border: 1px solid #2ecc71; border-radius: 8px; padding: 8px 12px; text-align: center; }
    .api-status-offline { background: #e74c3c33; border: 1px solid #e74c3c; border-radius: 8px; padding: 8px 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

def check_api_connection():
    """Check if FastAPI backend is available."""
    try:
        import httpx
        r = httpx.get(f"{FRONTEND_API_URL}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_resource
def load_predictor(model_name):
    """Load and cache the fraud predictor (local fallback)."""
    try:
        from src.predictor import FraudPredictor
        return FraudPredictor(model_name=model_name)
    except FileNotFoundError:
        return None


def predict_via_api(transaction_dict, model_name=None, threshold=None):
    """Make prediction via FastAPI backend. Returns None on failure."""
    import httpx
    params = {}
    if model_name:
        params["model"] = model_name
    if threshold is not None:
        params["threshold"] = threshold
    try:
        r = httpx.post(f"{FRONTEND_API_URL}/predict", json=transaction_dict, params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None  # Caller will fall back to local prediction


def explain_via_api(transaction_dict, model_name=None):
    """Get SHAP explanation via FastAPI backend."""
    import httpx
    params = {}
    if model_name:
        params["model"] = model_name
    r = httpx.post(f"{FRONTEND_API_URL}/predict/explain", json=transaction_dict, params=params, timeout=30.0)
    r.raise_for_status()
    return r.json()


def render_prediction_result(result):
    """Render the prediction result with visual styling."""
    is_fraud = result.get("is_fraud", False)
    confidence = result.get("confidence", 0)
    fraud_prob = result.get("fraud_probability", 0)
    threshold = result.get("threshold_used", 0.5)

    if is_fraud:
        st.markdown(f"""
        <div class="fraud-card">
            <h1 style="color: #e74c3c; font-size: 3rem; margin: 0;">🚨 FRAUD DETECTED</h1>
            <p style="color: #eaeaea; font-size: 1.3rem; margin-top: 15px;">
                Confidence: <strong style="color: #e74c3c;">{confidence:.1%}</strong>
            </p>
            <p style="color: #a0a0b0; font-size: 0.95rem;">
                Fraud Probability: {fraud_prob:.4f} | Threshold: {threshold:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="legit-card">
            <h1 style="color: #2ecc71; font-size: 3rem; margin: 0;">✅ LEGITIMATE</h1>
            <p style="color: #eaeaea; font-size: 1.3rem; margin-top: 15px;">
                Confidence: <strong style="color: #2ecc71;">{confidence:.1%}</strong>
            </p>
            <p style="color: #a0a0b0; font-size: 0.95rem;">
                Fraud Probability: {fraud_prob:.4f} | Threshold: {threshold:.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Legitimate Prob.", f"{result.get('legitimate_probability', 1-fraud_prob):.4f}")
    col2.metric("Fraud Prob.", f"{fraud_prob:.4f}")
    col3.metric("Confidence", f"{confidence:.1%}")


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # API status
    api_online = check_api_connection()
    if api_online:
        st.markdown('<div class="api-status-online">🟢 API Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status-offline">🔴 API Offline (Local Mode)</div>', unsafe_allow_html=True)

    st.markdown("")

    # Model selection
    model_choice = st.selectbox(
        "🤖 Select Model",
        options=["xgboost", "random_forest"],
        format_func=lambda x: "XGBoost" if x == "xgboost" else "Random Forest",
        index=0,
    )

    # Load predictor (local fallback)
    predictor = load_predictor(model_choice)
    if predictor is None and not api_online:
        st.error("❌ Model not found! Run `python train.py` first.")
        st.stop()

    # Model info
    if predictor:
        st.markdown("### 📊 Model Info")
        info = predictor.get_model_info()
        st.markdown(f"""
        <div class="metric-card">
            <p><strong>Model:</strong> {info['model_type']}</p>
            <p><strong>Features:</strong> {info['n_features']}</p>
            <p><strong>Threshold:</strong> {info['optimal_threshold']:.4f}</p>
            <p><strong>Scaler:</strong> {'✅ Loaded' if info['has_scaler'] else '❌ None'}</p>
        </div>
        """, unsafe_allow_html=True)

        if info.get("training_metrics"):
            st.markdown("### 🏆 Training Metrics")
            m = info["training_metrics"]
            for metric_name, value in m.items():
                if isinstance(value, float):
                    st.metric(metric_name.replace("_", " ").title(), f"{value:.4f}")

    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit + FastAPI\n\n"
        "[View on GitHub](https://github.com) • "
        "[Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)"
    )


# =============================================================================
# Main Content
# =============================================================================

st.markdown('<h1 class="main-header">🔍 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time transaction analysis powered by machine learning</p>',
            unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Single Prediction",
    "📋 Batch Prediction",
    "🧪 Quick Test",
    "🔬 SHAP Explainability",
    "🎚️ Threshold Tuning",
    "📊 Model Dashboard"
])


# =============================================================================
# Tab 1: Single Prediction
# =============================================================================

with tab1:
    st.markdown("### Enter Transaction Details")
    st.markdown("Input the transaction features below to get a real-time fraud prediction.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            time_val = st.number_input("⏰ Time (seconds since first transaction)",
                                        min_value=0.0, max_value=200000.0, value=0.0, step=100.0)
            amount_val = st.number_input("💰 Amount ($)",
                                          min_value=0.0, max_value=30000.0, value=100.0, step=10.0)

        with col2:
            st.markdown("**PCA Features (V1-V28)**")
            st.caption("These are the PCA-transformed features from the original data.")

        with st.expander("📐 PCA Features (V1-V28)", expanded=False):
            v_values = {}
            cols = st.columns(4)
            for i in range(1, 29):
                col_idx = (i - 1) % 4
                with cols[col_idx]:
                    v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1,
                                                         format="%.4f", key=f"v{i}_input")

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        transaction = {"Time": time_val, "Amount": amount_val}
        transaction.update(v_values)

        with st.spinner("Analyzing transaction..."):
            if api_online:
                result = predict_via_api(transaction, model_name=model_choice)
                if result is None and predictor is not None:
                    result = predictor.predict_single(transaction)
            else:
                result = predictor.predict_single(transaction) if predictor is not None else None

        st.markdown("---")
        render_prediction_result(result)


# =============================================================================
# Tab 2: Batch Prediction
# =============================================================================

with tab2:
    st.markdown("### Upload CSV for Batch Prediction")
    st.markdown("Upload a CSV file with transaction data to get predictions for all rows.")
    st.caption(f"Required columns: {', '.join(FEATURE_COLS[:5])}... (all {len(FEATURE_COLS)} features)")

    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"],
                                      help="Upload a CSV or JSON file containing transaction features.")

    if uploaded_file is not None:
        try:
            # Read raw content to determine format
            raw_data = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Try to decode as string
            try:
                decoded = raw_data.decode("utf-8").strip()
            except Exception:
                decoded = ""
                
            # Parse based on content type
            if decoded.startswith("{") or decoded.startswith("["):
                import json
                parsed_json = json.loads(decoded)
                if isinstance(parsed_json, dict):
                    df = pd.DataFrame([parsed_json])
                else:
                    df = pd.DataFrame(parsed_json)
            else:
                df = pd.read_csv(uploaded_file)
                
            st.markdown(f"**Loaded:** {len(df):,} transactions")

            missing = set(FEATURE_COLS) - set(df.columns)
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                if st.button("🔍 Run Batch Prediction", key="batch_predict"):
                    with st.spinner(f"Processing {len(df):,} transactions..."):
                        if predictor:
                            results = predictor.predict_batch(df)
                        else:
                            st.error("Model not available for batch prediction")
                            results = None

                    if results is not None:
                        n_fraud = results["Is_Fraud"].sum()
                        n_legit = len(results) - n_fraud

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Transactions", f"{len(results):,}")
                        col2.metric("🟢 Legitimate", f"{n_legit:,}")
                        col3.metric("🔴 Fraudulent", f"{n_fraud:,}")

                        st.markdown("### Results")
                        display_cols = ["Amount", "Prediction", "Fraud_Probability", "Confidence"]
                        available_cols = [c for c in display_cols if c in results.columns]
                        st.dataframe(results[available_cols], use_container_width=True, height=400)

                        csv = results.to_csv(index=False)
                        st.download_button("📥 Download Results CSV", csv,
                                            "fraud_predictions.csv", "text/csv",
                                            use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


# =============================================================================
# Tab 3: Quick Test
# =============================================================================

with tab3:
    st.markdown("### Quick Test with Sample Transactions")
    st.markdown("Test the model with pre-built sample transactions.")

    from src.predictor import get_sample_transactions
    samples = get_sample_transactions()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Legitimate Transaction Example")
        st.json({k: round(v, 3) for k, v in list(samples["legitimate"].items())[:6]})
        st.caption(f"Amount: ${samples['legitimate']['Amount']:.2f}")

        if st.button("Test Legitimate Example", key="test_legit", use_container_width=True):
            if api_online:
                result = predict_via_api(samples["legitimate"], model_name=model_choice)
                if result is None and predictor is not None:
                    result = predictor.predict_single(samples["legitimate"])
            else:
                result = predictor.predict_single(samples["legitimate"]) if predictor is not None else None
            render_prediction_result(result)

    with col2:
        st.markdown("#### 🔴 Fraud Transaction Example")
        st.json({k: round(v, 3) for k, v in list(samples["fraud"].items())[:6]})
        st.caption(f"Amount: ${samples['fraud']['Amount']:.2f}")

        if st.button("Test Fraud Example", key="test_fraud", use_container_width=True):
            if api_online:
                result = predict_via_api(samples["fraud"], model_name=model_choice)
                if result is None and predictor is not None:
                    result = predictor.predict_single(samples["fraud"])
            else:
                result = predictor.predict_single(samples["fraud"]) if predictor is not None else None
            render_prediction_result(result)


# =============================================================================
# Tab 4: SHAP Explainability
# =============================================================================

with tab4:
    st.markdown("### 🔬 SHAP Explainability")
    st.markdown("Understand which features drive the model's predictions using SHAP values.")

    # Global SHAP plots
    st.markdown("#### Global Feature Importance")
    shap_files = {
        "SHAP Summary (XGBoost)": "shap_summary_xgboost.png",
        "SHAP Summary (Random Forest)": "shap_summary_random_forest.png",
        "SHAP Importance (XGBoost)": "shap_importance_xgboost.png",
        "SHAP Importance (Random Forest)": "shap_importance_random_forest.png",
    }

    available_shap = {}
    for name, filename in shap_files.items():
        path = FIGURES_DIR / filename
        if path.exists():
            available_shap[name] = path

    if available_shap:
        selected_shap = st.selectbox("Select SHAP Visualization", list(available_shap.keys()))
        if selected_shap:
            st.image(str(available_shap[selected_shap]), caption=selected_shap,
                     use_container_width=True)
    else:
        st.info("📊 No SHAP plots found. Run `python train.py` to generate SHAP analysis.")

    # Per-prediction explanation
    st.markdown("---")
    st.markdown("#### Explain a Single Prediction")
    st.markdown("Enter a transaction to see which features contributed most to the prediction.")

    with st.form("shap_explain_form"):
        shap_col1, shap_col2 = st.columns(2)
        with shap_col1:
            shap_time = st.number_input("⏰ Time", min_value=0.0, value=406.0, key="shap_time")
            shap_amount = st.number_input("💰 Amount", min_value=0.0, value=44.50, key="shap_amount")
        with shap_col2:
            use_sample = st.selectbox("Or use a sample:", ["Custom", "Legitimate Sample", "Fraud Sample"],
                                       key="shap_sample")

        shap_submitted = st.form_submit_button("🔍 Explain Prediction", use_container_width=True)

    if shap_submitted:
        if use_sample == "Legitimate Sample":
            explain_transaction = samples["legitimate"]
        elif use_sample == "Fraud Sample":
            explain_transaction = samples["fraud"]
        else:
            explain_transaction = {"Time": shap_time, "Amount": shap_amount}
            for i in range(1, 29):
                explain_transaction[f"V{i}"] = 0.0

        with st.spinner("Computing SHAP explanation..."):
            try:
                if api_online:
                    explanation = explain_via_api(explain_transaction, model_name=model_choice)
                    pred = explanation["prediction"]
                    render_prediction_result(pred)

                    st.markdown("#### Top Contributing Features")
                    top_features = explanation["top_features"]
                    if top_features:
                        feat_df = pd.DataFrame(top_features)
                        feat_df = feat_df.sort_values("abs_shap_value", ascending=True)

                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in feat_df["shap_value"]]
                        ax.barh(feat_df["feature"], feat_df["shap_value"], color=colors)
                        ax.set_xlabel("SHAP Value (impact on fraud probability)")
                        ax.set_title("Feature Contributions to Prediction")
                        ax.axvline(x=0, color="white", linewidth=0.8)
                        fig.patch.set_facecolor("#1a1a2e")
                        ax.set_facecolor("#16213e")
                        ax.tick_params(colors="#eaeaea")
                        ax.xaxis.label.set_color("#eaeaea")
                        ax.title.set_color("#eaeaea")
                        st.pyplot(fig)
                else:
                    # Local fallback
                    from src.explainability import FraudExplainer
                    explainer = FraudExplainer(predictor.model, model_name=model_choice)
                    explanation = explainer.explain_single(explain_transaction)
                    result = predictor.predict_single(explain_transaction)
                    render_prediction_result(result)

                    st.markdown("#### Top Contributing Features")
                    feat_df = pd.DataFrame(explanation["top_features"])
                    if len(feat_df) > 0:
                        feat_df = feat_df.sort_values("abs_shap_value", ascending=True)

                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in feat_df["shap_value"]]
                        ax.barh(feat_df["feature"], feat_df["shap_value"], color=colors)
                        ax.set_xlabel("SHAP Value")
                        ax.set_title("Feature Contributions")
                        ax.axvline(x=0, color="white", linewidth=0.8)
                        fig.patch.set_facecolor("#1a1a2e")
                        ax.set_facecolor("#16213e")
                        ax.tick_params(colors="#eaeaea")
                        ax.xaxis.label.set_color("#eaeaea")
                        ax.title.set_color("#eaeaea")
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")
                st.info("Make sure SHAP is installed: `pip install shap`")


# =============================================================================
# Tab 5: Threshold Tuning
# =============================================================================

with tab5:
    st.markdown("### 🎚️ Interactive Threshold Tuning")
    st.markdown("Adjust the decision threshold and see how it affects precision, recall, and F1-score in real-time.")

    threshold_val = st.slider("Decision Threshold", min_value=0.01, max_value=0.99,
                               value=predictor.optimal_threshold if predictor else 0.5,
                               step=0.01, key="threshold_slider")

    # Show threshold plot if available
    threshold_plot = FIGURES_DIR / "threshold_optimization.png"
    if threshold_plot.exists():
        st.image(str(threshold_plot), caption="F1-Score vs Decision Threshold",
                 use_container_width=True)

    # Show saved threshold strategies
    if predictor:
        from src.threshold import ThresholdOptimizer
        thresh_data = ThresholdOptimizer.load(predictor.model_name)
        if thresh_data:
            st.markdown("#### Saved Threshold Strategies")
            strategies_df = pd.DataFrame([
                {"Strategy": k, "Threshold": v}
                for k, v in thresh_data.get("all_results", {}).items()
            ])
            if len(strategies_df) > 0:
                st.dataframe(strategies_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Test with Custom Threshold")
    st.markdown(f"Current threshold: **{threshold_val:.3f}** | "
                f"Model default: **{predictor.optimal_threshold:.3f}**" if predictor else "")

    if st.button("🧪 Test Threshold on Sample Data", key="test_threshold"):
        test_transactions = [samples["legitimate"], samples["fraud"]]
        for i, tx in enumerate(test_transactions):
            tx_type = "Legitimate" if i == 0 else "Fraud"
            st.markdown(f"##### {tx_type} Sample (threshold={threshold_val:.3f})")
            if predictor:
                result = predictor.predict_single(tx, threshold=threshold_val)
                render_prediction_result(result)


# =============================================================================
# Tab 6: Model Dashboard
# =============================================================================

with tab6:
    st.markdown("### Model Performance Dashboard")

    figure_files = {
        "Confusion Matrices": "confusion_matrices.png",
        "ROC Curves": "roc_curves.png",
        "Precision-Recall Curves": "precision_recall_curves.png",
        "Metrics Comparison": "metrics_comparison.png",
        "Feature Importance (Random Forest)": "feature_importance_random_forest.png",
        "Feature Importance (XGBoost)": "feature_importance_xgboost.png",
        "Class Distribution": "class_distribution.png",
        "Amount Distribution": "amount_distribution.png",
        "Threshold Optimization": "threshold_optimization.png",
    }

    available_figures = {}
    for name, filename in figure_files.items():
        path = FIGURES_DIR / filename
        if path.exists():
            available_figures[name] = path

    if available_figures:
        selected_figure = st.selectbox("📊 Select Visualization",
                                        options=list(available_figures.keys()))
        if selected_figure:
            st.image(str(available_figures[selected_figure]), caption=selected_figure,
                     use_container_width=True)
    else:
        st.info("📊 No training figures found. Run `python train.py` to generate visualizations.")

    st.markdown("### 📈 Metrics Summary")
    metrics_data = load_metrics()
    if metrics_data:
        metrics_df = pd.DataFrame([metrics_data])
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Run `python train.py` to generate training metrics.")
