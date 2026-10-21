import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import BUSINESS_CONFIG, DATA_CONFIG, MODEL_CONFIG
from src.data_loader import BankMarketingDataLoader
from src.features import BehavioralFeatureEngineer, CampaignOptimaFeaturePipeline
from src.optimization import ThresholdOptimizer, CampaignBudgetOptimizer
from src.explainability import ModelExplainabilityEngine
import joblib

# Page Configuration
st.set_page_config(
    page_title="CampaignOptima | Marketing Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS - Softer Warm Ivory & Truffle Theme
st.markdown("""
<style>
    /* Global Page Body - Softer Warm Cream/Ivory */
    .stApp {
        background-color: #FBF8E6 !important;
        color: #38240D !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Headers & Text */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38240D !important;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #713600 !important;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #38240D !important;
    }

    /* Metric Cards - Optimized for 5-Column Row Alignment */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2D8B3 !important;
        border-radius: 10px !important;
        padding: 10px 8px !important;
        box-shadow: 0 2px 5px rgba(56, 36, 13, 0.05) !important;
        min-height: 105px !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #713600 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #38240D !important;
        font-weight: 800 !important;
        font-size: 1.25rem !important;
        white-space: nowrap !important;
    }

    /* Input Controls & Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #38240D !important;
        border: 1px solid #E2D8B3 !important;
    }
    input {
        background-color: #FFFFFF !important;
        color: #38240D !important;
        border: 1px solid #E2D8B3 !important;
        caret-color: transparent !important;
    }
    
    /* FIX: Remove selection highlight line '|' / box artifacts inside inputs */
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    input:focus, div[data-baseweb="select"] > div:focus-within {
        border-color: #C05800 !important;
        box-shadow: none !important;
    }

    /* Dropdown Arrow Styling */
    div[data-baseweb="select"] svg {
        fill: #38240D !important;
        color: #38240D !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* Number Input (+ / -) Buttons */
    div[data-testid="stNumberInput"] button {
        background-color: #F0E8C8 !important;
        color: #38240D !important;
        border: 1px solid #E2D8B3 !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background-color: #C05800 !important;
        color: #FFFFFF !important;
    }
    
    /* ==========================================================
       ROBUST DROPDOWN MENU FIX (BaseWeb Popover & Portal Layers)
       ========================================================== */
    div[data-baseweb="popover"], 
    div[data-baseweb="menu"], 
    ul[role="listbox"], 
    div[data-testid="stVirtualDropdown"] {
        background-color: #FFFDF5 !important;
        background: #FFFDF5 !important;
        border: 1px solid #E2D8B3 !important;
        box-shadow: 0 4px 12px rgba(56, 36, 13, 0.15) !important;
    }
    
    li[role="option"], 
    div[role="option"],
    ul[role="listbox"] li {
        background-color: #FFFDF5 !important;
        color: #38240D !important;
    }
    
    li[role="option"]:hover, 
    div[role="option"]:hover {
        background-color: #E2D8B3 !important;
        color: #38240D !important;
    }
    
    div[data-baseweb="popover"] * {
        color: #38240D !important;
    }

    div[data-testid="stForm"] {
        background-color: #FFFDF5 !important;
        border: 1px solid #E2D8B3 !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* High-Contrast Benchmark Scorecard Custom Table */
    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #FFFFFF;
        color: #38240D;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2D8B3;
        font-size: 0.95rem;
    }
    .benchmark-table th {
        background-color: #38240D;
        color: #FFFFFF !important;
        padding: 12px 14px;
        text-align: left;
        font-weight: 700;
    }
    .benchmark-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #E2D8B3;
        color: #38240D !important;
    }
    .benchmark-table tr:hover {
        background-color: #FFFDF5;
    }
    .highlight-cell {
        background-color: #FFE0B2 !important;
        color: #38240D !important;
        font-weight: 800 !important;
        border-radius: 4px;
    }

    /* Plotly Action Buttons Visibility Fix */
    .modebar-btn svg, [data-testid="stElementToolbar"] button {
        fill: #38240D !important;
        color: #38240D !important;
    }
    .modebar-container, [data-testid="stElementToolbar"] {
        background-color: rgba(226, 216, 179, 0.85) !important;
        border-radius: 6px !important;
        padding: 2px !important;
    }

    /* Decision Highlight Cards */
    .card-approved {
        background-color: #F0F7EE !important;
        border: 1px solid #B8E0B0 !important;
        border-radius: 10px !important;
        padding: 18px !important;
        color: #2E5A27 !important;
        margin-bottom: 15px !important;
    }
    .card-approved p, .card-approved h4, .card-approved b {
        color: #2E5A27 !important;
    }
    .card-declined {
        background-color: #FDF2F2 !important;
        border: 1px solid #F7C5C5 !important;
        border-radius: 10px !important;
        padding: 18px !important;
        color: #9E2A2B !important;
        margin-bottom: 15px !important;
    }
    .card-declined p, .card-declined h4, .card-declined b {
        color: #9E2A2B !important;
    }

    /* Buttons & Form Submit Buttons */
    .stButton > button, .stDownloadButton > button, div.stFormSubmitButton > button {
        background-color: #C05800 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, div.stFormSubmitButton > button:hover {
        background-color: #713600 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_bank_data():
    """Cached raw bank dataset loading."""
    loader = BankMarketingDataLoader()
    df = loader.load_raw_data()
    return df


@st.cache_resource
def load_model_and_pipeline():
    """Cached loading or rapid training of champion model and feature pipeline."""
    model_path = MODEL_CONFIG.model_artifact_path
    pipe_path = MODEL_CONFIG.pipeline_artifact_path

    if model_path.exists() and pipe_path.exists():
        model = joblib.load(model_path)
        pipeline = joblib.load(pipe_path)
        return model, pipeline

    df = load_bank_data()
    loader = BankMarketingDataLoader()
    X, y = loader.clean_and_prepare(df)
    X_train, X_test, y_train, y_test = loader.get_stratified_train_test_split(X, y)

    pipeline = CampaignOptimaFeaturePipeline()
    X_train_proc = pipeline.fit_transform(X_train)

    import xgboost as xgb
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos = float(neg_count) / float(max(pos_count, 1))

    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_proc, y_train)

    MODEL_CONFIG.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipe_path)

    return model, pipeline


def render_highlighted_benchmark_table(df):
    """Renders custom high-contrast HTML table with maximum value highlights."""
    max_pr = df["PR-AUC (Primary)"].max()
    max_roc = df["ROC-AUC"].max()
    max_f1 = df["F1-Score"].max()

    html = '<table class="benchmark-table"><thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'

    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            val = row[col]
            if isinstance(val, (float, int)):
                formatted_val = f"{val:.4f}"
                is_max = (
                    (col == "PR-AUC (Primary)" and val == max_pr) or
                    (col == "ROC-AUC" and val == max_roc) or
                    (col == "F1-Score" and val == max_f1)
                )
                if is_max:
                    html += f'<td class="highlight-cell">{formatted_val}</td>'
                else:
                    html += f'<td>{formatted_val}</td>'
            else:
                html += f'<td><b>{val}</b></td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html


# Title Section
st.markdown('<div class="main-title">CampaignOptima</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Driven Customer Conversion & Marketing Budget Optimization Engine</div>', unsafe_allow_html=True)

df_raw = load_bank_data()
model, pipeline = load_model_and_pipeline()

threshold_meta = {
    "optimal_threshold": 0.029,
    "cost_per_call": BUSINESS_CONFIG.cost_per_call,
    "conversion_revenue": BUSINESS_CONFIG.conversion_revenue,
    "profit_uplift_pct": 60.5
}
if MODEL_CONFIG.threshold_config_path.exists():
    try:
        with open(MODEL_CONFIG.threshold_config_path, "r") as f:
            threshold_meta.update(json.load(f))
    except Exception:
        pass

# Top Metric Cards (5 Columns Aligned)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_prospects = len(df_raw)
raw_conv_rate = (df_raw["y"] == "yes").mean() * 100

kpi1.metric("Total Prospects", f"{total_prospects:,}")
kpi2.metric("Base Conversion Rate", f"{raw_conv_rate:.1f}%")
kpi3.metric("Optimal Cutoff", f"{threshold_meta['optimal_threshold']:.3f}", delta=f"-{(0.5 - threshold_meta['optimal_threshold']):.3f} vs 0.50", delta_color="inverse")
kpi4.metric("Profit Uplift over Default", f"+{threshold_meta['profit_uplift_pct']:.1f}%", delta="Optimized")
kpi5.metric("Unit Economics", f"${int(threshold_meta['cost_per_call'])} /${int(threshold_meta['conversion_revenue'])}")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Executive KPI & Campaign Analytics",
    "🎯 Real-Time Conversion Predictor & SHAP",
    "🧮 Operations Research & Budget Simulator"
])

# ==============================================================================
# TAB 1: EXECUTIVE KPI & CAMPAIGN ANALYTICS (Increased Graph Spacing)
# ==============================================================================
with tab1:
    st.subheader("Campaign Behavioral Intelligence & Cohort Conversion Dynamics")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        job_conv = df_raw.groupby("job")["y"].apply(lambda s: (s == "yes").mean() * 100).reset_index()
        job_counts = df_raw["job"].value_counts().reset_index()
        job_counts.columns = ["job", "count"]
        job_conv = job_conv.merge(job_counts, on="job").sort_values(by="y", ascending=True)

        fig_job = px.bar(
            job_conv,
            x="y",
            y="job",
            orientation="h",
            color="y",
            color_continuous_scale=["#DFCF93", "#C05800", "#713600"],
            title="Conversion Rate by Profession / Job Category (%)",
            labels={"y": "Conversion Rate (%)", "job": "Job Category", "count": "Prospects"},
            hover_data=["count"]
        )
        fig_job.add_vline(x=raw_conv_rate, line_dash="dash", line_color="#9E2A2B", annotation_text="Avg Conv Rate")
        fig_job.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#38240D", size=12),
            title=dict(font=dict(color="#38240D", size=16)),
            xaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            yaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            legend=dict(font=dict(color="#38240D")),
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_job, use_container_width=True)

    with col_a2:
        df_raw["balance_tier"] = pd.qcut(
            df_raw["balance"],
            q=5,
            labels=["1. Deficit (<$50)", "2. Modest ($50-$400)", "3. Average ($400-$1k)", "4. High ($1k-$2.5k)", "5. Affluent (>$2.5k)"]
        )
        tier_conv = df_raw.groupby("balance_tier", observed=False)["y"].apply(lambda s: (s == "yes").mean() * 100).reset_index()

        fig_tier = px.bar(
            tier_conv,
            x="balance_tier",
            y="y",
            color="y",
            color_continuous_scale=["#E0D197", "#713600"],
            title="Conversion Rate Across Account Balance Tiers (%)",
            labels={"y": "Conversion Rate (%)", "balance_tier": "Balance Tier"}
        )
        fig_tier.add_hline(y=raw_conv_rate, line_dash="dash", line_color="#9E2A2B")
        fig_tier.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#38240D", size=12),
            title=dict(font=dict(color="#38240D", size=16)),
            xaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            yaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            legend=dict(font=dict(color="#38240D")),
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    # Increased vertical spacing between graph rows
    st.markdown("<br><br>", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month_df = df_raw.groupby("month")["y"].agg(
            conv_rate=lambda s: (s == "yes").mean() * 100,
            volume="count"
        ).reindex(month_order).dropna().reset_index()

        fig_month = make_subplots(specs=[[{"secondary_y": True}]])
        fig_month.add_trace(
            go.Bar(x=month_df["month"], y=month_df["volume"], name="Call Volume", marker_color="#DECF9A"),
            secondary_y=False
        )
        fig_month.add_trace(
            go.Scatter(x=month_df["month"], y=month_df["conv_rate"], name="Conversion Rate %", line=dict(color="#C05800", width=3)),
            secondary_y=True
        )
        fig_month.update_layout(
            title_text="Campaign Seasonality: Call Volume vs Conversion Efficiency",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#38240D", size=12),
            title=dict(font=dict(color="#38240D", size=16)),
            margin=dict(l=20, r=20, t=50, b=20),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1.05, font=dict(color="#38240D"))
        )
        fig_month.update_xaxes(tickfont=dict(color="#38240D"), title_font=dict(color="#38240D"))
        fig_month.update_yaxes(title_text="Total Contacts Placed", secondary_y=False, showgrid=False, title_font=dict(color="#38240D"), tickfont=dict(color="#38240D"))
        fig_month.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True, showgrid=True, gridcolor="#E3D292", title_font=dict(color="#38240D"), tickfont=dict(color="#38240D"))
        st.plotly_chart(fig_month, use_container_width=True)

    with col_b2:
        burn_df = df_raw.copy()
        burn_df["campaign_capped"] = np.clip(burn_df["campaign"], 1, 10)
        burn_stat = burn_df.groupby("campaign_capped")["y"].agg(
            conv_rate=lambda s: (s == "yes").mean() * 100,
            count="count"
        ).reset_index()

        fig_burn = px.line(
            burn_stat,
            x="campaign_capped",
            y="conv_rate",
            markers=True,
            title="Telemarketing Contact Fatigue: Conversion Decay vs Repeated Calls",
            labels={"campaign_capped": "Contacts Placed During Campaign", "conv_rate": "Conversion Rate (%)"}
        )
        fig_burn.update_traces(line_color="#9E2A2B", line_width=3, marker=dict(size=8))
        fig_burn.add_vrect(x0=3.5, x1=10, fillcolor="#9E2A2B", opacity=0.08, line_width=0, annotation_text="Burnout Zone (Diminishing Return)")
        fig_burn.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#38240D", size=12),
            title=dict(font=dict(color="#38240D", size=16)),
            xaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            yaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            legend=dict(font=dict(color="#38240D")),
            margin=dict(l=20, r=20, t=50, b=20),
            height=360
        )
        st.plotly_chart(fig_burn, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏆 Model Performance Benchmark Scorecard (Targeting PR-AUC)")
    benchmark_file = MODEL_CONFIG.model_artifact_path.parent / "benchmark_report.json"
    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            bench_dict = json.load(f)
        bench_df = pd.DataFrame(bench_dict).T.reset_index().rename(columns={"index": "Algorithm"})
        bench_df = bench_df[["Algorithm", "pr_auc", "roc_auc", "f1_score", "precision", "recall", "brier_score"]]
        bench_df.columns = ["Model Architecture", "PR-AUC (Primary)", "ROC-AUC", "F1-Score", "Precision", "Recall", "Brier Loss"]
        
        st.markdown(render_highlighted_benchmark_table(bench_df), unsafe_allow_html=True)


# ==============================================================================
# TAB 2: REAL-TIME CONVERSION PREDICTOR & SHAP EXPLAINABILITY (Gauge Layout Fix)
# ==============================================================================
with tab2:
    st.subheader("Customer Intelligence Scoring & Local SHAP Attribution")
    st.markdown("Adjust individual customer demographic and behavioral attributes to calculate **Real-Time Propensity** and view feature impact.")

    with st.form("customer_intake_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### Demographics")
            in_age = st.slider("Customer Age", min_value=18, max_value=90, value=38)
            in_job = st.selectbox("Occupation", options=[
                "management", "technician", "blue-collar", "admin.", "services",
                "retired", "self-employed", "student", "unemployed", "entrepreneur", "housemaid"
            ], index=0)
            in_marital = st.selectbox("Marital Status", ["married", "single", "divorced"], index=0)
            in_education = st.selectbox("Education Level", ["tertiary", "secondary", "primary", "unknown"], index=0)

        with col2:
            st.markdown("### Financial Liquidity")
            in_balance = st.number_input("Account Balance ($)", min_value=-2000, max_value=100000, value=3500, step=250)
            in_housing = st.selectbox("Housing Loan?", ["no", "yes"], index=0)
            in_loan = st.selectbox("Personal Loan?", ["no", "yes"], index=0)
            in_default = st.selectbox("Credit Default Record?", ["no", "yes"], index=0)

        with col3:
            st.markdown("### Contact Details")
            in_duration = st.slider("Call Duration (seconds)", min_value=0, max_value=1800, value=360)
            in_contact = st.selectbox("Contact Medium", ["cellular", "telephone", "unknown"], index=0)
            in_month = st.selectbox("Contact Month", ["may", "jul", "aug", "jun", "nov", "apr", "feb", "jan", "oct", "sep", "mar", "dec"], index=10)
            in_day = st.slider("Day of Month", min_value=1, max_value=31, value=15)

        with col4:
            st.markdown("### Campaign History")
            in_campaign = st.number_input("Contacts in Current Campaign", min_value=1, max_value=30, value=1)
            in_pdays = st.number_input("Days Since Prior Campaign (-1 if never)", min_value=-1, max_value=850, value=90)
            in_previous = st.number_input("Prior Campaign Contacts", min_value=0, max_value=25, value=2)
            in_poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "other", "unknown"], index=0)

        predict_btn = st.form_submit_button("⚡ Evaluate Prospect Propensity", use_container_width=True)

    input_df = pd.DataFrame([{
        "age": in_age, "job": in_job, "marital": in_marital, "education": in_education,
        "default": in_default, "balance": in_balance, "housing": in_housing, "loan": in_loan,
        "contact": in_contact, "day": in_day, "month": in_month, "duration": in_duration,
        "campaign": in_campaign, "pdays": in_pdays, "previous": in_previous, "poutcome": in_poutcome
    }])

    input_proc = pipeline.transform(input_df)
    prob = float(model.predict_proba(input_proc)[0, 1])
    opt_thresh = float(threshold_meta["optimal_threshold"])
    expected_val = (prob * float(threshold_meta["conversion_revenue"])) - float(threshold_meta["cost_per_call"])

    st.markdown("<br>", unsafe_allow_html=True)
    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.markdown("#### Predicted Conversion Propensity")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Conversion Probability (%)", 'font': {'size': 18, 'color': '#38240D'}},
            delta={'reference': opt_thresh * 100, 'increasing': {'color': "#2E5A27"}, 'decreasing': {'color': "#9E2A2B"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#713600"},
                'bar': {'color': "#C05800"},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#DFCF96",
                'steps': [
                    {'range': [0, opt_thresh * 100], 'color': "#F7B9B9"},
                    {'range': [opt_thresh * 100, 50], 'color': "#FFEC9E"},
                    {'range': [50, 100], 'color': "#D0F9C5"}
                ],
                'threshold': {
                    'line': {'color': "#9E2A2B", 'width': 3},
                    'thickness': 0.75,
                    'value': opt_thresh * 100
                }
            }
        ))
        # FIX: Increased top margin and height to prevent title overlap with the gauge chart
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#38240D"),
            height=290,
            margin=dict(l=20, r=20, t=60, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with p_col2:
        st.markdown("#### Strategic Decision Recommendation")
        is_opt_target = prob >= opt_thresh
        is_default_target = prob >= 0.50

        if is_opt_target:
            st.markdown(f"""
                <div class="card-approved">
                    <h4 style="margin-top:0;">✅ CAMPAIGN CALL APPROVED</h4>
                    <p><b>Propensity:</b> {prob:.1%} (Exceeds optimal decision threshold {opt_thresh:.1%})</p>
                    <hr style="border-color: #B8E0B0;">
                    <p><b>Expected Value:</b> ({prob:.2f} × ${threshold_meta['conversion_revenue']:.0f}) - ${threshold_meta['cost_per_call']:.0f} = <b>+${expected_val:,.2f}</b></p>
                    <p><b>Default Threshold (0.50) Decision:</b> {"TARGETED" if is_default_target else "IGNORED (FALSE NEGATIVE)"}</p>
                    <p><b>CampaignOptima Threshold ({opt_thresh:.3f}) Decision:</b> TARGETED (HIGH PROFITABILITY)</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="card-declined">
                    <h4 style="margin-top:0;">❌ DO NOT CONTACT</h4>
                    <p><b>Propensity:</b> {prob:.1%} (Below optimal decision threshold {opt_thresh:.1%})</p>
                    <hr style="border-color: #F7C5C5;">
                    <p><b>Expected Value:</b> ({prob:.2f} × ${threshold_meta['conversion_revenue']:.0f}) - ${threshold_meta['cost_per_call']:.0f} = <b>${expected_val:,.2f}</b></p>
                    <p><b>Decision Strategy:</b> SKIPPED TO PRESERVE MARKETING MARGIN</p>
                </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: OPERATIONS RESEARCH & BUDGET SIMULATOR
# ==============================================================================
with tab3:
    st.subheader("Operations Research: Marketing Budget Allocation & Knapsack Optimization")
    st.markdown("""
    Move beyond static scoring: Marketers face hard budget constraints and telemarketing fatigue caps. 
    This engine applies **Linear Programming (`scipy.optimize.linprog`)** to solve the bounded Knapsack problem:
    """)

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.markdown("#### ⚙️ Resource Constraints")
        sim_budget = st.slider("Campaign Dollar Budget ($)", min_value=5000, max_value=100000, value=25000, step=5000)
        sim_cost_call = st.number_input("Cost per Contact Call ($)", min_value=5.0, max_value=50.0, value=15.0, step=1.0)
        sim_conv_revenue = st.number_input("Term Deposit Value ($)", min_value=100.0, max_value=1000.0, value=300.0, step=25.0)
        sim_method = st.selectbox("Optimization Algorithm", ["Linear Programming (HiGHS)", "Greedy Marginal Knapsack"])
        apply_burnout_cap = st.checkbox("Enforce Telemarketing Contact Ceiling (Limit calls/prospect)", value=True)

        run_sim_btn = st.button("Run Campaign Optimization Engine", use_container_width=True)

    with sim_col2:
        X_raw, _ = BankMarketingDataLoader().clean_and_prepare(df_raw)
        X_raw_proc = pipeline.transform(X_raw)
        all_probs = model.predict_proba(X_raw_proc)[:, 1]

        opt_engine = CampaignBudgetOptimizer(
            cost_per_call=sim_cost_call,
            conversion_revenue=sim_conv_revenue
        )

        max_cap = int(sim_budget / sim_cost_call * 0.95) if apply_burnout_cap else None

        if "Greedy" in sim_method:
            sim_res = opt_engine.solve_greedy_knapsack(df_raw, all_probs, sim_budget, max_call_capacity=max_cap)
        else:
            sim_res = opt_engine.solve_budget_allocation_lp(df_raw, all_probs, sim_budget, max_call_capacity=max_cap)

        st.markdown("#### 📈 Optimized Financial Outcomes")
        o_c1, o_c2, o_c3, o_c4 = st.columns(4)
        o_c1.metric("Targeted Contacts", f"{sim_res['total_contacts_selected']:,}", f"{sim_res['budget_utilization_percent']:.0f}% Budget Used")
        o_c2.metric("Expected Conversions", f"{sim_res['expected_conversions']:.0f}")
        o_c3.metric("Net Projected Profit", f"${sim_res['expected_net_profit']:,.0f}")
        o_c4.metric("Campaign ROI", f"{sim_res['expected_roi_percent']:.1f}%")

        # Visualizing Budget vs Profit Frontier
        budgets_test = np.linspace(5000, 75000, 15)
        frontier_profits = []
        frontier_conversions = []

        for b in budgets_test:
            tmp_res = opt_engine.solve_greedy_knapsack(df_raw, all_probs, b)
            frontier_profits.append(tmp_res["expected_net_profit"])
            frontier_conversions.append(tmp_res["expected_conversions"])

        frontier_df = pd.DataFrame({
            "Budget": budgets_test,
            "Expected_Net_Profit": frontier_profits,
            "Conversions": frontier_conversions
        })

        fig_frontier = px.line(
            frontier_df,
            x="Budget",
            y="Expected_Net_Profit",
            markers=True,
            title="Efficient Marketing Frontier: Diminishing Marginal Returns across Budgets",
            labels={"Budget": "Allocated Budget ($)", "Expected_Net_Profit": "Net Expected Profit ($)"}
        )
        fig_frontier.add_vline(x=sim_budget, line_dash="dash", line_color="#9E2A2B", annotation_text="Current Budget")
        fig_frontier.update_traces(line_color="#C05800", line_width=3)
        fig_frontier.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#38240D", size=12),
            title=dict(font=dict(color="#38240D", size=16)),
            xaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            yaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
            legend=dict(font=dict(color="#38240D")),
            margin=dict(l=20, r=20, t=40, b=20),
            height=320
        )
        st.plotly_chart(fig_frontier, use_container_width=True)

    st.markdown("---")

    # Segment Allocation Breakdown
    st.markdown("#### 👥 Optimal Prospect Cohort Allocations")
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        if sim_res["job_targeting_breakdown"]:
            job_break_df = pd.DataFrame(list(sim_res["job_targeting_breakdown"].items()), columns=["Job", "Calls_Allocated"])
            fig_jbreak = px.bar(
                job_break_df.sort_values(by="Calls_Allocated", ascending=True),
                x="Calls_Allocated",
                y="Job",
                orientation="h",
                color="Calls_Allocated",
                color_continuous_scale=["#DFCF93", "#C05800", "#713600"],
                title="Call Volume Allocated by Professional Cohort"
            )
            fig_jbreak.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#38240D", size=12),
                title=dict(font=dict(color="#38240D", size=16)),
                xaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
                yaxis=dict(title_font=dict(color="#38240D"), tickfont=dict(color="#38240D")),
                legend=dict(font=dict(color="#38240D")),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_jbreak, use_container_width=True)

    with t_col2:
        if sim_res["education_targeting_breakdown"]:
            edu_break_df = pd.DataFrame(list(sim_res["education_targeting_breakdown"].items()), columns=["Education", "Calls_Allocated"])
            fig_ebreak = px.pie(
                edu_break_df,
                names="Education",
                values="Calls_Allocated",
                hole=0.45,
                title="Target Allocation by Education Level",
                color_discrete_sequence=["#C05800", "#713600", "#E1D094", "#9E2A2B"]
            )
            fig_ebreak.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#38240D", size=12),
                title=dict(font=dict(color="#38240D", size=16)),
                legend=dict(font=dict(color="#38240D")),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320
            )
            st.plotly_chart(fig_ebreak, use_container_width=True)

    # Export Targeted Call Sheet
    st.markdown("#### 📋 Prioritized Telemarketing Lead Call Sheet")
    targeted_data = sim_res["targeted_customers_df"]
    
    st.dataframe(
        targeted_data[targeted_data["is_targeted"] == 1][
            ["predicted_conversion_prob", "expected_net_profit", "age", "job", "balance", "duration", "campaign", "pdays"]
        ].head(50),
        use_container_width=True
    )

    csv_data = targeted_data[targeted_data["is_targeted"] == 1].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prioritized Telemarketing Lead Sheet (CSV)",
        data=csv_data,
        file_name="campaignoptima_targeted_leads.csv",
        mime="text/csv",
        use_container_width=True
    )
