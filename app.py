"""
Customer Churn Prediction — Streamlit Web App
=============================================
B.Tech Gen AI – Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #065A82;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .result-box-green {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #155724;
    }
    .result-box-red {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.3rem;
        font-weight: 700;
        color: #721c24;
    }
    .metric-card {
        background: #f0f8ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stSidebar > div:first-child {
        background-color: #f7fbff;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the pre-trained Random Forest model."""
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ── Encoding Maps (must match training) ──────────────────────────────────────
FREQUENT_FLYER_MAP      = {"No": 0, "Yes": 1}
ANNUAL_INCOME_MAP       = {"Low Income": 1, "Middle Income": 2, "High Income": 0}
ACCOUNT_SYNCED_MAP      = {"No": 0, "Yes": 1}
BOOKED_HOTEL_MAP        = {"No": 0, "Yes": 1}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=70)
    st.markdown("## 🔍 Input Features")
    st.markdown("Fill in the customer details below:")

    age = st.slider("🎂 Age", min_value=18, max_value=70, value=35, step=1)

    frequent_flyer = st.selectbox(
        "✈️ Frequent Flyer",
        options=["Yes", "No"],
        help="Is the customer a frequent flyer?"
    )

    annual_income_class = st.selectbox(
        "💰 Annual Income Class",
        options=["Low Income", "Middle Income", "High Income"]
    )

    services_opted = st.slider(
        "🛎️ Services Opted",
        min_value=1, max_value=9, value=4,
        help="Number of services the customer has opted for"
    )

    account_synced = st.selectbox(
        "📱 Account Synced to Social Media",
        options=["Yes", "No"]
    )

    booked_hotel = st.selectbox(
        "🏨 Booked Hotel or Not",
        options=["Yes", "No"]
    )

    predict_btn = st.button("🚀 Predict Churn", use_container_width=True, type="primary")

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">B.Tech Gen AI — Final Project &nbsp;|&nbsp; Random Forest Classifier</div>', unsafe_allow_html=True)
st.markdown("---")

# Overview
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌲 Algorithm", "Random Forest")
with col2:
    st.metric("📁 Training Samples", "763")
with col3:
    st.metric("🎯 Model Accuracy", "~88%")

st.markdown("---")

# ── Prediction Section ────────────────────────────────────────────────────────
if predict_btn:
    # Encode input
    input_data = np.array([[
        age,
        FREQUENT_FLYER_MAP[frequent_flyer],
        ANNUAL_INCOME_MAP[annual_income_class],
        services_opted,
        ACCOUNT_SYNCED_MAP[account_synced],
        BOOKED_HOTEL_MAP[booked_hotel]
    ]])

    prediction = model.predict(input_data)[0]
    proba      = model.predict_proba(input_data)[0]

    churn_prob    = proba[1] * 100
    no_churn_prob = proba[0] * 100

    st.subheader("🎯 Prediction Result")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        if prediction == 1:
            st.markdown(
                f'<div class="result-box-red">⚠️ Customer is likely to CHURN<br>'
                f'<span style="font-size:0.9rem;">Churn probability: {churn_prob:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.warning("💡 **Recommendation:** Offer a retention incentive — discounts, loyalty points, or personalized outreach.")
        else:
            st.markdown(
                f'<div class="result-box-green">✅ Customer is NOT likely to churn<br>'
                f'<span style="font-size:0.9rem;">Retention probability: {no_churn_prob:.1f}%</span></div>',
                unsafe_allow_html=True
            )
            st.success("💡 **Recommendation:** Continue providing excellent service to maintain satisfaction.")

    with col_b:
        # Probability gauge chart
        fig, ax = plt.subplots(figsize=(4.5, 4))
        bars = ax.barh(
            ['Not Churn', 'Churn'],
            [no_churn_prob, churn_prob],
            color=['#1C7293', '#F96167'],
            height=0.45,
            edgecolor='white'
        )
        for bar, val in zip(bars, [no_churn_prob, churn_prob]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 115)
        ax.set_xlabel('Probability (%)', fontsize=11)
        ax.set_title('Churn Probability', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Input Summary
    st.markdown("---")
    st.subheader("📋 Customer Profile Summary")
    summary_df = pd.DataFrame({
        "Feature": ["Age", "Frequent Flyer", "Annual Income Class",
                    "Services Opted", "Account Synced", "Booked Hotel"],
        "Value":   [age, frequent_flyer, annual_income_class,
                    services_opted, account_synced, booked_hotel]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    # Landing info
    st.markdown("""
    ### 👈 How to Use This App
    1. **Fill in** the customer details in the left sidebar.
    2. Click the **🚀 Predict Churn** button.
    3. View the churn prediction and probability score.
    4. Use the recommendation to take action!

    ---
    ### 📖 About This Project
    This application predicts whether a customer is likely to churn using a
    **Random Forest Classifier** trained on customer data including demographics,
    service usage, and account information.

    | Feature | Description |
    |---------|-------------|
    | Age | Customer's age |
    | Frequent Flyer | Whether customer flies frequently |
    | Annual Income Class | Income bracket (Low / Middle / High) |
    | Services Opted | Number of services subscribed |
    | Account Synced | Social media account sync status |
    | Booked Hotel | Whether customer has booked hotel |
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
    "Customer Churn Prediction App &nbsp;•&nbsp; B.Tech Gen AI 2nd Semester &nbsp;•&nbsp; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
