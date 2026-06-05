# ============================================================
# PROJECT 1 — CUSTOMER CHURN PREDICTOR
# FILE: streamlit_app.py  |  DAY 4 AFTERNOON
# Run: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #185FA5; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #5F5E5A; margin-bottom: 2rem;
    }
    .metric-card {
        background: #F1EFE8; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .risk-high {
        background: #FCEBEB; border: 2px solid #E24B4A;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .risk-low {
        background: #EAF3DE; border: 2px solid #639922;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model and scaler ─────────────────────────────────────
@st.cache_resource
def load_model():
    model  = pickle.load(open('xgb_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    with open('feature_columns.json') as f:
        columns = json.load(f)
    return model, scaler, columns

model, scaler, feature_columns = load_model()

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">Customer Churn Predictor</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict customer churn risk using XGBoost + SHAP explainability | Built for Canadian telecom & banking sectors</p>',
            unsafe_allow_html=True)

# ── Sidebar — Customer Inputs ─────────────────────────────────
st.sidebar.header("Customer Details")
st.sidebar.markdown("---")

# Account info
st.sidebar.subheader("Account")
tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract       = st.sidebar.selectbox("Contract Type",
                    ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Charges
st.sidebar.subheader("Charges")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges   = st.sidebar.slider("Total Charges ($)", 0, 9000,
                                     int(monthly_charges * tenure))

# Services
st.sidebar.subheader("Services")
internet_service = st.sidebar.selectbox("Internet Service",
                    ["DSL", "Fiber optic", "No"])
phone_service    = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines",
                    ["Yes", "No", "No phone service"])
online_security  = st.sidebar.selectbox("Online Security",
                    ["Yes", "No", "No internet service"])
online_backup    = st.sidebar.selectbox("Online Backup",
                    ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection",
                    ["Yes", "No", "No internet service"])
tech_support     = st.sidebar.selectbox("Tech Support",
                    ["Yes", "No", "No internet service"])
streaming_tv     = st.sidebar.selectbox("Streaming TV",
                    ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies",
                    ["Yes", "No", "No internet service"])

# Demographics
st.sidebar.subheader("Demographics")
gender      = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior      = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner     = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents  = st.sidebar.selectbox("Dependents", ["Yes", "No"])

predict_btn = st.sidebar.button("Predict Churn Risk", type="primary")

# ── Build input dataframe ─────────────────────────────────────
def build_input():
    row = {c: 0 for c in feature_columns}

    # Numeric
    row['tenure']          = tenure
    row['MonthlyCharges']  = monthly_charges
    row['TotalCharges']    = total_charges if total_charges > 0 else monthly_charges * tenure

    # Engineered features
    row['ChargePerTenure'] = monthly_charges / (tenure + 1)
    services = sum([
        1 if multiple_lines   == "Yes" else 0,
        1 if online_security  == "Yes" else 0,
        1 if online_backup    == "Yes" else 0,
        1 if device_protection== "Yes" else 0,
        1 if tech_support     == "Yes" else 0,
        1 if streaming_tv     == "Yes" else 0,
        1 if streaming_movies == "Yes" else 0,
        1 if phone_service    == "Yes" else 0,
        1 if internet_service != "No"  else 0,
    ])
    row['TotalServices']  = services
    row['IsNewCustomer']  = 1 if tenure < 12 else 0

    # Binary
    row['gender']          = 1 if gender == "Male" else 0
    row['SeniorCitizen']   = 1 if senior == "Yes" else 0
    row['Partner']         = 1 if partner == "Yes" else 0
    row['Dependents']      = 1 if dependents == "Yes" else 0
    row['PhoneService']    = 1 if phone_service == "Yes" else 0
    row['PaperlessBilling']= 1 if paperless == "Yes" else 0

    # One-hot: Contract
    if f'Contract_{contract}' in row:
        row[f'Contract_{contract}'] = 1

    # One-hot: PaymentMethod
    pm_key = f'PaymentMethod_{payment_method}'
    if pm_key in row:
        row[pm_key] = 1

    # One-hot: InternetService
    if f'InternetService_{internet_service}' in row:
        row[f'InternetService_{internet_service}'] = 1

    # One-hot: other services
    for feat, val in [
        ('MultipleLines', multiple_lines),
        ('OnlineSecurity', online_security),
        ('OnlineBackup', online_backup),
        ('DeviceProtection', device_protection),
        ('TechSupport', tech_support),
        ('StreamingTV', streaming_tv),
        ('StreamingMovies', streaming_movies),
    ]:
        key = f'{feat}_{val}'
        if key in row:
            row[key] = 1

    df = pd.DataFrame([row])
    # Scale numeric columns
    num_cols = ['tenure','MonthlyCharges','TotalCharges','ChargePerTenure','TotalServices']
    valid_num = [c for c in num_cols if c in df.columns]
    df[valid_num] = scaler.transform(df[valid_num])
    return df

# ── Main area — results ───────────────────────────────────────
if not predict_btn:
    st.info("Configure customer details in the sidebar, then click **Predict Churn Risk**.")

    # Show dataset overview
    st.subheader("About This Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", "5,634")
    col2.metric("Features Used", "35+")
    col3.metric("Model", "XGBoost")
    col4.metric("Test AUC-ROC", "~0.85")

    st.markdown("""
    **How it works:**
    1. Fill in customer details in the sidebar
    2. Click **Predict Churn Risk**
    3. See the churn probability + SHAP explanation showing exactly WHY

    **Built with:** Python · XGBoost · SHAP · MLflow · Streamlit

    **Dataset:** IBM Telco Customer Churn (7,043 customers)
    """)

else:
    input_df = build_input()
    prob     = model.predict_proba(input_df)[0][1]
    pred     = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

    # ── Row 1: Key metrics ──
    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        if prob >= 0.5:
            st.markdown(f"""
            <div class="risk-high">
                <h2 style="color:#A32D2D;margin:0">{prob*100:.1f}%</h2>
                <p style="color:#A32D2D;font-weight:600;margin:0">Churn Probability</p>
                <p style="color:#E24B4A;margin:4px 0 0 0">⚠ HIGH RISK — Likely to churn</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2 style="color:#27500A;margin:0">{prob*100:.1f}%</h2>
                <p style="color:#27500A;font-weight:600;margin:0">Churn Probability</p>
                <p style="color:#639922;margin:4px 0 0 0">✓ LOW RISK — Likely to stay</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.metric("Tenure",           f"{tenure} months")
        st.metric("Monthly Charges",  f"${monthly_charges}")

    with col3:
        st.metric("Contract Type",    contract)
        st.metric("Total Services",   f"{sum([1 if multiple_lines=='Yes' else 0, 1 if online_security=='Yes' else 0, 1 if online_backup=='Yes' else 0, 1 if device_protection=='Yes' else 0, 1 if tech_support=='Yes' else 0, 1 if streaming_tv=='Yes' else 0, 1 if streaming_movies=='Yes' else 0])}")

    st.markdown("---")

    # ── Row 2: SHAP waterfall ──
    st.subheader("Why This Prediction? (SHAP Explanation)")
    st.caption("Red bars = features pushing toward churn. Blue bars = features pushing toward staying.")

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(12, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns.tolist()
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP plot could not render: {e}")

    st.markdown("---")

    # ── Row 3: Risk factors summary ──
    st.subheader("Key Risk Factors for This Customer")
    risk_factors = []
    if contract == "Month-to-month":
        risk_factors.append("Month-to-month contract → highest churn risk contract type")
    if tenure < 12:
        risk_factors.append(f"New customer ({tenure} months) → early-tenure customers churn most")
    if monthly_charges > 70:
        risk_factors.append(f"High monthly charges (${monthly_charges}) → above average cost drives churn")
    if internet_service == "Fiber optic":
        risk_factors.append("Fiber optic service → historically highest churn internet tier")
    if payment_method == "Electronic check":
        risk_factors.append("Electronic check payment → manual payment linked to higher churn")
    if not risk_factors:
        risk_factors.append("No major risk flags detected for this customer")

    for factor in risk_factors:
        st.markdown(f"- {factor}")

    # ── Row 4: Retention recommendation ──
    st.subheader("Retention Recommendation")
    if prob >= 0.7:
        st.error("""
        **Immediate action required.**
        Offer: 20% discount on upgrade to 1-year contract.
        Priority: Assign dedicated account manager.
        Timeline: Contact within 48 hours.
        """)
    elif prob >= 0.5:
        st.warning("""
        **Proactive outreach recommended.**
        Offer: Free tech support upgrade for 3 months.
        Priority: Include in next retention campaign.
        Timeline: Contact within 2 weeks.
        """)
    else:
        st.success("""
        **Low risk — standard engagement.**
        Action: Include in loyalty rewards program.
        Monitor: Flag if monthly charges increase significantly.
        """)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Bhavana Aswin | XGBoost + SHAP + MLflow + Streamlit | IBM Telco Dataset")
