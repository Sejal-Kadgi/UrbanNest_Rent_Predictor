# Build your Streamlit application here

import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* Title styling */
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem !important;
    }

    /* Subheader styling */
    h2, h3 {
        font-weight: 600 !important;
        color: #c4b5fd !important;
        margin-top: 0.5rem !important;
    }

    /* Card-like section boxes */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Input labels */
    label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #a5b4fc !important;
        letter-spacing: 0.02em;
    }

    /* Input fields */
    input, select {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #6d28d9, #4338ca) !important;
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.6) !important;
        transform: translateY(-2px) !important;
    }

    /* Success box */
    div[data-testid="stAlert"] {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 12px !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #6ee7b7 !important;
    }

    /* Error box */
    div[data-testid="stNotification"] {
        border-radius: 12px !important;
    }

    /* Divider */
    hr {
        border-color: rgba(167, 139, 250, 0.2) !important;
        margin: 1rem 0 !important;
    }

    /* Caption */
    div[data-testid="stCaptionContainer"] p {
        color: #818cf8 !important;
        font-size: 0.85rem !important;
    }

    /* Number input arrows */
    button[kind="stepper"] {
        background: rgba(167, 139, 250, 0.2) !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_rf_model.pkl")
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, label_encoders, feature_cols

model, label_encoders, feature_cols = load_artifacts()

# Build city → locations mapping from training data
train_raw = pd.read_csv("Dataset/train.csv")
city_to_locations = train_raw.groupby("city")["location"].apply(list).to_dict()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UrbanNest Rent Predictor", page_icon="🏠", layout="centered")
st.title("🏠 UrbanNest Analytics — Rent Predictor")
st.caption("Dynamic House Rent Prediction Engine · Mumbai · Pune · Delhi · Hisar")
st.markdown("---")

# ── Row 1: Property Identity ──────────────────────────────────────────────────
st.subheader("🏡 Property Identity")
col1, col2 = st.columns(2)

with col1:
    city          = st.selectbox("City",              options=list(label_encoders['city'].classes_))
    available_locations = sorted(set(city_to_locations.get(city, list(label_encoders['location'].classes_))))
    location = st.selectbox("Location", options=available_locations)

with col2:
    status        = st.selectbox("Furnishing Status", options=list(label_encoders['Status'].classes_))
    property_type = st.selectbox("Property Type",     options=list(label_encoders['property_type'].classes_))

st.markdown("---")

# ── Row 2: Size & Rooms ───────────────────────────────────────────────────────
st.subheader("📐 Size & Rooms")
col3, col4, col5 = st.columns(3)

with col3:
    size      = st.number_input("Size (ft²)",  min_value=100, max_value=10000, value=800, step=50)
    bhk       = st.number_input("BHK",         min_value=1,   max_value=10,    value=2,   step=1)

with col4:
    rooms_num = st.number_input("Total Rooms", min_value=1,   max_value=20,    value=3,   step=1)
    num_bath  = st.number_input("Bathrooms",   min_value=1,   max_value=10,    value=2,   step=1)

with col5:
    num_bal    = st.number_input("Balconies",         min_value=0, max_value=10,  value=1, step=1)
    verif_days = st.number_input("Verification Days", min_value=0, max_value=365, value=7, step=1)

st.markdown("---")

# ── Row 3: Financial & Location ───────────────────────────────────────────────
st.subheader("💰 Financial & Location")
col6, col7 = st.columns(2)

with col6:
    security      = st.number_input("Security Deposit (₹)", min_value=0, max_value=500000, value=50000, step=5000)
    is_negotiable = st.selectbox("Is Negotiable?",          options=[0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")

with col7:
    latitude  = st.number_input("Latitude",  value=19.076, format="%.4f")
    longitude = st.number_input("Longitude", value=72.877, format="%.4f")

st.markdown("---")

if st.button("🔍 Predict Rent", use_container_width=True):
    try:
        # Encode categoricals using saved LabelEncoders
        city_enc   = label_encoders['city'].transform([city])[0]
        loc_enc    = label_encoders['location'].transform([location])[0]
        status_enc = label_encoders['Status'].transform([status])[0]
        ptype_enc  = label_encoders['property_type'].transform([property_type])[0]

        # Build feature vector in exact training order
        input_map = {
            'location':          loc_enc,
            'city':              city_enc,
            'latitude':          latitude,
            'longitude':         longitude,
            'numBathrooms':      num_bath,
            'numBalconies':      num_bal,
            'isNegotiable':      is_negotiable,
            'SecurityDeposit':   security,
            'Status':            status_enc,
            'Size_ft²':          size,
            'BHK':               bhk,
            'rooms_num':         rooms_num,
            'property_type':     ptype_enc,
            'verification_days': verif_days,
        }

        input_vector = np.array([[input_map[c] for c in feature_cols]])
        prediction   = model.predict(input_vector)[0]

        st.success(f"💰 Estimated Monthly Rent: ₹{prediction:,.0f}")
        st.caption(f"Model: Random Forest · Features used: {len(feature_cols)}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")