# Build your Streamlit application here

import streamlit as st
import joblib
import pickle
import numpy as np

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
    location      = st.selectbox("Location",          options=list(label_encoders['location'].classes_))

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
    num_bal    = st.number_input("Balconies",          min_value=0, max_value=10,  value=1, step=1)
    verif_days = st.number_input("Verification Days",  min_value=0, max_value=365, value=7, step=1)

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