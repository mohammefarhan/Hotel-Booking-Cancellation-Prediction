# ====================================
# HOTEL BOOKING CANCELLATION PREDICTOR
# Developed by Farhan
# ====================================

import streamlit as st
import pandas as pd
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="centered"
)

# -------------------------
# LOAD MODEL + ENCODER
# -------------------------
model = joblib.load("hotel_rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------------
# HEADER WITH SMALL LOGOS
# -------------------------
c1, c2, c3 = st.columns([1,6,1])

with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/139/139899.png", width=35)

with c2:
    st.title("Hotel Booking Cancellation Predictor")
    st.caption("Developed by Farhan")

with c3:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=35)

st.write(
"""
This application predicts whether a hotel reservation is likely to be cancelled
using a tuned Random Forest machine learning model.
"""
)

st.divider()

# -------------------------
# BOOKING DETAILS
# -------------------------
st.subheader("🧾 Booking Details")

lead_time = st.number_input("Lead Time", 0, 500, 45)
avg_price = st.number_input("Avg Price Per Room", 0.0, 500.0, 120.0)
special_requests = st.selectbox("Special Requests", [0,1,2,3,4,5])

week_nights = st.number_input("Week Nights", 0, 20, 2)
weekend_nights = st.number_input("Weekend Nights", 0, 10, 1)
repeated_guest = st.selectbox("Repeated Guest", [0,1])

meal_plan = st.selectbox(
    "Meal Plan",
    ["Meal Plan 1","Meal Plan 2","Meal Plan 3","Not Selected"]
)

room_type = st.selectbox(
    "Room Type",
    ["Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4",
     "Room_Type 5","Room_Type 6","Room_Type 7"]
)

market_segment = st.selectbox(
    "Market Segment",
    ["Online","Offline","Corporate","Complementary","Aviation"]
)

st.divider()

# -------------------------
# BUILD INPUT DATA
# -------------------------
input_data = pd.DataFrame([{
    "lead_time": lead_time,
    "avg_price_per_room": avg_price,
    "no_of_special_requests": special_requests,
    "no_of_week_nights": week_nights,
    "no_of_weekend_nights": weekend_nights,
    "repeated_guest": repeated_guest,
    "total_nights": week_nights + weekend_nights
}])

# safer encoding (only activate existing columns)
for col in model_columns:
    if col == f"type_of_meal_plan_{meal_plan}":
        input_data[col] = 1
    if col == f"room_type_reserved_{room_type}":
        input_data[col] = 1
    if col == f"market_segment_type_{market_segment}":
        input_data[col] = 1

# add missing columns
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

input_data = input_data[model_columns]

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Booking Status"):

    prediction = model.predict(input_data)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    if label == "Canceled":
        st.error("⚠️ Booking Likely to be Cancelled")
    else:
        st.success("✅ Booking Likely to be Confirmed")

    prob = model.predict_proba(input_data)[0]
    st.caption(f"Prediction confidence: {prob}")