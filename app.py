import streamlit as st
import joblib
import os
from helper import prepare_features, reverse_mapping

# Load the saved model + metadata once
@st.cache_resource
def load_model_data():

    path = os.path.join("Artifacts", "best_model.joblib")
    return joblib.load(path)

model_data = load_model_data()
model = model_data['model']

st.set_page_config(page_title="CodeX BevIntel™ 🍷: Price Predictor", layout="wide")
st.title("CodeX BevIntel™ 🍷: Price Predictor")

with st.form("bevintel_form", clear_on_submit=False):
    col1, col2, col3, col4 = st.columns(4)

    # Row 1
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=18, step=1, key="age")
    with col2:
        gender = st.selectbox("Gender", ["", "M", "F"], index=0, key="gender")
    with col3:
        zone = st.selectbox("Zone", ["", "Metro", "Urban", "Rural", "Semi-Urban"], index=0, key="zone")
    with col4:
        occupation = st.selectbox("Occupation", ["", "Entrepreneur", "Working Professional", "Student", "Retired"], index=0, key="occupation")

    # Row 2
    with col1:
        income = st.selectbox("Income Level (in lakhs)", ["", "Not_reported", "<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"], index=0, key="income")
    with col2:
        consume_freq = st.selectbox("Consume Frequency (weekly)", ["", "0-2 times", "3-4 times", "5-7 times"], index=0, key="consume_freq")
    with col3:
        current_brand = st.selectbox("Current Brand", ["", "Newcomer", "Established"], index=0, key="current_brand")
    with col4:
        size_pref = st.selectbox("Preferable Consumption Size", ["", "Small (250 ml)", "Medium (500 ml)", "Large (1 L)"], index=0, key="size_pref")

    # Row 3
    with col1:
        awareness = st.selectbox("Awareness of other brands", ["", "0 to 1", "2 to 4", "above 4"], index=0, key="awareness")
    with col2:
        reason = st.selectbox("Reasons for choosing brands", ["", "Price", "Quality", "Availability", "Brand Reputation"], index=0, key="reason")
    with col3:
        flavor = st.selectbox("Flavor Preference", ["", "Traditional", "Exotic"], index=0, key="flavor")
    with col4:
        channel = st.selectbox("Purchase Channel", ["", "Online", "Retail Store"], index=0, key="channel")

    # Row 4 — remaining fields
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        packaging = st.selectbox("Packaging Preference", ["", "Simple", "Premium", "Eco-Friendly"], index=0, key="packaging")
    with col6:
        health = st.selectbox("Health Concerns", ["", "Low (Not very concerned)", "Medium (Moderately health-conscious)", "High (Very health-conscious)"], index=0, key="health")
    with col7:
        situation = st.selectbox("Typical Consumption Situations", ["", "Active (eg. Sports, gym)", "Casual (eg. At home)", "Social (eg. Parties)"], index=0, key="situation")


    submit = st.form_submit_button("Calculate Price Range")

if submit:
    # validation logic: ensure all fields are filled / valid
    # age is automatically valid since number_input restricts range
    # check that none of the selectboxes have the default empty value ""
    all_filled = True
    required_keys = ["gender","zone","occupation","income","consume_freq",
                     "current_brand","size_pref","awareness","reason",
                     "flavor","channel","packaging","health","situation"]
    for key in required_keys:
        if st.session_state.get(key, "") == "":
            all_filled = False
            break

    if not all_filled:
        st.error("Please enter relevant values for all the fields")
    else:
        # collect raw inputs into a dict
        raw_input = {
            'age': age,
            'gender': gender,
            'zone': zone,
            'occupation': occupation,
            'income_level': income,
            'consume_frequency': consume_freq,
            'current_brand': current_brand,
            'preferable_consumption_size': size_pref,
            'awareness_of_other_brands': awareness,
            'reasons_for_choosing_brands': reason,
            'flavor_preference': flavor,
            'purchase_channel': channel,
            'packaging_preference': packaging,
            'health_concerns': health,
            'typical_consumption_situations': situation
        }

        # prepare features
        X_new = prepare_features(raw_input, model_data)

        # get prediction
        pred = model.predict(X_new)

        # display result
        st.success(f"Predicted Price Range: {reverse_mapping(pred)}")

