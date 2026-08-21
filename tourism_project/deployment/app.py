"""
app.py
-------
Streamlit front-end for the "Visit with Us" Wellness Tourism Package predictor.

The app loads the model that the GitHub Actions pipeline trained and committed
to this folder, collects customer details from the user into a single-row
dataframe, and predicts whether the customer is likely to purchase the package.
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🌴")

st.title("🌴 Wellness Tourism Package Predictor")
st.write(
    "Predict whether a customer is likely to purchase the **Wellness Tourism "
    "Package** before contacting them. Fill in the customer details below and "
    "click **Predict**."
)

model = load_model()

# ---------------- Collect user inputs ----------------
st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input(
        "Number Of Persons Visiting", min_value=1, max_value=10, value=3
    )
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input(
        "Number Of Trips (per year)", min_value=0, max_value=30, value=2
    )

with col2:
    Passport = st.selectbox("Holds Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    OwnCar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    NumberOfChildrenVisiting = st.number_input(
        "Number Of Children Visiting (below 5)", min_value=0, max_value=5, value=0
    )
    Designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    MonthlyIncome = st.number_input(
        "Monthly Income", min_value=1000, max_value=100000, value=20000
    )
    ProductPitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    NumberOfFollowups = st.number_input(
        "Number Of Follow-ups", min_value=0, max_value=10, value=3
    )
    DurationOfPitch = st.number_input(
        "Duration Of Pitch (minutes)", min_value=1, max_value=60, value=15
    )

# Assemble the inputs into a single-row dataframe with the training columns.
input_df = pd.DataFrame(
    [
        {
            "Age": Age,
            "TypeofContact": TypeofContact,
            "CityTier": CityTier,
            "DurationOfPitch": DurationOfPitch,
            "Occupation": Occupation,
            "Gender": Gender,
            "NumberOfPersonVisiting": NumberOfPersonVisiting,
            "NumberOfFollowups": NumberOfFollowups,
            "ProductPitched": ProductPitched,
            "PreferredPropertyStar": PreferredPropertyStar,
            "MaritalStatus": MaritalStatus,
            "NumberOfTrips": NumberOfTrips,
            "Passport": Passport,
            "PitchSatisfactionScore": PitchSatisfactionScore,
            "OwnCar": OwnCar,
            "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
            "Designation": Designation,
            "MonthlyIncome": MonthlyIncome,
        }
    ]
)

st.subheader("Model Input")
st.dataframe(input_df)

# Use the same decision threshold that the model was evaluated with in training.
classification_threshold = 0.45

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= classification_threshold else 0

    if prediction == 1:
        st.success(
            f"✅ This customer is **likely to purchase** the Wellness Tourism "
            f"Package (probability: {proba:.2%})."
        )
    else:
        st.info(
            f"❌ This customer is **unlikely to purchase** the Wellness Tourism "
            f"Package (probability: {proba:.2%})."
        )
