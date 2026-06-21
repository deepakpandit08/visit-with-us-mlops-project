
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = "deepakpandit08/visit-with-us-wellness-model"

st.set_page_config(
    page_title="Visit with Us - Wellness Package Predictor",
    layout="centered"
)

st.title("Visit with Us - Wellness Tourism Package Predictor")

st.write(
    "This app predicts whether a customer is likely to purchase the Wellness Tourism Package."
)

model_file = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename="best_model.joblib",
    repo_type="model"
)

model = joblib.load(model_file)

st.subheader("Enter Customer Details")

Age = st.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0.0, value=15.0)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0.0, value=3.0)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1.0, max_value=7.0, value=3.0)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0.0, value=2.0)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0.0, value=1.0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)

input_data = pd.DataFrame({
    "Age": [Age],
    "TypeofContact": [TypeofContact],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "ProductPitched": [ProductPitched],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [MaritalStatus],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome]
})

st.write("Input data preview:")
st.dataframe(input_data)

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Customer is likely to buy the package.")
        st.write(f"Purchase Probability: {probability:.2%}")
    else:
        st.warning(f"Prediction: Customer is not likely to buy the package.")
        st.write(f"Purchase Probability: {probability:.2%}")
