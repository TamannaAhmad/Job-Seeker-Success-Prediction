import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('hiring_success_model.pkl')

st.title("Hiring Success Prediction App 🚀")

st.write("Fill in candidate details below:")

# Input fields
age = st.slider('Age', 20, 50, 30)
gender = st.radio('Gender', ['Male', 'Female'])
education = st.selectbox('Education Level', ['Bachelor\'s Type 1', 'Bachelor\'s Type 2', 'Master\'s', 'PhD'])
experience = st.slider('Years of Experience', 0, 15, 5)
previous_companies = st.slider('Previous Companies Worked', 1, 5, 2)
distance = st.slider('Distance from Company (km)', 1.0, 50.0, 10.0)
interview_score = st.slider('Interview Score', 0, 100, 75)
skill_score = st.slider('Skill Score', 0, 100, 80)
personality_score = st.slider('Personality Score', 0, 100, 70)
strategy = st.selectbox('Recruitment Strategy', ['Aggressive', 'Moderate', 'Conservative'])

# Mapping categorical inputs
gender_map = {'Male': 0, 'Female': 1}
education_map = {'Bachelor\'s Type 1': 1, 'Bachelor\'s Type 2': 2, 'Master\'s': 3, 'PhD': 4}
strategy_map = {'Aggressive': 1, 'Moderate': 2, 'Conservative': 3}

candidate_data = {
    'Age': age,
    'Gender': gender_map[gender],
    'EducationLevel': education_map[education],
    'ExperienceYears': experience,
    'PreviousCompanies': previous_companies,
    'DistanceFromCompany': distance,
    'InterviewScore': interview_score,
    'SkillScore': skill_score,
    'PersonalityScore': personality_score,
    'RecruitmentStrategy': strategy_map[strategy]
}

if st.button('Predict Hiring Success'):
    candidate_df = pd.DataFrame([candidate_data])
    probability = model.predict_proba(candidate_df)[0,1]
    st.success(f"Predicted Hiring Success Probability: {probability:.2%}")
