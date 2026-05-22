import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('one_hot_encoder_geo.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    le = pickle.load(f)

##streamlit app
st.title("Customer Churn Prediction from BANk data")
#user input
credit_score = st.number_input("Credit Score",min_value=300,max_value=900,value=650)
geography = st.selectbox("Select Geography", ohe.categories_[0])
gender=st.selectbox("Gender",le.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender':   gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
#ohe geography and label
geography_encoded = ohe.transform([[input_data['Geography']]]).toarray()
gender_encoded = le.transform([input_data['Gender']])[0]
input_df = pd.DataFrame([input_data])
input_df['Gender'] = gender_encoded
input_df = pd.concat([input_df.drop(['Geography'], axis=1), pd.DataFrame(geography_encoded, columns=ohe.get_feature_names_out(['Geography']))], axis=1)
input_scaled = scaler.transform(input_df)
#prediction
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]
if prediction_probability > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_probability:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {prediction_probability:.2f}")
    
