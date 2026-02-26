# =====================================
# TITANIC STREAMLIT APP
# =====================================

import streamlit as st
import numpy as np
import pickle
import os


# =====================================
# LOAD MODEL
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "titanic_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


# =====================================
# UI
# =====================================

st.title("Titanic Survival Predictor")


pclass = st.selectbox("Passenger Class", [1,2,3])

sex = st.selectbox("Sex", ["Male", "Female"])

age = st.slider("Age", 0, 80, 25)

fare = st.slider("Fare", 0, 500, 50)


# encode sex
sex_encoded = 0 if sex == "Male" else 1


# =====================================
# PREDICTION
# =====================================

input_data = np.array([[pclass, sex_encoded, age, fare]])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

prob = model.predict_proba(input_scaled)


# =====================================
# OUTPUT
# =====================================

if prediction[0] == 1:
    st.success("Survived")
else:
    st.error("Not Survived")


st.write("Probability Survived:", prob[0][1])
