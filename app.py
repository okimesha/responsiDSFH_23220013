import streamlit as st
import joblib
import numpy as np

# ===============================
# Load Model dan Scaler
# ===============================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Cryotherapy Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Cryotherapy Treatment Prediction")
st.write("Prediksi keberhasilan terapi Cryotherapy menggunakan Machine Learning.")

st.divider()

# ===============================
# Input Data
# ===============================

sex = st.selectbox(
    "Sex",
    [1, 2],
    help="1 = Male, 2 = Female"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

time = st.number_input(
    "Time (Months)",
    min_value=0.0,
    value=6.0
)

number_of_warts = st.number_input(
    "Number of Warts",
    min_value=1,
    value=5
)

type_wart = st.number_input(
    "Type",
    min_value=1,
    max_value=3,
    value=1
)

area = st.number_input(
    "Area",
    min_value=1,
    value=100
)

st.divider()

# ===============================
# Prediksi
# ===============================

if st.button("Predict"):

    data = np.array([
        [
            sex,
            age,
            time,
            number_of_warts,
            type_wart,
            area
        ]
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Treatment Successful")
    else:
        st.error("❌ Treatment Failed")

    if probability is not None:
        st.write("### Probability")

        st.write(f"Failure : **{probability[0]*100:.2f}%**")
        st.write(f"Success : **{probability[1]*100:.2f}%**")

st.divider()

st.caption("Machine Learning Prediction using Best Model")
