import streamlit as st
import traceback
import joblib

try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception:
    st.code(traceback.format_exc())
    st.stop()
