import streamlit as st
import traceback
import joblib

try:
    model = joblib.load("best_model (1).pkl")
    scaler = joblib.load("scaler (1).pkl")
except Exception:
    st.code(traceback.format_exc())
    st.stop()
