import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import datetime
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from train import (
    getModel, getAccuracy, get_test_data, 
    get_predictions, get_feature_names, get_full_data
)

# --- SET PAGE CONFIG ---
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- HTML Report Generation Function ---
def create_html_report(name, input_data, result, probability):
    style = """
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
        .content { background-color: #f9f9f9; padding: 20px; border-radius: 8px; }
        .result-diabetic { color: #D32F2F; font-weight: bold; font-size: 1.2em; }
        .result-nondiabetic { color: #388E3C; font-weight: bold; font-size: 1.2em; }
        .disclaimer { font-size: 0.8em; color: #777; text-align: center; margin-top: 20px; }
        ul { list-style-type: none; padding-left: 0; }
        li { margin-bottom: 8px; }
        li strong { display: inline-block; width: 250px; }
    </style>
    """
    result_class = "result-diabetic" if result == "Diabetic" else "result-nondiabetic"
    details_html = "".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in input_data.items()])

    return f"""
    <html><head>{style}</head><body>
        <h1>Diabetes Prediction Report</h1>
        <div class="content">
            <h2>Patient Name: {name}</h2>
            <p><strong>Report Date:</strong> {datetime.date.today().strftime('%Y-%m-%d')}</p>
            <h2>Patient Details</h2><ul>{details_html}</ul>
            <h2>Prediction Result</h2>
            <p class="result {result_class}">Prediction: {result}</p>
            <p><strong>Probability of Diabetes:</strong> {probability:.2%}</p>
        </div>
        <p><strong>Disclaimer:</strong> This tool is an AI-powered risk assessment system that uses past data to provide early diabetes predictions, intended for educational purposes and not as a substitute for professional medical diagnosis.</p>
    </body></html>
    """

# --- Background Image Functions ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), bin_file)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None

def set_bg_from_local_file(file_name):
    base64_str = get_base64_of_bin_file(file_name)
    if base64_str:
        file_ext = os.path.splitext(file_name)[1].lower()
        mime_type = "image/png" if file_ext == ".png" else "image/jpeg"
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:{mime_type};base64,{base64_str}");
            background-size: cover; background-position: center; background-attachment: fixed;
        }}
        .form-container {{ background-color: rgba(255, 255, 255, 0.9); border-radius: 10px; padding: 25px; border: 1px solid #ddd; }}
        
        /* UPDATED: Left-align headers and metrics */
        [data-testid="stAppViewContainer"] .main h1, 
        [data-testid="stAppViewContainer"] .main h2 {{ 
            color: #ffffff !important; 
            text-shadow: 2px 2px 4px #000000 !important; 
            text-align: left !important; 
        }}

        [data-testid="stMetric"] {{
            text-align: left !important;
            display: flex;
            flex-direction: column;
            align-items: flex-start !important;
        }}
        
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: #ffffff !important;
            text-shadow: 1px 1px 3px #000000 !important;
            justify-content: flex-start !important;
            text-align: left !important;
        }}
        </style>
        """, unsafe_allow_html=True)

# --- Initialize Session State ---
for key, default in [('show_input_dialog', False), ('show_result_dialog', False), ('input_data', {}), ('prediction_result', ""), ('prediction_proba', 0.0)]:
    if key not in st.session_state: st.session_state[key] = default

# --- Load Model and Data ---
model, accuracy = getModel(), getAccuracy()
X_test, y_test = get_test_data()
y_pred, y_pred_proba = get_predictions()
feature_names, data = get_feature_names(), get_full_data()

# --- Sidebar ---
st.sidebar.header("App Options")
theme_choice = st.sidebar.radio("Select Theme", ["Custom Background", "Default Dark"])
if theme_choice == "Custom Background": set_bg_from_local_file('bg1.jpg')

# --- Main Page UI ---
if model is None or data.empty:
    st.error("Error: Model or data could not be loaded.")
else:
    # --- STATE 1: Input Form ---
    if st.session_state.show_input_dialog:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Patient Data")
        col1, col2 = st.columns(2)
        with col1:
            preg = st.number_input("Pregnancies", 0, 20, 0)
            gluc = st.number_input("Glucose", 0.0, 2000.0, 120.0)
            bp = st.number_input("Blood Pressure", 0.0, 400.0, 70.0)
            skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
        with col2:
            ins = st.number_input("Insulin", 0.0, 2000.0, 80.0)
            bmi = st.number_input("BMI", 0.0, 70.0, 30.0)
            age = st.number_input("Age", 1.0, 130.0, 30.0)
            dpf = 0.4

        if st.button("Predict"):
            st.session_state.input_data = {'Pregnancies': preg, 'Glucose': gluc, 'BloodPressure': bp, 'SkinThickness': skin, 'Insulin': ins, 'BMI': bmi, 'DiabetesPedigreeFunction': dpf, 'Age': age}
            input_df = pd.DataFrame([st.session_state.input_data])
            st.session_state.prediction_result = "Diabetic" if model.predict(input_df)[0] == 1 else "Non-Diabetic"
            st.session_state.prediction_proba = model.predict_proba(input_df)[0][1]
            st.session_state.show_input_dialog, st.session_state.show_result_dialog = False, True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STATE 2: Result Page ---
    elif st.session_state.show_result_dialog:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        res, prob = st.session_state.prediction_result, st.session_state.prediction_proba
        if res == "Diabetic": st.error(f"Prediction: {res}")
        else: st.success(f"Prediction: {res}")
        st.metric("Probability of Diabetes", f"{prob:.2%}")
        
        person_name = st.text_input("Enter Name for Report")
        if person_name:
            html_rep = create_html_report(person_name, st.session_state.input_data, res, prob)
            st.download_button("Download Report (.html)", html_rep, file_name=f"Report_{person_name}.html", mime="text/html")
        
        col_back, _, col_end = st.columns(3)
        if col_back.button("Go Back"):
            st.session_state.show_result_dialog, st.session_state.show_input_dialog = False, True
            st.rerun()
        if col_end.button("End"):
            st.session_state.show_result_dialog, st.session_state.show_input_dialog = False, False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STATE 3: Main Page Dashboard ---
    else:
        st.title("Diabetes Prediction App")
        if st.button("Start Prediction", key="start_button"):
            st.session_state.show_input_dialog = True
            st.rerun()

        st.header("Model Performance")
        
        # FIXED: Using columns to force the metric to the left
        m_col1, m_col2 = st.columns([1, 4]) 
        with m_col1:
            st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

        tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])
        
        with tabs[0]:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
            st.pyplot(fig)

        with tabs[1]:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.legend()
            st.pyplot(fig)

        with tabs[2]:
            try:
                importance = model.named_steps['model'].feature_importances_
                feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance')
                fig, ax = plt.subplots()
                ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='teal')
                st.pyplot(fig)
            except: st.error("Importance data unavailable.")



