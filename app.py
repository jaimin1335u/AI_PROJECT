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
    if result == "Diabetic":
        result_class = "result-diabetic"
    else:
        result_class = "result-nondiabetic"

    details_html = "<ul>"
    for key, value in input_data.items():
        details_html += f"<li><strong>{key}:</strong> {value}</li>"
    details_html += "</ul>"

    html_content = f"""
    <html>
    <head>
        <title>Diabetes Prediction Report</title>
        {style}
    </head>
    <body>
        <h1>Diabetes Prediction Report</h1>
        <div class="content">
            <h2>Patient Name: {name}</h2>
            <p><strong>Report Date:</strong> {datetime.date.today().strftime('%Y-%m-%d')}</p>
            <h2>Patient Details</h2>
            {details_html}
            <h2>Prediction Result</h2>
            <p class="result {result_class}">Prediction: {result}</p>
            <p><strong>Probability of Diabetes:</strong> {probability:.2%}</p>
        </div>
        <p class="disclaimer">
            Disclaimer: This prediction is based on a machine learning model and is not a substitute for 
            professional medical advice, diagnosis, or treatment.
        </p>
    </body>
    </html>
    """
    return html_content

# --- Background Image Functions ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), bin_file)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

def set_bg_from_local_file(file_name):
    base64_str = get_base64_of_bin_file(file_name)
    if base64_str is not None:
        file_ext = os.path.splitext(file_name)[1].lower()
        mime_type = "image/png" if file_ext == ".png" else "image/jpeg"

        page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:{mime_type};base64,{base64_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .form-container {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
        }}
        [data-testid="stAppViewContainer"] .main h1, 
        [data-testid="stAppViewContainer"] .main h2, 
        [data-testid="stAppViewContainer"] .main h3, 
        [data-testid="stAppViewContainer"] .main .stMetric {{
            color: #333333 !important;
            text-shadow: 1px 1px 3px #FFFFFF !important;
            text-align: center;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'show_input_dialog' not in st.session_state:
    st.session_state.show_input_dialog = False
if 'show_result_dialog' not in st.session_state:
    st.session_state.show_result_dialog = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = ""
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = 0.0

# --- Load Model and Data ---
model = getModel()
accuracy = getAccuracy()
X_test, y_test = get_test_data()
y_pred, y_pred_proba = get_predictions()
feature_names = get_feature_names()
data = get_full_data()

# --- Sidebar ---
st.sidebar.header("App Options")
theme_choice = st.sidebar.radio("Select Theme", ["Custom Background", "Default Dark"], index=0)
show_charts = st.sidebar.checkbox("Show Data Exploration", value=True)

if theme_choice == "Custom Background":
    set_bg_from_local_file('bg1.jpg')

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
            preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
            gluc = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=0.1)
            bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
            skin = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        with col2:
            ins = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=0.1)
            bmi = st.number_input("BMI (kg/m^2)", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
            age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=30.0, step=1.0)
            dpf = 0.4 # Hardcoded value from your code

        if st.button("Predict", key="predict_dialog_button"):
            st.session_state.input_data = {
                'Pregnancies': preg, 'Glucose': gluc, 'BloodPressure': bp,
                'SkinThickness': skin, 'Insulin': ins, 'BMI': bmi,
                'DiabetesPedigreeFunction': dpf, 'Age': age
            }
            
            # --- FIX: Convert to DataFrame to avoid SimpleImputer errors ---
            input_df = pd.DataFrame([st.session_state.input_data])
            
            prediction_val = model.predict(input_df)[0]
            probability_val = model.predict_proba(input_df)[0][1]
            
            st.session_state.prediction_result = "Diabetic" if prediction_val == 1 else "Non-Diabetic"
            st.session_state.prediction_proba = probability_val
            st.session_state.show_input_dialog = False
            st.session_state.show_result_dialog = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STATE 2: Result Page ---
    elif st.session_state.show_result_dialog:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        result = st.session_state.prediction_result
        proba = st.session_state.prediction_proba
        
        if result == "Diabetic":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
        
        st.metric(label="Probability of Diabetes", value=f"{proba:.2%}")
        st.divider()
        
        st.subheader("Download Report")
        person_name = st.text_input("Enter Person's Name for Report", key="person_name_input")
        
        html_data = ""
        download_disabled = True
        if person_name:
            html_data = create_html_report(person_name, st.session_state.input_data, result, proba)
            download_disabled = False
        
        col_back, col_dl, col_end = st.columns(3)
        with col_back:
            if st.button("Go Back"):
                st.session_state.show_result_dialog = False
                st.session_state.show_input_dialog = True
                st.rerun()
        with col_dl:
            st.download_button(
                label="Download Report (.html)",
                data=html_data,
                file_name=f"Diabetes_Report_{person_name.replace(' ', '_')}.html",
                mime="text/html",
                disabled=download_disabled
            )
        with col_end:
            if st.button("End"):
                st.session_state.show_result_dialog = False
                st.session_state.show_input_dialog = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- STATE 3: Main Page ---
    else:
        st.title("Diabetes Prediction App")
        if st.button("Start Prediction", key="start_button"):
            st.session_state.show_input_dialog = True
            st.rerun()

        st.header("Model Performance")
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

        tab_list = ["Confusion Matrix", "ROC Curve", "Feature Importance"]
        if show_charts: tab_list.append("Data Exploration")
        tabs = st.tabs(tab_list)
        
        with tabs[0]:
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
            st.pyplot(fig)

        with tabs[1]:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.legend(loc="lower right")
            st.pyplot(fig)

        with tabs[2]:
            st.subheader("Feature Importance")
            try:
                importance = model.named_steps['model'].feature_importances_
                feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)
                fig, ax = plt.subplots()
                ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                st.pyplot(fig)
            except:
                st.error("Could not retrieve feature importance.")

        if show_charts:
            with tabs[3]:
                selected_feature = st.selectbox("Select a feature:", feature_names)
                fig, ax = plt.subplots()
                ax.hist(data[data['Outcome'] == 0][selected_feature], alpha=0.5, label='Non-Diabetic', color='blue')
                ax.hist(data[data['Outcome'] == 1][selected_feature], alpha=0.5, label='Diabetic', color='red')
                ax.legend()
                st.pyplot(fig)
