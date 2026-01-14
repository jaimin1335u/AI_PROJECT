import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
import datetime
# No FPDF import needed
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from train import (
    getModel, getAccuracy, get_test_data, 
    get_predictions, get_feature_names, get_full_data
)

# --- SET PAGE CONFIG ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
    # Removed 'theme="light"' argument as it's not supported in your Streamlit version
)

# --- NEW: HTML Report Generation Function ---
def create_html_report(name, input_data, result, probability):
    """Generates an HTML report string."""
    
    # Simple styling
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
    
    # Result styling
    if result == "Diabetic":
        result_class = "result-diabetic"
    else:
        result_class = "result-nondiabetic"

    # Build patient details list
    details_html = "<ul>"
    for key, value in input_data.items():
        details_html += f"<li><strong>{key}:</strong> {value}</li>"
    details_html += "</ul>"

    # Build the full HTML
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
            professional medical advice, diagnosis, or treatment. Always seek the advice of your 
            physician or other qualified health provider.
        </p>
    </body>
    </html>
    """
    return html_content

# --- Background Image Functions ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """ Gets the Base64 string of a binary file. """
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), bin_file)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

def set_bg_from_local_file(file_name):
    """ Sets the background image from a local file. """
    base64_str = get_base64_of_bin_file(file_name)
    if base64_str is not None:
        # --- Auto-detect file type for correct MIME ---
        file_ext = os.path.splitext(file_name)[1].lower()
        mime_type = ""
        if file_ext == ".png":
            mime_type = "image/png"
        elif file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        else:
            st.warning(f"Unsupported background image format: {file_ext}. Please use PNG or JPG.")
            return
        # --- End of auto-detect ---

        page_bg_img = f"""
        <style>
        /* --- Main Background (Targeting the root app container) --- */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:{mime_type};base64,{base64_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* --- Remove default white boxes on main page --- */
        [data-testid="stAppViewContainer"] .main [data-testid="stBlock"] {{
            background-color: transparent;
            padding: 0;
            margin: 0;
        }}
        
        /* --- Style containers for INPUT and RESULT pages ONLY --- */
        .form-container {{
            background-color: rgba(255, 255, 255, 0.9); /* Slightly opaque white */
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #ddd; /* Light border for definition */
        }}

        /* --- Main Page Text Styling (Title, Headers, Metrics) --- */
        [data-testid="stAppViewContainer"] .main h1, 
        [data-testid="stAppViewContainer"] .main h2, 
        [data-testid="stAppViewContainer"] .main h3, 
        [data-testid="stAppViewContainer"] .main .stMetric {{
            color: #333333 !important; /* MOD: Dark grey text, FORCED */
            text-shadow: 1px 1px 3px #FFFFFF !important; /* MOD: Light shadow, FORCED */
            text-align: center;
        }}
        [data-testid="stAppViewContainer"] .main .stMetric [data-testid="stMetricLabel"], 
        [data-testid="stAppViewContainer"] .main .stMetric [data-testid="stMetricValue"] {{
            color: #333333 !important; /* MOD: Dark grey text, FORCED */
        }}
        [data-testid="stAppViewContainer"] .main h1 {{ padding-top: 20px; }}

        /* --- "Start Prediction" Button Styling --- */
        [data-testid="stAppViewContainer"] .main .stButton > button {{
            font-size: 1.25rem;
            padding: 15px 30px;
            width: 50%; /* Make button 50% width, not 100% */
            border-radius: 8px;
            margin: 10px auto; /* Center the button */
            display: block;
        }}
        
        /* --- Center-align the button block --- */
        [data-testid="stAppViewContainer"] .main [data-testid="stBlock"]:has(.stButton) {{
            text-align: center;
        }}

        /* --- Chart/Tabs Styling --- */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: rgba(255, 255, 255, 0.7); /* Light, semi-transparent tab bar */
            border-radius: 8px;
            padding: 3px;
        }}
        .stTabs [data-baseweb="tab"] {{
             background-color: transparent;
             color: black !important; /* MOD: Black text for tabs, FORCED */
             text-shadow: none !important; /* MOD: No shadow, FORCED */
        }}
        /* Keep chart plots on a light background for readability */
        .stpyplot {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
        }}

        /* --- NEW: Download Button Styling --- */
        [data-testid="stAppViewContainer"] .main [data-testid="stFileDownloadButton"] > button {{
            background-color: #4CAF50 !important; /* Darker green background */
            color: white !important; /* White text */
            border: 1px solid #388E3C !important; /* Darker green border */
            padding: 10px 20px !important;
            border-radius: 5px !important;
            font-size: 1rem !important;
        }}
        [data-testid="stAppViewContainer"] .main [data-testid="stFileDownloadButton"] > button:hover {{
            background-color: #388E3C !important; /* Even darker on hover */
            border-color: #1B5E20 !important;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        # We print a warning to the console if the file isn't found
        #st.warning(f"Could not find background image: {file_name}")
        newvaridkwhy = 0  

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

# --- NEW: Sidebar options ---
# Moved to the top of the main UI logic to be accessible by background function
st.sidebar.header("App Options")
theme_choice = st.sidebar.radio(
    "Select Theme",
    ["Custom Background", "Default Dark"],
    index=0  # Default to Custom Background
)
show_charts = st.sidebar.checkbox("Show Data Exploration", value=True)

# --- Set Background ---
# --- Pointing to your local 'bg1.jpg' file ---
# --- NEW: Only set background if user selected it ---
if theme_choice == "Custom Background":
    set_bg_from_local_file('bg1.jpg')
# If "Default Dark" is chosen, this function is skipped, 
# and the app will use the standard Streamlit theme.

# --- Main Page UI ---
if model is None or data.empty:
    st.error("Error: Model or data ('diabetes.csv') could not be loaded. Please check 'train.py' and ensure 'diabetes.csv' exists.")
else:
    
    # --- STATE 1: Show Input Form ---
    if st.session_state.show_input_dialog:
        # Wrap in a div with the 'form-container' class
        # This class only has an effect if the custom background is loaded
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("Patient Data")
            
        # Use columns for a slightly better layout
        col1, col2 = st.columns(2)
        
        with col1:
            preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
            gluc = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=0.1)
            bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
            skin = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        
        with col2:
            ins = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, step=0.1)
            bmi = st.number_input("BMI (kg/m^2)", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
            #dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4, step=0.01)
            age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, value=30.0, step=1.0)

            # Hardcoded dpf value as per your code
            dpf = 0.4 

        if st.button("Predict", key="predict_dialog_button"):
             st.session_state.input_data = {
                'Pregnancies': preg,
                'Glucose': gluc,
                'BloodPressure': bp,
                'SkinThickness': skin,
                'Insulin': ins,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age}
    
            # Create a DataFrame instead of a list
            # This ensures the Imputer knows which column is which
            input_df = pd.DataFrame([st.session_state.input_data])
    
            # Make prediction using the DataFrame
            prediction_val = model.predict(input_df)[0]
            probability_val = model.predict_proba(input_df)[0][1]
            
            # Prepare list for model
            input_list = [preg, gluc, bp, skin, ins, bmi, dpf, age]
            
            # Make prediction
            prediction_val = model.predict([input_list])[0]
            probability_val = model.predict_proba([input_list])[0][1]
            
            # Save results to session state
            st.session_state.prediction_result = "Diabetic" if prediction_val == 1 else "Non-Diabetic"
            st.session_state.prediction_proba = probability_val
            
            # Change state flags
            st.session_state.show_input_dialog = False
            st.session_state.show_result_dialog = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True) # Close the div

    # --- STATE 2: Show Result Page ---
    elif st.session_state.show_result_dialog:
        # Wrap in a div with the 'form-container' class
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
        
        # --- Download Section (Re-enabled with HTML) ---
        st.subheader("Download Report")
        person_name = st.text_input("Enter Person's Name for Report", key="person_name_input")
        
        html_data = ""
        download_disabled = True
        
        if person_name:
            # Generate the HTML report
            html_data = create_html_report(person_name, st.session_state.input_data, result, proba)
            download_disabled = False
        else:
            st.caption("Please enter a name to enable download.")
        
        # --- Page Buttons (Changed back to 3 columns) ---
        col_back, col_dl, col_end = st.columns(3)
        
        with col_back:
            if st.button("Go Back"):
                st.session_state.show_result_dialog = False
                st.session_state.show_input_dialog = True
                st.rerun()
        
        with col_dl:
            # Download button now serves an HTML file
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
        st.markdown('</div>', unsafe_allow_html=True) # Close the div

    # --- STATE 3: Show Main Page (Default) ---
    else:
        st.title("Diabetes Prediction App")
        
        if st.button("Start Prediction", key="start_button"):
            st.session_state.show_input_dialog = True
            st.session_state.show_result_dialog = False
            st.rerun()

        # --- Main Page Visualizations (Displayed below the button) ---
        st.header("Model Performance")
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

        # --- Sidebar options are already defined above ---
        
        # --- Define tabs ---
        tab_list = [
            "Confusion Matrix", 
            "ROC Curve", 
            "Feature Importance"
        ]
        if show_charts:
            tab_list.append("Data Exploration")
            
        tabs = st.tabs(tab_list)
        
        # --- Populate tabs ---
        with tabs[0]:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        with tabs[1]:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            st.pyplot(fig)

        with tabs[2]:
            st.subheader("Feature Importance")
            try:
                # Access the feature importances from the 'model' step of the pipeline
                importance = model.named_steps['model'].feature_importances_
            except AttributeError:
                st.error("Could not retrieve feature importance. Is the model a pipeline?")
                importance = [0] * len(feature_names)
                
            feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            feature_importance = feature_importance.sort_values(by='Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis([i/float(len(feature_importance['Importance'])) for i in range(len(feature_importance['Importance']))])
            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance Value")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            st.pyplot(fig)

        # --- NEW: Conditional Tab ---
        if show_charts:
            with tabs[3]:
                st.header("Explore the Data")
                selected_feature = st.selectbox("Select a feature to visualize:", feature_names)
                
                st.subheader(f"Distribution of {selected_feature}")
                fig, ax = plt.subplots()
                data_0 = data[data['Outcome'] == 0][selected_feature]
                data_1 = data[data['Outcome'] == 1][selected_feature]
                ax.hist(data_0, bins=30, alpha=0.7, label='Non-Diabetic (0)', color='blue')
                ax.hist(data_1, bins=30, alpha=0.7, label='Diabetic (1)', color='red')
                ax.set_title(f"Distribution of {selected_feature} by Outcome")
                ax.set_xlabel(selected_feature)
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)

                st.subheader("Feature Relationships (Scatter Plot)")
                col1_sc, col2_sc = st.columns(2)
                with col1_sc:
                    x_feat = st.selectbox("Select X-axis feature:", feature_names, index=feature_names.index('Glucose'))
                with col2_sc:
                    y_feat = st.selectbox("Select Y-axis feature:", feature_names, index=feature_names.index('BMI'))

                fig, ax = plt.subplots()
                data_0_scatter = data[data['Outcome'] == 0]
                data_1_scatter = data[data['Outcome'] == 1]
                ax.scatter(data_0_scatter[x_feat], data_0_scatter[y_feat], alpha=0.6, label='Non-Diabetic (0)', color='blue')
                ax.scatter(data_1_scatter[x_feat], data_1_scatter[y_feat], alpha=0.6, label='Diabetic (1)', color='red')
                ax.set_title(f"{y_feat} vs. {x_feat}")
                ax.set_xlabel(x_feat)
                ax.set_ylabel(y_feat)
                ax.legend()

                st.pyplot(fig)


