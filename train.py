import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_FILE = "diabetes.csv"
MODEL_FILE = "diabetes_model.joblib"

# Global variables to be accessed by app.py
model_pipeline = None
accuracy = 0
X_test, y_test = None, None
y_pred, y_pred_proba = None, None
feature_names = []
data = pd.DataFrame()

def load_and_train():
    global model_pipeline, accuracy, X_test, y_test, y_pred, y_pred_proba, feature_names, data
    
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: '{DATA_FILE}' not found.")
        return

    # Preprocessing: Replace 0s with NaN in clinical features
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # --- Load or Train Logic ---
    if os.path.exists(MODEL_FILE):
        print(f"Loading saved model from '{MODEL_FILE}'...")
        try:
            loaded_model = joblib.load(MODEL_FILE)
            # Test prediction to catch Version/Attribute Errors immediately
            loaded_model.predict(X_test.head(1))
            model_pipeline = loaded_model
            print("Model loaded and verified successfully.")
        except Exception as e:
            print(f"Model incompatible or broken ({e}). Retraining...")
            os.remove(MODEL_FILE)
            model_pipeline = None

    if model_pipeline is None:
        print("Training new model (this may take a moment)...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
        
        # Simplified grid for faster cloud deployment
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.1],
            'model__max_depth': [3, 5]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model_pipeline = grid_search.best_estimator_
        joblib.dump(model_pipeline, MODEL_FILE)
        print("New model saved.")

    # Generate test results for the dashboard
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

# Run initialization once on import
load_and_train()

# Export functions for app.py
def getModel(): return model_pipeline
def getAccuracy(): return accuracy
def get_test_data(): return X_test, y_test
def get_predictions(): return y_pred, y_pred_proba
def get_feature_names(): return feature_names
def get_full_data(): return data
