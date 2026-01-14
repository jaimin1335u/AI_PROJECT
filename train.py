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

# --- Define global variables so they can be exported ---
model_pipeline = None
accuracy = 0
X_test, y_test = None, None
y_pred, y_pred_proba = None, None
feature_names = []
data = pd.DataFrame()

def load_and_train():
    global model_pipeline, accuracy, X_test, y_test, y_pred, y_pred_proba, feature_names, data
    
    # Load dataset
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: '{DATA_FILE}' not found.")
        return

    # Preprocessing
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # --- Load or Train ---
    if os.path.exists(MODEL_FILE):
        print(f"Loading saved model from '{MODEL_FILE}'...")
        try:
            model_pipeline = joblib.load(MODEL_FILE)
        except Exception as e:
            print(f"Failed to load model ({e}). Re-training...")
            model_pipeline = None

    if model_pipeline is None:
        print("Training a new model...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
        
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model_pipeline = grid_search.best_estimator_
        joblib.dump(model_pipeline, MODEL_FILE)

    # --- Run Performance check ---
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

# --- Initialize the data when the module is imported ---
load_and_train()

# --- Export Functions ---
def getModel(): return model_pipeline
def getAccuracy(): return accuracy
def get_test_data(): return X_test, y_test
def get_predictions(): return y_pred, y_pred_proba
def get_feature_names(): return feature_names
def get_full_data(): return data

if __name__ == "__main__":
    print(f"Model Training Script Completed. Accuracy: {accuracy:.4f}")
