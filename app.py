import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
MODEL_METADATA_PATH = ROOT / "model_metadata.json"
TRAINING_METRICS_PATH = ROOT / "training_metrics.json"


@st.cache_resource
def load_model(model_name: str):
    model_filename = model_name.lower().replace(" ", "_") + ".joblib"
    return joblib.load(MODEL_DIR / model_filename)


@st.cache_data
def load_metadata():
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_metrics_table():
    return pd.read_json(TRAINING_METRICS_PATH)


@st.cache_data
def load_test_data():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError("test_data.csv was not found. Run python train_models.py first.")
    return pd.read_csv(TEST_DATA_PATH)


def model_evaluation_metrics(model_name: str, X: pd.DataFrame, y: pd.Series):
    model = load_model(model_name)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob, multi_class="ovo", average="macro", labels=model.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


st.set_page_config(page_title="Dry Bean Model Evaluation", page_icon="🌱", layout="wide")
st.title("Dry Bean Variety Classification")
st.caption("Deployment-ready Streamlit app that evaluates the saved models in this repository.")

if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
    st.error("Model files were not found. Run python train_models.py before deploying this app.")
    st.stop()

if not MODEL_METADATA_PATH.exists() or not TRAINING_METRICS_PATH.exists() or not TEST_DATA_PATH.exists():
    st.error("Missing required files: model_metadata.json, training_metrics.json, or test_data.csv.")
    st.stop()

metadata = load_metadata()
metrics_table = load_metrics_table()
model_names = sorted([p.stem.replace("_", " ").title() for p in MODEL_DIR.glob("*.joblib")])
selected_model = st.selectbox("Choose a trained model", model_names)

test_df = load_test_data()
X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

st.subheader(f"Model summary: {selected_model}")
st.dataframe(metrics_table, use_container_width=True)

selected_metrics = model_evaluation_metrics(selected_model, X_test, y_test)

st.subheader("Evaluation on held-out test data")
st.dataframe(pd.DataFrame([selected_metrics]).T.rename_axis("Metric").reset_index().rename(columns={"index": "Metric", 0: "Value"}), use_container_width=True)

st.subheader("Class distribution in the test set")
st.bar_chart(test_df["Class"].value_counts())

st.sidebar.header("About")
st.sidebar.write(f"Target column: {metadata.get('target_column', 'Class')}")
st.sidebar.write(f"Feature columns: {len(X_test.columns)}")
st.sidebar.write("This app uses the model files saved in the repository and does not retrain from uploaded data.")