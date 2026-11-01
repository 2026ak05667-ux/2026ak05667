"""Train the five required classifiers and save models plus held-out test data."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dry_bean_dataset.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Class"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset(path=DATA_PATH):
    """Load the CSV and validate its target and numeric feature columns."""
    if not path.exists():
        raise FileNotFoundError("Put the downloaded dataset at dry_bean_dataset.csv.")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Columns: {df.columns.tolist()}")
    df = df.dropna(subset=[TARGET]).copy()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    if X.isnull().any().any():
        raise ValueError("Missing feature values found; inspect and clean the CSV.")
    if len(X.select_dtypes(exclude="number").columns):
        raise ValueError("This Dry Bean workflow expects numeric input features.")
    return df, X, y


def build_classifiers():
    """Create the five models; scaling is fitted only on training data via Pipeline."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=2),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate_model(model, X_test, y_test):
    """Return the six assignment metrics; multiclass AUC uses macro one-vs-one."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob, multi_class="ovo", average="macro",
                             labels=model.classes_),
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }


def main():
    """Train, evaluate, save fitted models, and export test/report files."""
    MODEL_DIR.mkdir(exist_ok=True)
    df, X, y = load_dataset()
    print(f"Dataset shape: {df.shape}; input features: {X.shape[1]}; classes: {y.nunique()}")
    print("\nClass distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rows = []
    for name, model in build_classifiers().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        scores = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **scores})
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)
        print({key: round(value, 4) for key, value in scores.items()})

    # Export labelled held-out rows; Streamlit uses these to assess performance.
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "training_metrics.csv", index=False)
    results.to_json(ROOT / "training_metrics.json", orient="records", indent=2)
    metadata = {
        "target_column": TARGET,
        "feature_columns": X.columns.tolist(),
        "class_labels": sorted(y.astype(str).unique().tolist()),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "auc": "macro-averaged one-vs-one",
        "precision_recall_f1": "macro average",
    }
    (ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nComparison table:\n", results.round(4).to_string(index=False))
    print("\nCreated model/, test_data.csv, training_metrics.csv/json, model_metadata.json.")


if __name__ == "__main__":
    main()
