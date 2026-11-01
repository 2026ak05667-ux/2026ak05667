# Dry Bean Variety Classification — Starter Project

## a. Problem statement
Compare five classifiers for predicting dry bean variety from numerical measurements.

## b. Dataset description
Dataset: UCI Dry Bean Dataset. Confirm details against the official UCI page in your report. The target column used by this project is `Class`.

## c. GitHub Repository Link
Replace this with your own public repository URL after publishing and testing your repository.

## d. Models used
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

Run `python train_models.py` to create `training_metrics.csv` and `training_metrics.json`. Transfer the actual measured values into the assignment's comparison table.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9956 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8884 | 0.9548 | 0.9038 | 0.9041 | 0.9036 | 0.8652 |
| KNN | 0.9166 | 0.9842 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9690 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9957 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

### Observations
The Support Vector Machine produced the strongest results overall, with the highest Accuracy, AUC, Precision, Recall, F1, and MCC. Logistic Regression was a close second across every metric and is the strongest simpler interpretable baseline. KNN also performed well but was slightly behind the top two models. Decision Tree was clearly weaker than SVM, Logistic Regression, and KNN. Gaussian Naive Bayes had the lowest classification performance, although it retained a high AUC.

### Overall winner
The Support Vector Machine is the overall winner based on all six measured metrics. It achieved Accuracy 0.9221, macro-averaged one-vs-one AUC 0.9957, and the highest Precision, Recall, F1, and MCC. These balanced metrics make it preferable because correct classification across all seven bean classes is important and the macro averages give each class equal weight.

## Run locally
1. Install: `python -m pip install -r requirements.txt`
2. Save the downloaded dataset as `dry_bean_dataset.csv` in this folder; target column must be `Class`.
3. Train models: `python train_models.py`
4. Launch app: `streamlit run app.py`
5. Upload the generated `test_data.csv` into the app.

The app loads saved models and evaluates uploaded labelled test data; it does not retrain on uploaded data. Customize, run, verify, and understand the code before submission.
