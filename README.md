# Machine Learning Assignment 2: Dry Bean Classification

## 1. Problem Statement

The objective is to classify dry beans into seven varieties using numerical measurements of their size and shape. Five classification models are compared using the same training and held-out test partitions. A Streamlit application loads the saved models and evaluates them on uploaded, labelled test data without retraining.

## 2. Project Links

- GitHub repository: https://github.com/2026ak05717-del/ML_Assignement_2
- Live Streamlit application: **Not added yet. Replace this text with the verified public application URL before submission.**

## 3. Dataset Description

The project uses the public UCI Dry Bean dataset.

- Dataset page: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- DOI: https://doi.org/10.24432/C50S4B
- Original instances: 13,611.
- Input features: 16 numerical measurements.
- Target column: `Class`.
- Classes: `BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER` and `SIRA`.
- Exact duplicate rows removed before splitting: 68.
- Unique rows used: 13,543.
- Training rows: 10,834.
- Held-out test rows: 2,709.

The reviewed dataset contains no missing feature values. The split is stratified, with 80% used for training and 20% for testing, using random state `42`.

The feature names are preserved exactly as supplied by UCI:

```text
Area, Perimeter, MajorAxisLength, MinorAxisLength,
AspectRation, Eccentricity, ConvexArea, EquivDiameter,
Extent, Solidity, roundness, Compactness,
ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
```

## 4. Models and Evaluation Method

The five models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine.

Each saved artifact is a fitted scikit-learn pipeline containing preprocessing and its classifier. Preprocessing is fitted on training data only. Standardization is used for the scale-sensitive Logistic Regression, KNN and SVM models.

Stratified three-fold cross-validation is performed on the training partition only. Macro F1 is the primary comparison metric because it gives each bean class equal weight. The finalized models are fitted on the full training partition and evaluated on the same held-out rows. The test results are not used for further parameter tuning.

### Metric definitions

- **Accuracy:** proportion of correctly classified examples.
- **AUC:** macro-averaged one-vs-rest ROC AUC, calculated from class probabilities in the fitted estimator's `classes_` order.
- **Precision, Recall and F1:** macro averages over the fixed seven class labels.
- **MCC:** multiclass Matthews correlation coefficient.

Undefined class-level Precision, Recall and F1 values use `zero_division=0`. If uploaded test data omits any of the seven classes, full seven-class macro AUC is reported as unavailable rather than replaced with zero. Other metrics are still calculated using their stated conventions.

## 5. Held-Out Test Results

All five models are evaluated on the same 2,709 held-out rows. Values below are rounded to four decimal places; `model_metrics.csv` retains the full precision.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.9011 | 0.9703 | 0.9160 | 0.9152 | 0.9155 | 0.8804 |
| KNN | 0.9136 | 0.9854 | 0.9268 | 0.9237 | 0.9250 | 0.8956 |
| Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| Support Vector Machine | **0.9210** | 0.9945 | **0.9342** | **0.9317** | **0.9328** | **0.9045** |

### Model observations

| Model | Observation |
|---|---|
| Logistic Regression | A strong baseline with macro F1 of 0.9302 and the highest AUC, 0.9947. Its classification scores are close to SVM, suggesting that a linear decision function is already useful for these features. |
| Decision Tree | Achieves macro F1 of 0.9155. It performs below Logistic Regression, KNN and SVM with the evaluated settings, but offers an interpretable threshold-based model. |
| KNN | Achieves macro F1 of 0.9250, making it a competitive distance-based baseline. Feature scaling is important because raw feature magnitudes differ substantially. |
| Naive Bayes | Has the lowest macro F1, 0.7669, despite an AUC of 0.9670. High probability-ranking performance does not necessarily produce equally strong final class predictions. Its conditional-independence and Gaussian assumptions may limit performance, but the scores alone do not prove the cause. |
| Support Vector Machine | Has the highest macro F1, accuracy, precision, recall and MCC among the five evaluated models. Its macro F1 is 0.9328, compared with 0.9302 for Logistic Regression. |

### Overall conclusion

SVM is the best-performing model in this experiment according to the preselected primary metric, macro F1. It also led the training-only cross-validation comparison. Logistic Regression is a close alternative and has the highest held-out AUC.

The SVM advantage over Logistic Regression is small: approximately 0.0026 in macro F1. These results describe the evaluated settings and split; they do not establish statistical significance or prove that SVM will outperform every alternative on future data.

## 6. Repository Structure

```text
ML_Assignement_2/
    README.md
    app.py
    requirements.txt
    ML_Assignment2_DryBean_Review.ipynb
    DryBean_Streamlit_Setup.ipynb
    test_data.csv
    model_metrics.csv
    model/
        manifest.json
        logistic_regression.joblib
        decision_tree.joblib
        knn.joblib
        naive_bayes.joblib
        svm.joblib
```

The manifest records the expected features, class labels, model filenames, dataset provenance, split settings and training package versions.

## 7. Running the Application

From the repository folder, install the application dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved pipelines should be loaded with package versions compatible with those used during training, as recorded in `model/manifest.json`. The application requirements are not a complete notebook environment; notebook execution also requires Jupyter, Matplotlib and openpyxl.

### Using the app

1. Upload the supplied `test_data.csv` using the sidebar.
2. Select one of the five saved models.
3. Review the six metrics, confusion matrix, classification report and predictions.
4. Optionally select the comparison checkbox to evaluate all five models on the same uploaded rows.
5. Download predictions or the comparison table if required.

The CSV must contain exactly the 16 feature columns and the `Class` target column. Column order may differ. Feature values must be numeric and finite, and target labels must match the seven supported classes. The app rejects missing or unexpected columns, duplicate headers, invalid feature values and unknown labels.

The app's confusion matrix shows raw counts. The notebook's displayed confusion matrices are normalized by actual-class counts. Both use actual classes as rows and predicted classes as columns.

The application loads only repository-provided model artifacts. It does not retrain models or accept uploaded joblib/pickle files.

## 8. Streamlit Community Cloud Deployment

Connect the GitHub repository to Streamlit Community Cloud, select the `main` branch and set the entry-point file to `app.py`. Ensure that `requirements.txt` and the complete `model` folder are committed.

After deployment, verify the public application with `test_data.csv` and add its working URL to Section 2 before submission.

## 9. References

- UCI Dry Bean dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- scikit-learn documentation: https://scikit-learn.org/stable/
- Streamlit documentation: https://docs.streamlit.io/
