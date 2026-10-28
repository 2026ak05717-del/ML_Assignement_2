import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
)
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def read_manifest(project_dir):
    manifest_path = Path(project_dir) / "model" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["features"]
    classes = manifest["classes"]
    models = manifest["models"]
    if (
        not isinstance(features, list)
        or len(features) != 16
        or not all(isinstance(value, str) and value for value in features)
        or len(set(features)) != 16
    ):
        raise ValueError("The manifest must contain 16 distinct feature names.")
    if (
        not isinstance(classes, list)
        or len(classes) != 7
        or not all(isinstance(value, str) and value for value in classes)
        or len(set(classes)) != 7
    ):
        raise ValueError("The manifest must contain seven distinct class labels.")
    if manifest["target"] != "Class" or "Class" in features:
        raise ValueError("The manifest target must be Class, separate from the features.")
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError("The manifest must identify all five required models.")
    for filename in models.values():
        if (
            not isinstance(filename, str)
            or "/" in filename
            or "\\" in filename
            or Path(filename).suffix != ".joblib"
        ):
            raise ValueError("Model filenames must be repository-local .joblib files.")
    return manifest


def read_test_csv(payload, manifest):
    if not payload or not payload.strip():
        raise ValueError("The uploaded CSV is empty.")
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, [])
        if not header:
            raise ValueError("The uploaded CSV has no column header.")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate column names are not allowed.")
        required = manifest["features"] + [manifest["target"]]
        missing = [column for column in required if column not in header]
        extra = [column for column in header if column not in required]
        if missing or extra:
            details = []
            if missing:
                details.append("Missing required columns: " + ", ".join(missing))
            if extra:
                details.append("Unexpected columns: " + ", ".join(extra))
            raise ValueError(". ".join(details))
        for row_number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} fields; "
                    f"expected {len(header)}."
                )
        frame = pd.read_csv(
            io.StringIO(text), dtype=str, keep_default_na=False, on_bad_lines="error"
        )
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"Cannot read the CSV: {error}") from error
    if frame.empty:
        raise ValueError("The uploaded CSV contains no test rows.")
    features = frame.loc[:, manifest["features"]].copy()
    for column in manifest["features"]:
        try:
            features[column] = features[column].astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Invalid numerical values in {column}; all feature values must be numeric."
            ) from error
        if not np.isfinite(features[column].to_numpy()).all():
            raise ValueError(
                f"Invalid numerical values in {column}; missing values and infinity "
                "are not allowed."
            )
    labels = frame[manifest["target"]]
    unknown = sorted(set(labels) - set(manifest["classes"]))
    if unknown:
        raise ValueError("Unknown class labels: " + ", ".join(repr(value) for value in unknown))
    return features, labels


def load_pipeline(project_dir, name, manifest):
    if name not in MODEL_NAMES:
        raise ValueError("Unknown model selection.")
    artifact = Path(project_dir) / "model" / manifest["models"][name]
    # Only repository artifacts are loaded; uploaded files are always CSV data.
    pipeline = joblib.load(artifact)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"{name} must be a complete fitted sklearn Pipeline.")
    check_is_fitted(pipeline)
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise TypeError(f"{name} must support predict_proba for multiclass AUC.")
    if set(pipeline.classes_) != set(manifest["classes"]):
        raise ValueError(f"{name} class labels do not match the manifest.")
    if list(pipeline.feature_names_in_) != manifest["features"]:
        raise ValueError(f"{name} feature names do not match the manifest.")
    return pipeline


def evaluate_pipeline(pipeline, features, actual_labels, classes):
    predicted_labels = np.asarray(pipeline.predict(features))
    probabilities = np.asarray(pipeline.predict_proba(features))
    model_labels = np.asarray(pipeline.classes_)
    actual_labels = np.asarray(actual_labels)
    if len(actual_labels) == 0:
        raise ValueError("At least one test row is required.")
    if set(model_labels) != set(classes):
        raise ValueError("The model must be trained on all seven manifest classes.")
    if not set(actual_labels).issubset(set(model_labels)):
        raise ValueError("The test data contains an unknown bean class.")
    if predicted_labels.shape != actual_labels.shape:
        raise ValueError("Prediction dimensions do not match the test labels.")
    if not set(predicted_labels).issubset(set(model_labels)):
        raise ValueError("The model returned an unknown class label.")
    if probabilities.shape != (len(actual_labels), len(model_labels)):
        raise ValueError("Probability dimensions do not match the test data.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain missing or infinite values.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Class probabilities for each row must sum to 1.")
    if set(actual_labels) == set(model_labels):
        auc = roc_auc_score(
            actual_labels, probabilities, labels=model_labels,
            multi_class="ovr", average="macro",
        )
        auc_note = "Macro one-vs-rest AUC calculated for all seven classes."
    else:
        auc = np.nan
        auc_note = (
            "Full multiclass AUC unavailable: the test data does not contain "
            "all seven classes."
        )
    averaging = {"labels": classes, "average": "macro", "zero_division": 0}
    metrics = {
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "AUC": auc,
        "Precision": precision_score(actual_labels, predicted_labels, **averaging),
        "Recall": recall_score(actual_labels, predicted_labels, **averaging),
        "F1": f1_score(actual_labels, predicted_labels, **averaging),
        "MCC": matthews_corrcoef(actual_labels, predicted_labels),
    }
    return metrics, predicted_labels, auc_note


def main():
    import streamlit as st

    st.set_page_config(page_title="Dry Bean Model Explorer", layout="wide")
    st.title("Dry Bean Model Explorer")
    st.caption("Evaluate saved pipelines on held-out CSV data. No training occurs in this app.")
    project_dir = Path(__file__).resolve().parent
    try:
        manifest = read_manifest(project_dir)
    except (OSError, ValueError, TypeError, KeyError) as error:
        st.error(f"Cannot read model/manifest.json: {error}")
        st.stop()

    with st.sidebar:
        st.header("Test your models")
        uploaded = st.file_uploader("Upload held-out test data", type=["csv"])
        selected = st.selectbox("Saved model", list(MODEL_NAMES))
        compare = st.checkbox("Compare all five models on these rows")
        st.caption("CSV only. Never upload joblib or pickle files from an untrusted source.")
    st.info(
        "Use test_data.csv or other genuinely held-out data. The CSV must contain "
        "the exact 16 feature columns and Class. Column order may differ."
    )
    st.caption(
        "Accuracy: fraction correct. Precision/Recall/F1: macro average over all "
        "seven manifest classes, zero_division=0. AUC: macro one-vs-rest from "
        "class probabilities. MCC: multiclass Matthews correlation coefficient."
    )
    if uploaded is None:
        st.write("Upload your test CSV using the sidebar to begin.")
        with st.expander("Required CSV columns"):
            st.code(", ".join(manifest["features"] + [manifest["target"]]))
        return
    try:
        features, actual = read_test_csv(uploaded.getvalue(), manifest)
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"CSV validation failed: {error}")
        st.stop()
    st.success(f"Validated {len(features):,} test rows and 16 features.")
    with st.expander("Preview validated data"):
        preview = features.copy()
        preview[manifest["target"]] = actual
        st.dataframe(preview.head(50), hide_index=True)

    try:
        pipeline = load_pipeline(project_dir, selected, manifest)
        metrics, predicted, auc_note = evaluate_pipeline(
            pipeline, features, actual, manifest["classes"]
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
        st.error(f"Cannot evaluate {selected}: {error}")
        st.stop()
    st.subheader(selected)
    for offset in (0, 3):
        for column, metric in zip(st.columns(3), METRIC_COLUMNS[offset:offset + 3]):
            value = metrics[metric]
            column.metric(metric, "Unavailable" if pd.isna(value) else f"{value:.4f}")
    if pd.isna(metrics["AUC"]):
        st.warning(auc_note)
    else:
        st.caption(auc_note)
    confusion_tab, report_tab, prediction_tab = st.tabs(
        ["Confusion matrix", "Classification report", "Predictions"]
    )
    with confusion_tab:
        counts = confusion_matrix(actual, predicted, labels=manifest["classes"])
        st.caption("Rows: actual class. Columns: predicted class.")
        st.dataframe(pd.DataFrame(
            counts,
            index=pd.Index(manifest["classes"], name="Actual"),
            columns=pd.Index(manifest["classes"], name="Predicted"),
        ))
    with report_tab:
        report = classification_report(
            actual, predicted, labels=manifest["classes"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4))
    with prediction_tab:
        results = features.copy()
        results["Actual Class"] = actual.to_numpy()
        results["Predicted Class"] = predicted
        st.dataframe(results, hide_index=True)
        st.download_button(
            "Download predictions", results.to_csv(index=False).encode("utf-8"),
            file_name="dry_bean_predictions.csv", mime="text/csv",
        )
    if compare:
        rows = []
        try:
            with st.spinner("Evaluating all five saved pipelines on the same rows..."):
                for name in MODEL_NAMES:
                    if name == selected:
                        model_metrics = metrics
                    else:
                        other = load_pipeline(project_dir, name, manifest)
                        model_metrics, _, _ = evaluate_pipeline(
                            other, features, actual, manifest["classes"]
                        )
                    rows.append({"ML Model Name": name, **model_metrics})
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError) as error:
            st.error(f"Model comparison failed for {name}: {error}")
            st.stop()
        table = pd.DataFrame(rows).set_index("ML Model Name")
        st.subheader("Same-data model comparison")
        st.dataframe(table[METRIC_COLUMNS].round(4))
        st.caption("AUC is shown as missing when a class is absent; the other metrics remain valid.")
        st.download_button(
            "Download comparison", table.to_csv().encode("utf-8"),
            file_name="uploaded_data_metrics.csv", mime="text/csv",
        )


if __name__ == "__main__":
    main()
