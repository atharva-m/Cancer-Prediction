# train.py — Train a Random Forest or SVC on the Breast Cancer dataset with optional top-15 feature subset

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "svc"], required=True,
                        help="rf = RandomForest, svc = Support Vector")
    parser.add_argument("--top15", action="store_true",
                        help="use only top 15 features")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="proportion of data for test set")
    parser.add_argument("--out_model", default="models/best_model.pkl",
                        help="where to save trained model")
    parser.add_argument("--out_report", default="reports/metrics.json",
                        help="where to save metrics JSON")
    args = parser.parse_args()

    # Prepare directories
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Load dataset
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Optional feature selection
    if args.top15:
        # These are the 15 most important features
        top15 = [
            "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
            "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
            "mean fractal dimension", "radius error", "texture error", "perimeter error",
            "area error", "smoothness error",
        ]
        X_train = X_train[top15]
        X_test  = X_test[top15]

    # Build pipeline and parameter grid
    if args.model == "rf":
        model = RandomForestClassifier(random_state=42)
        pipe = Pipeline([("scale", StandardScaler()), ("clf", model)])
        grid = {
            "clf__n_estimators": [100, 500, 1000],
            "clf__max_depth": [None, 5, 10]
        }
    else:
        model = SVC(probability=True, random_state=42)
        pipe = Pipeline([("scale", StandardScaler()), ("clf", model)])
        grid = {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"]
        }

    # Hyperparameter search
    search = GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Evaluate on test set
    y_pred = best.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "best_params": search.best_params_
    }

    # Print metrics
    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
    disp.plot(cmap="Blues")
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Feature Importances plot
    if args.model == "rf":
        feature_importances = best.named_steps['clf'].feature_importances_
        feature_names = X_train.columns
        # Create a DataFrame for plotting
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig("reports/feature_importances.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Save metrics and model
    Path(args.out_report).write_text(json.dumps(metrics, indent=2))
    joblib.dump(best, args.out_model)
    print(f"Saved model to {args.out_model} and report to {args.out_report}")

if __name__ == "__main__":
    main()
