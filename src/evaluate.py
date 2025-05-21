#!/usr/bin/env python3
# evaluate.py — Load & evaluate a saved model.

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to .pkl model")
    p.add_argument("--test_size", type=float, default=0.2)
    args = p.parse_args()

    model = joblib.load(args.model)
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=args.test_size,
        stratify=data.target, random_state=42
    )

    preds = model.predict(X_test)
    print("accuracy:", accuracy_score(y_test, preds))
    print("precision:", precision_score(y_test, preds))
    print("recall:", recall_score(y_test, preds))
    print("f1:", f1_score(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
    disp.plot(cmap="Blues")
    plt.show()

if __name__ == "__main__":
    main()
