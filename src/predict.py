#!/usr/bin/env python3
# predict.py — Batch inference on new CSV data.

import argparse
from pathlib import Path

import joblib
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to .pkl model")
    p.add_argument("--input", required=True, help="CSV file with features")
    p.add_argument("--output", default="predictions.csv")
    args = p.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    df["prediction"] = model.predict(df)
    Path(args.output).write_text(df.to_csv(index=False))
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
