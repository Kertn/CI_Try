"""
Example usage: python predict.py 5.1 3.5 1.4 0.2
"""
import sys, joblib
import numpy as np
from sklearn.datasets import load_iris
from pathlib import Path

MODEL_PATH = Path("artifacts/iris_dt.pkl")

def predict(sample):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train first.")
    clf = joblib.load(MODEL_PATH)
    pred = clf.predict(np.array(sample).reshape(1, -1))[0]
    return load_iris().target_names[pred]

if __name__ == "__main__":
    floats = list(map(float, sys.argv[1:]))
    if len(floats) != 4:
        sys.exit("Provide 4 numeric features.")
    print(predict(floats))