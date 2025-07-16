"""
Train a Decision-Tree classifier on Iris and save it to disk.
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import json, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "iris_dt.pkl"
TEST_PATH = MODEL_DIR / "test_data.joblib"  \

def load_data():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train():
    X_train, X_test, y_train, y_test = load_data()
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    logging.info("Validation accuracy = %.4f", acc)

    # persist model and test set for downstream evaluation
    metrics = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DecisionTreeClassifier",
        "accuracy": acc,
        "features": list(X_train.columns),
    }
    METRICS_PATH = MODEL_DIR / "metrics.json"
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logging.info("Wrote metrics to %s", METRICS_PATH)

    return acc

if __name__ == "__main__":
    train()