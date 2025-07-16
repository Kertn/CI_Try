"""
Load the trained model + hold-out set and report accuracy.
"""
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

MODEL_PATH = Path("artifacts/iris_dt.pkl")
TEST_PATH = Path("artifacts/test_data.joblib")

def evaluate():
    if not MODEL_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Model or test data not found. Run train.py first.")

    clf = joblib.load(MODEL_PATH)
    data = joblib.load(TEST_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy on hold-out set: {acc:.3f}")
    return acc

if __name__ == "__main__":
    evaluate()