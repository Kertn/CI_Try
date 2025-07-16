"""
Train a Decision-Tree classifier on Iris and save it to disk.
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

    # persist model and test set for downstream evaluation
    joblib.dump(clf, MODEL_PATH)
    joblib.dump({"X_test": X_test, "y_test": y_test}, TEST_PATH)
    print(f"Saved model to {MODEL_PATH} with accuracy={acc:.3f}")

    return acc

if __name__ == "__main__":
    train()