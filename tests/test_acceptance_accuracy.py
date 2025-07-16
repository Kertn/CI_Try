import train
import evaluate
import pytest

THRESHOLD = 0.9  # business requirement

@pytest.fixture(scope="session", autouse=True)
def _run_training_once():
    # Train model once at session start if not present
    if not train.MODEL_PATH.exists():
        train.train()

def test_accuracy_meets_threshold():
    acc = evaluate.evaluate()
    assert acc >= THRESHOLD, f"Accuracy {acc:.3f} below threshold {THRESHOLD}"