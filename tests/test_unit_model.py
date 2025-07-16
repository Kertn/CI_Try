import joblib, os
from train import train, MODEL_PATH

def test_model_training_and_persistence(tmp_path, monkeypatch):
    # train() writes to artifacts; redirect to temp dir during test
    monkeypatch.setattr("train.MODEL_DIR", tmp_path)
    monkeypatch.setattr("train.MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr("train.TEST_PATH", tmp_path / "test.joblib")

    acc = train()
    assert acc > 0.8, "Baseline accuracy too low"
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "test.joblib").exists()