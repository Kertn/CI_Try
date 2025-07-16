from train import load_data
import pandas as pd
import pytest

def test_load_data_shapes():
    X_train, X_test, y_train, y_test = load_data()
    # 80/20 split on 150 samples -> 120 / 30
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert isinstance(X_train, pd.DataFrame)

def test_stratification():
    _, X_test, _, y_test = load_data()
    # Ensure all three classes appear in test set
    assert set(y_test.unique()) == {0, 1, 2}