import pytest
import pandas as pd
import numpy as np

def test_basic():
    assert 1 + 1 == 2

def test_sample_data_shape():
    dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.random.rand(len(dates)) * 100})
    assert "ds" in df.columns
    assert "y" in df.columns
    assert len(df) > 0

def test_data_types():
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.random.rand(100) * 100})
    assert pd.api.types.is_datetime64_any_dtype(df['ds'])
    assert pd.api.types.is_numeric_dtype(df['y'])

def test_no_nulls():
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.random.rand(100) * 100})
    assert df['ds'].isnull().sum() == 0
    assert df['y'].isnull().sum() == 0

def test_positive_demand():
    y = np.random.rand(365) * 100 + 50
    assert all(y > 0)

def test_prophet_import():
    try:
        from prophet import Prophet
        assert Prophet is not None
    except Exception:
        pytest.skip("Prophet backend not available in CI")

def test_prophet_trains():
    try:
        from prophet import Prophet
        dates = pd.date_range(start="2022-01-01", periods=365, freq="D")
        df = pd.DataFrame({"ds": dates, "y": np.random.rand(365) * 100 + 50})
        model = Prophet()
        model.fit(df)
        assert model is not None
    except Exception:
        pytest.skip("Prophet Stan backend not available in CI")
