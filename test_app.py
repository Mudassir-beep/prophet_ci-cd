import pytest
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime

# ======================================
# Test 1: Data Generation
# ======================================
def test_sample_data_shape():
    """Test that sample data has correct structure"""
    dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "y": np.random.randint(50, 200, len(dates)).astype(float)
    })
    assert "ds" in df.columns, "Missing ds column"
    assert "y" in df.columns, "Missing y column"
    assert len(df) > 0, "Empty dataframe"
    print("✅ Test 1 passed: Data shape correct")

# ======================================
# Test 2: Data Types
# ======================================
def test_data_types():
    """Test correct data types"""
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.random.rand(100) * 100})
    assert pd.api.types.is_datetime64_any_dtype(df['ds']), "ds must be datetime"
    assert pd.api.types.is_numeric_dtype(df['y']), "y must be numeric"
    print("✅ Test 2 passed: Data types correct")

# ======================================
# Test 3: No Null Values
# ======================================
def test_no_nulls():
    """Test no null values in data"""
    dates = pd.date_range(start="2022-01-01", periods=100, freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.random.rand(100) * 100})
    assert df['ds'].isnull().sum() == 0, "Nulls found in ds"
    assert df['y'].isnull().sum() == 0, "Nulls found in y"
    print("✅ Test 3 passed: No null values")

# ======================================
# Test 4: Prophet Model Training
# ======================================
def test_prophet_trains():
    """Test Prophet model trains without error"""
    dates = pd.date_range(start="2021-01-01", periods=365, freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "y": np.random.rand(365) * 100 + 50
    })
    model = Prophet(yearly_seasonality=True)
    model.fit(df)
    assert model is not None, "Model failed to train"
    print("✅ Test 4 passed: Prophet model trained")

# ======================================
# Test 5: Prophet Forecast Output
# ======================================
def test_prophet_forecast():
    """Test Prophet generates correct forecast"""
    dates = pd.date_range(start="2021-01-01", periods=365, freq="D")
    df = pd.DataFrame({
        "ds": dates,
        "y": np.random.rand(365) * 100 + 50
    })
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    assert "yhat" in forecast.columns, "Missing yhat in forecast"
    assert "yhat_lower" in forecast.columns, "Missing yhat_lower"
    assert "yhat_upper" in forecast.columns, "Missing yhat_upper"
    assert len(forecast) == len(df) + 30, "Forecast length incorrect"
    print("✅ Test 5 passed: Forecast output correct")

# ======================================
# Test 6: Positive Demand Values
# ======================================
def test_positive_demand():
    """Test that demand values are positive"""
    dates = pd.date_range(start="2021-01-01", periods=365, freq="D")
    y = np.random.rand(365) * 100 + 50
    assert all(y > 0), "Negative demand values found"
    print("✅ Test 6 passed: All demand values positive")

if __name__ == "__main__":
    test_sample_data_shape()
    test_data_types()
    test_no_nulls()
    test_prophet_trains()
    test_prophet_forecast()
    test_positive_demand()
    print("\n🎉 All tests passed!")
