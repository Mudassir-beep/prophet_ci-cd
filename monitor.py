import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ================================================
# CONFIGURATION
# ================================================
REFERENCE_DATA_PATH = "data/train_data.csv"
LIVE_DATA_PATH      = "data/live_data.csv"
REPORT_PATH         = "reports/drift_report.html"

# ================================================
# STEP 1 - Load Reference Data
# ================================================
def load_reference_data():
    if os.path.exists(REFERENCE_DATA_PATH):
        print("Loading reference data from file...")
        df = pd.read_csv(REFERENCE_DATA_PATH)
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        print("Generating sample reference data...")
        np.random.seed(42)
        dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
        trend = np.linspace(100, 150, len(dates))
        yearly = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        noise = np.random.normal(0, 8, len(dates))
        df = pd.DataFrame({
            "ds":        dates,
            "y":         (trend + yearly + noise).clip(min=10),
            "month":     dates.month,
            "dayofweek": dates.dayofweek,
            "quarter":   dates.quarter
        })
    return df

# ================================================
# STEP 2 - Load Live Data
# ================================================
def load_live_data():
    if os.path.exists(LIVE_DATA_PATH):
        print("Loading live data from file...")
        df = pd.read_csv(LIVE_DATA_PATH)
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        print("Generating sample live data (simulating drift)...")
        np.random.seed(99)
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        trend = np.linspace(200, 250, len(dates))
        noise = np.random.normal(0, 20, len(dates))
        df = pd.DataFrame({
            "ds":        dates,
            "y":         (trend + noise).clip(min=10),
            "month":     dates.month,
            "dayofweek": dates.dayofweek,
            "quarter":   dates.quarter
        })
    return df

# ================================================
# STEP 3 - Run Drift Detection
# ================================================
def run_drift_detection(reference_df, live_df):
    print("Running drift detection...")

    feature_cols = ['y', 'month', 'dayofweek', 'quarter']
    ref  = reference_df[feature_cols].copy()
    curr = live_df[feature_cols].copy()

    drift_results = {}
    drifted_count = 0

    for col in feature_cols:
        ref_mean  = ref[col].mean()
        curr_mean = curr[col].mean()
        ref_std   = ref[col].std()

        # Simple z-score drift detection
        if ref_std > 0:
            z_score = abs(curr_mean - ref_mean) / ref_std
            drifted = z_score > 2.0
        else:
            drifted = False

        drift_results[col] = {
            "ref_mean":  round(ref_mean, 4),
            "curr_mean": round(curr_mean, 4),
            "drifted":   drifted
        }

        if drifted:
            drifted_count += 1
            print(f"  DRIFT in {col}: ref={ref_mean:.2f} curr={curr_mean:.2f}")
        else:
            print(f"  OK    {col}: ref={ref_mean:.2f} curr={curr_mean:.2f}")

    drift_share    = drifted_count / len(feature_cols)
    drift_detected = drift_share > 0.5

    return {
        "drift_detected": drift_detected,
        "drift_share":    round(drift_share, 4),
        "feature_drift":  drift_results
    }

# ================================================
# STEP 4 - Get Forecast Metrics
# ================================================
def get_forecast_metrics(reference_df, live_df):
    print("Calculating forecast metrics...")
    try:
        from prophet import Prophet
        model = Prophet(
            seasonality_mode="multiplicative",
            yearly_seasonality=True
        )
        model.fit(reference_df[['ds', 'y']])
        future   = model.make_future_dataframe(periods=len(live_df), freq='D')
        forecast = model.predict(future)
        forecast_live = forecast.tail(len(live_df))

        actual    = live_df['y'].values
        predicted = forecast_live['yhat'].values
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae  = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MAPE: {mape:.4f}%")

        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 4)}

    except Exception as e:
        print(f"  Prophet metrics skipped: {e}")
        return {"rmse": None, "mae": None, "mape": None}

# ================================================
# STEP 5 - Save HTML Report
# ================================================
def save_report(drift_results, forecast_metrics):
    os.makedirs("reports", exist_ok=True)

    drift_color = "red" if drift_results['drift_detected'] else "green"
    drift_text  = "DRIFT DETECTED" if drift_results['drift_detected'] else "NO DRIFT"

    rows = ""
    for col, result in drift_results['feature_drift'].items():
        color = "red" if result['drifted'] else "green"
        rows += f"""
        <tr>
            <td>{col}</td>
            <td>{result['ref_mean']}</td>
            <td>{result['curr_mean']}</td>
            <td style='color:{color}'>{result['drifted']}</td>
        </tr>"""

    html = f"""
    <html>
    <head>
        <title>Drift Report</title>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .status {{ font-size: 24px; font-weight: bold; color: {drift_color}; }}
        </style>
    </head>
    <body>
        <h1>Travel Prophet — Drift Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Overall Status</h2>
        <p class='status'>{drift_text}</p>
        <p>Drifted Features: {drift_results['drift_share']*100:.1f}%</p>

        <h2>Forecast Metrics on Live Data</h2>
        <p>RMSE: {forecast_metrics['rmse']}</p>
        <p>MAE:  {forecast_metrics['mae']}</p>
        <p>MAPE: {forecast_metrics['mape']}%</p>

        <h2>Feature Drift Details</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Reference Mean</th>
                <th>Current Mean</th>
                <th>Drifted</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """

    with open(REPORT_PATH, "w") as f:
        f.write(html)
    print(f"Report saved to {REPORT_PATH}")

# ================================================
# STEP 6 - Send Alert
# ================================================
def send_alert(drift_results, forecast_metrics):
    message = f"""
    DRIFT ALERT!
    Time:     {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Drift:    {drift_results['drift_share']*100:.1f}% features drifted
    RMSE:     {forecast_metrics['rmse']}
    Action:   Retrain model and redeploy!
    """
    print(message)

# ================================================
# MAIN
# ================================================
def main():
    print("=" * 50)
    print("  Travel Prophet Drift Monitor")
    print(f"  {datetime.now()}")
    print("=" * 50)

    reference_df     = load_reference_data()
    live_df          = load_live_data()

    print(f"\nReference: {len(reference_df)} rows")
    print(f"Live:      {len(live_df)} rows")

    print("\n--- Forecast Metrics ---")
    forecast_metrics = get_forecast_metrics(reference_df, live_df)

    print("\n--- Drift Detection ---")
    drift_results    = run_drift_detection(reference_df, live_df)

    print("\n--- Results ---")
    if drift_results['drift_detected']:
        print("DRIFT DETECTED!")
        send_alert(drift_results, forecast_metrics)
    else:
        print("No drift detected - model healthy!")

    save_report(drift_results, forecast_metrics)

    summary = {
        "timestamp":      datetime.now().isoformat(),
        "drift_detected": drift_results['drift_detected'],
        "drift_share":    drift_results['drift_share'],
        "rmse":           forecast_metrics['rmse'],
        "mae":            forecast_metrics['mae'],
        "mape":           forecast_metrics['mape']
    }

    with open("reports/monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nMonitoring complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
