import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from prophet import Prophet
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

# ================================================
# CONFIGURATION
# ================================================
REFERENCE_DATA_PATH = "data/train_data.csv"   # original training data
LIVE_DATA_PATH      = "data/live_data.csv"    # recent production data
REPORT_PATH         = "reports/drift_report.html"
DRIFT_THRESHOLD     = 0.5                     # 50% features drifted = alert

# ================================================
# STEP 1 - Generate or Load Reference Data
# (In production this comes from S3)
# ================================================
def load_reference_data():
    """Load original training data used to train Prophet"""
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
            "ds":     dates,
            "y":      (trend + yearly + noise).clip(min=10),
            "month":  dates.month,
            "dayofweek": dates.dayofweek,
            "quarter": dates.quarter
        })
    return df

# ================================================
# STEP 2 - Generate or Load Live Production Data
# (In production this comes from S3 daily logs)
# ================================================
def load_live_data():
    """Load recent production data — last 30 days"""
    if os.path.exists(LIVE_DATA_PATH):
        print("Loading live data from file...")
        df = pd.read_csv(LIVE_DATA_PATH)
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        print("Generating sample live data (simulating drift)...")
        np.random.seed(99)
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")

        # Simulating drift — different distribution than training
        trend = np.linspace(200, 250, len(dates))   # higher demand (drift!)
        noise = np.random.normal(0, 20, len(dates))  # more noise (drift!)
        df = pd.DataFrame({
            "ds":        dates,
            "y":         (trend + noise).clip(min=10),
            "month":     dates.month,
            "dayofweek": dates.dayofweek,
            "quarter":   dates.quarter
        })
    return df

# ================================================
# STEP 3 - Run Prophet Forecast on Live Data
# ================================================
def get_forecast_metrics(reference_df, live_df):
    """Train Prophet on reference data and evaluate on live data"""
    print("Training Prophet model on reference data...")

    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(reference_df[['ds', 'y']])

    # Forecast for live period
    future = model.make_future_dataframe(
        periods=len(live_df),
        freq='D'
    )
    forecast = model.predict(future)
    forecast_live = forecast.tail(len(live_df))

    # Calculate metrics
    actual    = live_df['y'].values
    predicted = forecast_live['yhat'].values

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae  = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")

    return {
        "rmse": round(rmse, 4),
        "mae":  round(mae, 4),
        "mape": round(mape, 4)
    }

# ================================================
# STEP 4 - Run Evidently Drift Report
# ================================================
def run_drift_report(reference_df, live_df):
    """Run Evidently AI drift detection"""
    print("Running Evidently AI drift detection...")

    # Use only numeric feature columns
    feature_cols = ['y', 'month', 'dayofweek', 'quarter']

    reference_features = reference_df[feature_cols].copy()
    live_features      = live_df[feature_cols].copy()

    # Run Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric()
    ])

    report.run(
        reference_data=reference_features,
        current_data=live_features
    )

    # Extract drift results
    report_dict   = report.as_dict()
    drift_detected = report_dict['metrics'][1]['result']['dataset_drift']
    drift_share    = report_dict['metrics'][1]['result']['share_of_drifted_columns']

    # Save HTML report
    os.makedirs("reports", exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"Drift report saved to: {REPORT_PATH}")

    return {
        "drift_detected": drift_detected,
        "drift_share":    round(drift_share, 4),
        "report_path":    REPORT_PATH
    }

# ================================================
# STEP 5 - Send Alert (Email/Slack/SNS)
# ================================================
def send_alert(drift_results, forecast_metrics):
    """Send alert when drift is detected"""

    message = f"""
    ╔══════════════════════════════════════╗
    ║   🚨 DATA DRIFT ALERT DETECTED!     ║
    ╚══════════════════════════════════════╝

    Detected at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    📊 Drift Results:
    ├── Drift Detected: {drift_results['drift_detected']}
    └── Drifted Features: {drift_results['drift_share']*100:.1f}%

    📈 Prophet Forecast Metrics on Live Data:
    ├── RMSE: {forecast_metrics['rmse']}
    ├── MAE:  {forecast_metrics['mae']}
    └── MAPE: {forecast_metrics['mape']}%

    📋 Full Report: {drift_results['report_path']}

    ⚡ Action Required:
    1. Review drift report
    2. Check if retraining is needed
    3. Push updated code to GitHub
       to trigger CI/CD retraining
    """

    print(message)

    # ── In production on AWS uncomment this: ──
    # import boto3
    # sns = boto3.client('sns', region_name='us-east-1')
    # sns.publish(
    #     TopicArn='arn:aws:sns:us-east-1:YOUR_ID:drift-alerts',
    #     Subject='Data Drift Detected in Travel Prophet!',
    #     Message=message
    # )

    # ── For Slack notifications uncomment this: ──
    # import requests
    # requests.post(
    #     os.environ['SLACK_WEBHOOK_URL'],
    #     json={"text": message}
    # )

# ================================================
# MAIN — Run Full Monitoring Pipeline
# ================================================
def main():
    print("=" * 50)
    print("  Travel Prophet — Drift Monitor")
    print(f"  Running at: {datetime.now()}")
    print("=" * 50)

    # Load data
    reference_df = load_reference_data()
    live_df      = load_live_data()

    print(f"\nReference data: {len(reference_df)} rows")
    print(f"Live data:      {len(live_df)} rows")

    # Get forecast metrics
    print("\n--- Forecast Metrics ---")
    forecast_metrics = get_forecast_metrics(reference_df, live_df)

    # Run drift detection
    print("\n--- Drift Detection ---")
    drift_results = run_drift_report(reference_df, live_df)

    # Check results and alert
    print("\n--- Results ---")
    if drift_results['drift_detected']:
        print("🚨 DRIFT DETECTED — Sending alert!")
        send_alert(drift_results, forecast_metrics)
    else:
        print("✅ No drift detected — model is healthy!")
        print(f"   Drifted features: {drift_results['drift_share']*100:.1f}%")
        print(f"   RMSE: {forecast_metrics['rmse']}")

    # Save results summary
    summary = {
        "timestamp":        datetime.now().isoformat(),
        "drift_detected":   drift_results['drift_detected'],
        "drift_share":      drift_results['drift_share'],
        "rmse":             forecast_metrics['rmse'],
        "mae":              forecast_metrics['mae'],
        "mape":             forecast_metrics['mape']
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nMonitoring summary saved to reports/monitoring_summary.json")
    print("=" * 50)
    print("Monitoring complete!")

if __name__ == "__main__":
    main()
