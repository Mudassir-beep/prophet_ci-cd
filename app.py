import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ======================================
# Page Config
# ======================================
st.set_page_config(
    page_title="✈️ Travel Demand Forecaster",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Travel Demand Forecaster")
st.markdown("**Powered by Facebook Prophet** — Upload your travel data or use sample data to forecast future demand.")

# ======================================
# Sidebar Controls
# ======================================
st.sidebar.header("⚙️ Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Period (days)", 
    min_value=30, 
    max_value=365, 
    value=90,
    step=30
)

seasonality_mode = st.sidebar.selectbox(
    "Seasonality Mode",
    ["multiplicative", "additive"],
    index=0
)

yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)
daily_seasonality = st.sidebar.checkbox("Daily Seasonality", value=False)

# ======================================
# Data Input
# ======================================
st.subheader("📂 Data Input")

data_option = st.radio(
    "Choose data source:",
    ["Use Sample Travel Data", "Upload CSV File"],
    horizontal=True
)

def generate_sample_data():
    """Generate realistic travel demand sample data"""
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
    
    # Base demand with trend
    trend = np.linspace(100, 150, len(dates))
    
    # Yearly seasonality (peak in summer and holidays)
    yearly = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365 - np.pi/2)
    
    # Weekly seasonality (weekends higher)
    weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Random noise
    noise = np.random.normal(0, 8, len(dates))
    
    demand = trend + yearly + weekly + noise
    demand = np.maximum(demand, 10)  # no negative demand
    
    df = pd.DataFrame({"ds": dates, "y": demand.round(0)})
    return df

if data_option == "Use Sample Travel Data":
    df = generate_sample_data()
    st.success("✅ Sample travel demand data loaded! (Jan 2021 — Dec 2023)")
    st.dataframe(df.tail(10), use_container_width=True)

else:
    uploaded_file = st.file_uploader(
        "Upload CSV with 'ds' (date) and 'y' (value) columns",
        type=["csv"]
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['ds'] = pd.to_datetime(df['ds'])
        st.success(f"✅ Loaded {len(df)} rows of data!")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("👆 Please upload a CSV file with columns: **ds** (date) and **y** (value)")
        st.stop()

# ======================================
# Train & Forecast
# ======================================
st.divider()
st.subheader("🔮 Forecast")

if st.button("🚀 Generate Forecast", type="primary"):
    with st.spinner("Training Prophet model..."):
        
        # Train model
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        st.success(f"✅ Forecast generated for next {forecast_days} days!")

    # ======================================
    # Plot Results
    # ======================================
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Forecasted Demand",
            f"{forecast['yhat'].tail(forecast_days).mean():.0f}",
            delta=f"+{forecast['yhat'].tail(forecast_days).mean() - df['y'].mean():.0f} vs historical"
        )

    with col2:
        st.metric(
            "Peak Forecasted Demand",
            f"{forecast['yhat'].tail(forecast_days).max():.0f}",
            delta="Next period peak"
        )

    # Main forecast plot
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        name='Historical',
        mode='lines',
        line=dict(color='#1f77b4', width=1.5)
    ))

    # Forecast
    forecast_future = forecast[forecast['ds'] > df['ds'].max()]
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat'],
        name='Forecast',
        mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_future['ds'], forecast_future['ds'][::-1]]),
        y=pd.concat([forecast_future['yhat_upper'], forecast_future['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title="Travel Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Demand",
        hovermode='x unified',
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # Components plot
    st.subheader("📊 Forecast Components")
    
    col1, col2 = st.columns(2)

    with col1:
        # Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['trend'],
            mode='lines', line=dict(color='green', width=2)
        ))
        fig_trend.update_layout(title="Trend", height=300)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        # Yearly seasonality
        if yearly_seasonality:
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yearly'],
                mode='lines', line=dict(color='purple', width=2)
            ))
            fig_yearly.update_layout(title="Yearly Seasonality", height=300)
            st.plotly_chart(fig_yearly, use_container_width=True)

    # Forecast table
    st.subheader("📋 Forecast Data")
    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
    forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    forecast_display = forecast_display.round(2)
    st.dataframe(forecast_display, use_container_width=True)

    # Download button
    csv = forecast_display.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="travel_forecast.csv",
        mime="text/csv"
    )
