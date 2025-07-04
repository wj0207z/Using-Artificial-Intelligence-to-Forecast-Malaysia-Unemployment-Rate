import streamlit as st
import pandas as pd
import plotly.express as px
import pmdarima as pm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# === Load and preprocess dataset ===
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")

# Convert and sort date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# Select unemployment rate
u_rate = df['u_rate'].dropna()

# === Build ARIMA model using auto_arima ===
model = pm.auto_arima(
    u_rate,
    seasonal=True,
    m=4,  # Quarterly data
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

# Forecast next 8 quarters
n_periods = 8
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# Forecast dates
last_date = u_rate.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')

# Forecast DataFrame
forecast_df = pd.DataFrame({
    "Forecast Date": forecast_dates,
    "Forecasted Unemployment Rate": forecast,
    "Lower CI": conf_int[:, 0],
    "Upper CI": conf_int[:, 1]
})

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🧮 Tab 2", "📊 Tab 3"])

# === 📈 Forecast Tab ===
with tab1:
    st.title("📈 ARIMA Forecast: Malaysia Quarterly Unemployment Rate")

    st.markdown("""
    This forecast uses an **ARIMA model with seasonality** to predict the unemployment rate over the next **2 years** (8 quarters).
    
    - The blue line shows the historical unemployment rate from your dataset.
    - The orange line shows the ARIMA-predicted rate.
    - The shaded area is the **95% confidence interval**, giving a range of possible future values.
    
    This forecast can help policymakers and analysts identify future labor market trends in Malaysia.
    """)

    # Combine historical + forecast
    actual_df = u_rate.reset_index().rename(columns={"date": "Date", "u_rate": "Unemployment Rate"})
    forecast_renamed = forecast_df.rename(columns={
        "Forecast Date": "Date",
        "Forecasted Unemployment Rate": "Unemployment Rate"
    })
    chart_df = pd.concat([actual_df, forecast_renamed], axis=0).reset_index(drop=True)

    # Chart
    fig = px.line(chart_df, x="Date", y="Unemployment Rate",
                  title="Actual vs Forecasted Unemployment Rate (Quarterly)",
                  markers=True)

    # Confidence interval shading
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Upper CI"],
                    mode="lines", name="Upper CI", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Lower CI"],
                    mode="lines", name="Lower CI", fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False)

    # Layout
    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title="Unemployment Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

# === Tab 2 (Empty Placeholder) ===
with tab2:
    st.title("🧮 Placeholder Tab 2")
    st.markdown("This space is reserved for future analysis or visualizations.")

# === Tab 3 (Empty Placeholder) ===
with tab3:
    st.title("📊 Placeholder Tab 3")
    st.markdown("This space is reserved for more data or interpretation.")

