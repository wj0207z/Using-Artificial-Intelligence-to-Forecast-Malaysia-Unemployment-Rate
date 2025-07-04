import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === Load and preprocess dataset ===
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# === Select metric to forecast ===
metrics = {
    "Labour Force": "lf",
    "Employed": "lf_employed",
    "Unemployed": "lf_unemployed",
    "Outside Labour Force": "lf_outside",
    "Participation Rate (%)": "p_rate",
    "Employment to Population Ratio (%)": "ep_ratio",
    "Unemployment Rate (%)": "u_rate"
}
selected_metric_label = st.selectbox("Choose a metric to forecast:", list(metrics.keys()), index=6)
selected_metric = metrics[selected_metric_label]

# Select series
y = df[selected_metric].dropna()
series = y.copy()

# === Feature Engineering: Lag features for time series ===
def create_lagged_features(series, n_lags=8):
    df_lag = pd.DataFrame(series)
    for lag in range(1, n_lags+1):
        df_lag[f"lag_{lag}"] = df_lag[selected_metric].shift(lag)
    df_lag = df_lag.dropna()
    return df_lag

n_lags = 8
lagged = create_lagged_features(series, n_lags)

X = lagged.drop(columns=[selected_metric])
y_lagged = lagged[selected_metric]

# Train/test split (use all for training, as we want to forecast future)
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y_lagged)

# === Forecast next 8 quarters ===
last_known = list(series[-n_lags:])
forecast = []
for i in range(8):
    input_feats = np.array(last_known[-n_lags:]).reshape(1, -1)
    pred = rf.predict(input_feats)[0]
    forecast.append(pred)
    last_known.append(pred)

# Forecast dates
last_date = series.index[-1]
if not isinstance(last_date, pd.Timestamp):
    last_date = pd.to_datetime(last_date)
forecast_dates = pd.date_range(start=last_date + pd.offsets.QuarterEnd(), periods=8, freq='Q')

# Forecast DataFrame
forecast_df = pd.DataFrame({
    "Forecast Date": forecast_dates,
    f"Forecasted {selected_metric_label}": forecast
})

# === Tabs ===
tab1, tab2, tab3 = st.tabs([f"🌲 Forecast ({selected_metric_label})", "🧮 Tab 2", "📊 Tab 3"])

# === 🌲 Forecast Tab ===
with tab1:
    st.title(f"🌲 Random Forest Forecast: Malaysia Quarterly {selected_metric_label}")

    st.markdown(f"""
    This forecast uses a **Random Forest Regressor** to predict the {selected_metric_label.lower()} over the next **2 years** (8 quarters).
    
    - The blue line shows the historical {selected_metric_label.lower()} from your dataset.
    - The green line shows the Random Forest-predicted value.
    
    This forecast can help policymakers and analysts identify future trends in Malaysia's labour market.
    """)

    # Combine historical + forecast
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    forecast_renamed = forecast_df.rename(columns={
        "Forecast Date": "Date",
        f"Forecasted {selected_metric_label}": selected_metric_label
    })
    chart_df = pd.concat([actual_df, forecast_renamed], axis=0).reset_index(drop=True)

    # Chart
    fig = px.line(chart_df, x="Date", y=selected_metric_label,
                  title=f"Actual vs Forecasted {selected_metric_label} (Quarterly)",
                  markers=True)
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df[f"Forecasted {selected_metric_label}"],
                    mode="lines+markers", name="Forecast", line=dict(color="green", width=2))

    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title=selected_metric_label,
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