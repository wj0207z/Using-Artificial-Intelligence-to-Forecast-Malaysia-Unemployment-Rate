import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

# === Load and preprocess dataset ===
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# === Metric selection ===
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

# === Select and plot time series ===
series = df[selected_metric].dropna()
st.subheader("📊 Historical Time Series")
st.line_chart(series)

# === Train/Test Split Configuration ===
total_points = len(series)
test_size = st.slider("Test Set Size (quarters):", min_value=4, max_value=min(20, total_points//4), value=8, step=1)
train_size = total_points - test_size

st.markdown(f"**Training Set:** {train_size} quarters | **Test Set:** {test_size} quarters")

# === LSTM Configuration ===
st.markdown("### ⚙️ LSTM Model Settings")
col1, col2 = st.columns(2)
with col1:
    n_lags = st.slider("Number of Lag Quarters (input window):", min_value=4, max_value=16, value=8, step=1)
    n_epochs = st.slider("Epochs:", min_value=10, max_value=200, value=50, step=10)
with col2:
    n_units = st.slider("LSTM Units:", min_value=8, max_value=128, value=32, step=8)
    dropout = st.slider("Dropout (regularization):", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

# === Prepare Data for LSTM ===
def create_lagged_sequences(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Scale data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

# Split
train_scaled = series_scaled[:train_size]
test_scaled = series_scaled[train_size-n_lags:]  # include lags from end of train

# Create sequences
X_train, y_train = create_lagged_sequences(train_scaled, n_lags)
X_test, y_test = create_lagged_sequences(test_scaled, n_lags)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Build and Train LSTM Model ===
model = Sequential()
model.add(LSTM(n_units, input_shape=(n_lags, 1)))
model.add(Dropout(dropout))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=n_epochs,
    batch_size=8,
    validation_split=0.2,
    callbacks=[es],
    verbose=0
)

# === Forecast on Test Set ===
y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred)

# === Forecast Dates ===
forecast_dates = series.index[train_size:]
forecast_dates = pd.to_datetime(forecast_dates).strftime('%Y-%m-%d')

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs([
    f"🔮 LSTM Forecast ({selected_metric_label})",
    "📊 Model Diagnostics",
    "📋 Model Summary",
    "🧠 LSTM Explanation"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"🔮 LSTM Forecast for {selected_metric_label}")
    
    # Plot actual vs forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=y_test_actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=y_pred,
        mode='lines+markers',
        name='LSTM Forecast',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(title="LSTM Forecast vs Actual", xaxis_title="Date", yaxis_title=selected_metric_label)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("MAPE (%)", f"{mape:.2f}")
    with col4:
        st.metric("R²", f"{r2:.3f}")
    
    # Download forecast
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Actual": y_test_actual,
        "LSTM Forecast": y_pred
    })
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.strftime('%Y-%m-%d')
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "lstm_forecast.csv", "text/csv")

# === Tab 2: Model Diagnostics ===
with tab2:
    st.title("📊 Model Diagnostics")
    
    # Loss curve
    st.subheader("Training & Validation Loss Curve")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss'))
    fig_loss.update_layout(title="Loss Curve", xaxis_title="Epoch", yaxis_title="MSE Loss")
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Residuals
    st.subheader("Residuals (Actual - Forecast)")
    residuals = y_test_actual - y_pred
    fig_resid = px.line(x=forecast_dates, y=residuals, labels={'x': 'Date', 'y': 'Residuals'}, title="Residuals Over Time")
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)
    
    # Residual distribution
    st.subheader("Residual Distribution")
    fig_hist = px.histogram(residuals, nbins=20, title="Residual Distribution")
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)

# === Tab 3: Model Summary ===
with tab3:
    st.title("📋 LSTM Model Summary")
    st.subheader("Model Architecture")
    st.text(model.summary())
    
    st.subheader("Model Parameters")
    st.markdown(f"""
    - **Input Window (Lags):** {n_lags}
    - **LSTM Units:** {n_units}
    - **Dropout:** {dropout}
    - **Epochs:** {len(history.history['loss'])}
    - **Batch Size:** 8
    - **Early Stopping:** Patience 10
    """)
    
    st.subheader("Performance Metrics")
    st.markdown(f"""
    - **RMSE:** {rmse:.2f}
    - **MAE:** {mae:.2f}
    - **MAPE:** {mape:.2f}%
    - **R²:** {r2:.3f}
    """)

# === Tab 4: LSTM Explanation ===
with tab4:
    st.title("🧠 Understanding LSTM for Time Series Forecasting")
    st.markdown("""
    **LSTM (Long Short-Term Memory)** networks are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data.

    ### How LSTM Works:
    - **Input Window:** Uses a sliding window of past values (lags) to predict the next value.
    - **Memory Cells:** LSTM cells can remember information for long periods, making them ideal for time series.
    - **Non-linear Modeling:** Can capture complex, non-linear relationships in the data.
    - **Regularization:** Dropout helps prevent overfitting.
    - **Early Stopping:** Stops training when validation loss stops improving.

    ### Why LSTM for Unemployment Forecasting?
    - **Captures Trends & Seasonality:** Learns both short-term and long-term patterns.
    - **Handles Non-linearity:** Useful for economic data with complex dynamics.
    - **Flexible:** Can be extended to multivariate or multi-step forecasting.

    ### Best Practices:
    - **Scale Data:** LSTMs work best with normalized data.
    - **Tune Hyperparameters:** Number of lags, units, dropout, and epochs can affect performance.
    - **Monitor Overfitting:** Use validation loss and early stopping.
    """) 