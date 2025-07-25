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
test_pct = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5)
test_size = int(len(series) * test_pct / 100)
train_size = len(series) - test_size
st.markdown(f"**Training Set:** {train_size} quarters | **Test Set:** {test_size} quarters")

# === LSTM Configuration ===
st.markdown("### ⚙️ LSTM Model Settings")
col1, col2 = st.columns(2)
with col1:
    n_lags = st.slider("Number of Lag Quarters (input window):", min_value=4, max_value=16, value=8, step=1)
    with st.popover("❓"):
        st.markdown("""
        **Number of Lag Quarters (input window)**
        How many past quarters the model uses to predict the next value.
        - More lags can capture longer-term dependencies in the data.
        - Too many lags may add noise or cause overfitting.
        - Typical values: 4-12 for quarterly data.
        """)
    n_epochs = st.slider("Epochs:", min_value=10, max_value=200, value=50, step=10)
    with st.popover("❓"):
        st.markdown("""
        **Epochs**
        Number of times the model sees the entire training data.
        - More epochs can improve learning but may overfit.
        """)
    batch_size = st.slider("Batch Size:", min_value=4, max_value=64, value=8, step=4)
    with st.popover("❓"):
        st.markdown("""
        **Batch Size**
        Number of samples processed before the model is updated.
        - Smaller batch sizes can improve generalization.
        """)
    learning_rate = st.slider("Learning Rate:", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
    with st.popover("❓"):
        st.markdown("""
        **Learning Rate**
        Step size for updating model weights.
        - Lower values make learning slower but more stable.
        """)
with col2:
    n_units = st.slider("LSTM Units:", min_value=8, max_value=128, value=32, step=8)
    with st.popover("❓"):
        st.markdown("""
        **LSTM Units**
        The number of memory cells in the LSTM layer.
        - More units allow the model to learn more complex patterns.
        - Too many units may increase overfitting and training time.
        - Typical values: 16-64 for most time series problems.
        """)
    dropout = st.slider("Dropout (regularization):", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    with st.popover("❓"):
        st.markdown("""
        **Dropout**
        Fraction of units dropped for regularization.
        - Helps prevent overfitting.
        """)
    val_split = st.slider("Validation Split:", min_value=0.05, max_value=0.4, value=0.2, step=0.05)
    with st.popover("❓"):
        st.markdown("""
        **Validation Split**
        Fraction of training data used for validation.
        - Used for early stopping and monitoring overfitting.
        """)
    n_periods = st.slider("Number of quarters to forecast:", min_value=4, max_value=16, value=8, step=4)
    with st.popover("❓"):
        st.markdown("""
        **Forecast Horizon**
        Number of future quarters to forecast beyond the last available data.
        """)

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
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(LSTM(n_units, input_shape=(n_lags, 1)))
model.add(Dropout(dropout))
model.add(Dense(1))
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_split=val_split,
    callbacks=[es],
    verbose=0
)

# === Forecast on Test Set ===
y_pred_scaled = model.predict(X_test).flatten()
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# === Multi-step Future Forecasting ===
def forecast_future(model, scaler, series, n_lags, n_periods):
    last_values = series[-n_lags:]
    preds = []
    current_input = last_values.copy()
    for _ in range(n_periods):
        scaled_input = scaler.transform(current_input.reshape(-1, 1)).flatten()
        X_input = scaled_input.reshape((1, n_lags, 1))
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(pred)
        current_input = np.append(current_input[1:], pred)
    return np.array(preds)

future_forecast = forecast_future(model, scaler, series.values, n_lags, n_periods)
future_dates = pd.date_range(start=series.index[-1] + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')
future_dates = future_dates.strftime('%Y-%m-%d')

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
r2 = r2_score(y_test_actual, y_pred)

# === Confidence Intervals ===
residuals = y_test_actual - y_pred
forecast_std = np.std(residuals)
future_lower = future_forecast - 1.96 * forecast_std
future_upper = future_forecast + 1.96 * forecast_std

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
    
    # Plot actual, test forecast, and future forecast
    fig = go.Figure()
    # Actual
    fig.add_trace(go.Scatter(
        x=series.index[:train_size],
        y=series.values[:train_size],
        mode='lines',
        name='Train',
        line=dict(color='gray', width=2, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=series.index[train_size:],
        y=y_test_actual,
        mode='lines+markers',
        name='Test Actual',
        line=dict(color='black', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=series.index[train_size:],
        y=y_pred,
        mode='lines+markers',
        name='Test Forecast',
        line=dict(color='blue', width=2)
    ))
    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_forecast,
        mode='lines+markers',
        name='Future Forecast',
        line=dict(color='green', width=2)
    ))
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_upper,
        mode='lines',
        name='Upper CI',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_lower,
        mode='lines',
        name='Lower CI',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0),
        showlegend=False
    ))
    fig.update_layout(title="LSTM Forecast vs Actual", xaxis_title="Date", yaxis_title=selected_metric_label)
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    forecast_table = pd.DataFrame({
        "Date": future_dates,
        "LSTM Forecast": future_forecast,
        "Lower CI": future_lower,
        "Upper CI": future_upper
    })
    forecast_table["Date"] = pd.to_datetime(forecast_table["Date"]).dt.strftime('%Y-%m-%d')
    forecast_table.index = range(1, len(forecast_table) + 1)
    forecast_table.index.name = 'Index'
    st.dataframe(forecast_table, use_container_width=True)
    csv = forecast_table.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "lstm_forecast.csv", "text/csv")
    
    # Metrics table
    metrics_df = pd.DataFrame({
        "Set": ["Test"],
        "RMSE": [rmse],
        "MAE": [mae],
        "MAPE (%)": [mape],
        "R²": [r2]
    })
    st.dataframe(metrics_df, use_container_width=True)

# === Tab 2: Model Diagnostics ===
with tab2:
    st.title("📊 LSTM Model Diagnostics")
    
    # Introduction to LSTM diagnostics
    st.markdown("""
    **🔍 What are LSTM Residual Diagnostics?**
    
    **Residuals** are the differences between your actual data and what your LSTM model predicted. 
    For LSTM models, these diagnostics help assess how well the neural network captures temporal patterns.
    
    **Why are they important for LSTM?**
    - **Memory validation**: Check if LSTM's memory cells are capturing long-term dependencies
    - **Overfitting detection**: Neural networks can easily overfit to training data
    - **Temporal pattern assessment**: Verify if LSTM is learning meaningful sequential relationships
    - **Forecast reliability**: Poor residuals indicate unreliable future predictions
    """)
    
    # Loss curve
    st.subheader("📈 Training & Validation Loss Curve")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss'))
    fig_loss.update_layout(title="LSTM Loss Curve", xaxis_title="Epoch", yaxis_title="MSE Loss")
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Loss curve interpretation
    st.markdown("""
    **🔍 LSTM Loss Curve Interpretation:**
    
    **✅ Good Signs:**
    - **Converging loss**: Both train and validation loss decrease and stabilize
    - **Close curves**: Train and validation loss follow similar patterns
    - **No divergence**: Validation loss doesn't increase while train loss decreases
    - **Early stopping**: Model stops before overfitting (if early stopping was used)
    
    **⚠️ Warning Signs:**
    - **Overfitting**: Validation loss increases while train loss continues decreasing
    - **Underfitting**: Both losses remain high and don't converge
    - **Unstable training**: Loss curves are very noisy or oscillating
    - **Poor convergence**: Loss doesn't stabilize after many epochs
    
    **🎯 LSTM-Specific Usage:**
    - **Memory cell validation**: Stable loss suggests LSTM cells are working properly
    - **Gradient flow**: Smooth curves indicate good gradient flow through LSTM gates
    - **Capacity assessment**: Helps determine if LSTM has enough units for the task
    - **Regularization check**: Dropout effectiveness can be seen in validation loss
    """)
    
    # Residuals
    st.subheader("🟣 LSTM Residuals Over Time")
    fig_resid = px.line(x=series.index[train_size:], y=residuals, labels={'x': 'Date', 'y': 'Residuals'}, title="LSTM Residuals Over Time")
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)
    
    # LSTM residuals interpretation
    st.markdown("""
    **🔍 LSTM Residuals Interpretation:**
    
    **✅ Good Signs:**
    - **Random scatter**: No obvious temporal patterns (LSTM captured all dependencies)
    - **Mean close to zero**: No systematic bias in predictions
    - **Constant variance**: Homoscedastic residuals (LSTM handles all time periods equally)
    - **No outliers**: No extreme prediction errors
    
    **⚠️ Warning Signs:**
    - **Temporal patterns**: If residuals show trends or seasonality, LSTM missed temporal dependencies
    - **Heteroskedasticity**: If error variance changes over time, LSTM may need more units or different architecture
    - **Systematic bias**: If residuals are consistently positive/negative, LSTM has prediction bias
    - **Large outliers**: Extreme errors may indicate LSTM struggling with certain patterns
    
    **🎯 LSTM-Specific Usage:**
    - **Memory cell assessment**: Random residuals suggest LSTM memory cells are working well
    - **Temporal dependency check**: Patterns in residuals indicate missed long-term dependencies
    - **Architecture validation**: Helps determine if LSTM has enough capacity (units) for the task
    - **Training quality**: Good residuals indicate effective training and gradient flow
    """)
    
    # Residual distribution
    st.subheader("📊 LSTM Residual Distribution")
    fig_hist = px.histogram(residuals, nbins=20, title="LSTM Residual Distribution")
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Residual stats
    st.markdown(f"**Mean:** {np.mean(residuals):.4f} | **Std:** {np.std(residuals):.4f} | **Skewness:** {pd.Series(residuals).skew():.3f} | **Kurtosis:** {pd.Series(residuals).kurtosis():.3f}")
    
    # LSTM distribution interpretation
    st.markdown("""
    **🔍 LSTM Residual Distribution Interpretation:**
    
    **✅ Good Signs:**
    - **Bell-shaped curve**: Roughly normal distribution (LSTM predictions are well-behaved)
    - **Centered at zero**: No systematic bias in LSTM predictions
    - **Reasonable spread**: Standard deviation indicates prediction uncertainty
    - **Low skewness**: Symmetric distribution around zero
    
    **⚠️ Warning Signs:**
    - **Skewed distribution**: Asymmetric residuals may indicate LSTM bias
    - **Heavy tails**: Too many extreme errors suggest LSTM struggles with outliers
    - **Multiple peaks**: Bimodal distribution may indicate LSTM learning different patterns for different regimes
    - **Very wide spread**: High standard deviation suggests LSTM uncertainty
    
    **🎯 LSTM-Specific Usage:**
    - **Prediction quality**: Normal distribution suggests LSTM is making reliable predictions
    - **Bias detection**: Skewness indicates if LSTM systematically over/under-predicts
    - **Outlier handling**: Heavy tails suggest LSTM may need more training data or regularization
    - **Confidence assessment**: Distribution shape affects confidence interval reliability
    """)
    
    # Actual vs Predicted
    st.subheader("📈 LSTM Actual vs Predicted (Test Set)")
    fig_scatter = px.scatter(x=y_test_actual, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="LSTM: Actual vs Predicted")
    fig_scatter.add_trace(go.Scatter(x=[y_test_actual.min(), y_test_actual.max()], y=[y_test_actual.min(), y_test_actual.max()], mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Actual vs Predicted interpretation
    st.markdown("""
    **🔍 LSTM Actual vs Predicted Interpretation:**
    
    **✅ Good Signs:**
    - **Points close to diagonal**: LSTM predictions closely match actual values
    - **Random scatter**: No systematic patterns in prediction errors
    - **Good coverage**: Points spread across the range of actual values
    - **No clustering**: Predictions don't cluster in specific regions
    
    **⚠️ Warning Signs:**
    - **Systematic deviation**: Points consistently above/below diagonal indicate bias
    - **Poor coverage**: LSTM struggles with certain ranges of values
    - **Clustering**: Predictions cluster in specific regions (LSTM may be overfitting)
    - **Outliers**: Points far from diagonal indicate LSTM struggles with certain patterns
    
    **🎯 LSTM-Specific Usage:**
    - **Prediction accuracy**: Distance from diagonal shows LSTM prediction quality
    - **Bias assessment**: Systematic deviation indicates LSTM learning bias
    - **Range coverage**: Helps identify if LSTM handles all value ranges equally well
    - **Overfitting check**: Clustering may indicate LSTM memorizing rather than generalizing
    """)
    
    # Comprehensive LSTM diagnostics guide
    st.subheader("📚 LSTM-Specific Diagnostics Guide")
    
    st.markdown("""
    **🎯 When to Use Each LSTM Diagnostic:**
    
    **1. Loss Curve:**
    - **Use when**: You want to assess LSTM training quality and overfitting
    - **Look for**: Converging loss, close train/validation curves
    - **Action**: If overfitting, increase dropout or reduce units; if underfitting, increase units or epochs
    
    **2. Residuals Over Time:**
    - **Use when**: You want to check if LSTM captured all temporal dependencies
    - **Look for**: Random scatter, no temporal patterns
    - **Action**: If patterns exist, increase LSTM units or add more lags
    
    **3. Residual Distribution:**
    - **Use when**: You want to validate LSTM prediction quality and bias
    - **Look for**: Normal distribution, centered at zero
    - **Action**: If skewed, check data scaling or model architecture
    
    **4. Actual vs Predicted:**
    - **Use when**: You want to assess LSTM prediction accuracy across value ranges
    - **Look for**: Points close to diagonal, good coverage
    - **Action**: If poor coverage, consider different LSTM architecture or more training data
    """)
    
    st.markdown("""
    **🔬 LSTM-Specific Model Validation:**
    
    **Step 1: Training Assessment**
    - Monitor loss curves for convergence and overfitting
    - Check if early stopping was triggered appropriately
    - Validate that dropout is working (validation loss should be close to training loss)
    
    **Step 2: Temporal Pattern Validation**
    - Ensure residuals show no temporal patterns (LSTM captured all dependencies)
    - Check if LSTM handles different time periods equally well
    - Verify that memory cells are working (no systematic bias)
    
    **Step 3: Prediction Quality Assessment**
    - Validate that predictions are normally distributed around actual values
    - Check for systematic bias in predictions
    - Assess prediction accuracy across different value ranges
    
    **Step 4: Architecture Validation**
    - Determine if LSTM has enough capacity (units) for the task
    - Check if the number of lags is appropriate
    - Validate that regularization (dropout) is effective
    """)
    
    st.markdown("""
    **💡 LSTM-Specific Best Practices:**
    
    **For Data Scientists:**
    - Always scale data before LSTM training (MinMaxScaler or StandardScaler)
    - Use early stopping to prevent overfitting
    - Monitor both training and validation loss curves
    - Consider LSTM architecture complexity vs. data size
    
    **For Researchers:**
    - Document LSTM hyperparameters and training process
    - Compare LSTM diagnostics with other models (ARIMA, GRU, etc.)
    - Use diagnostics to guide LSTM architecture improvements
    - Validate LSTM assumptions about temporal dependencies
    
    **For Business Users:**
    - Understand that LSTM confidence intervals are approximate
    - Monitor LSTM performance over time as new data becomes available
    - Consider ensemble methods combining LSTM with other models
    - Use LSTM diagnostics to assess forecast reliability
    """)
    
    st.info("""
    **📌 LSTM-Specific Notes:** 
    - LSTM diagnostics focus on temporal pattern capture and neural network training quality
    - Unlike statistical models, LSTM doesn't assume specific residual distributions
    - Focus on temporal dependency capture and prediction accuracy rather than strict statistical assumptions
    - LSTM confidence intervals are based on residual variance and may not capture all uncertainty sources
    """)

# === Tab 3: Model Summary ===
with tab3:
    st.title("📋 LSTM Model Summary & Explanation")
    st.subheader("Model Architecture")
    st.text(model.summary())
    
    st.subheader("Model Parameters")
    st.markdown(f"""
    - **Input Window (Lags):** {n_lags}
    - **LSTM Units:** {n_units}
    - **Dropout:** {dropout}
    - **Epochs:** {len(history.history['loss'])}
    - **Batch Size:** {batch_size}
    - **Learning Rate:** {learning_rate}
    - **Validation Split:** {val_split}
    - **Early Stopping:** Patience 10
    """)
    
    st.subheader("Performance Metrics (Test Set)")
    st.markdown(f"""
    - **RMSE:** {rmse:.2f}
    - **MAE:** {mae:.2f}
    - **MAPE:** {mape:.2f}%
    - **R²:** {r2:.3f}
    """)

    st.subheader("🧑‍💻 Code Walkthrough & Model Explanation")
    st.markdown("""
    **Step 1: Data Preparation**
    ```python
    df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)
series = df[selected_metric].dropna()
    ```
    - Loads and sorts the data, sets the date as index, selects the metric to forecast.

    **Step 2: Scaling and Lag Feature Creation**
    ```python
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    # Create lagged sequences for supervised learning
    def create_lagged_sequences(series, n_lags):
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i-n_lags:i])
            y.append(series[i])
        return np.array(X), np.array(y)
    ```
    - Scales the data to [0, 1] for stable neural network training.
    - Creates input/output pairs for the LSTM using a sliding window of `n_lags`.

    **Step 3: Model Building**
    ```python
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(n_lags, 1)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    ```
    - Builds a simple LSTM network with one LSTM layer, dropout for regularization, and a dense output layer.
    - Uses Adam optimizer and mean squared error loss.

    **Step 4: Model Training**
    ```python
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=n_epochs,
    batch_size=batch_size,
    validation_split=val_split,
    callbacks=[es],
    verbose=0
)
    ```
    - Trains the model with early stopping to prevent overfitting.
    - Uses a validation split for monitoring.

    **Step 5: Forecasting**
    ```python
    # Multi-step recursive forecasting
def forecast_future(model, scaler, series, n_lags, n_periods):
    last_values = series[-n_lags:]
    preds = []
    current_input = last_values.copy()
    for _ in range(n_periods):
        scaled_input = scaler.transform(current_input.reshape(-1, 1)).flatten()
        X_input = scaled_input.reshape((1, n_lags, 1))
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(pred)
        current_input = np.append(current_input[1:], pred)
    return np.array(preds)
    ```
    - Forecasts future values recursively, using the last `n_lags` predictions as input for each step.

    **Step 6: Evaluation**
    ```python
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, y_pred)
    ```
    - Calculates standard regression metrics to evaluate forecast accuracy.

    **Model Parameter Explanations:**
    - **Input Window (Lags):** Number of past quarters used for each prediction. More lags = more context, but too many can overfit.
    - **LSTM Units:** Number of memory cells in the LSTM layer. More units = more capacity, but also more risk of overfitting.
    - **Dropout:** Fraction of units dropped during training to prevent overfitting.
    - **Epochs:** Number of times the model sees the training data.
    - **Batch Size:** Number of samples per gradient update.
    - **Learning Rate:** Step size for optimizer updates.
    - **Validation Split:** Fraction of training data used for validation.
    - **Early Stopping:** Stops training if validation loss doesn't improve.

    **Best Practices & Interpretation:**
    - Always scale your data before training LSTM/GRU models.
    - Use early stopping and dropout to prevent overfitting.
    - Tune lags, units, and learning rate for best results.
    - LSTM is powerful for capturing long-term dependencies and non-linear patterns in time series.
    - Recursive forecasting can accumulate error; monitor confidence intervals.

    **Limitations:**
    - LSTM models require more data and computation than classical models.
    - Can overfit if too complex or if data is too short.
    - Confidence intervals are approximate (based on residual std).
    - Not as interpretable as ARIMA/SARIMA.
    """)

# === Tab 4: LSTM Explanation ===
with tab4:
    st.title("🧠 LSTM Model Summary & Explanation")
    st.subheader("🔍 What is LSTM?")
    st.markdown("""
    **LSTM (Long Short-Term Memory)** networks are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data, overcoming the vanishing gradient problem of simple RNNs. They are especially powerful for time series forecasting with complex, non-linear, and long-memory patterns.
    """)
    st.subheader("📐 LSTM Model Components")
    st.markdown("""
    - **Input Window (Lags):** Number of past quarters used for each prediction. More lags = more context, but too many can overfit.
    - **LSTM Units:** Number of memory cells in the LSTM layer. More units = more capacity, but also more risk of overfitting.
    - **Dropout:** Fraction of units dropped during training to prevent overfitting.
    - **Epochs:** Number of times the model sees the training data.
    - **Batch Size:** Number of samples per gradient update.
    - **Learning Rate:** Step size for optimizer updates.
    - **Validation Split:** Fraction of training data used for validation.
    - **Early Stopping:** Stops training if validation loss doesn't improve.
    """)
    st.subheader("🔄 How LSTM Works")
    st.markdown("""
    1. **Data Preparation:**
       - Data is scaled (e.g., MinMaxScaler) for stable neural network training.
       - Lagged input windows are created (e.g., 8 previous quarters to predict the next).
    2. **Model Architecture:**
       - LSTM layer(s) process the input sequence, learning temporal dependencies.
       - Dropout layer(s) help prevent overfitting.
       - Dense layer outputs the forecast.
    3. **Training:**
       - Model is trained on the training set, with early stopping to avoid overfitting.
       - Validation set monitors generalization.
    4. **Forecasting:**
       - Model predicts on the test set and recursively for future quarters.
       - Inverse scaling returns predictions to original units.
    """)
    st.subheader("📊 Model Performance")
    st.markdown("""
    - **RMSE:** Average prediction error in original units (lower is better).
    - **MAE:** Average absolute error.
    - **MAPE:** Average percentage error.
    - **R²:** Proportion of variance explained (closer to 1 is better).
    """)
    st.subheader("🎯 How Each Parameter Affects Your Forecast")
    st.markdown("""
    - **Input Window (Lags):**
      - More lags allow the model to see further back in time, capturing longer-term patterns.
      - Too many lags can introduce noise and overfitting.
    - **LSTM Units:**
      - More units increase the model’s ability to learn complex relationships.
      - Too many can overfit, especially with limited data.
    - **Dropout:**
      - Higher dropout reduces overfitting but may underfit if too high.
    - **Epochs/Batch Size/Learning Rate:**
      - Affect training speed, convergence, and generalization.
    """)
    st.subheader("🔬 Why LSTM for This Data?")
    st.markdown("""
    - **Handles Non-Linearity:**
      - Captures complex, non-linear relationships in economic data.
    - **Long-Term Memory:**
      - Remembers patterns over many quarters, ideal for economic cycles.
    - **Robust to Noise:**
      - Can generalize well with proper regularization and validation.
    """)
    st.subheader("💡 Practical Tips")
    st.markdown("""
    **For Policy Makers:**
    - Use LSTM forecasts for planning when data shows complex, non-linear, or long-term patterns.
    - Monitor confidence intervals for risk assessment.

    **For Researchers:**
    - Tune hyperparameters (lags, units, dropout) for best results.
    - Compare LSTM with simpler models to justify complexity.

    **For Business Users:**
    - LSTM is powerful but less interpretable than ARIMA/SARIMA.
    - Use diagnostics and explainability tools (e.g., SHAP) to understand model decisions.
    """)
    st.subheader("✅ Best Practices")
    st.markdown("""
    - Always scale your data before training LSTM/GRU models.
    - Use early stopping and dropout to prevent overfitting.
    - Tune lags, units, and learning rate for best results.
    - Validate with out-of-sample data.
    """)
    st.subheader("⚠️ Limitations")
    st.markdown("""
    - Requires more data and computation than classical models.
    - Can overfit if too complex or if data is too short.
    - Confidence intervals are approximate (based on residual std).
    - Not as interpretable as ARIMA/SARIMA.
    """) 