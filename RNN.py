import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# === Load and preprocess dataset ===
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

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

series = df[selected_metric].dropna()

# === Train/Test Split Configuration ===
test_pct = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5)
test_size = int(len(series) * test_pct / 100)
train_size = len(series) - test_size
st.markdown(f"**Training Set:** {train_size} quarters | **Test Set:** {test_size} quarters")

# === Display historical time series ===
st.subheader('📊 Historical Time Series')
st.line_chart(series)

# === RNN Configuration ===
st.markdown("### ⚙️ RNN Model Settings")
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
    n_units = st.slider("RNN Units:", min_value=4, max_value=64, value=16, step=4)
    with st.popover("❓"):
        st.markdown("""
        **RNN Units**
        The number of units (neurons) in the RNN layer.
        - More units can learn more complex patterns.
        - Too many may increase overfitting and training time.
        - Typical values: 8-32.
        """)
    dropout = st.slider("Dropout Rate:", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    with st.popover("❓"):
        st.markdown("""
        **Dropout Rate**
        The fraction of neurons randomly dropped during training.
        - Helps prevent overfitting.
        - Typical values: 0.1-0.3.
        """)
with col2:
    epochs = st.slider("Epochs:", min_value=10, max_value=200, value=50, step=10)
    with st.popover("❓"):
        st.markdown("""
        **Epochs**
        The number of times the model sees the entire training set.
        - More epochs can improve learning but may overfit.
        - Use early stopping or monitor validation loss.
        """)
    batch_size = st.slider("Batch Size:", min_value=4, max_value=32, value=8, step=2)
    with st.popover("❓"):
        st.markdown("""
        **Batch Size**
        The number of samples processed before updating the model.
        - Smaller batches can improve generalization but may be slower.
        - Typical values: 8-32.
        """)
    learning_rate = st.number_input("Learning Rate:", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-4, format="%e")
    with st.popover("❓"):
        st.markdown("""
        **Learning Rate**
        Controls how much the model weights are updated during training.
        - Too high: model may not converge.
        - Too low: training may be very slow.
        - Typical values: 0.001-0.01.
        """)
    val_split = st.slider("Validation Split:", min_value=0.05, max_value=0.4, value=0.2, step=0.05)
    with st.popover("❓"):
        st.markdown("""
        **Validation Split**
        The fraction of training data used for validation.
        - Helps monitor overfitting.
        - Typical values: 0.1-0.3.
        """)
    n_periods = st.slider("Forecast Horizon (quarters):", min_value=4, max_value=16, value=8, step=1)
    with st.popover("❓"):
        st.markdown("""
        **Forecast Horizon**
        How many future quarters to predict.
        - Longer horizons increase uncertainty.
        - Typical values: 4-12.
        """)

# === Data Preparation ===
def create_lagged_data(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
X_all, y_all = create_lagged_data(scaled_series, n_lags)

X_train, y_train = X_all[:train_size-n_lags], y_all[:train_size-n_lags]
X_test, y_test = X_all[train_size-n_lags:], y_all[train_size-n_lags:]

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# === Model Building ===
model = Sequential([
    SimpleRNN(n_units, input_shape=(n_lags, 1), activation='tanh'),
    Dropout(dropout),
    Dense(32, activation='relu'),
    Dense(1)
])
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse')

# === Model Training ===
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=val_split,
    verbose=0
)

# === Forecasting ===
train_pred = model.predict(X_train).flatten()
test_pred = model.predict(X_test).flatten()

train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

def recursive_forecast(model, last_window, n_periods, scaler):
    preds = []
    window = last_window.copy()
    for _ in range(n_periods):
        x_input = window.reshape(1, -1, 1)
        pred = model.predict(x_input, verbose=0)[0, 0]
        preds.append(pred)
        window = np.roll(window, -1)
        window[-1] = pred
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_inv

last_window = scaled_series[-n_lags:]
future_forecast = recursive_forecast(model, last_window, n_periods, scaler)

forecast_dates = pd.date_range(start=series.index[-1] + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')
forecast_dates = forecast_dates.strftime('%Y-%m-%d')

residuals = y_test_inv - test_pred_inv
forecast_std = np.std(residuals)
conf_int_lower = future_forecast - 1.96 * forecast_std
conf_int_upper = future_forecast + 1.96 * forecast_std

rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
mae = mean_absolute_error(y_test_inv, test_pred_inv)
mape = np.mean(np.abs((y_test_inv - test_pred_inv) / y_test_inv)) * 100
r2 = r2_score(y_test_inv, test_pred_inv)

tab1, tab2, tab3, tab4 = st.tabs([
    f"🔮 RNN Forecast ({selected_metric_label})",
    "📊 Model Diagnostics",
    "📋 Model Summary",
    "🧠 RNN Explanation"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"🔮 RNN Forecast for {selected_metric_label}")
    # Actual + forecast plot
    actual_df = pd.DataFrame({
        "Date": series.index[n_lags:train_size],
        "Actual": y_train_inv
    })
    test_df = pd.DataFrame({
        "Date": series.index[train_size:],
        "Actual": y_test_inv,
        "RNN Forecast": test_pred_inv
    })
    future_df = pd.DataFrame({
        "Date": forecast_dates,
        "RNN Forecast": future_forecast,
        "Lower CI": conf_int_lower,
        "Upper CI": conf_int_upper
    })
    # Set index to start from 1
    actual_df.index = range(1, len(actual_df) + 1)
    actual_df.index.name = 'Index'
    test_df.index = range(1, len(test_df) + 1)
    test_df.index.name = 'Index'
    future_df.index = range(1, len(future_df) + 1)
    future_df.index.name = 'Index'
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df["Date"], y=actual_df["Actual"], mode="lines", name="Train Actual"))
    fig.add_trace(go.Scatter(x=test_df["Date"], y=test_df["Actual"], mode="lines+markers", name="Test Actual"))
    fig.add_trace(go.Scatter(x=test_df["Date"], y=test_df["RNN Forecast"], mode="lines+markers", name="Test Forecast"))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["RNN Forecast"], mode="lines+markers", name="Future Forecast"))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Upper CI"], mode="lines", name="Upper CI", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Lower CI"], mode="lines", name="Lower CI", fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False))
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)
    # Forecast table
    st.dataframe(future_df, use_container_width=True)
    csv = future_df.to_csv().encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "rnn_forecast.csv", "text/csv")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("MAPE (%)", f"{mape:.2f}")
    with col4:
        st.metric("R²", f"{r2:.3f}")

# === Tab 2: Diagnostics ===
with tab2:
    st.title("📊 RNN Model Diagnostics")
    
    # Introduction to RNN diagnostics
    st.markdown("""
    **🔍 What are RNN Residual Diagnostics?**
    
    **Residuals** are the differences between your actual data and what your RNN model predicted. 
    For RNN models, these diagnostics help assess how well the simple recurrent neural network captures temporal patterns.
    
    **Why are they important for RNN?**
    - **Vanishing gradient validation**: Check if RNN can effectively learn long-term dependencies
    - **Simple architecture assessment**: RNN has the simplest recurrent architecture compared to LSTM/GRU
    - **Temporal pattern assessment**: Verify if RNN is learning meaningful sequential relationships
    - **Forecast reliability**: Poor residuals indicate unreliable future predictions
    """)
    
    st.subheader("🟣 RNN Residuals Over Time")
    resid_fig = px.line(x=test_df["Date"], y=residuals, labels={'x': 'Date', 'y': 'Residuals'}, title="RNN Residuals (Test Set)")
    st.plotly_chart(resid_fig, use_container_width=True)
    
    # RNN residuals interpretation
    st.markdown("""
    **🔍 RNN Residuals Interpretation:**
    
    **✅ Good Signs:**
    - **Random scatter**: No obvious temporal patterns (RNN captured dependencies)
    - **Mean close to zero**: No systematic bias in predictions
    - **Constant variance**: Homoscedastic residuals (RNN handles all time periods equally)
    - **No outliers**: No extreme prediction errors
    
    **⚠️ Warning Signs:**
    - **Temporal patterns**: If residuals show trends or seasonality, RNN missed temporal dependencies
    - **Heteroskedasticity**: If error variance changes over time, RNN may need more units or different architecture
    - **Systematic bias**: If residuals are consistently positive/negative, RNN has prediction bias
    - **Large outliers**: Extreme errors may indicate RNN struggling with certain patterns
    
    **🎯 RNN-Specific Usage:**
    - **Vanishing gradient assessment**: Random residuals suggest RNN can learn temporal dependencies despite vanishing gradient issues
    - **Simple architecture validation**: RNN should capture basic temporal patterns with simple recurrent connections
    - **Long-term dependency check**: Patterns in residuals may indicate RNN struggling with long-term dependencies
    - **Architecture validation**: Helps determine if RNN has enough capacity (units) for the task
    """)
    st.subheader("📊 RNN Residual Distribution")
    fig_hist = px.histogram(residuals, nbins=20, title="RNN Residual Distribution")
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # RNN distribution interpretation
    st.markdown("""
    **🔍 RNN Residual Distribution Interpretation:**
    
    **✅ Good Signs:**
    - **Bell-shaped curve**: Roughly normal distribution (RNN predictions are well-behaved)
    - **Centered at zero**: No systematic bias in RNN predictions
    - **Reasonable spread**: Standard deviation indicates prediction uncertainty
    - **Low skewness**: Symmetric distribution around zero
    
    **⚠️ Warning Signs:**
    - **Skewed distribution**: Asymmetric residuals may indicate RNN bias
    - **Heavy tails**: Too many extreme errors suggest RNN struggles with outliers
    - **Multiple peaks**: Bimodal distribution may indicate RNN learning different patterns for different regimes
    - **Very wide spread**: High standard deviation suggests RNN uncertainty
    
    **🎯 RNN-Specific Usage:**
    - **Simple architecture quality**: Normal distribution suggests RNN's simple recurrent connections are working well
    - **Bias detection**: Skewness indicates if RNN systematically over/under-predicts
    - **Vanishing gradient assessment**: Distribution shape reflects RNN's ability to learn despite gradient issues
    - **Confidence assessment**: Distribution shape affects confidence interval reliability
    """)
    st.subheader("Q-Q Plot (Normality)")
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    fig_qq, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    st.pyplot(fig_qq)
    st.subheader("PACF Plot (Residuals)")
    from statsmodels.graphics.tsaplots import plot_pacf
    fig_pacf = plot_pacf(residuals, lags=min(20, len(residuals)//2-1))
    st.pyplot(fig_pacf.figure)
    st.subheader("Metrics Table")
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MAPE", "R²"],
        "Value": [rmse, mae, mape, r2]
    })
    st.dataframe(metrics_df, use_container_width=True)
    
    # Comprehensive RNN diagnostics guide
    st.subheader("📚 RNN-Specific Diagnostics Guide")
    
    st.markdown("""
    **🎯 When to Use Each RNN Diagnostic:**
    
    **1. Residuals Over Time:**
    - **Use when**: You want to check if RNN captured temporal dependencies despite vanishing gradient issues
    - **Look for**: Random scatter, no temporal patterns
    - **Action**: If patterns exist, consider LSTM/GRU for better long-term dependency capture
    
    **2. Residual Distribution:**
    - **Use when**: You want to validate RNN prediction quality with simple architecture
    - **Look for**: Normal distribution, centered at zero
    - **Action**: If skewed, check data scaling or consider more complex recurrent architectures
    
    **3. Q-Q Plot:**
    - **Use when**: You want to assess RNN prediction normality
    - **Look for**: Points following diagonal line
    - **Action**: If non-normal, consider data transformations or different model architecture
    
    **4. PACF Plot:**
    - **Use when**: You want to check if RNN captured all temporal dependencies
    - **Look for**: No significant spikes outside confidence bands
    - **Action**: If spikes exist, RNN may need more units or consider LSTM/GRU
    
    **5. Metrics Table:**
    - **Use when**: You want to quantify RNN performance on temporal pattern recognition
    - **Look for**: Low RMSE/MAE, high R²
    - **Action**: If metrics are poor, consider LSTM/GRU for better temporal dependency capture
    """)
    
    st.markdown("""
    **🔬 RNN-Specific Model Validation:**
    
    **Step 1: Vanishing Gradient Assessment**
    - Ensure residuals show no temporal patterns (RNN captured dependencies despite gradient issues)
    - Check if RNN handles different time periods equally well
    - Verify that simple recurrent connections are working effectively
    - Compare performance with LSTM/GRU for long-term dependency capture
    
    **Step 2: Simple Architecture Validation**
    - Check if RNN has enough units for the task
    - Validate that simple recurrent connections can capture the temporal patterns
    - Ensure RNN is not struggling with vanishing gradient problems
    - Consider if LSTM/GRU would be more appropriate for the data
    
    **Step 3: Prediction Quality Assessment**
    - Validate that predictions are normally distributed around actual values
    - Check for systematic bias in predictions
    - Assess prediction accuracy across different value ranges
    - Consider RNN's limitations with long-term dependencies
    
    **Step 4: Architecture Suitability**
    - Determine if RNN's simple architecture is sufficient for the task
    - Check if the number of lags is appropriate
    - Validate that dropout is effective for regularization
    - Assess if LSTM/GRU would provide better performance
    """)
    
    st.markdown("""
    **💡 RNN-Specific Best Practices:**
    
    **For Data Scientists:**
    - Always scale data before RNN training (MinMaxScaler or StandardScaler)
    - Use appropriate number of units for the task complexity
    - Consider LSTM/GRU if RNN struggles with long-term dependencies
    - Monitor for vanishing gradient issues during training
    
    **For Researchers:**
    - Document RNN hyperparameters and training process
    - Compare RNN diagnostics with LSTM/GRU for architecture comparison
    - Use diagnostics to guide architecture selection (RNN vs LSTM vs GRU)
    - Validate RNN assumptions about temporal dependencies
    
    **For Business Users:**
    - Understand that RNN has the simplest recurrent architecture
    - Monitor RNN performance over time as new data becomes available
    - Consider RNN's limitations with long-term dependencies
    - Use RNN diagnostics to assess if more complex models (LSTM/GRU) are needed
    """)
    
    st.info("""
    **📌 RNN-Specific Notes:** 
    - RNN diagnostics focus on simple recurrent architecture performance
    - RNN may struggle with long-term dependencies due to vanishing gradient issues
    - Focus on simple temporal pattern capture and gradient flow assessment
    - RNN confidence intervals are based on residual variance and may not capture all uncertainty sources
    - RNN is often used as a baseline before considering more complex architectures (LSTM/GRU)
    """)

# === Tab 3: Model Summary ===
with tab3:
    st.title("📋 RNN Model Summary & Explanation")
    st.subheader("Model Architecture")
    st.text(model.summary())
    st.subheader("Model Parameters")
    st.markdown(f"""
    - **Input Window (Lags):** {n_lags}
    - **RNN Units:** {n_units}
    - **Dropout:** {dropout}
    - **Epochs:** {epochs}
    - **Batch Size:** {batch_size}
    - **Learning Rate:** {learning_rate}
    - **Validation Split:** {val_split}
    """)
    st.subheader("🧑‍💻 Code Walkthrough & Model Explanation")
    st.markdown("""
    **Step-by-step RNN Forecasting Workflow:**
    1. **Data Preparation:**
        - Load and sort the unemployment data.
        - Select the metric to forecast.
        - Scale the data to [0, 1] using MinMaxScaler.
        - Create lagged input windows (X) and targets (y).
    2. **Model Building:**
        - Build a Sequential model with SimpleRNN, Dropout, and Dense layers.
        - The SimpleRNN layer captures sequential dependencies from the lagged input window.
        - Dropout helps prevent overfitting.
        - Dense layers map features to the forecast output.
    3. **Model Training:**
        - Train the model on the training set with the selected batch size, epochs, and validation split.
        - Use Adam optimizer with the chosen learning rate.
    4. **Forecasting:**
        - Predict on the test set and recursively forecast into the future.
        - Inverse scale predictions to original units.
    5. **Diagnostics & Metrics:**
        - Plot residuals, histograms, Q-Q, PACF, and actual vs predicted.
        - Show RMSE, MAE, MAPE, and R² metrics.
    6. **Interpretation:**
        - More units can capture more complex patterns but may overfit.
        - Dropout and validation split help prevent overfitting.
        - Use lag window size to control how much history the model sees.
    """)
    st.subheader("Best Practices & Limitations")
    st.markdown("""
    - **Best Practices:**
        - Tune lag window and RNN units for your data.
        - Use dropout and validation split to avoid overfitting.
        - Monitor residuals and metrics for model quality.
    - **Limitations:**
        - SimpleRNNs can struggle with long-term dependencies (consider LSTM/GRU for those).
        - Requires enough data for effective training.
    """)

# === Tab 4: RNN Explanation ===
with tab4:
    st.title("🧠 RNN Model Summary & Explanation")
    st.subheader("🔍 What is RNN?")
    st.markdown("""
    **RNN (Simple Recurrent Neural Network)** is the most basic form of recurrent neural network, designed to capture short-term dependencies in sequential data. RNNs process sequences by maintaining a hidden state that is updated at each time step, making them suitable for time series with simple temporal patterns.
    """)
    st.subheader("📐 RNN Model Components")
    st.markdown("""
    - **Input Window (Lags):** Number of past quarters used for each prediction.
    - **RNN Units:** Number of neurons in the RNN layer.
    - **Dropout:** Fraction of units dropped during training to prevent overfitting.
    - **Epochs:** Number of times the model sees the training data.
    - **Batch Size:** Number of samples per gradient update.
    - **Learning Rate:** Step size for optimizer updates.
    - **Validation Split:** Fraction of training data used for validation.
    - **Early Stopping:** Stops training if validation loss doesn't improve.
    """)
    st.subheader("🔄 How RNN Works")
    st.markdown("""
    1. **Data Preparation:**
       - Data is scaled for stable neural network training.
       - Lagged input windows are created.
    2. **Model Architecture:**
       - SimpleRNN layer(s) process the input sequence, learning short-term dependencies.
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
      - More lags allow the model to see further back in time, capturing more context.
      - Too many lags can introduce noise and overfitting.
    - **RNN Units:**
      - More units increase the model’s ability to learn patterns.
      - Too many can overfit, especially with limited data.
    - **Dropout:**
      - Higher dropout reduces overfitting but may underfit if too high.
    - **Epochs/Batch Size/Learning Rate:**
      - Affect training speed, convergence, and generalization.
    """)
    st.subheader("🔬 Why RNN for This Data?")
    st.markdown("""
    - **Simplicity:**
      - RNNs are easy to implement and interpret for simple time series.
    - **Short-Term Memory:**
      - Effective for capturing short-term dependencies in the data.
    - **Baseline Comparison:**
      - Useful as a baseline to compare with more complex models (LSTM, GRU).
    """)
    st.subheader("💡 Practical Tips")
    st.markdown("""
    **For Policy Makers:**
    - Use RNN forecasts for short-term planning and as a baseline.
    - Monitor confidence intervals for risk assessment.

    **For Researchers:**
    - Tune hyperparameters (lags, units, dropout) for best results.
    - Compare RNN with LSTM/GRU to justify model choice.

    **For Business Users:**
    - RNN is simple and fast, but may not capture complex patterns.
    - Use diagnostics and explainability tools to understand model decisions.
    """)
    st.subheader("✅ Best Practices")
    st.markdown("""
    - Always scale your data before training RNN models.
    - Use early stopping and dropout to prevent overfitting.
    - Tune lags, units, and learning rate for best results.
    - Validate with out-of-sample data.
    """)
    st.subheader("⚠️ Limitations")
    st.markdown("""
    - Prone to vanishing gradient, limiting long-term memory.
    - Can overfit if too complex or if data is too short.
    - Confidence intervals are approximate (based on residual std).
    - Not as interpretable as ARIMA/SARIMA.
    """) 