import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
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

# === Display historical time series ===
st.subheader('📊 Historical Time Series')
st.line_chart(series)

# === Train/Test Split Configuration ===
test_pct = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5)
test_size = int(len(series) * test_pct / 100)
train_size = len(series) - test_size
st.markdown(f"**Training Set:** {train_size} quarters | **Test Set:** {test_size} quarters")

# === CNN Configuration ===
st.markdown("### ⚙️ CNN Model Settings")
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
    n_filters = st.slider("Number of CNN Filters:", min_value=8, max_value=64, value=32, step=8)
    with st.popover("❓"):
        st.markdown("""
        **Number of CNN Filters**
        The number of convolutional filters (feature detectors) in the CNN layer.
        - More filters can capture more complex patterns.
        - Too many may increase overfitting and computation time.
        - Typical values: 16-64.
        """)
    kernel_size = st.slider("Kernel Size:", min_value=2, max_value=8, value=4, step=1)
    with st.popover("❓"):
        st.markdown("""
        **Kernel Size**
        The width of the convolutional filter (number of time steps).
        - Larger kernels capture broader patterns.
        - Too large may smooth out important details.
        - Typical values: 2-6.
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
    Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', input_shape=(n_lags, 1)),
    Dropout(dropout),
    Flatten(),
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
# In-sample prediction
train_pred = model.predict(X_train).flatten()
test_pred = model.predict(X_test).flatten()

# Inverse scale
train_pred_inv = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Multi-step future forecasting
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

# === Dates for plotting and tables ===
forecast_dates = pd.date_range(start=series.index[-1] + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')
forecast_dates = forecast_dates.strftime('%Y-%m-%d')

# === Confidence Intervals (approximate) ===
residuals = y_test_inv - test_pred_inv
forecast_std = np.std(residuals)
conf_int_lower = future_forecast - 1.96 * forecast_std
conf_int_upper = future_forecast + 1.96 * forecast_std

# === Metrics ===
rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
mae = mean_absolute_error(y_test_inv, test_pred_inv)
mape = np.mean(np.abs((y_test_inv - test_pred_inv) / y_test_inv)) * 100
r2 = r2_score(y_test_inv, test_pred_inv)

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs([
    f"🔮 CNN Forecast ({selected_metric_label})",
    "📊 Model Diagnostics",
    "📋 Model Summary",
    "🧠 CNN Explanation"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"🔮 CNN Forecast for {selected_metric_label}")
    # Actual + forecast plot
    actual_df = pd.DataFrame({
        "Date": series.index[n_lags:train_size],
        "Actual": y_train_inv
    })
    test_df = pd.DataFrame({
        "Date": series.index[train_size:],
        "Actual": y_test_inv,
        "CNN Forecast": test_pred_inv
    })
    future_df = pd.DataFrame({
        "Date": forecast_dates,
        "CNN Forecast": future_forecast,
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
    fig.add_trace(go.Scatter(x=test_df["Date"], y=test_df["CNN Forecast"], mode="lines+markers", name="Test Forecast"))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["CNN Forecast"], mode="lines+markers", name="Future Forecast"))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Upper CI"], mode="lines", name="Upper CI", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Lower CI"], mode="lines", name="Lower CI", fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False))
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)
    # Forecast table
    st.dataframe(future_df, use_container_width=True)
    csv = future_df.to_csv().encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "cnn_forecast.csv", "text/csv")
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

# === Tab 2: Diagnostics & Metrics ===
with tab2:
    st.title("📊 CNN Model Diagnostics")
    
    # Introduction to CNN diagnostics
    st.markdown("""
    **🔍 What are CNN Residual Diagnostics?**
    
    **Residuals** are the differences between your actual data and what your CNN model predicted. 
    For CNN models, these diagnostics help assess how well the convolutional neural network captures local temporal patterns.
    
    **Why are they important for CNN?**
    - **Local pattern validation**: Check if CNN's convolutional filters are capturing meaningful local temporal patterns
    - **Kernel effectiveness**: Verify if the kernel size and filters are appropriate for the data
    - **Temporal dependency assessment**: CNN focuses on local patterns rather than long-term dependencies
    - **Forecast reliability**: Poor residuals indicate unreliable future predictions
    """)
    
    # Residuals plot
    st.subheader("🟣 CNN Residuals Over Time")
    resid_fig = px.line(x=test_df["Date"], y=residuals, labels={'x': 'Date', 'y': 'Residuals'}, title="CNN Residuals (Test Set)")
    st.plotly_chart(resid_fig, use_container_width=True)
    
    # CNN residuals interpretation
    st.markdown("""
    **🔍 CNN Residuals Interpretation:**
    
    **✅ Good Signs:**
    - **Random scatter**: No obvious temporal patterns (CNN captured local dependencies)
    - **Mean close to zero**: No systematic bias in predictions
    - **Constant variance**: Homoscedastic residuals (CNN handles all time periods equally)
    - **No outliers**: No extreme prediction errors
    
    **⚠️ Warning Signs:**
    - **Temporal patterns**: If residuals show trends or seasonality, CNN missed important patterns
    - **Heteroskedasticity**: If error variance changes over time, CNN may need different kernel size or filters
    - **Systematic bias**: If residuals are consistently positive/negative, CNN has prediction bias
    - **Large outliers**: Extreme errors may indicate CNN struggling with certain local patterns
    
    **🎯 CNN-Specific Usage:**
    - **Local pattern assessment**: Random residuals suggest CNN filters are capturing local temporal patterns well
    - **Kernel size validation**: Patterns in residuals may indicate inappropriate kernel size
    - **Filter effectiveness**: Helps determine if CNN has enough filters for the task
    - **Local vs global patterns**: CNN focuses on local patterns; long-term dependencies may be missed
    """)
    # Histogram
    st.subheader("📊 CNN Residual Distribution")
    fig_hist = px.histogram(residuals, nbins=20, title="CNN Residual Distribution")
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # CNN distribution interpretation
    st.markdown("""
    **🔍 CNN Residual Distribution Interpretation:**
    
    **✅ Good Signs:**
    - **Bell-shaped curve**: Roughly normal distribution (CNN predictions are well-behaved)
    - **Centered at zero**: No systematic bias in CNN predictions
    - **Reasonable spread**: Standard deviation indicates prediction uncertainty
    - **Low skewness**: Symmetric distribution around zero
    
    **⚠️ Warning Signs:**
    - **Skewed distribution**: Asymmetric residuals may indicate CNN bias
    - **Heavy tails**: Too many extreme errors suggest CNN struggles with outliers
    - **Multiple peaks**: Bimodal distribution may indicate CNN learning different patterns for different regimes
    - **Very wide spread**: High standard deviation suggests CNN uncertainty
    
    **🎯 CNN-Specific Usage:**
    - **Local pattern quality**: Normal distribution suggests CNN filters are capturing local patterns well
    - **Bias detection**: Skewness indicates if CNN systematically over/under-predicts
    - **Kernel effectiveness**: Distribution shape reflects how well CNN kernels capture local temporal patterns
    - **Confidence assessment**: Distribution shape affects confidence interval reliability
    """)
    
    # Actual vs Predicted
    st.subheader("📈 CNN Actual vs Predicted (Test Set)")
    fig_scatter = px.scatter(x=y_test_inv, y=test_pred_inv, title="CNN: Actual vs Predicted")
    fig_scatter.add_trace(go.Scatter(x=[y_test_inv.min(), y_test_inv.max()], y=[y_test_inv.min(), y_test_inv.max()], mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Actual vs Predicted interpretation
    st.markdown("""
    **🔍 CNN Actual vs Predicted Interpretation:**
    
    **✅ Good Signs:**
    - **Points close to diagonal**: CNN predictions closely match actual values
    - **Random scatter**: No systematic patterns in prediction errors
    - **Good coverage**: Points spread across the range of actual values
    - **No clustering**: Predictions don't cluster in specific regions
    
    **⚠️ Warning Signs:**
    - **Systematic deviation**: Points consistently above/below diagonal indicate bias
    - **Poor coverage**: CNN struggles with certain ranges of values
    - **Clustering**: Predictions cluster in specific regions (CNN may be overfitting to local patterns)
    - **Outliers**: Points far from diagonal indicate CNN struggles with certain patterns
    
    **🎯 CNN-Specific Usage:**
    - **Local pattern accuracy**: Distance from diagonal shows CNN's ability to capture local temporal patterns
    - **Bias assessment**: Systematic deviation indicates CNN learning bias in local patterns
    - **Kernel size validation**: Poor coverage may indicate inappropriate kernel size
    - **Filter effectiveness**: Clustering may indicate CNN needs more or different filters
    """)
    
    # Comprehensive CNN diagnostics guide
    st.subheader("📚 CNN-Specific Diagnostics Guide")
    
    st.markdown("""
    **🎯 When to Use Each CNN Diagnostic:**
    
    **1. Residuals Over Time:**
    - **Use when**: You want to check if CNN captured local temporal patterns effectively
    - **Look for**: Random scatter, no temporal patterns
    - **Action**: If patterns exist, adjust kernel size or increase filters
    
    **2. Residual Distribution:**
    - **Use when**: You want to validate CNN prediction quality for local patterns
    - **Look for**: Normal distribution, centered at zero
    - **Action**: If skewed, check data scaling or adjust CNN architecture
    
    **3. Actual vs Predicted:**
    - **Use when**: You want to assess CNN's ability to capture local temporal patterns
    - **Look for**: Points close to diagonal, good coverage
    - **Action**: If poor coverage, consider different kernel size or more filters
    
    **4. Metrics Table:**
    - **Use when**: You want to quantify CNN performance on local pattern recognition
    - **Look for**: Low RMSE/MAE, high R²
    - **Action**: If metrics are poor, adjust CNN hyperparameters or consider different architecture
    """)
    
    st.markdown("""
    **🔬 CNN-Specific Model Validation:**
    
    **Step 1: Local Pattern Assessment**
    - Ensure residuals show no temporal patterns (CNN captured local dependencies)
    - Check if CNN handles different time periods equally well
    - Verify that convolutional filters are working effectively
    - Validate that kernel size is appropriate for the temporal patterns
    
    **Step 2: Filter Effectiveness Validation**
    - Check if CNN has enough filters to capture local patterns
    - Validate that kernel size matches the temporal scale of patterns
    - Ensure CNN is not overfitting to local patterns
    - Compare performance with other models for local vs global pattern capture
    
    **Step 3: Prediction Quality Assessment**
    - Validate that predictions are normally distributed around actual values
    - Check for systematic bias in predictions
    - Assess prediction accuracy across different value ranges
    - Consider CNN's focus on local patterns vs long-term dependencies
    
    **Step 4: Architecture Validation**
    - Determine if CNN has enough filters for the task
    - Check if the kernel size is appropriate for temporal patterns
    - Validate that dropout is effective for regularization
    - Assess CNN's suitability for the specific forecasting task
    """)
    
    st.markdown("""
    **💡 CNN-Specific Best Practices:**
    
    **For Data Scientists:**
    - Always scale data before CNN training (MinMaxScaler or StandardScaler)
    - Choose kernel size based on temporal pattern characteristics
    - Use appropriate number of filters for the complexity of local patterns
    - Consider CNN's limitations with long-term dependencies
    
    **For Researchers:**
    - Document CNN hyperparameters and training process
    - Compare CNN diagnostics with LSTM/GRU for local vs global pattern capture
    - Use diagnostics to guide CNN architecture improvements
    - Validate CNN assumptions about local temporal patterns
    
    **For Business Users:**
    - Understand that CNN focuses on local temporal patterns
    - Monitor CNN performance over time as new data becomes available
    - Consider CNN's strengths in capturing local patterns vs limitations with long-term dependencies
    - Use CNN diagnostics to assess forecast reliability for local pattern-based predictions
    """)
    
    st.info("""
    **📌 CNN-Specific Notes:** 
    - CNN diagnostics focus on local temporal pattern capture rather than long-term dependencies
    - CNN is effective for capturing local patterns but may miss long-term trends
    - Focus on kernel size effectiveness and filter performance
    - CNN confidence intervals are based on residual variance and may not capture all uncertainty sources
    - CNN is often preferred when local temporal patterns are more important than long-term dependencies
    """)
    # Metrics table
    st.subheader("📋 Metrics Table")
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MAPE", "R²"],
        "Value": [rmse, mae, mape, r2]
    })
    st.dataframe(metrics_df, use_container_width=True)

# === Tab 3: Model Summary ===
with tab3:
    st.title("📋 CNN Model Summary & Explanation")
    st.subheader("🔍 What is CNN?")
    st.markdown("""
    **CNN (Convolutional Neural Network)** is a deep learning model that uses convolutional layers to extract local temporal patterns from time series data. CNNs are effective for capturing short-term dependencies and local features, making them suitable for time series with strong local patterns or seasonality.
    """)
    st.subheader("📐 CNN Model Components")
    st.markdown("""
    - **Input Window (Lags):** Number of past quarters used for each prediction.
    - **CNN Filters:** Number of convolutional filters (feature detectors).
    - **Kernel Size:** Width of the convolutional filter (number of time steps).
    - **Dropout:** Fraction of units dropped during training to prevent overfitting.
    - **Epochs:** Number of times the model sees the training data.
    - **Batch Size:** Number of samples per gradient update.
    - **Learning Rate:** Step size for optimizer updates.
    - **Validation Split:** Fraction of training data used for validation.
    - **Early Stopping:** Stops training if validation loss doesn't improve.
    """)
    st.subheader("🔄 How CNN Works")
    st.markdown("""
    1. **Data Preparation:**
       - Data is scaled for stable neural network training.
       - Lagged input windows are created.
    2. **Model Architecture:**
       - Conv1D layer(s) extract local temporal features from the input window.
       - Dropout layer(s) help prevent overfitting.
       - Flatten and Dense layers output the forecast.
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
    - **CNN Filters:**
      - More filters allow the model to learn more complex local patterns.
      - Too many can overfit, especially with limited data.
    - **Kernel Size:**
      - Larger kernels capture broader patterns, but too large may smooth out important details.
    - **Dropout:**
      - Higher dropout reduces overfitting but may underfit if too high.
    - **Epochs/Batch Size/Learning Rate:**
      - Affect training speed, convergence, and generalization.
    """)
    st.subheader("🔬 Why CNN for This Data?")
    st.markdown("""
    - **Local Pattern Recognition:**
      - CNNs excel at capturing local, short-term patterns in time series.
    - **Parameter Sharing:**
      - Convolutional filters share parameters, making the model efficient.
    - **Translation Invariance:**
      - CNNs can recognize patterns regardless of their position in the input window.
    - **Handles Seasonality:**
      - Effective for time series with strong seasonal or local patterns.
    """)
    st.subheader("💡 Practical Tips")
    st.markdown("""
    **For Policy Makers:**
    - Use CNN forecasts for planning when data shows strong local or seasonal patterns.
    - Monitor confidence intervals for risk assessment.

    **For Researchers:**
    - Tune hyperparameters (lags, filters, kernel size, dropout) for best results.
    - Compare CNN with LSTM/GRU to justify model choice.

    **For Business Users:**
    - CNN is powerful for local pattern recognition, but less interpretable than ARIMA/SARIMA.
    - Use diagnostics and explainability tools to understand model decisions.
    """)
    st.subheader("✅ Best Practices")
    st.markdown("""
    - Always scale your data before training CNN models.
    - Use early stopping and dropout to prevent overfitting.
    - Tune lags, filters, kernel size, and learning rate for best results.
    - Validate with out-of-sample data.
    """)
    st.subheader("⚠️ Limitations")
    st.markdown("""
    - May miss long-term dependencies compared to LSTM/GRU.
    - Can overfit if too complex or if data is too short.
    - Confidence intervals are approximate (based on residual std).
    - Not as interpretable as ARIMA/SARIMA.
    """)

# === Tab 4: Explainability ===
with tab4:
    st.title("🧠 CNN Model Explainability")
    st.markdown("### Feature (Lag) Importance")
    try:
        import shap
        explainer = shap.DeepExplainer(model, X_train[:100])
        shap_values = explainer.shap_values(X_test[:10])
        import matplotlib.pyplot as plt
        plt.bar(range(n_lags), np.abs(shap_values[0][0]))
        plt.xlabel('Lag')
        plt.ylabel('SHAP Value')
        plt.title('Lag Importance for Forecast')
        st.pyplot(plt)
        st.markdown("Higher SHAP values for a lag mean that quarter was more influential for the forecast.")
    except Exception as e:
        st.warning("SHAP not available or failed. Showing permutation importance instead.")
        # Permutation importance fallback
        importances = np.zeros(n_lags)
        base_pred = model.predict(X_test).flatten()
        base_rmse = np.sqrt(np.mean((y_test - base_pred) ** 2))
        for i in range(n_lags):
            X_test_perm = X_test.copy()
            np.random.shuffle(X_test_perm[:, i, 0])
            perm_pred = model.predict(X_test_perm).flatten()
            perm_rmse = np.sqrt(np.mean((y_test - perm_pred) ** 2))
            importances[i] = perm_rmse - base_rmse
        fig = px.bar(x=list(range(1, n_lags+1)), y=importances, labels={"x": "Lag", "y": "Importance (RMSE increase)"}, title="Permutation Lag Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Higher values mean the model relies more on that lag for accurate forecasts.")
    st.markdown("""
    **What does this mean?**
    - Higher importance for a lag means the model relies more on that quarter for its forecast.
    - This helps you understand which past periods the model uses most.
    """)
    st.markdown("### Saliency Map (Gradient-based)")
    try:
        import tensorflow as tf
        # Use the first test sample
        x_input = tf.convert_to_tensor(X_test[:1])
        with tf.GradientTape() as tape:
            tape.watch(x_input)
            pred = model(x_input)
        grads = tape.gradient(pred, x_input).numpy().flatten()
        fig_sal = px.bar(x=list(range(1, n_lags+1)), y=np.abs(grads), labels={"x": "Lag", "y": "Gradient Magnitude"}, title="Saliency Map (Input Window)")
        st.plotly_chart(fig_sal, use_container_width=True)
        st.markdown("Higher gradient magnitude means the model's prediction is more sensitive to that lag.")
    except Exception as e:
        st.warning("Saliency map not available: " + str(e)) 