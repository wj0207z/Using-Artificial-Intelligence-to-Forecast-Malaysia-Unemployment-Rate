import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Time Series Models
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

# Machine Learning Models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

def create_lagged_data(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_lstm(series, test_size, n_lags=8, n_units=32, epochs=50, batch_size=8, learning_rate=0.001, return_future=False):
    try:
        # Use improved implementation similar to LSTM.py
        train_series = series.iloc[:-test_size]
        test_series = series.iloc[-test_size:]
        
        # Scale data
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        
        # Split with proper lag handling
        train_scaled = series_scaled[:len(train_series)]
        test_scaled = series_scaled[len(train_series)-n_lags:]  # include lags from end of train
        
        # Create sequences
        X_train, y_train = create_lagged_data(train_scaled, n_lags)
        X_test, y_test = create_lagged_data(test_scaled, n_lags)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with better architecture
        model = Sequential([
            LSTM(n_units, input_shape=(n_lags, 1)),
            Dropout(0.1),  # Reduced dropout for better performance
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        # Train with early stopping
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                 validation_split=0.2, callbacks=[es], verbose=0)
        
        # Forecast on test set
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Future forecast
        future_forecast = None
        if return_future:
            n_future = 4
            last_values = series.values[-n_lags:]
            preds = []
            current_input = last_values.copy()
            for _ in range(n_future):
                scaled_input = scaler.transform(current_input.reshape(-1, 1)).flatten()
                X_input = scaled_input.reshape((1, n_lags, 1))
                pred_scaled = model.predict(X_input, verbose=0)[0, 0]
                pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
                preds.append(pred)
                current_input = np.append(current_input[1:], pred)
            future_forecast = np.array(preds)
            return y_pred, None, future_forecast
        else:
            return y_pred, None
    except:
        if return_future:
            return None, None, None
        else:
            return None, None

def train_gru(series, test_size, n_lags=8, n_units=32, epochs=50, batch_size=8, learning_rate=0.001, return_future=False):
    try:
        # Use improved implementation similar to GRU.py
        train_series = series.iloc[:-test_size]
        test_series = series.iloc[-test_size:]
        
        # Scale data
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        
        # Split with proper lag handling
        train_scaled = series_scaled[:len(train_series)]
        test_scaled = series_scaled[len(train_series)-n_lags:]  # include lags from end of train
        
        # Create sequences
        X_train, y_train = create_lagged_data(train_scaled, n_lags)
        X_test, y_test = create_lagged_data(test_scaled, n_lags)
        
        # Reshape for GRU [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with better architecture
        model = Sequential([
            GRU(n_units, input_shape=(n_lags, 1)),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        # Train with early stopping
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                 validation_split=0.2, callbacks=[es], verbose=0)
        
        # Forecast on test set
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Future forecast
        future_forecast = None
        if return_future:
            n_future = 4
            last_values = series.values[-n_lags:]
            preds = []
            current_input = last_values.copy()
            for _ in range(n_future):
                scaled_input = scaler.transform(current_input.reshape(-1, 1)).flatten()
                X_input = scaled_input.reshape((1, n_lags, 1))
                pred_scaled = model.predict(X_input, verbose=0)[0, 0]
                pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
                preds.append(pred)
                current_input = np.append(current_input[1:], pred)
            future_forecast = np.array(preds)
            return y_pred, None, future_forecast
        else:
            return y_pred, None
    except:
        if return_future:
            return None, None, None
        else:
            return None, None

def train_cnn(series, test_size, n_lags=8, n_filters=32, kernel_size=4, epochs=50, batch_size=8, learning_rate=0.001, return_future=False):
    try:
        # Use improved implementation similar to CNN.py
        train_series = series.iloc[:-test_size]
        test_series = series.iloc[-test_size:]
        
        # Scale data
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        
        # Split with proper lag handling
        train_scaled = series_scaled[:len(train_series)]
        test_scaled = series_scaled[len(train_series)-n_lags:]  # include lags from end of train
        
        # Create sequences
        X_train, y_train = create_lagged_data(train_scaled, n_lags)
        X_test, y_test = create_lagged_data(test_scaled, n_lags)
        
        # Reshape for CNN [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with better architecture
        model = Sequential([
            Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', input_shape=(n_lags, 1)),
            Dropout(0.1),  # Reduced dropout
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        # Train with early stopping
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                 validation_split=0.2, callbacks=[es], verbose=0)
        
        # Forecast on test set
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Future forecast
        future_forecast = None
        if return_future:
            n_future = 4
            last_values = series.values[-n_lags:]
            preds = []
            current_input = last_values.copy()
            for _ in range(n_future):
                scaled_input = scaler.transform(current_input.reshape(-1, 1)).flatten()
                X_input = scaled_input.reshape((1, n_lags, 1))
                pred_scaled = model.predict(X_input, verbose=0)[0, 0]
                pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
                preds.append(pred)
                current_input = np.append(current_input[1:], pred)
            future_forecast = np.array(preds)
            return y_pred, None, future_forecast
        else:
            return y_pred, None
    except:
        if return_future:
            return None, None, None
        else:
            return None, None

def train_rnn(series, test_size, n_lags=8, n_units=16, epochs=50, batch_size=8, learning_rate=0.001, return_future=False):
    try:
        # Use improved implementation similar to RNN.py
        train_series = series.iloc[:-test_size]
        test_series = series.iloc[-test_size:]
        
        # Scale data
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        
        # Split with proper lag handling
        train_scaled = series_scaled[:len(train_series)]
        test_scaled = series_scaled[len(train_series)-n_lags:]  # include lags from end of train
        
        # Create sequences
        X_train, y_train = create_lagged_data(train_scaled, n_lags)
        X_test, y_test = create_lagged_data(test_scaled, n_lags)
        
        # Reshape for RNN [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build model with better architecture
        model = Sequential([
            SimpleRNN(n_units, input_shape=(n_lags, 1), activation='tanh'),
            Dropout(0.1),  # Reduced dropout
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        # Train with early stopping
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                 validation_split=0.2, callbacks=[es], verbose=0)
        
        # Forecast on test set
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Future forecast
        future_forecast = None
        if return_future:
            n_future = 4
            last_values = series.values[-n_lags:]
            preds = []
            current_input = last_values.copy()
            for _ in range(n_future):
                scaled_input = scaler.transform(current_input.reshape(-1, 1)).flatten()
                X_input = scaled_input.reshape((1, n_lags, 1))
                pred_scaled = model.predict(X_input, verbose=0)[0, 0]
                pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
                preds.append(pred)
                current_input = np.append(current_input[1:], pred)
            future_forecast = np.array(preds)
            return y_pred, None, future_forecast
        else:
            return y_pred, None
    except:
        if return_future:
            return None, None, None
        else:
            return None, None

def train_arima(series, test_size, force_seasonal=None, return_model=False):
    train_series = series.iloc[:-test_size]
    # Detect seasonality if not specified
    if force_seasonal is None:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(series, model='additive', period=4)
        seasonal_component = decomposition.seasonal
        seasonality_strength = abs(seasonal_component).max()
        force_seasonal = seasonality_strength > 0.5
    model = pm.auto_arima(
        train_series,
        seasonal=force_seasonal,
        m=4 if force_seasonal else 1,
        max_d=1,
        D=1 if force_seasonal else 0,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    forecast = model.predict(n_periods=test_size)
    aic = model.aic()
    if return_model:
        return forecast, aic, model
    else:
        return forecast, aic

def train_sarima(series, test_size, seasonal_period=4, return_model=False):
    train_series = series.iloc[:-test_size]
    model = pm.auto_arima(
        train_series,
        seasonal=True,
        m=seasonal_period,
        max_d=2,
        max_D=2,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        D=1,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    forecast = model.predict(n_periods=test_size)
    aic = model.aic()
    if return_model:
        return forecast, aic, model
    else:
        return forecast, aic

def train_exponential_smoothing(train_data, test_periods, return_model=False):
    """Train Exponential Smoothing model and make predictions (Holt-Winters only)"""
    try:
        model_hw = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=4)
        fitted_hw = model_hw.fit()
        forecast = fitted_hw.forecast(test_periods)
        if return_model:
            return forecast, fitted_hw.aic, fitted_hw
        else:
            return forecast, fitted_hw.aic
    except:
        if return_model:
            return None, None, None
        else:
            return None, None

def calculate_safe_mape(y_true, y_pred):
    """Calculate MAPE while handling zero values in y_true"""
    if len(y_true) == 0:
        return None
    # Remove pairs where true value is zero
    non_zero_mask = y_true != 0
    y_true_safe = y_true[non_zero_mask]
    y_pred_safe = y_pred[non_zero_mask]
    
    if len(y_true_safe) == 0:
        return None
    
    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100

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
st.markdown("### 🔄 Train/Test Split Configuration")
st.info("""
**Model Comparison Setup:**
This page will train all forecasting models on the same training data and evaluate them on the same test set for fair comparison.
""")

# Set default train set size to 80% of the data, test set to 20%
total_points = len(series)
train_size = int(total_points * 0.8)
test_size = total_points - train_size

st.markdown(f"""
**📊 Data Split:**
- **Total Observations**: {total_points}
- **Training Set**: {train_size} quarters ({train_size/total_points*100:.1f}%)
- **Test Set**: {test_size} quarters ({test_size/total_points*100:.1f}%)
- **Training Period**: {series.index[0].strftime('%Y-%m')} to {series.index[train_size-1].strftime('%Y-%m')}
- **Test Period**: {series.index[train_size].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')}
""")

# Split the data with proper date handling
train_series = series[:train_size].copy()
test_series = series[train_size:].copy()
test_dates = test_series.index
test_series.index = pd.to_datetime(test_series.index).strftime('%Y-%m-%d')

# Ensure data is properly scaled for each model type
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
train_scaled = series_scaled[:train_size]
test_scaled = series_scaled[train_size:]

# === Model Configuration ===
st.markdown("### ⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📈 Time Series Models:**")
    use_arima = st.checkbox("ARIMA", value=True)
    use_sarima = st.checkbox("SARIMA", value=True)
    use_exp_smooth = st.checkbox("Exponential Smoothing", value=True)

with col2:
    st.markdown("**🧠 Deep Learning Models:**")
    use_lstm = st.checkbox("LSTM", value=True)
    use_gru = st.checkbox("GRU", value=True)
    use_cnn = st.checkbox("CNN", value=True)
    use_rnn = st.checkbox("RNN", value=True)

# === Model Training Parameters ===
st.markdown("### 🛠️ Model Parameters")
with st.expander("Advanced Model Settings"):
    col1, col2 = st.columns(2)
    with col1:
        n_lags = st.slider("Number of Lag Quarters:", min_value=4, max_value=16, value=8, step=1)
        with st.popover(":red[❓]"):
            st.markdown("""
            **Number of Lag Quarters (input window):**
            How many past quarters the model uses to predict the next value.
            - More lags can capture longer-term dependencies in the data.
            - Too many lags may add noise or cause overfitting.
            - Typical values: 4-12 for quarterly data.
            **How it affects the model:**
            - **Too low:** The model may miss important patterns and dependencies, leading to underfitting and poor forecasts.
            - **Too high:** The model may overfit to noise, become less stable, and require more data to train effectively.
            - **Best practice:** Start with 4-8 for quarterly data and tune based on validation performance.
            """)
        n_epochs = st.slider("Training Epochs:", min_value=10, max_value=200, value=50, step=10)
        with st.popover(":red[❓]"):
            st.markdown("""
            **Epochs:**
            Number of times the model sees the entire training data.
            - More epochs can improve learning but may overfit.
            - Use early stopping to prevent overfitting.
            **How it affects the model:**
            - **Too low:** The model may underfit and not learn the data patterns well.
            - **Too high:** The model may overfit, memorizing noise instead of generalizing.
            - **Best practice:** Use early stopping and monitor validation loss to find the optimal number.
            """)
        batch_size = st.slider("Batch Size:", min_value=4, max_value=64, value=8, step=4)
        with st.popover(":red[❓]"):
            st.markdown("""
            **Batch Size:**
            Number of samples processed before the model is updated.
            - Smaller batch sizes can improve generalization.
            - Larger batch sizes can speed up training but may overfit.
            **How it affects the model:**
            - **Small batch:** Slower training, but can help the model generalize better and escape local minima.
            - **Large batch:** Faster training, but may lead to poorer generalization and overfitting.
            - **Best practice:** Try 8-32 for most time series problems.
            """)
        learning_rate = st.slider("Learning Rate:", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
        with st.popover(":red[❓]"):
            st.markdown("""
            **Learning Rate:**
            Step size for updating model weights.
            - Lower values make learning slower but more stable.
            - Too high values can cause the model to diverge.
            **How it affects the model:**
            - **Too low:** Training is slow and may get stuck before reaching a good solution.
            - **Too high:** The model may not converge and can oscillate or diverge.
            - **Best practice:** Start with 0.001 and adjust if the model is not learning or is unstable.
            """)
    with col2:
        seasonal_period = st.selectbox("Seasonal Period:", [4, 12], index=0, help="4 for quarterly data, 12 for monthly data")
        with st.popover(":red[❓]"):
            st.markdown("""
            **Seasonal Period (m):**
            The number of periods in a full seasonal cycle.
            - For quarterly data, use 4. For monthly data, use 12.
            - Controls the length of the repeating seasonal pattern.
            **How it affects the model:**
            - **Incorrect value:** The model may miss or misinterpret seasonal effects, leading to poor forecasts.
            - **Correct value:** The model can accurately capture and forecast repeating seasonal patterns.
            - **Best practice:** Match this to your data's true seasonality (e.g., 4 for quarterly, 12 for monthly).
            """)
        force_seasonal = st.checkbox("Force Seasonal Models", value=True, help="Use seasonal variants of models where applicable")
        with st.popover(":red[❓]"):
            st.markdown("""
            **Force Seasonal Models:**
            - If checked, ARIMA will be forced to use seasonal components.
            - If unchecked, ARIMA will be non-seasonal.
            - SARIMA always uses seasonality.
            **How it affects the model:**
            - **Enabled:** The model will try to capture repeating seasonal patterns, which is important for data with strong seasonality.
            - **Disabled:** The model will ignore seasonality, which may be better for non-seasonal data.
            - **Best practice:** Enable for data with clear seasonal cycles; disable for non-seasonal data.
            """)

# === Feature Engineering Function ===
def create_features(series, n_lags=8, include_seasonal=True):
    """Create comprehensive features for time series forecasting"""
    df_features = pd.DataFrame(series)
    
    # Lag features
    for lag in range(1, n_lags + 1):
        df_features[f'lag_{lag}'] = df_features.iloc[:, 0].shift(lag)
    
    # Rolling statistics
    df_features['rolling_mean_4'] = df_features.iloc[:, 0].rolling(window=4).mean()
    df_features['rolling_std_4'] = df_features.iloc[:, 0].rolling(window=4).std()
    df_features['rolling_mean_8'] = df_features.iloc[:, 0].rolling(window=8).mean()
    df_features['rolling_std_8'] = df_features.iloc[:, 0].rolling(window=8).std()
    
    # Trend features
    df_features['trend'] = range(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    
    # Seasonal features
    if include_seasonal:
        df_features['quarter'] = df_features.index.quarter
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    
    # Difference features
    df_features['diff_1'] = df_features.iloc[:, 0].diff()
    df_features['diff_2'] = df_features.iloc[:, 0].diff().diff()
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

# === Model Training and Prediction Functions ===
# --- Update train_arima ---
def train_arima(series, test_size, force_seasonal=None, return_model=False):
    train_series = series.iloc[:-test_size]
    # Detect seasonality if not specified
    if force_seasonal is None:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(series, model='additive', period=4)
        seasonal_component = decomposition.seasonal
        seasonality_strength = abs(seasonal_component).max()
        force_seasonal = seasonality_strength > 0.5
    model = pm.auto_arima(
        train_series,
        seasonal=force_seasonal,
        m=4 if force_seasonal else 1,
        max_d=1,
        D=1 if force_seasonal else 0,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    forecast = model.predict(n_periods=test_size)
    aic = model.aic()
    if return_model:
        return forecast, aic, model
    else:
        return forecast, aic

# --- Update train_sarima ---
def train_sarima(series, test_size, seasonal_period=4, return_model=False):
    train_series = series.iloc[:-test_size]
    model = pm.auto_arima(
        train_series,
        seasonal=True,
        m=seasonal_period,
        max_d=2,
        max_D=2,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        D=1,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    forecast = model.predict(n_periods=test_size)
    aic = model.aic()
    if return_model:
        return forecast, aic, model
    else:
        return forecast, aic

# --- Update train_exponential_smoothing ---
def train_exponential_smoothing(train_data, test_periods, return_model=False):
    """Train Exponential Smoothing model and make predictions (Holt-Winters only)"""
    try:
        model_hw = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=4)
        fitted_hw = model_hw.fit()
        forecast = fitted_hw.forecast(test_periods)
        if return_model:
            return forecast, fitted_hw.aic, fitted_hw
        else:
            return forecast, fitted_hw.aic
    except:
        if return_model:
            return None, None, None
        else:
            return None, None



# === Train All Models ===
st.markdown("### 🚀 Training Models...")

models_results = {}
progress_bar = st.progress(0)
status_text = st.empty()

progress_bar.progress(10)

# Train ARIMA
# --- ARIMA future forecast with full series ---
if use_arima:
    status_text.text("Training ARIMA...")
    forecast, aic, model_obj = train_arima(train_series, test_size, force_seasonal=False, return_model=True)
    if forecast is not None:
        # Fit a new model on the full series for future forecast
        try:
            model_full = pm.auto_arima(
                series,
                seasonal=False,
                m=1,
                max_d=1,
                D=0,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            future_forecast = model_full.predict(n_periods=4)
            print('ARIMA future forecast (full series):', future_forecast)
        except Exception as e:
            print('ARIMA future forecast error:', e)
            future_forecast = [None]*4
        models_results['ARIMA'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'model_obj': model_obj,
            'future_forecast': future_forecast
        }
    progress_bar.progress(14)

# Train SARIMA
# --- SARIMA future forecast with full series ---
if use_sarima:
    status_text.text("Training SARIMA...")
    forecast, aic, model_obj = train_sarima(train_series, test_size, seasonal_period=seasonal_period, return_model=True)
    if forecast is not None:
        try:
            model_full = pm.auto_arima(
                series,
                seasonal=True,
                m=seasonal_period,
                max_d=2,
                max_D=2,
                max_p=3,
                max_q=3,
                max_P=2,
                max_Q=2,
                D=1,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            future_forecast = model_full.predict(n_periods=4)
            print('SARIMA future forecast (full series):', future_forecast)
        except Exception as e:
            print('SARIMA future forecast error:', e)
            future_forecast = [None]*4
        models_results['SARIMA'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'model_obj': model_obj,
            'future_forecast': future_forecast
        }
    progress_bar.progress(28)

# Train Exponential Smoothing
# --- Exponential Smoothing future forecast with full series ---
if use_exp_smooth:
    status_text.text("Training Exponential Smoothing...")
    forecast, aic, model_obj = train_exponential_smoothing(train_series, test_size, return_model=True)
    if forecast is not None:
        try:
            model_hw = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_period)
            fitted_hw = model_hw.fit()
            future_forecast = fitted_hw.forecast(4)
            print('ExpSmooth future forecast (full series):', future_forecast)
        except Exception as e:
            print('ExpSmooth future forecast error:', e)
            future_forecast = [None]*4
        models_results['Exponential Smoothing'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'model_obj': model_obj,
            'future_forecast': future_forecast
        }
    progress_bar.progress(42)

# Train LSTM
if use_lstm:
    status_text.text("Training LSTM...")
    forecast, aic, future_forecast = train_lstm(series, test_size, n_lags=n_lags, n_units=32, 
                             epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, return_future=True)
    if forecast is not None:
        models_results['LSTM'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'future_forecast': future_forecast
        }
    progress_bar.progress(57)

# Train GRU
if use_gru:
    status_text.text("Training GRU...")
    forecast, aic, future_forecast = train_gru(series, test_size, n_lags=n_lags, n_units=32, 
                            epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, return_future=True)
    if forecast is not None:
        models_results['GRU'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'future_forecast': future_forecast
        }
    progress_bar.progress(71)

# Train CNN
if use_cnn:
    status_text.text("Training CNN...")
    forecast, aic, future_forecast = train_cnn(series, test_size, n_lags=n_lags, n_filters=32, kernel_size=4,
                             epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, return_future=True)
    if forecast is not None:
        models_results['CNN'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'future_forecast': future_forecast
        }
    progress_bar.progress(85)

# Train RNN
if use_rnn:
    status_text.text("Training RNN...")
    forecast, aic, future_forecast = train_rnn(series, test_size, n_lags=n_lags, n_units=16,
                             epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate, return_future=True)
    if forecast is not None:
        models_results['RNN'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': calculate_safe_mape(test_series.values, forecast),
            'r2': r2_score(test_series, forecast),
            'future_forecast': future_forecast
        }
    progress_bar.progress(100)

status_text.text("✅ All models trained successfully!")
progress_bar.empty()

# === Add Combined Forecast (Ensemble) ===
import numpy as np
all_forecasts = []
for model_name, results in models_results.items():
    if results is not None and 'forecast' in results:
        all_forecasts.append(np.array(results['forecast']))

if all_forecasts:
    combined_forecast = np.mean(all_forecasts, axis=0)
    # Compute metrics for combined forecast
    combined_rmse = np.sqrt(np.mean((test_series.values - combined_forecast) ** 2))
    combined_mae = np.mean(np.abs(test_series.values - combined_forecast))
    # Use safe MAPE function
    combined_mape = calculate_safe_mape(test_series.values, combined_forecast)

    models_results['Combined Forecast'] = {
        'forecast': combined_forecast,
        'aic': 'Not Applicable',
        'rmse': combined_rmse,
        'mae': combined_mae,
        'mape': combined_mape,
        'r2': None
    }

# Rebuild performance_df without R²
performance_data = []
for model_name, results in models_results.items():
    # Only add models that have valid results
    if results is not None and 'rmse' in results and 'mae' in results and 'mape' in results:
        model_metrics = {
            'Model': model_name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'MAPE (%)': results['mape']
        }
        
        # Add AIC only for statistical models
        if model_name in ['ARIMA', 'SARIMA', 'Exponential Smoothing']:
            model_metrics['AIC'] = results['aic'] if results['aic'] is not None else 'N/A'
        else:
            model_metrics['AIC'] = 'Not Applicable'  # For deep learning models
            
        performance_data.append(model_metrics)

if performance_data:
    performance_df = pd.DataFrame(performance_data)
    
    # Add tooltip explanation for AIC
    st.markdown("""
    <details>
    <summary>ℹ️ About AIC Values</summary>
    
    **AIC (Akaike Information Criterion):**
    - Available for statistical models (ARIMA, SARIMA, Exponential Smoothing)
    - Not applicable to deep learning models (LSTM, GRU, CNN, RNN)
    - Lower AIC values indicate better model fit
    
    Deep learning models use different metrics for model selection:
    - Validation loss
    - Cross-validation scores
    - Model complexity penalties
    </details>
    """, unsafe_allow_html=True)
else:
    performance_df = pd.DataFrame()

# === Tabs ===
tab_overview, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Overview Summary",
    # "📊 Performance Comparison",  # Tab 1 commented out
    "📈 Forecast Visualization",
    "🔍 Detailed Analysis",
    "🏆 Model Rankings",
    "📝 Methodology & Code"
])

# === Overview Summary Tab ===
with tab_overview:
    st.title("📝 Overview Summary")
    
    if not models_results:
        st.warning("No models were successfully trained.")
    else:
        # Handle cases where some models failed to train
        if not performance_df.empty and not performance_df['RMSE'].isna().all():
            best_rmse_model = performance_df.loc[performance_df['RMSE'].idxmin(), 'Model']
        else:
            best_rmse_model = "No valid models"
            
        if not performance_df.empty and not performance_df['MAPE (%)'].isna().all():
            best_mape_model = performance_df.loc[performance_df['MAPE (%)'].idxmin(), 'Model']
        else:
            best_mape_model = "No valid models"
            
        st.markdown(f"""
        This report provides a comprehensive comparison of multiple time series forecasting models, including ARIMA, SARIMA, Exponential Smoothing, LSTM, GRU, CNN, and RNN. Each model was trained and evaluated on the same historical data and test set to ensure a fair comparison.
        
        **Key Findings:**
        - The model with the lowest RMSE (Root Mean Squared Error) was **{best_rmse_model}**, indicating the most accurate predictions in terms of absolute error.
        - The model with the lowest MAPE (Mean Absolute Percentage Error) was **{best_mape_model}**, reflecting the best relative accuracy.
        
        **Model Strengths and Weaknesses:**
        - **ARIMA/SARIMA:** Strong for linear trends and seasonality, highly interpretable, but may miss nonlinear patterns.
        - **Exponential Smoothing:** Fast and robust for short-term and seasonal data, but limited for complex or nonlinear series.
        - **LSTM/GRU:** Excellent for capturing long-term dependencies and nonlinearities, but require more data and tuning.
        - **CNN:** Effective for local and short-term patterns, fast to train, but may miss long-term dependencies.
        - **RNN:** Simple baseline for sequential data, but less effective for long-term patterns due to vanishing gradients.
        
        **Practical Recommendations:**
        - For interpretability and transparency, ARIMA, SARIMA, and Exponential Smoothing are preferred.
        - For highest accuracy and complex patterns, LSTM or GRU are recommended if sufficient data and expertise are available.
        - Always validate model performance on out-of-sample data and consider both error metrics and interpretability for decision-making.
        
        Please refer to the other tabs for detailed performance metrics, visualizations, and methodology.
        """)
        
        # Show performance metrics if available
        if not performance_df.empty:
            st.subheader("📋 Performance Metrics")
            # Remove Combined Forecast from metrics table if present
            perf_df_no_combined = performance_df[performance_df['Model'] != 'Combined Forecast'] if 'Model' in performance_df.columns else performance_df
            st.dataframe(perf_df_no_combined, use_container_width=True)

            # --- Add bar charts for each metric ---
            import plotly.express as px
            st.subheader("📊 Model Comparison by Metric")
            # RMSE
            fig_rmse = px.bar(performance_df, x='Model', y='RMSE', title='RMSE by Model', text='RMSE')
            st.plotly_chart(fig_rmse, use_container_width=True)
            # MAE
            fig_mae = px.bar(performance_df, x='Model', y='MAE', title='MAE by Model', text='MAE')
            st.plotly_chart(fig_mae, use_container_width=True)
            # MAPE
            fig_mape = px.bar(performance_df, x='Model', y='MAPE (%)', title='MAPE (%) by Model', text='MAPE (%)')
            st.plotly_chart(fig_mape, use_container_width=True)
            # R² (if available)
            if 'R²' in performance_df.columns:
                fig_r2 = px.bar(performance_df, x='Model', y='R²', title='R² by Model', text='R²')
                st.plotly_chart(fig_r2, use_container_width=True)
        else:
            st.warning("No performance metrics available - all models failed to train.")

    # === Which Model is Better? (Only in Tab 1) ===
    st.markdown("---")
    st.header("🏆 Which Model is Better?")
    st.markdown("""
    ### **Model Selection Guidance**

    Choosing the "best" model depends on your goals, data characteristics, and how you value accuracy vs interpretability. Here’s a summary to help you decide:

    #### **1. Statistical Models**
    - **ARIMA**
      - **Best for:** Data with trend, little or no seasonality, high interpretability
      - **Strengths:** Simple, transparent, easy to explain
      - **Limitations:** May miss complex or nonlinear patterns
    - **SARIMA**
      - **Best for:** Data with strong, regular seasonality (e.g., quarterly cycles)
      - **Strengths:** Captures both trend and seasonal patterns, interpretable
      - **Limitations:** Still linear, may miss nonlinearities
    - **Exponential Smoothing**
      - **Best for:** Short-term forecasts, stable or moderately seasonal data
      - **Strengths:** Fast, robust, interpretable, good baseline
      - **Limitations:** Limited for complex or highly nonlinear data

    #### **2. Deep Learning Models**
    - **LSTM**
      - **Best for:** Data with long-term dependencies, nonlinear patterns, or when accuracy is critical
      - **Strengths:** Captures complex, nonlinear, and long-memory effects
      - **Limitations:** Less interpretable, needs more data and tuning
    - **GRU**
      - **Best for:** Similar to LSTM, but when you want faster training and fewer parameters
      - **Strengths:** Efficient, good for moderate to long-term patterns
      - **Limitations:** Slightly less expressive than LSTM, still less interpretable
    - **CNN**
      - **Best for:** Data with strong local/seasonal patterns, short-term dependencies
      - **Strengths:** Excels at local pattern recognition, fast
      - **Limitations:** May miss long-term dependencies
    - **RNN**
      - **Best for:** Simple sequential patterns, as a baseline for comparison
      - **Strengths:** Simple, fast, interpretable for basic patterns
      - **Limitations:** Struggles with long-term dependencies (vanishing gradient)

    ---

    ### **Performance-Based Recommendation**

    Below are the models ranked by their test set RMSE and MAPE (lower is better):
    """)
    try:
        best_rmse_model = performance_df.loc[performance_df['RMSE'].idxmin(), 'Model']
        best_mape_model = performance_df.loc[performance_df['MAPE (%)'].idxmin(), 'Model']
        st.success(f"**Best RMSE:** {best_rmse_model} | **Best MAPE:** {best_mape_model}")
    except Exception:
        st.info("Performance metrics table not available.")
    st.markdown("""
    #### **Practical Recommendations**
    - **If you want the most interpretable model:** Use **ARIMA** or **Exponential Smoothing**.
    - **If your data has strong seasonality:** Use **SARIMA** or **Holt-Winters Exponential Smoothing**.
    - **If you want the best accuracy and can handle complexity:** Try **LSTM** or **GRU**.
    - **If you care about local/short-term patterns:** **CNN** is a strong choice.
    - **For a simple deep learning baseline:** Use **RNN**.

    #### **For Policy Makers:**
    - Prefer interpretable models (ARIMA, SARIMA, Exponential Smoothing) for transparency.
    - Use deep learning models for highest accuracy if you can explain/justify their use.

    #### **For Researchers:**
    - Compare all models, report both accuracy and interpretability.
    - Use deep learning for complex, nonlinear, or long-memory data.

    #### **For Business Users:**
    - Use the model with the best test set performance, but monitor for overfitting.
    - Consider confidence intervals and diagnostics for risk management.

    ---

    **No single model is always best.** Use the diagnostics, performance metrics, and your domain knowledge to select the most appropriate model for your needs.
    """)

# === Tab 2: Forecast Visualization ===
with tab2:
    st.title("📈 Forecast Visualization")
    
    # Create forecast comparison plot
    import plotly.graph_objects as go
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=test_series.values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='black', width=3),
        marker=dict(size=8)
    ))
    
    # Add forecasts for each model
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, results) in enumerate(models_results.items()):
        if model_name == 'Combined Forecast':
            continue
        fig.add_trace(go.Scatter(
            x=test_series.index,
            y=results['forecast'],
            mode='lines+markers',
            name=f"{model_name} (RMSE: {results['rmse']:.2f})",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"Forecast Comparison - {selected_metric_label}",
        xaxis_title="Date",
        yaxis_title=selected_metric_label,
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add improved grouped bar chart grid for metrics comparison
    st.markdown(" ")  # Add space above
    st.subheader("📊 Model Metrics Comparison Grid")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Prepare data for grid
    model_names = []
    rmse_vals = []
    mae_vals = []
    mape_vals = []
    r2_vals = []
    for model_name, results in models_results.items():
        if all(k in results for k in ['rmse', 'mae', 'mape', 'r2']):
            model_names.append(model_name)
            rmse_vals.append(results['rmse'])
            mae_vals.append(results['mae'])
            mape_vals.append(results['mape'])
            r2_vals.append(results['r2'] if results['r2'] is not None else float('nan'))

    fig_grid = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "<b>RMSE Comparison (Lower is Better)</b>",
            "<b>MAPE Comparison (Lower is Better)</b>",
            "<b>MAE Comparison (Lower is Better)</b>",
            "<b>R² Comparison (Higher is Better)</b>"
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.3  # Increased vertical spacing
    )

    bar_style = dict(marker_line_width=2, marker_line_color='white')

    # RMSE
    fig_grid.add_trace(
        go.Bar(
            x=model_names, y=rmse_vals,
            marker=dict(color=rmse_vals, colorscale='Reds', line=dict(width=2, color='white')),
            text=[f"{v:.3f}" for v in rmse_vals], textposition='outside',
            name="RMSE"
        ),
        row=1, col=1
    )
    # MAPE
    fig_grid.add_trace(
        go.Bar(
            x=model_names, y=mape_vals,
            marker=dict(color=mape_vals, colorscale='Reds', line=dict(width=2, color='white')),
            text=[f"{v:.2f}%" for v in mape_vals], textposition='outside',
            name="MAPE"
        ),
        row=1, col=2
    )
    # MAE
    fig_grid.add_trace(
        go.Bar(
            x=model_names, y=mae_vals,
            marker=dict(color=mae_vals, colorscale='Reds', line=dict(width=2, color='white')),
            text=[f"{v:.3f}" for v in mae_vals], textposition='outside',
            name="MAE"
        ),
        row=2, col=1
    )
    # R²
    fig_grid.add_trace(
        go.Bar(
            x=model_names, y=r2_vals,
            marker=dict(color=r2_vals, colorscale='Greens', line=dict(width=2, color='white')),
            text=[f"{v:.2f}" for v in r2_vals], textposition='outside',
            name="R²"
        ),
        row=2, col=2
    )

    fig_grid.update_layout(
        height=950, width=1100,
        showlegend=False,
        title_text="<b>Model Comparison Across Metrics</b>",
        title_font_size=22,
        margin=dict(t=100, l=40, r=40, b=80),  # More top/bottom margin
        font=dict(size=16),
        bargap=0.1,  # Slightly smaller space between bars
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Rotate x-axis labels for all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig_grid.update_xaxes(tickangle=30, row=i, col=j)

    st.plotly_chart(fig_grid, use_container_width=True)
    st.markdown(" ")  # Add space below

    # Add metrics table below the plot
    st.subheader("📋 Performance Metrics")
    # Remove Combined Forecast from metrics table if present
    if not performance_df.empty and 'Model' in performance_df.columns:
        perf_df_no_combined = performance_df[performance_df['Model'] != 'Combined Forecast']
        st.dataframe(perf_df_no_combined, use_container_width=True)
    else:
        st.dataframe(performance_df, use_container_width=True)
    
    # Individual model plots
    st.subheader("🔍 Individual Model Forecasts")
    
    n_models = len([k for k in models_results if k != 'Combined Forecast'])
    cols = st.columns(min(3, n_models))
    
    for i, (model_name, results) in enumerate(models_results.items()):
        if model_name == 'Combined Forecast':
            continue
        col_idx = i % 3
        
        with cols[col_idx]:
            fig_ind = go.Figure()
            
            # Actual values
            fig_ind.add_trace(go.Scatter(
                x=test_series.index,
                y=test_series.values,
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            # Forecast
            fig_ind.add_trace(go.Scatter(
                x=test_series.index,
                y=results['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2)
            ))
            
            fig_ind.update_layout(
                title=f"{model_name}",
                xaxis_title="Date",
                yaxis_title=selected_metric_label,
                height=400
            )
            
            st.plotly_chart(fig_ind, use_container_width=True)
            
            # Model metrics
            st.markdown(f"""
            **Metrics:**
            - RMSE: {results['rmse']:.3f}
            - MAE: {results['mae']:.3f}
            - MAPE: {results['mape']:.2f}%
            """)

# === Tab 3: Detailed Analysis ===
with tab3:
    st.title("🔍 Detailed Analysis")
    
    # Residual analysis
    st.subheader("📊 Residual Analysis")
    
    # Create residual comparison
    fig_residuals = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'teal']  # Add 'teal' for RNN
    for i, (model_name, results) in enumerate(models_results.items()):
        residuals = test_series.values - results['forecast']
        # Assign a unique color for RNN
        if model_name == 'RNN':
            color = 'teal'
        elif model_name == 'ARIMA':
            color = 'red'
        else:
            color = colors[i % len(colors)]
        fig_residuals.add_trace(go.Scatter(
            x=test_series.index,
            y=residuals,
            mode='lines+markers',
            name=f"{model_name} (Std: {np.std(residuals):.3f})",
            line=dict(color=color, width=2)
        ))
    
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
    fig_residuals.update_layout(
        title="Residuals Comparison",
        xaxis_title="Date",
        yaxis_title="Residuals",
        height=500
    )
    
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Residual statistics
    st.subheader("📈 Residual Statistics")
    
    residual_stats = []
    for model_name, results in models_results.items():
        if model_name == 'Combined Forecast':
            continue
        residuals = test_series.values - results['forecast']
        residual_stats.append({
            'Model': model_name,
            'Mean Residual': np.mean(residuals),
            'Std Residual': np.std(residuals),
            'Min Residual': np.min(residuals),
            'Max Residual': np.max(residuals),
            'Skewness': pd.Series(residuals).skew(),
            'Kurtosis': pd.Series(residuals).kurtosis()
        })
    
    residual_df = pd.DataFrame(residual_stats)
    # Remove Combined Forecast from residual statistics table if present
    if not residual_df.empty and 'Model' in residual_df.columns:
        residual_df_no_combined = residual_df[residual_df['Model'] != 'Combined Forecast']
        st.dataframe(residual_df_no_combined, use_container_width=True)
    else:
        st.dataframe(residual_df, use_container_width=True)
    
    # Error distribution
    st.subheader("📊 Error Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RMSE vs MAE scatter
        fig_error_scatter = px.scatter(
            performance_df, 
            x='RMSE', 
            y='MAE',
            text='Model',
            title="RMSE vs MAE Comparison"
        )
        fig_error_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_error_scatter, use_container_width=True)
    
    with col2:
        # MAPE vs AIC scatter (handle mixed data types)
        aic_df_plot = performance_df[performance_df['AIC'] != 'N/A'].copy()
        if not aic_df_plot.empty:
            aic_df_plot['AIC'] = pd.to_numeric(aic_df_plot['AIC'], errors='coerce')
            fig_mape_aic = px.scatter(
                aic_df_plot, 
                x='MAPE (%)', 
                y='AIC',
                text='Model',
                title="MAPE vs AIC Comparison"
            )
            fig_mape_aic.update_traces(textposition="top center")
            st.plotly_chart(fig_mape_aic, use_container_width=True)
        else:
            st.markdown("**MAPE vs AIC:** No models with valid AIC values available for plotting.")

# === Tab 4: Model Rankings ===
with tab4:
    st.title("🏆 Model Rankings")
    
    # Create rankings
    st.subheader("🥇 Model Rankings by Performance Metric")
    
    # RMSE ranking
    rmse_ranking = performance_df.sort_values('RMSE')
    st.markdown("**📊 RMSE Ranking (Lower is Better):**")
    for i, (_, row) in enumerate(rmse_ranking.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        st.markdown(f"{medal} **{row['Model']}**: {row['RMSE']:.3f}")
    
    st.markdown("---")
    
    # MAPE ranking
    mape_ranking = performance_df.sort_values('MAPE (%)')
    st.markdown("**📊 MAPE Ranking (Lower is Better):**")
    for i, (_, row) in enumerate(mape_ranking.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        st.markdown(f"{medal} **{row['Model']}**: {row['MAPE (%)']:.2f}%")
    
    st.markdown("---")
    
    # AIC ranking (statistical models only)
    statistical_models = performance_df[performance_df['Model'].isin(['ARIMA', 'SARIMA', 'Exponential Smoothing'])].copy()
    valid_aic_models = statistical_models[statistical_models['AIC'] != 'N/A'].copy()
    
    if not valid_aic_models.empty:
        # Convert AIC to numeric for sorting
        valid_aic_models['AIC'] = pd.to_numeric(valid_aic_models['AIC'], errors='coerce')
        aic_ranking = valid_aic_models.sort_values('AIC')
        st.markdown("**📊 AIC Ranking for Statistical Models (Lower is Better):**")
        for i, (_, row) in enumerate(aic_ranking.iterrows()):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            st.markdown(f"{medal} **{row['Model']}**: {row['AIC']:.2f}")
    else:
        st.markdown("**📊 AIC Ranking:** No statistical models with valid AIC values available.")
    
    # Show deep learning models separately
    dl_models = performance_df[~performance_df['Model'].isin(['ARIMA', 'SARIMA', 'Exponential Smoothing'])]
    if not dl_models.empty:
        st.markdown("**📊 Deep Learning Models (AIC Not Applicable):**")
        for _, row in dl_models.iterrows():
            st.markdown(f"• **{row['Model']}**: Uses validation loss and other metrics for model selection")
    
    # Overall ranking
    st.subheader("🏆 Overall Performance Ranking")
    
    # Calculate overall score (normalized and combined)
    performance_df_normalized = performance_df.copy()
    
    # Normalize metrics (0-1 scale, lower is better for error metrics)
    performance_df_normalized['RMSE_norm'] = (performance_df['RMSE'] - performance_df['RMSE'].min()) / (performance_df['RMSE'].max() - performance_df['RMSE'].min())
    performance_df_normalized['MAE_norm'] = (performance_df['MAE'] - performance_df['MAE'].min()) / (performance_df['MAE'].max() - performance_df['MAE'].min())
    performance_df_normalized['MAPE_norm'] = (performance_df['MAPE (%)'] - performance_df['MAPE (%)'].min()) / (performance_df['MAPE (%)'].max() - performance_df['MAPE (%)'].min())
    
    # Calculate overall score (lower is better)
    performance_df_normalized['Overall_Score'] = (
        performance_df_normalized['RMSE_norm'] + 
        performance_df_normalized['MAE_norm'] + 
        performance_df_normalized['MAPE_norm']
    ) / 3
    
    overall_ranking = performance_df_normalized.sort_values('Overall_Score')
    
    st.markdown("**🏆 Overall Ranking (Combined Score):**")
    for i, (_, row) in enumerate(overall_ranking.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        st.markdown(f"{medal} **{row['Model']}**: Score = {row['Overall_Score']:.3f}")
    
    # Best model recommendation
    best_model = overall_ranking.iloc[0]['Model']
    st.success(f"""
    **🎯 Recommendation:**
    
    Based on the comprehensive analysis, **{best_model}** appears to be the best performing model for forecasting {selected_metric_label}.
    
    **Why {best_model} is recommended:**
    - Lowest overall error score
    - Best balance of accuracy and reliability
    - Consistent performance across multiple metrics
    """)
    
    # Download results
    st.subheader("📥 Download Results")
    
    # Create comprehensive results dataframe
    download_data = []
    for model_name, results in models_results.items():
        if model_name == 'Combined Forecast':
            continue
        download_data.append({
            'Model': model_name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'MAPE (%)': results['mape'],
            'AIC': results['aic'] if results['aic'] is not None else 'N/A'
        })
    
    download_df = pd.DataFrame(download_data)
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Performance Comparison CSV",
        csv,
        "model_comparison_results.csv",
        "text/csv"
    ) 

# === Future Forecasts for Next 4 Quarters (Starting from Next Year Q1) ===
# (Remove the entire block for future forecast table and related logic)
# (No code for next 4 quarters' forecast table remains)

# === Tab 5: Methodology & Code ===
with tab5:
    st.title("📝 Model Comparison Methodology & Code")
    st.markdown("""
    ## Overview
    This tab explains the methodology used for comparing all forecasting models in this app. It covers the data split, fairness, evaluation metrics, and the actual code used for splitting and evaluation.
    
    ---
    
    ### 1. **Chronological Train/Test Split**
    - The time series is split into a training set and a test set based on time order (not randomly).
    - This simulates real-world forecasting, where you predict future values based on past data.
    - **Example:**
    ```python
    test_pct = 20
    test_size = int(len(series) * test_pct / 100)
    train_size = len(series) - test_size
    train_series = series.iloc[:train_size]
    test_series = series.iloc[train_size:]
    ```
    - The same split is used for all models to ensure a fair comparison.
    
    ---
    
    ### 2. **Fairness and Consistency**
    - All models are trained on the same training data and evaluated on the same test data.
    - No model is allowed to see the test set during training or hyperparameter tuning.
    - Deep learning models may use a validation split from the training set for early stopping, but the test set remains untouched until final evaluation.
    
    ---
    
    ### 3. **Evaluation Metrics**
    - **RMSE (Root Mean Squared Error):** Measures average prediction error in original units.
    - **MAE (Mean Absolute Error):** Average absolute error.
    - **MAPE (Mean Absolute Percentage Error):** Average percentage error.
    - **R² (Coefficient of Determination):** Proportion of variance explained.
    - **AIC/BIC:** Used for statistical models to compare model fit and complexity.
    
    - **Example code for metrics:**
    ```python
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    mae = mean_absolute_error(test_series, forecast)
    mape = np.mean(np.abs((test_series - forecast) / test_series)) * 100
    r2 = r2_score(test_series, forecast)
    ```
    
    ---
    
    ### 4. **Model Training and Forecasting**
    - Each model is trained only on the training set.
    - Forecasts are generated for the test set period and compared to actual values.
    - **Example (ARIMA):**
    ```python
    import pmdarima as pm
    model = pm.auto_arima(train_series, seasonal=True, m=4, stepwise=True)
    forecast = model.predict(n_periods=test_size)
    ```
    - **Example (LSTM):**
    ```python
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    # ... create lagged data ...
    model = Sequential([
        LSTM(32, input_shape=(n_lags, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    forecast = model.predict(X_test).flatten()
    forecast_inv = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    ```
    
    ---
    
    ### 5. **Summary Table**
    | Step                | Description                                                                 |
    |---------------------|-----------------------------------------------------------------------------|
    | Data Split          | Chronological (time-based), same for all models                             |
    | Training            | Only on training set, no test set leakage                                   |
    | Forecasting         | Predict test set period, compare to actuals                                 |
    | Metrics             | RMSE, MAE, MAPE, R², AIC/BIC (where applicable)                             |
    | Fairness            | Identical data and metrics for all models                                   |
    | Visualization       | Tables, plots, and summary recommendations                                  |
    
    ---
    
    **This methodology ensures that your model comparison is fair, robust, and reflects real-world forecasting performance.**
    """) 