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

warnings.filterwarnings("ignore")

def create_lagged_data(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_lstm(series, test_size, n_lags=8, n_units=32, epochs=50, batch_size=8, learning_rate=0.001):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = create_lagged_data(scaled, n_lags)
    X = X[..., np.newaxis]
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    model = Sequential([
        LSTM(n_units, input_shape=(n_lags, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    pred = model.predict(X_test).flatten()
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred_inv, None

def train_gru(series, test_size, n_lags=8, n_units=32, epochs=50, batch_size=8, learning_rate=0.001):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = create_lagged_data(scaled, n_lags)
    X = X[..., np.newaxis]
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    model = Sequential([
        GRU(n_units, input_shape=(n_lags, 1)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    pred = model.predict(X_test).flatten()
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred_inv, None

def train_cnn(series, test_size, n_lags=8, n_filters=32, kernel_size=4, epochs=50, batch_size=8, learning_rate=0.001):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = create_lagged_data(scaled, n_lags)
    X = X[..., np.newaxis]
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    model = Sequential([
        Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu', input_shape=(n_lags, 1)),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    pred = model.predict(X_test).flatten()
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred_inv, None

def train_rnn(series, test_size, n_lags=8, n_units=16, epochs=50, batch_size=8, learning_rate=0.001):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = create_lagged_data(scaled, n_lags)
    X = X[..., np.newaxis]
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    model = Sequential([
        SimpleRNN(n_units, input_shape=(n_lags, 1), activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    pred = model.predict(X_test).flatten()
    pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred_inv, None

def train_arima(series, test_size, force_seasonal=None):
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
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
    return forecast, aic

def train_sarima(series, test_size, seasonal_period=4):
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
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
    return forecast, aic

def train_exponential_smoothing(train_data, test_periods):
    """Train Exponential Smoothing model and make predictions"""
    try:
        # Try different models and select best one
        models = {}
        
        # Simple Exponential Smoothing
        try:
            model_simple = ExponentialSmoothing(train_data, trend=None, seasonal=None)
            fitted_simple = model_simple.fit()
            models['Simple'] = fitted_simple
        except:
            pass
        
        # Holt's method
        try:
            model_holt = ExponentialSmoothing(train_data, trend='add', seasonal=None)
            fitted_holt = model_holt.fit()
            models['Holt'] = fitted_holt
        except:
            pass
        
        # Holt-Winters
        try:
            model_hw = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=4)
            fitted_hw = model_hw.fit()
            models['Holt-Winters'] = fitted_hw
        except:
            pass
        
        # Select best model
        if models:
            best_model_name = min(models.keys(), key=lambda x: models[x].aic)
            best_model = models[best_model_name]
            forecast = best_model.forecast(test_periods)
            return forecast, best_model.aic
        else:
            return None, None
    except:
        return None, None

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

# Split configuration
total_points = len(series)
test_size = st.slider("Test Set Size (quarters):", min_value=4, max_value=min(20, total_points//4), value=8, step=1)
train_size = total_points - test_size

st.markdown(f"""
**📊 Data Split:**
- **Total Observations**: {total_points}
- **Training Set**: {train_size} quarters ({train_size/total_points*100:.1f}%)
- **Test Set**: {test_size} quarters ({test_size/total_points*100:.1f}%)
- **Training Period**: {series.index[0].strftime('%Y-%m')} to {series.index[train_size-1].strftime('%Y-%m')}
- **Test Period**: {series.index[train_size].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')}
""")

# Split the data
train_series = series[:train_size]
test_series = series[train_size:]
test_series.index = pd.to_datetime(test_series.index).strftime('%Y-%m-%d')

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
def train_arima(series, test_size, force_seasonal=None):
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
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
    return forecast, aic

def train_sarima(series, test_size, seasonal_period=4):
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
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
    return forecast, aic

def train_exponential_smoothing(train_data, test_periods):
    """Train Exponential Smoothing model and make predictions"""
    try:
        # Try different models and select best one
        models = {}
        
        # Simple Exponential Smoothing
        try:
            model_simple = ExponentialSmoothing(train_data, trend=None, seasonal=None)
            fitted_simple = model_simple.fit()
            models['Simple'] = fitted_simple
        except:
            pass
        
        # Holt's method
        try:
            model_holt = ExponentialSmoothing(train_data, trend='add', seasonal=None)
            fitted_holt = model_holt.fit()
            models['Holt'] = fitted_holt
        except:
            pass
        
        # Holt-Winters
        try:
            model_hw = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=4)
            fitted_hw = model_hw.fit()
            models['Holt-Winters'] = fitted_hw
        except:
            pass
        
        # Select best model
        if models:
            best_model_name = min(models.keys(), key=lambda x: models[x].aic)
            best_model = models[best_model_name]
            forecast = best_model.forecast(test_periods)
            return forecast, best_model.aic
        else:
            return None, None
    except:
        return None, None

# === Train All Models ===
st.markdown("### 🚀 Training Models...")

models_results = {}
progress_bar = st.progress(0)
status_text = st.empty()

# Train ARIMA
if use_arima:
    status_text.text("Training ARIMA...")
    forecast, aic = train_arima(train_series, test_size)
    if forecast is not None:
        models_results['ARIMA'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(14)

# Train SARIMA
if use_sarima:
    status_text.text("Training SARIMA...")
    forecast, aic = train_sarima(train_series, test_size)
    if forecast is not None:
        models_results['SARIMA'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(28)

# Train Exponential Smoothing
if use_exp_smooth:
    status_text.text("Training Exponential Smoothing...")
    forecast, aic = train_exponential_smoothing(train_series, test_size)
    if forecast is not None:
        models_results['Exponential Smoothing'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(42)

# Train LSTM
if use_lstm:
    status_text.text("Training LSTM...")
    forecast, aic = train_lstm(train_series, test_size)
    if forecast is not None:
        models_results['LSTM'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(57)

# Train GRU
if use_gru:
    status_text.text("Training GRU...")
    forecast, aic = train_gru(train_series, test_size)
    if forecast is not None:
        models_results['GRU'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(71)

# Train CNN
if use_cnn:
    status_text.text("Training CNN...")
    forecast, aic = train_cnn(train_series, test_size)
    if forecast is not None:
        models_results['CNN'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(85)

# Train RNN
if use_rnn:
    status_text.text("Training RNN...")
    forecast, aic = train_rnn(train_series, test_size)
    if forecast is not None:
        models_results['RNN'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(100)

status_text.text("✅ All models trained successfully!")
progress_bar.empty()

# === Results Display ===
if not models_results:
    st.error("❌ No models were successfully trained. Please check your configuration.")
    st.stop()

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Comparison",
    "📈 Forecast Visualization",
    "🔍 Detailed Analysis",
    "🏆 Model Rankings",
    "📝 Methodology & Code"
])

# === Tab 1: Performance Comparison ===
with tab1:
    st.title("📊 Model Performance Comparison")
    
    # Create performance comparison table
    performance_data = []
    for model_name, results in models_results.items():
        performance_data.append({
            'Model': model_name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'MAPE (%)': results['mape'],
            'R²': results['r2'],
            'AIC': results['aic'] if results['aic'] is not None else 'N/A'
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Display performance table
    st.subheader("📋 Performance Metrics")
    st.dataframe(performance_df, use_container_width=True)
    
    # Performance comparison charts
    st.subheader("📊 Performance Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RMSE comparison
        fig_rmse = px.bar(performance_df, x='Model', y='RMSE', 
                          title="RMSE Comparison (Lower is Better)",
                          color='RMSE', color_continuous_scale='Reds_r')
        st.plotly_chart(fig_rmse, use_container_width=True)
        
        # MAE comparison
        fig_mae = px.bar(performance_df, x='Model', y='MAE',
                         title="MAE Comparison (Lower is Better)",
                         color='MAE', color_continuous_scale='Reds_r')
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col2:
        # MAPE comparison
        fig_mape = px.bar(performance_df, x='Model', y='MAPE (%)',
                          title="MAPE Comparison (Lower is Better)",
                          color='MAPE (%)', color_continuous_scale='Reds_r')
        st.plotly_chart(fig_mape, use_container_width=True)
        
        # R² comparison
        fig_r2 = px.bar(performance_df, x='Model', y='R²',
                        title="R² Comparison (Higher is Better)",
                        color='R²', color_continuous_scale='Greens')
        st.plotly_chart(fig_r2, use_container_width=True)

# === Tab 2: Forecast Visualization ===
with tab2:
    st.title("📈 Forecast Visualization")
    
    # Create forecast comparison plot
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
    
    # Add forecasts
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, results) in enumerate(models_results.items()):
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
    
    # Individual model plots
    st.subheader("🔍 Individual Model Forecasts")
    
    n_models = len(models_results)
    cols = st.columns(min(3, n_models))
    
    for i, (model_name, results) in enumerate(models_results.items()):
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
            - R²: {results['r2']:.3f}
            """)

# === Tab 3: Detailed Analysis ===
with tab3:
    st.title("🔍 Detailed Analysis")
    
    # Residual analysis
    st.subheader("📊 Residual Analysis")
    
    # Create residual comparison
    fig_residuals = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, results) in enumerate(models_results.items()):
        residuals = test_series.values - results['forecast']
        fig_residuals.add_trace(go.Scatter(
            x=test_series.index,
            y=residuals,
            mode='lines+markers',
            name=f"{model_name} (Std: {np.std(residuals):.3f})",
            line=dict(color=colors[i % len(colors)], width=2)
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
        # MAPE vs R² scatter
        fig_mape_r2 = px.scatter(
            performance_df, 
            x='MAPE (%)', 
            y='R²',
            text='Model',
            title="MAPE vs R² Comparison"
        )
        fig_mape_r2.update_traces(textposition="top center")
        st.plotly_chart(fig_mape_r2, use_container_width=True)

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
    
    # R² ranking
    r2_ranking = performance_df.sort_values('R²', ascending=False)
    st.markdown("**📊 R² Ranking (Higher is Better):**")
    for i, (_, row) in enumerate(r2_ranking.iterrows()):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        st.markdown(f"{medal} **{row['Model']}**: {row['R²']:.3f}")
    
    # Overall ranking
    st.subheader("🏆 Overall Performance Ranking")
    
    # Calculate overall score (normalized and combined)
    performance_df_normalized = performance_df.copy()
    
    # Normalize metrics (0-1 scale, lower is better for error metrics)
    performance_df_normalized['RMSE_norm'] = (performance_df['RMSE'] - performance_df['RMSE'].min()) / (performance_df['RMSE'].max() - performance_df['RMSE'].min())
    performance_df_normalized['MAE_norm'] = (performance_df['MAE'] - performance_df['MAE'].min()) / (performance_df['MAE'].max() - performance_df['MAE'].min())
    performance_df_normalized['MAPE_norm'] = (performance_df['MAPE (%)'] - performance_df['MAPE (%)'].min()) / (performance_df['MAPE (%)'].max() - performance_df['MAPE (%)'].min())
    performance_df_normalized['R2_norm'] = (performance_df['R²'].max() - performance_df['R²']) / (performance_df['R²'].max() - performance_df['R²'].min())
    
    # Calculate overall score (lower is better)
    performance_df_normalized['Overall_Score'] = (
        performance_df_normalized['RMSE_norm'] + 
        performance_df_normalized['MAE_norm'] + 
        performance_df_normalized['MAPE_norm'] + 
        performance_df_normalized['R2_norm']
    ) / 4
    
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
        download_data.append({
            'Model': model_name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'MAPE (%)': results['mape'],
            'R²': results['r2'],
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

# === Which Model is Better? ===
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

# Assume you have a DataFrame 'results_df' with columns: 'Model', 'RMSE', 'MAPE', etc.
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