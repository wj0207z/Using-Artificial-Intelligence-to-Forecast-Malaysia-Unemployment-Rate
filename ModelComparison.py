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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

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

# === Model Configuration ===
st.markdown("### ⚙️ Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📈 Time Series Models:**")
    use_arima = st.checkbox("ARIMA", value=True)
    use_sarima = st.checkbox("SARIMA", value=True)
    use_exp_smooth = st.checkbox("Exponential Smoothing", value=True)

with col2:
    st.markdown("**🤖 Machine Learning Models:**")
    use_rf = st.checkbox("Random Forest", value=True)
    use_xgb = st.checkbox("XGBoost", value=True)
    use_svr = st.checkbox("Support Vector Regression", value=True)

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
def train_arima(train_data, test_periods):
    """Train ARIMA model and make predictions"""
    try:
        # Auto ARIMA
        model = pm.auto_arima(
            train_data,
            seasonal=False,
            max_d=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Forecast
        forecast = model.predict(n_periods=test_periods)
        return forecast, model.aic()
    except:
        return None, None

def train_sarima(train_data, test_periods):
    """Train SARIMA model and make predictions"""
    try:
        # Auto SARIMA
        model = pm.auto_arima(
            train_data,
            seasonal=True,
            m=4,
            max_d=2,
            max_D=2,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Forecast
        forecast = model.predict(n_periods=test_periods)
        return forecast, model.aic()
    except:
        return None, None

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

def train_random_forest(train_data, test_periods):
    """Train Random Forest model and make predictions"""
    try:
        # Create features
        features_df = create_features(train_data, n_lags=8, include_seasonal=True)
        X = features_df.drop(columns=[features_df.columns[0]])
        y = features_df.iloc[:, 0]
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X, y)
        
        # Generate forecast
        last_features = X.iloc[-1].to_dict()
        forecast = generate_ml_forecast(rf_model, last_features, test_periods, n_lags=8)
        
        return forecast, None
    except:
        return None, None

def train_xgboost(train_data, test_periods):
    """Train XGBoost model and make predictions"""
    try:
        # Create features
        features_df = create_features(train_data, n_lags=8, include_seasonal=True)
        X = features_df.drop(columns=[features_df.columns[0]])
        y = features_df.iloc[:, 0]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_scaled, y)
        
        # Generate forecast
        last_features = X.iloc[-1].to_dict()
        forecast = generate_ml_forecast_scaled(xgb_model, scaler, last_features, test_periods, n_lags=8)
        
        return forecast, None
    except:
        return None, None

def train_svr(train_data, test_periods):
    """Train SVR model and make predictions"""
    try:
        # Create features
        features_df = create_features(train_data, n_lags=8, include_seasonal=True)
        X = features_df.drop(columns=[features_df.columns[0]])
        y = features_df.iloc[:, 0]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr_model.fit(X_scaled, y)
        
        # Generate forecast
        last_features = X.iloc[-1].to_dict()
        forecast = generate_ml_forecast_scaled(svr_model, scaler, last_features, test_periods, n_lags=8)
        
        return forecast, None
    except:
        return None, None

def generate_ml_forecast(model, last_features, n_periods, n_lags=8):
    """Generate multi-step forecast for ML models"""
    forecast = []
    current_features = last_features.copy()
    
    for i in range(n_periods):
        # Convert to feature array
        feature_values = list(current_features.values())
        pred = model.predict([feature_values])[0]
        forecast.append(pred)
        
        # Update features for next prediction
        for lag in range(n_lags, 1, -1):
            current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        current_features['lag_1'] = pred
        
        # Update other features
        if 'rolling_mean_4' in current_features:
            current_features['rolling_mean_4'] = pred
        if 'rolling_mean_8' in current_features:
            current_features['rolling_mean_8'] = pred
        
        current_features['trend'] += 1
        current_features['trend_squared'] = current_features['trend'] ** 2
        
        if 'quarter' in current_features:
            quarter = (current_features['quarter'] % 4) + 1
            current_features['quarter'] = quarter
            current_features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
            current_features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    return forecast

def generate_ml_forecast_scaled(model, scaler, last_features, n_periods, n_lags=8):
    """Generate multi-step forecast for scaled ML models"""
    forecast = []
    current_features = last_features.copy()
    
    for i in range(n_periods):
        # Convert to feature array and scale
        feature_values = list(current_features.values())
        feature_values_scaled = scaler.transform([feature_values])
        pred = model.predict(feature_values_scaled)[0]
        forecast.append(pred)
        
        # Update features for next prediction
        for lag in range(n_lags, 1, -1):
            current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        current_features['lag_1'] = pred
        
        # Update other features
        if 'rolling_mean_4' in current_features:
            current_features['rolling_mean_4'] = pred
        if 'rolling_mean_8' in current_features:
            current_features['rolling_mean_8'] = pred
        
        current_features['trend'] += 1
        current_features['trend_squared'] = current_features['trend'] ** 2
        
        if 'quarter' in current_features:
            quarter = (current_features['quarter'] % 4) + 1
            current_features['quarter'] = quarter
            current_features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
            current_features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    return forecast

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
    progress_bar.progress(16)

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
    progress_bar.progress(33)

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
    progress_bar.progress(50)

# Train Random Forest
if use_rf:
    status_text.text("Training Random Forest...")
    forecast, aic = train_random_forest(train_series, test_size)
    if forecast is not None:
        models_results['Random Forest'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(67)

# Train XGBoost
if use_xgb:
    status_text.text("Training XGBoost...")
    forecast, aic = train_xgboost(train_series, test_size)
    if forecast is not None:
        models_results['XGBoost'] = {
            'forecast': forecast,
            'aic': aic,
            'rmse': np.sqrt(mean_squared_error(test_series, forecast)),
            'mae': mean_absolute_error(test_series, forecast),
            'mape': np.mean(np.abs((test_series - forecast) / test_series)) * 100,
            'r2': r2_score(test_series, forecast)
        }
    progress_bar.progress(83)

# Train SVR
if use_svr:
    status_text.text("Training SVR...")
    forecast, aic = train_svr(train_series, test_size)
    if forecast is not None:
        models_results['SVR'] = {
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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance Comparison",
    "📈 Forecast Visualization",
    "🔍 Detailed Analysis",
    "🏆 Model Rankings"
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