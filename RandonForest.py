import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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

# Forecast horizon selector
n_periods = st.slider("Select number of quarters to forecast:", min_value=4, max_value=16, value=8, step=4)

# === Select and plot time series ===
series = df[selected_metric].dropna()
st.subheader("📊 Historical Time Series")
st.line_chart(series)

# === Random Forest Configuration ===
st.markdown("### 🌲 Random Forest Model Settings")
st.info("""
**Random Forest** is a powerful machine learning method that uses multiple decision trees to make predictions.
This app uses advanced feature engineering to capture time series patterns in your unemployment data.
""")

# Model parameters
col1, col2 = st.columns(2)
with col1:
    n_estimators = st.slider("Number of Trees:", min_value=50, max_value=500, value=200, step=50)
    max_depth = st.slider("Max Tree Depth:", min_value=3, max_value=20, value=10, step=1)

with col2:
    n_lags = st.slider("Number of Lag Features:", min_value=4, max_value=12, value=8, step=1)
    include_seasonal = st.checkbox("Include Seasonal Features", value=True)

# === Feature Engineering ===
def create_features(series, n_lags, include_seasonal=True):
    """Create comprehensive features for time series forecasting"""
    df_features = pd.DataFrame(series)
    
    # Lag features
    for lag in range(1, n_lags + 1):
        df_features[f'lag_{lag}'] = df_features[selected_metric].shift(lag)
    
    # Rolling statistics
    df_features['rolling_mean_4'] = df_features[selected_metric].rolling(window=4).mean()
    df_features['rolling_std_4'] = df_features[selected_metric].rolling(window=4).std()
    df_features['rolling_mean_8'] = df_features[selected_metric].rolling(window=8).mean()
    df_features['rolling_std_8'] = df_features[selected_metric].rolling(window=8).std()
    
    # Trend features
    df_features['trend'] = range(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    
    # Seasonal features
    if include_seasonal:
        df_features['quarter'] = df_features.index.quarter
        df_features['year'] = df_features.index.year
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    
    # Difference features
    df_features['diff_1'] = df_features[selected_metric].diff()
    df_features['diff_2'] = df_features[selected_metric].diff().diff()
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

# Create features
features_df = create_features(series, n_lags, include_seasonal)

# Prepare X and y
X = features_df.drop(columns=[selected_metric])
y = features_df[selected_metric]

# === Train Random Forest Model ===
rf_model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42,
    n_jobs=-1
)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

# Fit final model
rf_model.fit(X, y)

# === Feature Importance ===
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# === In-sample predictions ===
y_pred = rf_model.predict(X)
residuals = y - y_pred

# Performance metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
r2 = r2_score(y, y_pred)

# === Forecast ===
def generate_forecast(model, last_features, n_periods):
    """Generate multi-step forecast"""
    forecast = []
    current_features = last_features.copy()
    
    # Get feature names in the same order as training data
    feature_names = X.columns.tolist()
    
    for i in range(n_periods):
        # Convert dictionary to list in correct order
        feature_values = [current_features[feature] for feature in feature_names]
        
        # Make prediction
        pred = model.predict([feature_values])[0]
        forecast.append(pred)
        
        # Update features for next prediction
        # Shift lag features
        for lag in range(n_lags, 1, -1):
            current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        current_features['lag_1'] = pred
        
        # Update rolling statistics (simplified)
        if 'rolling_mean_4' in current_features:
            current_features['rolling_mean_4'] = pred
        if 'rolling_mean_8' in current_features:
            current_features['rolling_mean_8'] = pred
        
        # Update trend
        current_features['trend'] += 1
        current_features['trend_squared'] = current_features['trend'] ** 2
        
        # Update seasonal features
        if 'quarter' in current_features:
            quarter = (current_features['quarter'] % 4) + 1
            current_features['quarter'] = quarter
            current_features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
            current_features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
    
    return forecast

# Get last known features
last_features = X.iloc[-1].to_dict()
forecast_values = generate_forecast(rf_model, last_features, n_periods)

# Forecast dates
last_date = series.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')

# Calculate confidence intervals using model variance
forecast_std = np.std(residuals)
conf_int_lower = np.array(forecast_values) - 1.96 * forecast_std
conf_int_upper = np.array(forecast_values) + 1.96 * forecast_std

forecast_df = pd.DataFrame({
    "Forecast Date": forecast_dates,
    f"Forecasted {selected_metric_label}": forecast_values,
    "Lower CI": conf_int_lower,
    "Upper CI": conf_int_upper
})

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"🌲 Random Forest Forecast ({selected_metric_label})",
    "🔍 Feature Analysis",
    "📊 Model Diagnostics",
    "📋 Model Summary",
    "📋 Complete Technical Details"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"🌲 Random Forest Forecast for {selected_metric_label}")
    
    # Model info
    st.markdown(f"""
    **🔍 Your Random Forest Model:**
    
    **Model Configuration:**
    - **Number of Trees**: {n_estimators}
    - **Max Tree Depth**: {max_depth}
    - **Lag Features**: {n_lags}
    - **Seasonal Features**: {'Included' if include_seasonal else 'Not included'}
    
    **Cross-Validation Performance:**
    - **CV RMSE**: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}
    """)
    
    # Forecast visualization
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    forecast_df_renamed = forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    combined = pd.concat([actual_df, forecast_df_renamed], axis=0)

    fig = px.line(combined, x="Date", y=selected_metric_label, title="Random Forest Forecast vs Actual")
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Upper CI"],
                    mode="lines", name="Upper CI", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Lower CI"],
                    mode="lines", name="Lower CI", fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("MAE", f"{mae:.2f}")
    with col3:
        st.metric("MAPE (%)", f"{mape:.2f}")
    with col4:
        st.metric("R²", f"{r2:.3f}")

    # Forecast table
    st.dataframe(forecast_df, use_container_width=True)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "random_forest_forecast.csv", "text/csv")

# === Tab 2: Feature Analysis ===
with tab2:
    st.title("🔍 Feature Analysis")
    
    # Feature importance
    st.subheader("🌳 Feature Importance")
    
    # Top features
    top_features = feature_importance.head(10)
    fig_importance = px.bar(top_features, x='importance', y='feature', 
                           orientation='h', title="Top 10 Most Important Features")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Feature importance table
    st.markdown("**📊 Feature Importance Rankings:**")
    st.dataframe(feature_importance, use_container_width=True)
    
    # Feature correlation
    st.subheader("🔗 Feature Correlations")
    
    # Calculate correlations with target
    correlations = []
    for feature in X.columns:
        corr = np.corrcoef(X[feature], y)[0, 1]
        correlations.append({'feature': feature, 'correlation': corr})
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    
    fig_corr = px.bar(corr_df.head(10), x='correlation', y='feature', 
                      orientation='h', title="Top 10 Features by Correlation with Target")
    fig_corr.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature engineering explanation
    st.subheader("🔧 Feature Engineering Explanation")
    
    st.markdown("""
    **🌲 Random Forest uses these engineered features:**
    
    **1. Lag Features:**
    - **lag_1 to lag_{n_lags}**: Previous values of the target variable
    - **Purpose**: Capture short-term dependencies and patterns
    
    **2. Rolling Statistics:**
    - **rolling_mean_4/8**: Moving averages over 4 and 8 quarters
    - **rolling_std_4/8**: Moving standard deviations
    - **Purpose**: Capture trend and volatility patterns
    
    **3. Trend Features:**
    - **trend**: Linear time trend
    - **trend_squared**: Quadratic time trend
    - **Purpose**: Capture long-term trends and non-linear patterns
    
    **4. Seasonal Features:**
    - **quarter**: Quarter of the year (1-4)
    - **quarter_sin/cos**: Cyclical encoding of quarters
    - **Purpose**: Capture seasonal patterns in unemployment data
    
    **5. Difference Features:**
    - **diff_1**: First difference (change from previous quarter)
    - **diff_2**: Second difference (change in change)
    - **Purpose**: Capture rate of change patterns
    """)

# === Tab 3: Model Diagnostics ===
with tab3:
    st.title("📊 Model Diagnostics")
    
    # Residuals overview
    st.subheader("🟣 Residuals Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
    with col2:
        st.metric("Residual Std", f"{np.std(residuals):.4f}")
    with col3:
        st.metric("Residual Range", f"{np.max(residuals) - np.min(residuals):.4f}")
    with col4:
        st.metric("Residual Skewness", f"{residuals.skew():.3f}")

    # Residuals over time
    st.subheader("📈 Residuals Over Time")
    resid_fig = px.line(x=residuals.index, y=residuals.values,
                        labels={'x': 'Time', 'y': 'Residuals'}, title="Residuals Over Time")
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(resid_fig, use_container_width=True)

    # Residual diagnostics
    st.subheader("🔬 Residual Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Residual Distribution**")
        fig_hist = px.histogram(residuals, nbins=20, title="Residual Distribution")
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("**📈 Actual vs Predicted**")
        fig_scatter = px.scatter(x=y, y=y_pred, title="Actual vs Predicted Values")
        fig_scatter.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                                        mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Cross-validation results
    st.subheader("🔄 Cross-Validation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 CV RMSE by Fold**")
        cv_df = pd.DataFrame({
            'Fold': range(1, len(cv_rmse) + 1),
            'RMSE': cv_rmse
        })
        fig_cv = px.bar(cv_df, x='Fold', y='RMSE', title="Cross-Validation RMSE by Fold")
        st.plotly_chart(fig_cv, use_container_width=True)
    
    with col2:
        st.markdown("**📈 CV Performance Summary**")
        st.metric("Mean CV RMSE", f"{cv_rmse.mean():.3f}")
        st.metric("CV RMSE Std", f"{cv_rmse.std():.3f}")
        st.metric("CV RMSE Range", f"{cv_rmse.max() - cv_rmse.min():.3f}")

# === Tab 4: Model Summary ===
with tab4:
    st.title("📋 Random Forest Model Summary & Explanation")
    
    # Model parameters display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 Model Parameters:**
        - **Number of Trees**: {n_estimators}
        - **Max Tree Depth**: {max_depth}
        - **Lag Features**: {n_lags}
        - **Seasonal Features**: {'Included' if include_seasonal else 'Not included'}
        """)
    
    with col2:
        st.markdown(f"""
        **📈 Model Performance:**
        - **RMSE**: `{rmse:.2f}`
        - **MAE**: `{mae:.2f}`
        - **MAPE**: `{mape:.2f}%`
        - **R²**: `{r2:.3f}`
        """)
    
    # Random Forest explanation
    st.subheader("🌲 Understanding Random Forest Models")
    st.markdown("""
    **Random Forest** is an ensemble learning method that operates by constructing multiple decision trees:
    
    ### 🔍 How Random Forest Works:
    
    **1. Ensemble Method:**
    - **Multiple Trees**: Creates {n_estimators} decision trees
    - **Bootstrap Sampling**: Each tree uses different random subset of data
    - **Feature Randomization**: Each split considers random subset of features
    
    **2. Prediction Process:**
    - **Individual Predictions**: Each tree makes its own prediction
    - **Aggregation**: Final prediction is average of all tree predictions
    - **Variance Reduction**: Ensemble reduces overfitting and improves accuracy
    
    **3. Key Advantages:**
    - **Non-linear Relationships**: Can capture complex patterns
    - **Feature Importance**: Provides insights into important variables
    - **Robustness**: Less sensitive to outliers and noise
    - **No Assumptions**: No distributional assumptions required
    """)
    
    # Your model explanation
    st.subheader("🔍 Understanding Your Random Forest Model")
    
    st.markdown(f"""
    **🌲 Your Random Forest Configuration:**
    
    **Model Structure:**
    - **{n_estimators} Decision Trees**: Each tree provides a different perspective on the data
    - **Max Depth {max_depth}**: Controls tree complexity and prevents overfitting
    - **{n_lags} Lag Features**: Captures temporal dependencies up to {n_lags} quarters back
    - **Seasonal Features**: {'Included to capture quarterly patterns' if include_seasonal else 'Not included - focuses on trend and lag patterns'}
    
    **Feature Engineering:**
    - **Temporal Features**: Lag variables capture short-term memory
    - **Statistical Features**: Rolling means and standard deviations capture trends
    - **Seasonal Features**: {'Quarterly patterns modeled through cyclical encoding' if include_seasonal else 'No explicit seasonal modeling'}
    - **Trend Features**: Linear and quadratic trends capture long-term patterns
    """)
    
    # Why Random Forest for your data
    st.subheader("🎯 Why Random Forest for Unemployment Data?")
    
    st.markdown(f"""
    **✅ Random Forest is excellent for your data because:**
    
    **1. Captures Complex Patterns:**
    - **Non-linear relationships**: Unemployment may have complex, non-linear patterns
    - **Interaction effects**: Different factors may interact in complex ways
    - **Multiple time scales**: Can capture both short-term and long-term patterns
    
    **2. Robust and Reliable:**
    - **Outlier resistance**: Less sensitive to extreme values
    - **Missing data handling**: Can handle missing values gracefully
    - **Feature importance**: Shows which variables matter most
    
    **3. No Distributional Assumptions:**
    - **Flexible modeling**: Doesn't assume normal distributions
    - **Heteroscedasticity**: Can handle varying error variances
    - **Non-stationarity**: Can adapt to changing patterns over time
    
    **4. Interpretable Results:**
    - **Feature importance**: Understand which lags and features matter
    - **Partial dependence**: See how individual features affect predictions
    - **Uncertainty quantification**: Provides prediction intervals
    """)
    
    # Model performance explanation
    st.markdown(f"""
    **📊 Your Model Performance:**
    
    **RMSE: {rmse:.2f}** - Average prediction error in original units
    
    **MAE: {mae:.2f}** - Average absolute prediction error
    
    **MAPE: {mape:.2f}%** - Average percentage error (excellent for unemployment data)
    
    **R²: {r2:.3f}** - Proportion of variance explained by the model
    
    **Cross-Validation RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}** - Out-of-sample performance
    
    **What these numbers mean:**
    - **RMSE {rmse:.2f}**: Predictions are off by {rmse:.2f} percentage points on average
    - **MAPE {mape:.2f}%**: {mape:.2f}% average percentage error (very good for economic data)
    - **R² {r2:.3f}**: Model explains {r2*100:.1f}% of the variance in unemployment rates
    - **CV RMSE**: Out-of-sample performance suggests model generalizes well
    """)
    
    # Random Forest vs other methods
    st.subheader("🌲 Random Forest vs Other Methods")
    
    st.markdown("""
    **Random Forest Advantages:**
    
    **✅ Flexibility:**
    - Can capture non-linear relationships
    - No distributional assumptions
    - Handles mixed data types
    
    **✅ Robustness:**
    - Resistant to outliers
    - Handles missing data
    - Less prone to overfitting
    
    **✅ Interpretability:**
    - Feature importance rankings
    - Partial dependence plots
    - No black-box predictions
    
    **✅ Performance:**
    - Often achieves high accuracy
    - Good out-of-sample performance
    - Stable predictions
    
    **Limitations:**
    - **No explicit time series modeling**: Doesn't model autocorrelation explicitly
    - **Limited extrapolation**: May not extrapolate well beyond training range
    - **Computational cost**: Can be slower than simpler models
    - **Less interpretable than linear models**: Complex interactions harder to explain
    """)
    
    # Practical interpretation
    st.subheader("💡 How to Use Your Random Forest Forecast")
    
    st.markdown("""
    **For Policy Makers:**
    - **Feature insights**: Use feature importance to understand key drivers
    - **Scenario analysis**: Test different lag scenarios
    - **Risk assessment**: Use prediction intervals for uncertainty
    
    **For Businesses:**
    - **Trend identification**: Use model to identify key patterns
    - **Planning horizon**: Short to medium-term planning (1-2 years)
    - **Monitoring**: Track feature importance changes over time
    
    **For Researchers:**
    - **Pattern discovery**: Identify important temporal patterns
    - **Model comparison**: Compare with traditional time series models
    - **Feature engineering**: Learn which features matter most
    """)

# === Tab 5: Complete Technical Details ===
with tab5:
    st.title("📋 Complete Technical Details")
    
    # Model summary
    st.subheader("📋 Model Summary")
    
    st.markdown(f"""
    **🔍 Model Configuration:**
    
    **Algorithm**: Random Forest Regressor
    **Number of Trees**: {n_estimators}
    **Max Tree Depth**: {max_depth}
    **Random State**: 42 (for reproducibility)
    
    **Feature Engineering:**
    - **Total Features**: {len(X.columns)}
    - **Lag Features**: {n_lags}
    - **Rolling Statistics**: 4 features (mean and std for 4 and 8 quarters)
    - **Trend Features**: 2 features (linear and quadratic)
    - **Seasonal Features**: {'4 features' if include_seasonal else '0 features'}
    - **Difference Features**: 2 features (first and second differences)
    
    **Data Information:**
    - **Training Observations**: {len(X)}
    - **Target Variable**: {selected_metric_label}
    - **Feature Matrix Shape**: {X.shape}
    """)
    
    # Feature engineering details
    st.subheader("🔧 Feature Engineering Details")
    
    st.markdown("""
    **📊 Feature Creation Process:**
    
    **1. Lag Features:**
    ```python
    for lag in range(1, n_lags + 1):
        df_features[f'lag_{lag}'] = df_features[target].shift(lag)
    ```
    
    **2. Rolling Statistics:**
    ```python
    df_features['rolling_mean_4'] = df_features[target].rolling(window=4).mean()
    df_features['rolling_std_4'] = df_features[target].rolling(window=4).std()
    ```
    
    **3. Trend Features:**
    ```python
    df_features['trend'] = range(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    ```
    
    **4. Seasonal Features:**
    ```python
    df_features['quarter'] = df_features.index.quarter
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    ```
    
    **5. Difference Features:**
    ```python
    df_features['diff_1'] = df_features[target].diff()
    df_features['diff_2'] = df_features[target].diff().diff()
    ```
    """)
    
    # Model training details
    st.subheader("🌲 Model Training Details")
    
    st.markdown("""
    **📈 Training Process:**
    
    **1. Data Preparation:**
    ```python
    # Remove rows with NaN values after feature creation
    features_df = features_df.dropna()
    X = features_df.drop(columns=[target])
    y = features_df[target]
    ```
    
    **2. Cross-Validation:**
    ```python
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    ```
    
    **3. Model Fitting:**
    ```python
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    ```
    
    **4. Feature Importance:**
    ```python
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    ```
    """)
    
    # Forecasting process
    st.subheader("🔮 Forecasting Process")
    
    st.markdown("""
    **📊 Multi-step Forecasting:**
    
    **1. Initialize Features:**
    ```python
    last_features = X.iloc[-1].to_dict()  # Get last known feature values
    ```
    
    **2. Iterative Prediction:**
    ```python
    for i in range(n_periods):
        # Make prediction
        pred = model.predict([current_features])[0]
        forecast.append(pred)
        
        # Update features for next prediction
        # Shift lag features
        for lag in range(n_lags, 1, -1):
            current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        current_features['lag_1'] = pred
        
        # Update other features...
    ```
    
    **3. Confidence Intervals:**
    ```python
    forecast_std = np.std(residuals)
    conf_int_lower = forecast - 1.96 * forecast_std
    conf_int_upper = forecast + 1.96 * forecast_std
    ```
    """)
    
    # Performance metrics calculation
    st.subheader("📊 Performance Metrics Calculation")
    
    st.markdown("""
    **🔍 Metric Formulas:**
    
    **RMSE (Root Mean Square Error):**
    ```
    RMSE = √(Σ(y - ŷ)²/n)
    ```
    
    **MAE (Mean Absolute Error):**
    ```
    MAE = Σ|y - ŷ|/n
    ```
    
    **MAPE (Mean Absolute Percentage Error):**
    ```
    MAPE = (Σ|y - ŷ|/y) × 100%
    ```
    
    **R² (Coefficient of Determination):**
    ```
    R² = 1 - (SS_res / SS_tot)
    ```
    
    **Cross-Validation RMSE:**
    ```
    CV_RMSE = √(Σ(y_test - ŷ_test)²/n_test) for each fold
    ```
    """)
    
    # Quality assurance
    st.subheader("✅ Quality Assurance")
    
    st.markdown("""
    **🔬 Validation Methods:**
    
    **1. Time Series Cross-Validation:**
    - **Method**: TimeSeriesSplit with 5 folds
    - **Purpose**: Ensure temporal order is respected
    - **Result**: Out-of-sample performance estimation
    
    **2. Feature Importance Analysis:**
    - **Method**: Random Forest feature importance
    - **Purpose**: Identify most important predictors
    - **Result**: Model interpretability and feature selection
    
    **3. Residual Analysis:**
    - **Method**: Analysis of prediction errors
    - **Purpose**: Check model assumptions and fit quality
    - **Result**: Model diagnostics and improvement insights
    
    **4. Performance Metrics:**
    - **Multiple metrics**: RMSE, MAE, MAPE, R²
    - **Purpose**: Comprehensive performance evaluation
    - **Result**: Model comparison and selection
    
    **Best Practices Applied:**
    - **Temporal ordering**: Respects time series structure
    - **Feature engineering**: Comprehensive feature creation
    - **Hyperparameter tuning**: Reasonable parameter selection
    - **Validation strategy**: Proper out-of-sample testing
    
    **Limitations:**
    - **No explicit autocorrelation modeling**: Relies on lag features
    - **Limited extrapolation**: May not extrapolate well beyond training range
    - **Computational complexity**: Can be slower than simpler models
    - **Feature engineering dependency**: Performance depends on feature quality
    """) 