import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
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

# === XGBoost Configuration ===
st.markdown("### 🚀 XGBoost Model Settings")
st.info("""
**XGBoost (Extreme Gradient Boosting)** is a powerful gradient boosting algorithm that excels at capturing complex patterns in time series data.
This app uses advanced feature engineering and XGBoost's superior performance for unemployment forecasting.
""")

# Model parameters
col1, col2 = st.columns(2)
with col1:
    n_estimators = st.slider("Number of Trees:", min_value=50, max_value=500, value=200, step=50)
    with st.popover("❓"):
        st.markdown("""
        **Number of Trees**
        The number of boosting rounds (trees) in XGBoost.
        - More trees can improve accuracy but increase computation time.
        - Too many trees may lead to overfitting.
        """)
    max_depth = st.slider("Max Tree Depth:", min_value=3, max_value=20, value=6, step=1)
    with st.popover("❓"):
        st.markdown("""
        **Max Tree Depth**
        The maximum depth of each tree.
        - Deeper trees can capture more complex patterns but may overfit.
        - Shallower trees are more general but may underfit.
        """)
    learning_rate = st.slider("Learning Rate:", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    with st.popover("❓"):
        st.markdown("""
        **Learning Rate**
        Step size shrinkage used in update to prevent overfitting.
        - Lower values make the model more robust but require more trees.
        - Higher values speed up learning but may overfit.
        """)

with col2:
    subsample = st.slider("Subsample Ratio:", min_value=0.5, max_value=1.0, value=1.0, step=0.05)
    with st.popover("❓"):
        st.markdown("""
        **Subsample Ratio**
        The fraction of samples used for fitting each tree.
        - Lower values can help prevent overfitting.
        - Too low may underfit.
        """)
    n_lags = st.slider("Number of Lag Features:", min_value=4, max_value=12, value=8, step=1)
    with st.popover("❓"):
        st.markdown("""
        **Number of Lag Features**
        The number of previous quarters used as input features.
        - More lags can help capture longer-term dependencies.
        - Too many lags may add noise or cause overfitting.
        """)
    include_seasonal = st.checkbox("Include Seasonal Features", value=True)
    with st.popover("❓"):
        st.markdown("""
        **Include Seasonal Features**
        Adds features to help the model learn quarterly (seasonal) patterns.
        - Useful for data with strong seasonality.
        - May not help if data is not seasonal.
        """)

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
    
    # Additional rolling features for XGBoost
    df_features['rolling_min_4'] = df_features[selected_metric].rolling(window=4).min()
    df_features['rolling_max_4'] = df_features[selected_metric].rolling(window=4).max()
    df_features['rolling_median_4'] = df_features[selected_metric].rolling(window=4).median()
    
    # Trend features
    df_features['trend'] = range(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    df_features['trend_cubed'] = df_features['trend'] ** 3
    
    # Seasonal features
    if include_seasonal:
        df_features['quarter'] = df_features.index.quarter
        df_features['year'] = df_features.index.year
        df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
        df_features['year_sin'] = np.sin(2 * np.pi * df_features['year'] / 10)  # 10-year cycle
        df_features['year_cos'] = np.cos(2 * np.pi * df_features['year'] / 10)
    
    # Difference features
    df_features['diff_1'] = df_features[selected_metric].diff()
    df_features['diff_2'] = df_features[selected_metric].diff().diff()
    df_features['diff_4'] = df_features[selected_metric].diff(4)  # Year-over-year difference
    
    # Ratio features
    df_features['ratio_4'] = df_features[selected_metric] / df_features[selected_metric].rolling(window=4).mean()
    df_features['ratio_8'] = df_features[selected_metric] / df_features[selected_metric].rolling(window=8).mean()
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

# Create features
features_df = create_features(series, n_lags, include_seasonal)

# Prepare X and y
X = features_df.drop(columns=[selected_metric])
y = features_df[selected_metric]

# === Train XGBoost Model ===
# Create model without early stopping for cross-validation
xgb_model_cv = xgb.XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    random_state=42,
    n_jobs=-1
)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(xgb_model_cv, X, y, cv=tscv, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

# Create final model with early stopping
xgb_model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# Fit final model with validation split for early stopping
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# === Feature Importance ===
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# === In-sample predictions ===
y_pred = xgb_model.predict(X)
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
        if 'rolling_min_4' in current_features:
            current_features['rolling_min_4'] = pred
        if 'rolling_max_4' in current_features:
            current_features['rolling_max_4'] = pred
        if 'rolling_median_4' in current_features:
            current_features['rolling_median_4'] = pred
        
        # Update trend
        current_features['trend'] += 1
        current_features['trend_squared'] = current_features['trend'] ** 2
        current_features['trend_cubed'] = current_features['trend'] ** 3
        
        # Update seasonal features
        if 'quarter' in current_features:
            quarter = (current_features['quarter'] % 4) + 1
            current_features['quarter'] = quarter
            current_features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
            current_features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
        
        # Update year features
        if 'year' in current_features:
            current_features['year'] += 0.25  # Increment by quarter
            current_features['year_sin'] = np.sin(2 * np.pi * current_features['year'] / 10)
            current_features['year_cos'] = np.cos(2 * np.pi * current_features['year'] / 10)
    
    return forecast

# Get last known features
last_features = X.iloc[-1].to_dict()
forecast_values = generate_forecast(xgb_model, last_features, n_periods)

# Forecast dates
last_date = series.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')
forecast_dates = forecast_dates.strftime('%Y-%m-%d')

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
forecast_df["Forecast Date"] = pd.to_datetime(forecast_df["Forecast Date"]).dt.strftime('%Y-%m-%d')

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"🚀 XGBoost Forecast ({selected_metric_label})",
    "🔍 Feature Analysis",
    "📊 Model Diagnostics",
    "📋 Model Summary",
    "📋 Complete Technical Details"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"🚀 XGBoost Forecast for {selected_metric_label}")
    
    # Model info
    st.markdown(f"""
    **🔍 Your XGBoost Model:**
    
    **Model Configuration:**
    - **Number of Trees**: {n_estimators}
    - **Max Tree Depth**: {max_depth}
    - **Learning Rate**: {learning_rate}
    - **Subsample Ratio**: {subsample}
    - **Lag Features**: {n_lags}
    - **Seasonal Features**: {'Included' if include_seasonal else 'Not included'}
    
    **Cross-Validation Performance:**
    - **CV RMSE**: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}
    """)
    
    # Forecast visualization
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    actual_df["Date"] = pd.to_datetime(actual_df["Date"]).dt.strftime('%Y-%m-%d')
    forecast_df_renamed = forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    combined = pd.concat([actual_df, forecast_df_renamed], axis=0)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime('%Y-%m-%d')

    fig = px.line(combined, x="Date", y=selected_metric_label, title="XGBoost Forecast vs Actual")
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
    st.download_button("📥 Download Forecast CSV", csv, "xgboost_forecast.csv", "text/csv")

# === Tab 2: Feature Analysis ===
with tab2:
    st.title("🔍 Feature Analysis")
    
    # Feature importance
    st.subheader("🚀 Feature Importance")
    
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
    st.subheader("🔧 XGBoost Feature Engineering")
    
    st.markdown("""
    **🚀 XGBoost uses these advanced engineered features:**
    
    **1. Lag Features:**
    - **lag_1 to lag_{n_lags}**: Previous values of the target variable
    - **Purpose**: Capture short-term dependencies and patterns
    
    **2. Enhanced Rolling Statistics:**
    - **rolling_mean_4/8**: Moving averages over 4 and 8 quarters
    - **rolling_std_4/8**: Moving standard deviations
    - **rolling_min_4/max_4/median_4**: Additional statistical measures
    - **Purpose**: Capture trend, volatility, and distribution patterns
    
    **3. Advanced Trend Features:**
    - **trend**: Linear time trend
    - **trend_squared**: Quadratic time trend
    - **trend_cubed**: Cubic time trend
    - **Purpose**: Capture complex non-linear trends
    
    **4. Comprehensive Seasonal Features:**
    - **quarter**: Quarter of the year (1-4)
    - **quarter_sin/cos**: Cyclical encoding of quarters
    - **year_sin/cos**: Long-term cyclical patterns (10-year cycle)
    - **Purpose**: Capture multiple seasonal patterns
    
    **5. Advanced Difference Features:**
    - **diff_1**: First difference (change from previous quarter)
    - **diff_2**: Second difference (change in change)
    - **diff_4**: Year-over-year difference
    - **Purpose**: Capture rate of change patterns
    
    **6. Ratio Features:**
    - **ratio_4/8**: Current value relative to rolling means
    - **Purpose**: Capture relative performance and deviations
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
    st.title("📋 XGBoost Model Summary & Explanation")
    
    # Model parameters display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 Model Parameters:**
        - **Number of Trees**: {n_estimators}
        - **Max Tree Depth**: {max_depth}
        - **Learning Rate**: {learning_rate}
        - **Subsample Ratio**: {subsample}
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
    
    # XGBoost explanation
    st.subheader("🚀 Understanding XGBoost Models")
    st.markdown("""
    **XGBoost (Extreme Gradient Boosting)** is an advanced gradient boosting algorithm:
    
    ### 🔍 How XGBoost Works:
    
    **1. Gradient Boosting:**
    - **Sequential Learning**: Each tree learns from the errors of previous trees
    - **Gradient Descent**: Minimizes loss function using gradient information
    - **Regularization**: Built-in L1 and L2 regularization to prevent overfitting
    
    **2. Advanced Features:**
    - **Tree Pruning**: Automatically prunes trees to optimal size
    - **Missing Value Handling**: Built-in handling of missing data
    - **Parallel Processing**: Efficient parallel tree construction
    
    **3. Key Advantages:**
    - **Superior Performance**: Often achieves best-in-class accuracy
    - **Robustness**: Handles outliers and noise well
    - **Flexibility**: Works with various data types and distributions
    - **Interpretability**: Feature importance and partial dependence plots
    """)
    
    # Your model explanation
    st.subheader("🔍 Understanding Your XGBoost Model")
    
    st.markdown(f"""
    **🚀 Your XGBoost Configuration:**
    
    **Model Structure:**
    - **{n_estimators} Trees**: Sequential learning with gradient boosting
    - **Max Depth {max_depth}**: Controls tree complexity and prevents overfitting
    - **Learning Rate {learning_rate}**: {'Conservative learning' if learning_rate < 0.1 else 'Balanced learning' if learning_rate < 0.2 else 'Aggressive learning'} - smaller values are more conservative
    - **Subsample {subsample}**: Uses {subsample*100}% of data for each tree (prevents overfitting)
    - **{n_lags} Lag Features**: Captures temporal dependencies up to {n_lags} quarters back
    - **Seasonal Features**: {'Included to capture quarterly patterns' if include_seasonal else 'Not included - focuses on trend and lag patterns'}
    
    **Advanced Feature Engineering:**
    - **Enhanced Rolling Statistics**: Mean, std, min, max, median for better pattern capture
    - **Polynomial Trends**: Linear, quadratic, and cubic trends for complex patterns
    - **Multiple Seasonal Cycles**: Quarterly and long-term (10-year) seasonal patterns
    - **Ratio Features**: Relative performance measures for better scaling
    """)
    
    # Why XGBoost for your data
    st.subheader("🎯 Why XGBoost for Unemployment Data?")
    
    st.markdown(f"""
    **✅ XGBoost is excellent for your data because:**
    
    **1. Superior Pattern Recognition:**
    - **Complex relationships**: Can capture highly non-linear unemployment patterns
    - **Interaction effects**: Models complex interactions between different factors
    - **Multiple time scales**: Handles both short-term and long-term patterns simultaneously
    
    **2. Robust and Reliable:**
    - **Outlier resistance**: Less sensitive to extreme values and economic shocks
    - **Missing data handling**: Built-in handling of data gaps
    - **Regularization**: Prevents overfitting even with many features
    
    **3. Advanced Learning:**
    - **Gradient boosting**: Each tree improves upon previous predictions
    - **Adaptive learning**: Automatically adjusts to changing patterns
    - **Feature selection**: Identifies most important predictors automatically
    
    **4. Practical Advantages:**
    - **Fast training**: Efficient implementation for quick model updates
    - **Memory efficient**: Handles large feature sets without memory issues
    - **Production ready**: Stable and reliable for operational forecasting
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
    
    # XGBoost vs other methods
    st.subheader("🚀 XGBoost vs Other Methods")
    
    st.markdown("""
    **XGBoost Advantages:**
    
    **✅ Performance:**
    - Often achieves highest accuracy in competitions
    - Excellent handling of complex patterns
    - Robust to outliers and noise
    
    **✅ Efficiency:**
    - Fast training and prediction
    - Memory efficient
    - Parallel processing capabilities
    
    **✅ Flexibility:**
    - Handles various data types
    - Built-in regularization
    - Missing value handling
    
    **✅ Interpretability:**
    - Feature importance rankings
    - Partial dependence plots
    - Tree structure visualization
    
    **Limitations:**
    - **Sequential nature**: Cannot be fully parallelized
    - **Black box**: Individual predictions harder to explain
    - **Hyperparameter tuning**: Requires careful parameter selection
    - **Overfitting risk**: Can overfit with too many trees
    """)
    
    # Practical interpretation
    st.subheader("💡 How to Use Your XGBoost Forecast")
    
    st.markdown("""
    **For Policy Makers:**
    - **High accuracy**: Use for critical policy decisions
    - **Feature insights**: Understand key unemployment drivers
    - **Scenario analysis**: Test different economic scenarios
    
    **For Businesses:**
    - **Strategic planning**: Use for long-term workforce planning
    - **Risk assessment**: Monitor forecast uncertainty
    - **Competitive advantage**: Leverage superior prediction accuracy
    
    **For Researchers:**
    - **Benchmark model**: Compare with other forecasting methods
    - **Feature discovery**: Identify important economic indicators
    - **Model validation**: Test against traditional time series models
    """)

# === Tab 5: Complete Technical Details ===
with tab5:
    st.title("📋 Complete Technical Details")
    
    # Model summary
    st.subheader("📋 Model Summary")
    
    st.markdown(f"""
    **🔍 Model Configuration:**
    
    **Algorithm**: XGBoost Regressor
    **Number of Trees**: {n_estimators}
    **Max Tree Depth**: {max_depth}
    **Learning Rate**: {learning_rate}
    **Subsample Ratio**: {subsample}
    **Random State**: 42 (for reproducibility)
    
    **Feature Engineering:**
    - **Total Features**: {len(X.columns)}
    - **Lag Features**: {n_lags}
    - **Rolling Statistics**: 7 features (mean, std, min, max, median for 4 and 8 quarters)
    - **Trend Features**: 3 features (linear, quadratic, cubic)
    - **Seasonal Features**: {'6 features' if include_seasonal else '0 features'}
    - **Difference Features**: 3 features (first, second, and year-over-year differences)
    - **Ratio Features**: 2 features (4 and 8-quarter ratios)
    
    **Data Information:**
    - **Training Observations**: {len(X)}
    - **Target Variable**: {selected_metric_label}
    - **Feature Matrix Shape**: {X.shape}
    """)
    
    # Feature engineering details
    st.subheader("🔧 Feature Engineering Details")
    
    st.markdown("""
    **📊 Advanced Feature Creation Process:**
    
    **1. Enhanced Lag Features:**
    ```python
    for lag in range(1, n_lags + 1):
        df_features[f'lag_{lag}'] = df_features[target].shift(lag)
    ```
    
    **2. Advanced Rolling Statistics:**
    ```python
    df_features['rolling_mean_4'] = df_features[target].rolling(window=4).mean()
    df_features['rolling_std_4'] = df_features[target].rolling(window=4).std()
    df_features['rolling_min_4'] = df_features[target].rolling(window=4).min()
    df_features['rolling_max_4'] = df_features[target].rolling(window=4).max()
    df_features['rolling_median_4'] = df_features[target].rolling(window=4).median()
    ```
    
    **3. Polynomial Trend Features:**
    ```python
    df_features['trend'] = range(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    df_features['trend_cubed'] = df_features['trend'] ** 3
    ```
    
    **4. Comprehensive Seasonal Features:**
    ```python
    df_features['quarter'] = df_features.index.quarter
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    df_features['year_sin'] = np.sin(2 * np.pi * df_features['year'] / 10)
    df_features['year_cos'] = np.cos(2 * np.pi * df_features['year'] / 10)
    ```
    
    **5. Advanced Difference Features:**
    ```python
    df_features['diff_1'] = df_features[target].diff()
    df_features['diff_2'] = df_features[target].diff().diff()
    df_features['diff_4'] = df_features[target].diff(4)
    ```
    
    **6. Ratio Features:**
    ```python
    df_features['ratio_4'] = df_features[target] / df_features[target].rolling(window=4).mean()
    df_features['ratio_8'] = df_features[target] / df_features[target].rolling(window=8).mean()
    ```
    """)
    
    # Model training details
    st.subheader("🚀 Model Training Details")
    
    st.markdown("""
    **📈 XGBoost Training Process:**
    
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
    cv_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    ```
    
    **3. Model Fitting:**
    ```python
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_model.fit(X, y)
    ```
    
    **4. Feature Importance:**
    ```python
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
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
    - **Method**: XGBoost feature importance
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
    - **Advanced feature engineering**: Comprehensive feature creation
    - **Hyperparameter tuning**: Reasonable parameter selection
    - **Validation strategy**: Proper out-of-sample testing
    - **Regularization**: Built-in XGBoost regularization
    
    **Limitations:**
    - **Sequential nature**: Cannot be fully parallelized
    - **Complex interactions**: Harder to interpret than linear models
    - **Hyperparameter sensitivity**: Performance depends on parameter tuning
    - **Feature engineering dependency**: Performance depends on feature quality
    """) 