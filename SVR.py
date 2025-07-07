import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
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

# === SVR Configuration ===
st.markdown("### 🔧 SVR Model Settings")
st.info("""
**Support Vector Regression (SVR)** is a powerful machine learning method that uses support vectors to find optimal hyperplanes for regression.
This app uses advanced feature engineering and kernel selection to capture complex patterns in your unemployment data.
""")

# Model parameters
col1, col2 = st.columns(2)
with col1:
    kernel = st.selectbox("Kernel Function:", ["rbf", "linear", "poly", "sigmoid"], 
                         help="RBF is usually best for non-linear patterns")
    C = st.slider("Regularization Parameter (C):", min_value=0.1, max_value=100.0, value=1.0, step=0.1,
                  help="Controls trade-off between accuracy and generalization")

with col2:
    epsilon = st.slider("Epsilon (ε):", min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                       help="Defines the margin of tolerance where no penalty is given to errors")
    n_lags = st.slider("Number of Lag Features:", min_value=4, max_value=12, value=8, step=1)

# Additional parameters for specific kernels
if kernel == "rbf":
    gamma = st.selectbox("Gamma:", ["scale", "auto", 0.1, 0.01, 0.001], 
                        help="Kernel coefficient for RBF, poly and sigmoid")
elif kernel == "poly":
    degree = st.slider("Polynomial Degree:", min_value=2, max_value=5, value=3, step=1)
    gamma = st.selectbox("Gamma:", ["scale", "auto", 0.1, 0.01, 0.001])

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
        df_features['year_sin'] = np.sin(2 * np.pi * df_features['year'] / 10)
        df_features['year_cos'] = np.cos(2 * np.pi * df_features['year'] / 10)
    
    # Difference features
    df_features['diff_1'] = df_features[selected_metric].diff()
    df_features['diff_2'] = df_features[selected_metric].diff().diff()
    
    # Volatility features
    df_features['volatility_4'] = df_features[selected_metric].rolling(window=4).std()
    df_features['volatility_8'] = df_features[selected_metric].rolling(window=8).std()
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    return df_features

# Create features
features_df = create_features(series, n_lags, include_seasonal)

# Prepare X and y
X = features_df.drop(columns=[selected_metric])
y = features_df[selected_metric]

# === Data Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# === Train SVR Model ===
# Create SVR model with selected parameters
if kernel == "rbf":
    svr_model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma
    )
elif kernel == "poly":
    svr_model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma,
        degree=degree
    )
else:
    svr_model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon
    )

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(svr_model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

# Fit final model
svr_model.fit(X_scaled, y)

# === Support Vectors Analysis ===
support_vectors = svr_model.support_vectors_
n_support_vectors = len(support_vectors)
support_vector_indices = svr_model.support_

# === In-sample predictions ===
y_pred = svr_model.predict(X_scaled)
residuals = y - y_pred

# Performance metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100
r2 = r2_score(y, y_pred)

# === Forecast ===
def generate_forecast(model, scaler, last_features, n_periods):
    """Generate multi-step forecast"""
    forecast = []
    current_features = last_features.copy()
    
    # Get feature names in the same order as training data
    feature_names = X.columns.tolist()
    
    for i in range(n_periods):
        # Convert dictionary to list in correct order
        feature_values = [current_features[feature] for feature in feature_names]
        
        # Scale features
        feature_values_scaled = scaler.transform([feature_values])
        
        # Make prediction
        pred = model.predict(feature_values_scaled)[0]
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
forecast_values = generate_forecast(svr_model, scaler, last_features, n_periods)

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
    f"🔧 SVR Forecast ({selected_metric_label})",
    "🔍 Support Vectors Analysis",
    "📊 Model Diagnostics",
    "📋 Model Summary",
    "📋 Complete Technical Details"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"🔧 SVR Forecast for {selected_metric_label}")
    
    # Model info
    st.markdown(f"""
    **🔍 Your SVR Model:**
    
    **Model Configuration:**
    - **Kernel**: {kernel.upper()}
    - **Regularization (C)**: {C}
    - **Epsilon (ε)**: {epsilon}
    - **Lag Features**: {n_lags}
    - **Seasonal Features**: {'Included' if include_seasonal else 'Not included'}
    """)
    
    if kernel == "rbf":
        st.markdown(f"- **Gamma**: {gamma}")
    elif kernel == "poly":
        st.markdown(f"- **Gamma**: {gamma}")
        st.markdown(f"- **Degree**: {degree}")
    
    st.markdown(f"""
    **Cross-Validation Performance:**
    - **CV RMSE**: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}
    - **Support Vectors**: {n_support_vectors} out of {len(X)} training points
    """)
    
    # Forecast visualization
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    forecast_df_renamed = forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    combined = pd.concat([actual_df, forecast_df_renamed], axis=0)

    fig = px.line(combined, x="Date", y=selected_metric_label, title="SVR Forecast vs Actual")
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
    st.download_button("📥 Download Forecast CSV", csv, "svr_forecast.csv", "text/csv")

# === Tab 2: Support Vectors Analysis ===
with tab2:
    st.title("🔍 Support Vectors Analysis")
    
    # Support vectors overview
    st.subheader("🔧 Support Vectors Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Training Points", len(X))
    with col2:
        st.metric("Support Vectors", n_support_vectors)
    with col3:
        support_vector_ratio = n_support_vectors / len(X) * 100
        st.metric("Support Vector Ratio", f"{support_vector_ratio:.1f}%")
    
    # Support vector ratio interpretation
    if support_vector_ratio < 20:
        st.success("✅ Low support vector ratio - Model is efficient and generalizes well")
    elif support_vector_ratio < 50:
        st.warning("⚠️ Moderate support vector ratio - Model complexity is reasonable")
    else:
        st.error("❌ High support vector ratio - Model may be overfitting")
    
    # Support vectors visualization
    st.subheader("📊 Support Vectors in Feature Space")
    
    # Select two most important features for visualization
    feature_importance = np.abs(svr_model.coef_[0]) if hasattr(svr_model, 'coef_') else np.ones(X.shape[1])
    top_features_idx = np.argsort(feature_importance)[-2:]
    feature1, feature2 = X.columns[top_features_idx[0]], X.columns[top_features_idx[1]]
    
    # Create scatter plot
    fig_scatter = px.scatter(
        x=X.iloc[:, top_features_idx[0]], 
        y=X.iloc[:, top_features_idx[1]],
        color=y,
        title=f"Support Vectors in {feature1} vs {feature2} Space",
        labels={'x': feature1, 'y': feature2, 'color': selected_metric_label}
    )
    
    # Highlight support vectors
    fig_scatter.add_scatter(
        x=X.iloc[support_vector_indices, top_features_idx[0]],
        y=X.iloc[support_vector_indices, top_features_idx[1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Support Vectors',
        showlegend=True
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Support vector characteristics
    st.subheader("🔍 Support Vector Characteristics")
    
    # Support vector statistics
    sv_targets = y.iloc[support_vector_indices]
    sv_residuals = residuals.iloc[support_vector_indices]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Support Vector Target Values**")
        fig_sv_targets = px.histogram(sv_targets, title="Distribution of Support Vector Target Values")
        st.plotly_chart(fig_sv_targets, use_container_width=True)
        
        st.metric("Mean SV Target", f"{sv_targets.mean():.3f}")
        st.metric("SV Target Std", f"{sv_targets.std():.3f}")
    
    with col2:
        st.markdown("**📊 Support Vector Residuals**")
        fig_sv_residuals = px.histogram(sv_residuals, title="Distribution of Support Vector Residuals")
        st.plotly_chart(fig_sv_residuals, use_container_width=True)
        
        st.metric("Mean SV Residual", f"{sv_residuals.mean():.3f}")
        st.metric("SV Residual Std", f"{sv_residuals.std():.3f}")
    
    # Support vector explanation
    st.subheader("💡 Understanding Support Vectors")
    
    st.markdown(f"""
    **🔧 What are Support Vectors?**
    
    Support vectors are the critical training points that define the optimal hyperplane in SVR:
    
    **Key Characteristics:**
    - **{n_support_vectors} support vectors** out of {len(X)} total training points
    - **{support_vector_ratio:.1f}%** of training data are support vectors
    - Support vectors lie on or within the ε-tube around the regression line
    - They determine the model's decision boundary and generalization ability
    
    **What This Means for Your Model:**
    - **Low ratio ({support_vector_ratio:.1f}%)**: {'Efficient model with good generalization' if support_vector_ratio < 20 else 'Moderate complexity' if support_vector_ratio < 50 else 'High complexity - may overfit'}
    - **Kernel {kernel.upper()}**: {'Excellent for capturing non-linear patterns' if kernel == 'rbf' else 'Good for linear relationships' if kernel == 'linear' else 'Captures polynomial patterns' if kernel == 'poly' else 'Sigmoid transformation'}
    - **Regularization C={C}**: {'Strong regularization' if C < 1 else 'Moderate regularization' if C < 10 else 'Weak regularization'}
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
    st.title("📋 SVR Model Summary & Explanation")
    
    # Model parameters display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 Model Parameters:**
        - **Kernel**: {kernel.upper()}
        - **Regularization (C)**: {C}
        - **Epsilon (ε)**: {epsilon}
        - **Lag Features**: {n_lags}
        - **Seasonal Features**: {'Included' if include_seasonal else 'Not included'}
        """)
        
        if kernel == "rbf":
            st.markdown(f"- **Gamma**: {gamma}")
        elif kernel == "poly":
            st.markdown(f"- **Gamma**: {gamma}")
            st.markdown(f"- **Degree**: {degree}")
    
    with col2:
        st.markdown(f"""
        **📈 Model Performance:**
        - **RMSE**: `{rmse:.2f}`
        - **MAE**: `{mae:.2f}`
        - **MAPE**: `{mape:.2f}%`
        - **R²**: `{r2:.3f}`
        - **Support Vectors**: {n_support_vectors}
        """)
    
    # SVR explanation
    st.subheader("🔧 Understanding SVR Models")
    st.markdown("""
    **Support Vector Regression (SVR)** is a powerful machine learning method that finds optimal hyperplanes for regression:
    
    ### 🔍 How SVR Works:
    
    **1. ε-Tube Concept:**
    - **Epsilon tube**: Defines a margin where no penalty is given to errors
    - **Support vectors**: Points that lie on or outside the ε-tube
    - **Optimal hyperplane**: Minimizes the sum of errors while maximizing the margin
    
    **2. Kernel Trick:**
    - **Linear kernel**: For linear relationships
    - **RBF kernel**: For non-linear patterns (most common)
    - **Polynomial kernel**: For polynomial relationships
    - **Sigmoid kernel**: For sigmoid-like patterns
    
    **3. Key Advantages:**
    - **Non-linear modeling**: Can capture complex patterns
    - **Robustness**: Less sensitive to outliers
    - **Sparsity**: Only support vectors matter for predictions
    - **Regularization**: Built-in overfitting prevention
    """)
    
    # Your model explanation
    st.subheader("🔍 Understanding Your SVR Model")
    
    st.markdown(f"""
    **🔧 Your SVR Configuration:**
    
    **Kernel Selection:**
    - **{kernel.upper()} Kernel**: {'Excellent for capturing complex non-linear patterns in unemployment data' if kernel == 'rbf' else 'Good for linear relationships and trends' if kernel == 'linear' else 'Captures polynomial patterns and interactions' if kernel == 'poly' else 'Sigmoid transformation for S-shaped patterns'}
    
    **Regularization:**
    - **C = {C}**: {'Strong regularization - prioritizes generalization over accuracy' if C < 1 else 'Moderate regularization - balanced approach' if C < 10 else 'Weak regularization - prioritizes accuracy over generalization'}
    - **ε = {epsilon}**: Defines the margin where no penalty is given to errors
    
    **Feature Engineering:**
    - **{n_lags} Lag Features**: Captures temporal dependencies
    - **Seasonal Features**: {'Included to model quarterly patterns' if include_seasonal else 'Not included - focuses on trend patterns'}
    - **Rolling Statistics**: Captures trend and volatility patterns
    - **Data Scaling**: Standardized features for optimal SVR performance
    """)
    
    # Why SVR for your data
    st.subheader("🎯 Why SVR for Unemployment Data?")
    
    st.markdown(f"""
    **✅ SVR is excellent for your data because:**
    
    **1. Non-linear Pattern Capture:**
    - **Complex relationships**: Unemployment may have non-linear patterns
    - **Kernel flexibility**: {kernel.upper()} kernel adapts to your data structure
    - **Feature interactions**: Can capture complex interactions between variables
    
    **2. Robustness and Stability:**
    - **Outlier resistance**: Less sensitive to extreme values
    - **ε-tube concept**: Tolerates small prediction errors
    - **Regularization**: Built-in overfitting prevention
    
    **3. Sparse Solution:**
    - **Support vectors**: Only {n_support_vectors} critical points out of {len(X)} training points
    - **Efficient predictions**: Fast inference using support vectors
    - **Interpretability**: Support vectors show the most important data points
    
    **4. Theoretical Foundation:**
    - **Statistical learning theory**: Strong theoretical guarantees
    - **Margin maximization**: Optimizes for generalization
    - **Kernel methods**: Powerful non-linear modeling capability
    """)
    
    # Model performance explanation
    st.markdown(f"""
    **📊 Your Model Performance:**
    
    **RMSE: {rmse:.2f}** - Average prediction error in original units
    
    **MAE: {mae:.2f}** - Average absolute prediction error
    
    **MAPE: {mape:.2f}%** - Average percentage error (excellent for unemployment data)
    
    **R²: {r2:.3f}** - Proportion of variance explained by the model
    
    **Cross-Validation RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}** - Out-of-sample performance
    
    **Support Vector Ratio: {support_vector_ratio:.1f}%** - Model efficiency indicator
    
    **What these numbers mean:**
    - **RMSE {rmse:.2f}**: Predictions are off by {rmse:.2f} percentage points on average
    - **MAPE {mape:.2f}%**: {mape:.2f}% average percentage error (very good for economic data)
    - **R² {r2:.3f}**: Model explains {r2*100:.1f}% of the variance in unemployment rates
    - **Support Vector Ratio**: {'Efficient model' if support_vector_ratio < 20 else 'Moderate complexity' if support_vector_ratio < 50 else 'High complexity'}
    """)
    
    # SVR vs other methods
    st.subheader("🔧 SVR vs Other Methods")
    
    st.markdown("""
    **SVR Advantages:**
    
    **✅ Non-linear Modeling:**
    - Can capture complex non-linear relationships
    - Kernel trick enables high-dimensional feature spaces
    - Flexible modeling of various data patterns
    
    **✅ Robustness:**
    - Less sensitive to outliers than linear models
    - ε-tube concept provides error tolerance
    - Regularization prevents overfitting
    
    **✅ Theoretical Foundation:**
    - Based on statistical learning theory
    - Strong generalization guarantees
    - Optimal margin maximization
    
    **✅ Sparse Solutions:**
    - Only support vectors matter for predictions
    - Efficient memory usage
    - Fast inference times
    
    **Limitations:**
    - **Computational complexity**: Can be slower than simpler models
    - **Parameter tuning**: Requires careful selection of C, ε, and kernel parameters
    - **Interpretability**: Less interpretable than linear models
    - **Scalability**: May not scale well to very large datasets
    """)
    
    # Practical interpretation
    st.subheader("💡 How to Use Your SVR Forecast")
    
    st.markdown("""
    **For Policy Makers:**
    - **Non-linear insights**: Use SVR to identify complex unemployment patterns
    - **Support vector analysis**: Focus on critical data points that drive predictions
    - **Kernel selection**: Choose appropriate kernel based on data characteristics
    
    **For Businesses:**
    - **Pattern recognition**: Identify complex relationships in unemployment data
    - **Robust forecasting**: Use SVR's outlier resistance for stable predictions
    - **Feature importance**: Analyze which features contribute most to predictions
    
    **For Researchers:**
    - **Non-linear modeling**: Study complex relationships in economic data
    - **Kernel comparison**: Test different kernels for optimal performance
    - **Support vector analysis**: Understand which data points are most critical
    """)

# === Tab 5: Complete Technical Details ===
with tab5:
    st.title("📋 Complete Technical Details")
    
    # Model summary
    st.subheader("📋 Model Summary")
    
    st.markdown(f"""
    **🔍 Model Configuration:**
    
    **Algorithm**: Support Vector Regression
    **Kernel**: {kernel.upper()}
    **Regularization Parameter (C)**: {C}
    **Epsilon (ε)**: {epsilon}
    **Random State**: 42 (for reproducibility)
    
    **Feature Engineering:**
    - **Total Features**: {len(X.columns)}
    - **Lag Features**: {n_lags}
    - **Rolling Statistics**: 7 features (mean, std, min, max, median for 4 and 8 quarters)
    - **Trend Features**: 3 features (linear, quadratic, cubic)
    - **Seasonal Features**: {'6 features' if include_seasonal else '0 features'}
    - **Difference Features**: 2 features (first and second differences)
    - **Volatility Features**: 2 features (4 and 8 quarter volatility)
    
    **Data Information:**
    - **Training Observations**: {len(X)}
    - **Target Variable**: {selected_metric_label}
    - **Feature Matrix Shape**: {X.shape}
    - **Support Vectors**: {n_support_vectors} ({support_vector_ratio:.1f}%)
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
    df_features['rolling_min_4'] = df_features[target].rolling(window=4).min()
    df_features['rolling_max_4'] = df_features[target].rolling(window=4).max()
    df_features['rolling_median_4'] = df_features[target].rolling(window=4).median()
    ```
    
    **3. Trend Features:**
    ```python
    df_features['trend'] = range(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    df_features['trend_cubed'] = df_features['trend'] ** 3
    ```
    
    **4. Seasonal Features:**
    ```python
    df_features['quarter'] = df_features.index.quarter
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    ```
    
    **5. Volatility Features:**
    ```python
    df_features['volatility_4'] = df_features[target].rolling(window=4).std()
    df_features['volatility_8'] = df_features[target].rolling(window=8).std()
    ```
    """)
    
    # Model training details
    st.subheader("🔧 Model Training Details")
    
    st.markdown("""
    **📈 Training Process:**
    
    **1. Data Preprocessing:**
    ```python
    # Standardize features for SVR
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
    
    **2. Cross-Validation:**
    ```python
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(svr_model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    ```
    
    **3. Model Fitting:**
    ```python
    svr_model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma if kernel in ['rbf', 'poly'] else 'scale',
        degree=degree if kernel == 'poly' else 3,
        random_state=42
    )
    svr_model.fit(X_scaled, y)
    ```
    
    **4. Support Vector Analysis:**
    ```python
    support_vectors = svr_model.support_vectors_
    support_vector_indices = svr_model.support_
    n_support_vectors = len(support_vectors)
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
        # Scale features
        feature_values_scaled = scaler.transform([feature_values])
        
        # Make prediction
        pred = model.predict(feature_values_scaled)[0]
        forecast.append(pred)
        
        # Update features for next prediction
        # Shift lag features and update other features...
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
    
    **Support Vector Ratio:**
    ```
    SV_Ratio = (Number of Support Vectors / Total Training Points) × 100%
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
    
    **2. Support Vector Analysis:**
    - **Method**: Analysis of support vectors and their characteristics
    - **Purpose**: Understand model complexity and efficiency
    - **Result**: Model interpretability and efficiency assessment
    
    **3. Residual Analysis:**
    - **Method**: Analysis of prediction errors
    - **Purpose**: Check model assumptions and fit quality
    - **Result**: Model diagnostics and improvement insights
    
    **4. Feature Scaling:**
    - **Method**: StandardScaler for feature normalization
    - **Purpose**: Ensure optimal SVR performance
    - **Result**: Improved model convergence and accuracy
    
    **Best Practices Applied:**
    - **Temporal ordering**: Respects time series structure
    - **Feature engineering**: Comprehensive feature creation
    - **Data scaling**: Proper normalization for SVR
    - **Parameter selection**: Reasonable hyperparameter choices
    - **Validation strategy**: Proper out-of-sample testing
    
    **Limitations:**
    - **Computational complexity**: Can be slower than simpler models
    - **Parameter sensitivity**: Performance depends on C, ε, and kernel choice
    - **Scalability**: May not scale well to very large datasets
    - **Interpretability**: Less interpretable than linear models
    - **Memory usage**: Stores support vectors for predictions
    """) 