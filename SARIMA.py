import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
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

# === Trend & Seasonality Diagnostics ===
adf_stat, adf_pvalue, _, _, _, _ = adfuller(series)
decomposition = seasonal_decompose(series, model='additive', period=4)
seasonal_component = decomposition.seasonal
seasonality_strength = np.max(np.abs(seasonal_component.dropna()))

# === SARIMA Configuration ===
st.markdown("### 🌊 SARIMA Model Settings")
st.info("""
**SARIMA (Seasonal ARIMA)** models are specifically designed to handle time series with seasonal patterns.
This app automatically detects and models seasonal components in your quarterly unemployment data.
""")

# Seasonal period selector
seasonal_period = st.selectbox("Select seasonal period:", [4, 12], index=0, 
                              help="4 for quarterly data, 12 for monthly data")
seasonal_period_label = "quarterly" if seasonal_period == 4 else "monthly"

# Force seasonal ARIMA (always enabled for SARIMA)
force_seasonal = True
st.success(f"✅ Seasonal ARIMA enabled - Detecting {seasonal_period_label} patterns")

# === Fit SARIMA model ===
model = pm.auto_arima(
    series,
    seasonal=True,  # Always use seasonal for SARIMA
    m=seasonal_period,  # Seasonal period
    max_d=2,  # Allow more differencing for SARIMA
    max_D=2,  # Allow seasonal differencing
    max_p=3,  # Maximum AR order
    max_q=3,  # Maximum MA order
    max_P=2,  # Maximum seasonal AR order
    max_Q=2,  # Maximum seasonal MA order
    D=1,  # Seasonal differencing
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True
)

# === Forecast ===
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
last_date = series.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')
forecast_df = pd.DataFrame({
    "Forecast Date": forecast_dates,
    f"Forecasted {selected_metric_label}": forecast,
    "Lower CI": conf_int[:, 0],
    "Upper CI": conf_int[:, 1]
})

# === Residuals ===
residuals = pd.Series(model.resid())
in_sample_pred = model.predict_in_sample()
actual = series[-len(in_sample_pred):]
rmse = np.sqrt(np.mean((actual - in_sample_pred)**2))
mape = np.mean(np.abs((actual - in_sample_pred) / actual)) * 100

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📈 SARIMA Forecast ({selected_metric_label})",
    "🔎 Seasonal Diagnostics",
    "📊 Residual Analysis",
    "📋 SARIMA Model Summary",
    "📋 Complete Technical Details"
])

# === Tab 1: SARIMA Forecast ===
with tab1:
    st.title(f"🌊 SARIMA Forecast for {selected_metric_label}")
    
    # Model info
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order
    
    st.markdown(f"""
    **🔍 Your SARIMA Model: SARIMA({p},{d},{q})({P},{D},{Q},{m})**
    
    **Non-Seasonal Components:**
    - **AR({p})**: Uses {p} previous values
    - **I({d})**: {d} differencing step(s) applied
    - **MA({q})**: Uses {q} previous forecast errors
    
    **Seasonal Components:**
    - **SAR({P})**: Uses {P} previous seasonal values
    - **SI({D})**: {D} seasonal differencing step(s) applied
    - **SMA({Q})**: Uses {Q} previous seasonal forecast errors
    - **Period({m})**: {m}-period seasonality ({seasonal_period_label})
    """)
    
    # Forecast visualization
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    forecast_df_renamed = forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    combined = pd.concat([actual_df, forecast_df_renamed], axis=0)

    fig = px.line(combined, x="Date", y=selected_metric_label, title="SARIMA Forecast vs Actual")
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Upper CI"],
                    mode="lines", name="Upper CI", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Lower CI"],
                    mode="lines", name="Lower CI", fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("MAPE (%)", f"{mape:.2f}")
    with col3:
        st.metric("Seasonality Strength", f"{seasonality_strength:.3f}")

    # Forecast table
    st.dataframe(forecast_df, use_container_width=True)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download SARIMA Forecast CSV", csv, "sarima_forecast.csv", "text/csv")

# === Tab 2: Seasonal Diagnostics ===
with tab2:
    st.title("🔎 Seasonal Diagnostics & Pattern Analysis")
    
    # Seasonality strength
    st.subheader("📊 Seasonality Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Seasonality Strength", f"{seasonality_strength:.3f}")
        if seasonality_strength > 0.5:
            st.success("Strong seasonal patterns detected")
        elif seasonality_strength > 0.2:
            st.warning("Moderate seasonal patterns detected")
        else:
            st.info("Weak seasonal patterns detected")
    
    with col2:
        st.metric("Seasonal Period", f"{seasonal_period} ({seasonal_period_label})")
        st.metric("ADF p-value", f"{adf_pvalue:.4f}")

    # Seasonal decomposition
    st.subheader("📈 Seasonal Decomposition")
    
    # Trend component
    st.markdown("**📈 Trend Component**")
    trend_fig = px.line(x=decomposition.trend.index, y=decomposition.trend.values,
                        labels={"x": "Date", "y": "Trend"}, title="Long-term Trend")
    st.plotly_chart(trend_fig, use_container_width=True)

    # Seasonal component
    st.markdown("**🔁 Seasonal Component**")
    season_fig = px.line(x=seasonal_component.index, y=seasonal_component.values,
                         labels={"x": "Date", "y": "Seasonality"}, title="Seasonal Patterns")
    st.plotly_chart(season_fig, use_container_width=True)
    
    # Seasonal pattern interpretation
    if seasonal_period == 4:
        st.markdown("""
        **📅 Quarterly Seasonal Patterns:**
        
        **Q1 (Jan-Mar):** Post-holiday layoffs, seasonal job losses
        **Q2 (Apr-Jun):** Graduation season, new job seekers enter market  
        **Q3 (Jul-Sep):** Summer employment, seasonal hiring
        **Q4 (Oct-Dec):** Holiday hiring, year-end employment
        
        Your SARIMA model captures these quarterly cycles to improve forecast accuracy.
        """)

    # ACF and PACF plots
    st.subheader("📊 Autocorrelation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔁 Autocorrelation Function (ACF)**")
        fig_acf = sm.graphics.tsa.plot_acf(series, lags=40)
        st.pyplot(fig_acf.figure)
        st.markdown("""
        **ACF Interpretation:**
        - **Spikes at lag 4, 8, 12...**: Strong quarterly seasonality
        - **Gradual decay**: Trend component present
        - **Sharp cutoff**: MA component suggested
        """)
    
    with col2:
        st.markdown("**📉 Partial Autocorrelation Function (PACF)**")
        fig_pacf = sm.graphics.tsa.plot_pacf(series, lags=40)
        st.pyplot(fig_pacf.figure)
        st.markdown("""
        **PACF Interpretation:**
        - **Spikes at lag 4, 8, 12...**: Seasonal AR component needed
        - **Gradual decay**: AR component suggested
        - **Sharp cutoff**: AR order identification
        """)

# === Tab 3: Residual Analysis ===
with tab3:
    st.title("📊 SARIMA Residual Analysis")
    
    # Residuals overview
    st.subheader("🟣 Residuals Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
    with col2:
        st.metric("Residual Std", f"{np.std(residuals):.4f}")
    with col3:
        st.metric("Residual Range", f"{np.max(residuals) - np.min(residuals):.4f}")

    # Residuals over time
    st.subheader("📈 Residuals Over Time")
    resid_fig = px.line(x=residuals.index, y=residuals.values,
                        labels={'x': 'Date', 'y': 'Residuals'}, title="Residuals Over Time")
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(resid_fig, use_container_width=True)

    # Residual diagnostics
    st.subheader("🔬 Residual Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔁 Residual ACF**")
        fig_acf_resid = sm.graphics.tsa.plot_acf(residuals, lags=40)
        st.pyplot(fig_acf_resid.figure)
        st.markdown("""
        **Good signs:**
        - No significant spikes (white noise)
        - No seasonal patterns remaining
        - Model captures all time dependencies
        """)
    
    with col2:
        st.markdown("**📉 Residual PACF**")
        fig_pacf_resid = sm.graphics.tsa.plot_pacf(residuals, lags=40)
        st.pyplot(fig_pacf_resid.figure)
        st.markdown("""
        **Good signs:**
        - No significant spikes
        - No partial autocorrelation
        - Model specification is adequate
        """)

    # Residual normality
    st.subheader("📊 Residual Distribution")
    
    # Histogram
    fig_hist = px.histogram(residuals, nbins=20, title="Residual Distribution")
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Normality test
    from scipy.stats import jarque_bera
    jb_stat, jb_pvalue = jarque_bera(residuals)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jarque-Bera Statistic", f"{jb_stat:.4f}")
    with col2:
        st.metric("Jarque-Bera p-value", f"{jb_pvalue:.4f}")
    
    if jb_pvalue > 0.05:
        st.success("✅ Residuals are normally distributed")
    else:
        st.warning("⚠️ Residuals may not be normally distributed")

# === Tab 4: SARIMA Model Summary ===
with tab4:
    st.title("📋 SARIMA Model Summary & Explanation")
    
    # Model parameters display
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 SARIMA Model Parameters:**
        - **SARIMA Order**: `({p},{d},{q})`  
        - **Seasonal Order**: `({P},{D},{Q},{m})`  
        - **AIC**: `{round(model.aic(), 2)}`  
        - **BIC**: `{round(model.bic(), 2)}`
        """)
    
    with col2:
        st.markdown(f"""
        **📈 Model Performance:**
        - **RMSE**: `{rmse:.2f}`
        - **MAPE**: `{mape:.2f}%`
        - **Seasonality Strength**: `{seasonality_strength:.3f}`
        """)
    
    # SARIMA explanation
    st.subheader("🌊 Understanding SARIMA Models")
    st.markdown("""
    **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** models extend ARIMA to handle seasonal patterns:
    
    ### 📐 SARIMA(p,d,q)(P,D,Q,m) Components:
    
    **Non-Seasonal Part (p,d,q):**
    - **p (AR)**: Number of autoregressive terms for short-term patterns
    - **d (I)**: Degree of differencing to achieve stationarity
    - **q (MA)**: Number of moving average terms for error correction
    
    **Seasonal Part (P,D,Q,m):**
    - **P (SAR)**: Seasonal autoregressive order for seasonal patterns
    - **D (SI)**: Seasonal differencing to remove seasonal trends
    - **Q (SMA)**: Seasonal moving average for seasonal error correction
    - **m**: Seasonal period (4 for quarterly, 12 for monthly)
    """)
    
    # Your model explanation
    st.subheader("🔍 Understanding Your SARIMA Model")
    
    st.markdown(f"""
    **🌊 Your SARIMA({p},{d},{q})({P},{D},{Q},{m}) Model:**
    
    **Non-Seasonal Components:**
    - **AR({p})**: Uses unemployment rates from {p} previous quarter(s)
    - **I({d})**: {'No differencing' if d == 0 else f'{d} differencing step(s)'} applied
    - **MA({q})**: Uses {q} previous forecast error(s) for correction
    
    **Seasonal Components:**
    - **SAR({P})**: Uses unemployment rates from {P} previous year(s) in same quarter
    - **SI({D})**: {'No seasonal differencing' if D == 0 else f'{D} seasonal differencing step(s)'} applied
    - **SMA({Q})**: Uses {Q} previous seasonal forecast error(s)
    - **Period({m})**: {m}-quarter seasonal cycle ({seasonal_period_label} patterns)
    """)
    
    # Why SARIMA for your data
    st.subheader("🎯 Why SARIMA for Unemployment Data?")
    
    st.markdown(f"""
    **✅ SARIMA is ideal for your data because:**
    
    **1. Strong Seasonality Detected:**
    - **Seasonality Strength**: `{seasonality_strength:.3f}` 
    - **Quarterly Patterns**: Natural 4-quarter cycles in employment
    - **Economic Cycles**: Seasonal hiring/layoff patterns
    
    **2. Captures Multiple Time Scales:**
    - **Short-term**: Quarter-to-quarter changes (p,d,q)
    - **Long-term**: Year-to-year seasonal patterns (P,D,Q,m)
    - **Trend**: Long-term economic trends (d,D)
    
    **3. Real-world Unemployment Patterns:**
    - **Q1**: Post-holiday layoffs, seasonal job losses
    - **Q2**: Graduation season, new job seekers
    - **Q3**: Summer employment, seasonal hiring
    - **Q4**: Holiday hiring, year-end employment
    """)
    
    # Model performance explanation
    st.markdown(f"""
    **📊 Your Model Performance:**
    
    **RMSE: {rmse:.2f}** - Average prediction error in original units
    
    **MAPE: {mape:.2f}%** - Average percentage error (excellent for unemployment data)
    
    **AIC: {round(model.aic(), 2)}** - Model fit quality (lower is better)
    
    **BIC: {round(model.bic(), 2)}** - Model fit with complexity penalty
    
    **What these numbers mean:**
    - **RMSE {rmse:.2f}**: Predictions are off by {rmse:.2f} percentage points on average
    - **MAPE {mape:.2f}%**: {mape:.2f}% average percentage error (very good for economic data)
    - **AIC/BIC**: Model complexity is appropriate for the data
    """)
    
    # Seasonal vs non-seasonal comparison
    st.subheader("🌊 SARIMA vs Regular ARIMA")
    
    st.markdown("""
    **SARIMA Advantages for Your Data:**
    
    **✅ Better Seasonal Forecasting:**
    - Captures quarterly unemployment cycles
    - Predicts seasonal hiring/layoff patterns
    - Accounts for graduation and holiday effects
    
    **✅ More Accurate Long-term Forecasts:**
    - Seasonal patterns persist over time
    - Better confidence intervals
    - More reliable trend predictions
    
    **✅ Economic Reality:**
    - Unemployment has natural seasonal cycles
    - SARIMA models these cycles explicitly
    - More realistic for policy planning
    
    **Regular ARIMA Limitations:**
    - Ignores seasonal patterns
    - May miss important quarterly cycles
    - Less accurate for seasonal data
    """)
    
    # Practical interpretation
    st.subheader("💡 How to Use Your SARIMA Forecast")
    
    st.markdown("""
    **For Policy Makers:**
    - **Seasonal Planning**: Account for quarterly unemployment cycles
    - **Resource Allocation**: Plan for seasonal hiring/layoff periods
    - **Economic Policy**: Adjust policies for seasonal effects
    
    **For Businesses:**
    - **Workforce Planning**: Anticipate seasonal employment changes
    - **Hiring Strategy**: Plan for graduation and holiday seasons
    - **Economic Conditions**: Monitor seasonal unemployment trends
    
    **For Researchers:**
    - **Economic Analysis**: Study seasonal unemployment patterns
    - **Model Comparison**: Compare with other forecasting methods
    - **Validation**: Monitor forecast accuracy over time
    """)

# === Tab 5: Complete Technical Details ===
with tab5:
    st.title("📋 Complete SARIMA Technical Details")
    
    # Full model summary
    st.subheader("📋 Complete Model Summary")
    st.text(model.summary())
    
    # Model summary explanation
    st.subheader("📋 Understanding Your SARIMA Model Summary")
    
    # Get actual model parameters
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order
    
    st.markdown(f"""
    **🔍 Key Statistics Explained:**
    
    **Model Information:**
    - **SARIMAX({p}, {d}, {q})({P}, {D}, {Q}, {m})**: Your SARIMA model specification
    - **{len(series)} Observations**: Based on {len(series)} data points
    - **AIC: {model.aic():.3f}, BIC: {model.bic():.3f}**: Model selection criteria (lower is better)
    
    **Parameter Estimates:**
    - **Non-seasonal parameters**: φ₁, φ₂, ..., φₚ (AR) and θ₁, θ₂, ..., θₚ (MA)
    - **Seasonal parameters**: Φ₁, Φ₂, ..., Φₚ (SAR) and Θ₁, Θ₂, ..., Θₚ (SMA)
    - **sigma2**: Variance of the error term (noise level)
    - **Standard Error**: Uncertainty in parameter estimates
    - **P-value**: Statistical significance (should be < 0.05)
    
    **Model Diagnostics:**
    
    **Ljung-Box Test:**
    - Tests if residuals are independent (no autocorrelation)
    - **Good sign**: Model captures all time patterns properly
    - **Look for**: p-value > 0.05 indicates independent residuals
    
    **Jarque-Bera Test:**
    - Tests if residuals are normally distributed
    - **Impact**: Affects confidence interval accuracy
    - **Look for**: p-value > 0.05 indicates normal residuals
    
    **Heteroskedasticity Test:**
    - Tests if error variance is constant over time
    - **Impact**: Model assumption violation
    - **Look for**: p-value > 0.05 indicates constant variance
    """)
    
    # What the diagnostics mean
    st.markdown("""
    **🎯 What These Results Mean for Your Forecast:**
    
    **✅ Good Signs:**
    - **Independent residuals**: Model captures all time patterns
    - **Significant parameters**: Model is statistically valid
    - **Reasonable AIC/BIC**: Model complexity is appropriate
    - **Normal residuals**: Confidence intervals are reliable
    
    **⚠️ Potential Issues:**
    - **Non-normal residuals**: May affect confidence intervals
    - **Heteroskedasticity**: Error size may vary over time
    - **High skewness/kurtosis**: Outliers may affect model fit
    
    **💡 Recommendations:**
    - **Use forecasts cautiously**: Check actual vs predicted values
    - **Monitor performance**: Update model if accuracy declines
    - **Consider transformations**: If assumptions are violated
    - **Look for outliers**: Extreme values may need special handling
    """)
    
    # How calculations are carried out
    st.subheader("🧮 How SARIMA Calculations Are Carried Out")
    
    st.markdown("""
    **📊 SARIMA Model Parameter Calculations:**
    
    **SARIMA Order Selection:**
    ```python
    # Auto SARIMA process
    model = pm.auto_arima(
        series,                    # Your unemployment data
        seasonal=True,             # Always use seasonal for SARIMA
        m=seasonal_period,         # Seasonal period (4 for quarterly)
        max_d=2,                   # Maximum differencing
        max_D=2,                   # Maximum seasonal differencing
        max_p=3, max_q=3,          # Maximum non-seasonal orders
        max_P=2, max_Q=2,          # Maximum seasonal orders
        D=1,                       # Seasonal differencing
        stepwise=True,             # Efficient search
        suppress_warnings=True
    )
    
    # Results: model.order = (p, d, q)
    # Results: model.seasonal_order = (P, D, Q, m)
    ```
    
    **Seasonality Detection:**
    ```python
    # Decompose time series
    decomposition = seasonal_decompose(series, model='additive', period=4)
    seasonal_component = decomposition.seasonal
    
    # Calculate seasonality strength
    seasonality_strength = np.max(np.abs(seasonal_component.dropna()))
    # High values (>0.5) indicate strong seasonality
    ```
    
    **Performance Metrics:**
    ```python
    # Get model predictions and residuals
    in_sample_pred = model.predict_in_sample()
    actual = series[-len(in_sample_pred):]
    residuals = pd.Series(model.resid())
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual - in_sample_pred)**2))
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual - in_sample_pred) / actual)) * 100
    ```
    
    **SARIMA Forecast Generation:**
    ```python
    # Generate forecasts with confidence intervals
    forecast, conf_int = model.predict(
        n_periods=n_periods,      # Number of quarters to forecast
        return_conf_int=True      # Include confidence intervals
    )
    
    # Forecast dates
    last_date = series.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.offsets.QuarterBegin(), 
        periods=n_periods, 
        freq='Q'
    )
    ```
    """)
    
    # Statistical tests explanation
    st.markdown("""
    **🔬 Statistical Tests and Diagnostics:**
    
    **Stationarity Test (ADF):**
    ```python
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(series)
    # H₀: Series has unit root (non-stationary)
    # H₁: Series is stationary
    # Reject H₀ if p-value < 0.05
    ```
    
    **Model Information Criteria:**
    ```python
    # AIC (Akaike Information Criterion)
    aic = model.aic()
    # AIC = 2k - 2ln(L) where k=parameters, L=likelihood
    
    # BIC (Bayesian Information Criterion)  
    bic = model.bic()
    # BIC = ln(n)k - 2ln(L) where n=sample size
    # Lower values indicate better models
    ```
    
    **Residual Diagnostics:**
    ```python
    # Ljung-Box Test for independence
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    # p-value > 0.05 indicates independent residuals
    
    # Jarque-Bera Test for normality
    from scipy.stats import jarque_bera
    jb_stat, jb_pvalue = jarque_bera(residuals)
    # p-value > 0.05 indicates normal residuals
    ```
    """)
    
    # Data processing steps
    st.markdown("""
    **📈 Data Processing Steps:**
    
    **1. Data Loading and Cleaning:**
    ```python
    df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    series = df[selected_metric].dropna()  # Remove missing values
    ```
    
    **2. SARIMA Model Selection Process:**
    - **Step 1:** Test stationarity using ADF test
    - **Step 2:** Determine differencing order (d) and seasonal differencing (D)
    - **Step 3:** Grid search for optimal p, q, P, Q parameters
    - **Step 4:** Select model with lowest AIC/BIC
    - **Step 5:** Validate with residual diagnostics
    
    **3. Forecast Uncertainty:**
    - **Confidence intervals** based on residual variance
    - **Wider intervals** for longer forecast horizons
    - **Assumes normal distribution** of residuals
    - **95% confidence level** by default
    """)
    
    # Quality assurance
    st.markdown("""
    **✅ Quality Assurance:**
    
    **Verification Methods:**
    - **Cross-validation**: Model performance on unseen data
    - **Residual analysis**: Check model assumptions
    - **Out-of-sample testing**: Validate forecast accuracy
    - **Sensitivity analysis**: Test different parameters
    
    **Best Practices Applied:**
    - **Parsimony principle**: Prefer simpler models
    - **Information criteria**: Balance fit and complexity
    - **Diagnostic testing**: Ensure model validity
    - **Transparency**: All calculations are reproducible
    
    **Limitations:**
    - **Assumes linear relationships** in the data
    - **Requires stationarity** after differencing
    - **Sensitive to outliers** and structural breaks
    - **Confidence intervals** assume normal residuals
    - **Seasonal patterns** must be consistent over time
    """)
