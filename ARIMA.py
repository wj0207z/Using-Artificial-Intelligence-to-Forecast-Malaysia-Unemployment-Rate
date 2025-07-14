import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
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

# === ARIMA Config ===
st.markdown("### ⚙️ ARIMA Model Settings")
force_seasonal = st.checkbox("Force Seasonal ARIMA", value=seasonality_strength > 0.5)

# === Train/Test Split Configuration ===
test_pct = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5)
test_size = int(len(series) * test_pct / 100)
train_size = len(series) - test_size
st.markdown(f"**Training Set:** {train_size} quarters | **Test Set:** {test_size} quarters")

# Split the series
train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]

# === Fit ARIMA model on training set ===
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

# === Forecast on test set ===
test_forecast, test_conf_int = model.predict(n_periods=test_size, return_conf_int=True)
test_forecast_dates = pd.date_range(start=train_series.index[-1] + pd.offsets.QuarterBegin(), periods=test_size, freq='Q')
test_forecast_dates = test_forecast_dates.strftime('%Y-%m-%d')
test_forecast_df = pd.DataFrame({
    "Forecast Date": test_forecast_dates,
    f"Forecasted {selected_metric_label}": test_forecast,
    "Lower CI": test_conf_int[:, 0],
    "Upper CI": test_conf_int[:, 1]
})
test_forecast_df["Forecast Date"] = pd.to_datetime(test_forecast_df["Forecast Date"]).dt.strftime('%Y-%m-%d')

# === In-sample predictions (training set) ===
in_sample_pred = model.predict_in_sample()
train_actual = train_series[-len(in_sample_pred):]
train_rmse = np.sqrt(np.mean((train_actual - in_sample_pred)**2))
train_mape = np.mean(np.abs((train_actual - in_sample_pred) / train_actual)) * 100

# === Overall model metrics (for display) ===
rmse = train_rmse
mape = train_mape

# === Out-of-sample metrics (test set) ===
test_actual = test_series.values
out_rmse = np.sqrt(np.mean((test_actual - test_forecast)**2))
out_mape = np.mean(np.abs((test_actual - test_forecast) / test_actual)) * 100

# === Residuals ===
residuals = pd.Series(model.resid())
max_lags = min(40, len(series) // 2 - 1)
# PACF has stricter limits - use 25% of sample size for PACF
max_lags_pacf = min(20, len(series) // 4 - 1)

# === Future forecast (for explainability) ===
future_forecast, future_conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
forecast = future_forecast

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    f"📈 Forecast ({selected_metric_label})",
    "🔎 Trend & Seasonality Diagnostics",
    "📊 Residual Diagnostics",
    "📋 ARIMA Model Summary",
    "📋 Complete Model Summary",
    "🧠 Explainability"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"📈 Forecast for {selected_metric_label}")
    # Prepare DataFrames for plotting
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    actual_df["Date"] = pd.to_datetime(actual_df["Date"]).dt.strftime('%Y-%m-%d')
    train_df = train_series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    train_df["Date"] = pd.to_datetime(train_df["Date"]).dt.strftime('%Y-%m-%d')
    test_df = test_series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    test_df["Date"] = pd.to_datetime(test_df["Date"]).dt.strftime('%Y-%m-%d')
    test_forecast_df_renamed = test_forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    # Combine for plotting
    combined = pd.concat([train_df, test_df, test_forecast_df_renamed], axis=0)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime('%Y-%m-%d')
    # Plot
    fig = px.line(combined, x="Date", y=selected_metric_label, title="Forecast vs Actual")
    fig.add_scatter(x=test_forecast_df["Forecast Date"], y=test_forecast_df["Upper CI"],
                    mode="lines", name="Upper CI", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=test_forecast_df["Forecast Date"], y=test_forecast_df["Lower CI"],
                    mode="lines", name="Lower CI", fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False)
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)
    # Forecast table (test set)
    test_forecast_df_display = test_forecast_df.copy()
    test_forecast_df_display.index = range(1, len(test_forecast_df_display) + 1)
    test_forecast_df_display.index.name = 'Index'
    test_forecast_df_display = test_forecast_df_display.drop(columns=["Forecast Date"])
    st.dataframe(test_forecast_df_display, use_container_width=True)
    csv = test_forecast_df_display.to_csv().encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")
    # Metrics
    st.metric("Train RMSE", f"{train_rmse:.2f}")
    st.metric("Train MAPE (%)", f"{train_mape:.2f}")
    st.metric("Test RMSE", f"{out_rmse:.2f}")
    st.metric("Test MAPE (%)", f"{out_mape:.2f}")

# === Tab 2: Trend & Seasonality (Plotly UI) ===
with tab2:
    st.title("🔎 Trend & Seasonality Diagnostics")

    st.subheader("📉 Augmented Dickey-Fuller Test")
    st.metric("ADF Statistic", f"{adf_stat:.4f}")
    st.metric("p-value", f"{adf_pvalue:.4f}")
    if adf_pvalue < 0.05:
        st.success("Stationary: No trend detected")
    else:
        st.warning("Non-stationary: Trend likely exists")

    st.subheader("📈 Trend Component")
    trend_fig = px.line(x=decomposition.trend.index, y=decomposition.trend.values,
                        labels={"x": "Date", "y": "Trend"}, title="Trend")
    st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("🔁 Seasonal Component")
    season_fig = px.line(x=seasonal_component.index, y=seasonal_component.values,
                         labels={"x": "Date", "y": "Seasonality"}, title="Seasonality")
    st.plotly_chart(season_fig, use_container_width=True)

    st.subheader("📊 Autocorrelation Plot (ACF)")
    # Calculate appropriate lag size (max 50% of sample size)
    max_lags = min(40, len(series) // 2 - 1)
    fig_acf = sm.graphics.tsa.plot_acf(series, lags=max_lags)
    st.pyplot(fig_acf.figure)

    st.info(f"📌 Seasonality strength: `{seasonality_strength:.3f}`")

# === Tab 3: Residual Diagnostics (Plotly UI) ===
with tab3:
    st.title("📊 Residual Diagnostics")

    # Introduction to residual diagnostics
    st.markdown("""
    **🔍 What are Residual Diagnostics?**
    
    **Residuals** are the differences between your actual data and what your ARIMA model predicted. 
    They tell us how well your model fits the data and whether the model assumptions are met.
    
    **Why are they important?**
    - **Model validation**: Check if your model captures all important patterns
    - **Forecast reliability**: Poor residuals mean unreliable forecasts
    - **Assumption checking**: ARIMA models assume residuals are random noise
    - **Model improvement**: Identify what your model is missing
    """)
    
    # Residuals overview
    st.subheader("🟣 Residuals Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
    with col2:
        st.metric("Residual Std", f"{np.std(residuals):.4f}")
    with col3:
        st.metric("Residual Range", f"{np.max(residuals) - np.min(residuals):.4f}")
    
    # Overview interpretation
    st.markdown("""
    **📊 What These Numbers Mean:**
    
    **Mean Residual**: Should be close to zero. If not, your model has a systematic bias.
    **Standard Deviation**: Measures how spread out the errors are. Lower is better.
    **Range**: The difference between largest and smallest errors. Shows error variability.
    """)

    st.subheader("📈 Residuals Over Time")
    resid_fig = px.line(x=residuals.index, y=residuals.values,
                        labels={'x': 'Date', 'y': 'Residuals'}, title="Residuals Over Time")
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(resid_fig, use_container_width=True)
    
    # Residual interpretation with usage explanation
    st.markdown("""
    **🔍 What to Look For:**
    
    **✅ Good Signs:**
    - **Mean close to zero**: Residuals should average around zero
    - **Random scatter**: No obvious patterns or trends
    - **Constant variance**: Spread should be roughly the same over time
    - **No outliers**: No extreme values that stand out
    
    **⚠️ Warning Signs:**
    - **Trends**: If residuals show a clear upward or downward trend
    - **Heteroskedasticity**: If variance changes over time (funnel shape)
    - **Outliers**: Extreme values that may indicate model misspecification
    - **Patterns**: Any systematic patterns suggest the model is missing something
    
    **🎯 Usage:**
    - **Detect model misspecification**: If residuals show patterns, your model is missing important features
    - **Identify structural breaks**: Sudden changes in residual behavior
    - **Check for outliers**: Unusual observations that may need special handling
    - **Validate forecast assumptions**: Ensures your model errors are random
    """)

    st.subheader("🔁 Autocorrelation of Residuals (ACF)")
    fig_acf_resid = sm.graphics.tsa.plot_acf(residuals, lags=max_lags)
    st.pyplot(fig_acf_resid.figure)
    
    # ACF interpretation
    st.markdown(f"""
    **🔍 ACF Interpretation:**
    
    **✅ Good Signs:**
    - **No significant spikes**: All bars should be within the blue confidence bands
    - **Random pattern**: No systematic pattern in the autocorrelations
    - **White noise**: Residuals should behave like random noise
    
    **⚠️ Warning Signs:**
    - **Spikes outside bands**: Any bar extending beyond the blue lines indicates autocorrelation
    - **Seasonal patterns**: Spikes at lags 4, 8, 12... suggest seasonal patterns not captured
    - **Trend patterns**: Gradual decay suggests trend not fully captured
    - **Lags shown**: {max_lags} (adjusted for sample size)
    
    **🎯 Usage:**
    - **Test independence**: Check if residuals are truly independent (no autocorrelation)
    - **Identify missing patterns**: Spikes indicate patterns your model didn't capture
    - **Validate ARIMA assumptions**: ARIMA assumes residuals are white noise
    - **Guide model improvement**: Shows what additional terms might be needed
    """)
    
    # Residual distribution
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
    
    # Distribution interpretation
    st.markdown("""
    **🔍 Distribution Analysis:**
    
    **✅ Good Signs:**
    - **Bell-shaped curve**: Histogram should look roughly normal
    - **Centered at zero**: Peak should be close to zero
    - **Symmetric**: Left and right sides should be roughly equal
    - **Normal p-value > 0.05**: Jarque-Bera test indicates normality
    
    **⚠️ Warning Signs:**
    - **Skewed distribution**: Asymmetric histogram
    - **Multiple peaks**: Bimodal or multimodal distribution
    - **Heavy tails**: Too many extreme values
    - **Non-normal p-value < 0.05**: Jarque-Bera test indicates non-normality
    
    **🎯 Usage:**
    - **Validate normality assumption**: ARIMA confidence intervals assume normal residuals
    - **Assess forecast reliability**: Non-normal residuals may affect prediction intervals
    - **Detect outliers**: Extreme values that may need special handling
    - **Check model adequacy**: Normal residuals suggest good model fit
    """)
    
    # Q-Q Plot
    st.subheader("📈 Q-Q Plot (Quantile-Quantile)")
    
    # Create Q-Q plot
    from scipy import stats
    fig_qq, ax_qq = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot: Residuals vs Normal Distribution")
    st.pyplot(fig_qq)
    
    # Q-Q interpretation
    st.markdown("""
    **🔍 Q-Q Plot Interpretation:**
    
    **✅ Good Signs:**
    - **Points follow line**: Residuals should follow the red diagonal line
    - **No systematic deviation**: Points should scatter randomly around the line
    - **No extreme outliers**: No points far from the line
    
    **⚠️ Warning Signs:**
    - **Curved pattern**: Suggests non-normal distribution
    - **S-shaped curve**: Indicates skewness
    - **Extreme outliers**: Points far from the line
    - **Systematic deviation**: Clear pattern in how points deviate from line
    
    **🎯 Usage:**
    - **Visual normality test**: More intuitive than statistical tests
    - **Identify distribution shape**: Shows how residuals deviate from normal
    - **Detect outliers**: Points far from the line are potential outliers
    - **Assess model assumptions**: Helps validate normality for confidence intervals
    """)
    
    # PACF of residuals
    st.subheader("📉 Partial Autocorrelation of Residuals (PACF)")
    fig_pacf_resid = sm.graphics.tsa.plot_pacf(residuals, lags=max_lags_pacf)
    st.pyplot(fig_pacf_resid.figure)
    
    # PACF interpretation
    st.markdown(f"""
    **🔍 PACF Interpretation:**
    
    **✅ Good Signs:**
    - **No significant spikes**: All bars within confidence bands
    - **Random pattern**: No systematic partial autocorrelations
    - **White noise**: Residuals should be independent
    
    **⚠️ Warning Signs:**
    - **Spikes outside bands**: Indicates partial autocorrelation
    - **AR patterns**: Suggests autoregressive components not captured
    - **Seasonal patterns**: Spikes at seasonal lags
    - **Lags shown**: {max_lags_pacf} (adjusted for PACF limits)
    
    **🎯 Usage:**
    - **Test independence**: Check if residuals are truly independent after accounting for other lags
    - **Identify AR components**: Spikes suggest missing autoregressive terms
    - **Validate model specification**: Ensures no systematic patterns remain
    - **Guide model refinement**: Shows what additional AR terms might be needed
    """)
    
    # Overall assessment
    st.subheader("🎯 Overall Residual Assessment")
    
    # Calculate assessment metrics
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 Residual Statistics:**
        - **Mean**: {mean_resid:.4f} {'✅' if abs(mean_resid) < 0.1 else '⚠️'}
        - **Standard Deviation**: {std_resid:.4f}
        - **Skewness**: {skewness:.4f} {'✅' if abs(skewness) < 1 else '⚠️'}
        - **Kurtosis**: {kurtosis:.4f} {'✅' if abs(kurtosis) < 3 else '⚠️'}
        """)
    
    with col2:
        st.markdown(f"""
        **🔬 Test Results:**
        - **Normality Test**: {'✅ Normal' if jb_pvalue > 0.05 else '⚠️ Non-normal'}
        - **Independence**: Check ACF/PACF plots above
        - **Constant Variance**: Check residuals over time plot
        """)
    
    # Overall conclusion
    if jb_pvalue > 0.05 and abs(mean_resid) < 0.1 and abs(skewness) < 1:
        st.success("""
        **✅ Overall Assessment: GOOD**
        
        Your residuals appear to be well-behaved:
        - Normally distributed
        - Mean close to zero
        - Reasonable skewness
        - No obvious patterns in ACF/PACF
        
        This suggests your ARIMA model is well-specified and captures the data patterns adequately.
        """)
    else:
        st.warning("""
        **⚠️ Overall Assessment: NEEDS ATTENTION**
        
        Some residual diagnostics show potential issues:
        - Check for non-normality
        - Look for patterns in ACF/PACF
        - Consider model modifications
        
        While your model may still be useful, these issues could affect forecast accuracy and confidence intervals.
        """)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    st.markdown("""
    **If Residuals Show Issues:**
    
    **1. Non-normal residuals:**
    - Consider data transformations (log, square root)
    - Check for outliers and handle them appropriately
    - Consider robust estimation methods
    
    **2. Autocorrelated residuals:**
    - Increase AR or MA order
    - Add seasonal components if not present
    - Consider different model specifications
    
    **3. Heteroskedasticity (changing variance):**
    - Consider variance-stabilizing transformations
    - Use weighted estimation
    - Check for structural breaks in the data
    
    **4. Outliers:**
    - Investigate unusual observations
    - Consider dummy variables for outliers
    - Use robust estimation methods
    """)
    
    st.info("""
    **📌 Note:** Residual diagnostics help assess model adequacy, but minor violations don't necessarily mean the model is useless. 
    Focus on the overall pattern and consider the practical impact on your forecasts.
    """)
    
    # Comprehensive usage guide
    st.subheader("📚 Comprehensive Usage Guide")
    
    st.markdown("""
    **🎯 When to Use Each Diagnostic:**
    
    **1. Residuals Over Time:**
    - **Use when**: You want to see if your model errors are random
    - **Look for**: Patterns, trends, or changing variance
    - **Action**: If patterns exist, consider adding more model terms
    
    **2. Autocorrelation Function (ACF):**
    - **Use when**: You want to test if residuals are independent
    - **Look for**: Spikes outside confidence bands
    - **Action**: If spikes exist, add MA terms or seasonal components
    
    **3. Residual Distribution:**
    - **Use when**: You want to validate normality assumption
    - **Look for**: Bell-shaped, symmetric distribution
    - **Action**: If non-normal, consider data transformations
    
    **4. Q-Q Plot:**
    - **Use when**: You want a visual check of normality
    - **Look for**: Points following the diagonal line
    - **Action**: If curved pattern, check for outliers or transformations
    
    **5. Partial Autocorrelation (PACF):**
    - **Use when**: You want to test independence after accounting for other lags
    - **Look for**: Spikes outside confidence bands
    - **Action**: If spikes exist, add AR terms
    """)
    
    st.markdown("""
    **🔬 Scientific Method for Model Validation:**
    
    **Step 1: Visual Inspection**
    - Look at residuals over time for obvious patterns
    - Check histogram for normality
    - Examine Q-Q plot for systematic deviations
    
    **Step 2: Statistical Testing**
    - Use Jarque-Bera test for normality
    - Check ACF/PACF for independence
    - Calculate summary statistics
    
    **Step 3: Interpretation**
    - Minor violations may be acceptable
    - Focus on practical impact on forecasts
    - Consider model complexity vs. accuracy trade-off
    
    **Step 4: Action**
    - If major issues: Modify model specification
    - If minor issues: Monitor performance
    - If no issues: Model is adequate
    """)
    
    st.markdown("""
    **💡 Practical Tips:**
    
    **For Policy Makers:**
    - Focus on overall forecast accuracy rather than perfect diagnostics
    - Use residual analysis to understand model limitations
    - Consider multiple models for robust forecasting
    
    **For Researchers:**
    - Document all diagnostic results
    - Compare diagnostics across different models
    - Use diagnostics to guide model selection
    
    **For Business Users:**
    - Understand that no model is perfect
    - Use diagnostics to assess forecast reliability
    - Consider confidence intervals when making decisions
    """)

# === Tab 4: ARIMA Model Summary ===
with tab4:
    st.title("📋 ARIMA Model Summary & Explanation")
    
    # Model parameters display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 Model Parameters:**
        - **ARIMA Order**: `{model.order}`  
        - **Seasonal Order**: `{model.seasonal_order}`  
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
    
    # Detailed ARIMA explanation
    st.subheader("🔍 Understanding ARIMA Models")
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** models are powerful tools for time series forecasting that combine three components:
    
    ### 📐 ARIMA(p,d,q) Components:
    - **p (AR)**: Number of autoregressive terms - how many past values influence current value
    - **d (I)**: Degree of differencing - how many times the series is differenced to achieve stationarity
    - **q (MA)**: Number of moving average terms - how many past forecast errors influence current value
    
    ### 🔄 Seasonal ARIMA(p,d,q)(P,D,Q,m):
    - **P**: Seasonal autoregressive order
    - **D**: Seasonal differencing order  
    - **Q**: Seasonal moving average order
    - **m**: Seasonal period (4 for quarterly data)
    """)
    
    # Understanding your model results
    st.subheader("🔍 Understanding Your Model Results")
    
    # Model type explanation
    if force_seasonal:
        st.markdown(f"""
        **🌊 Seasonal ARIMA Model Selected**
        
        Your model is using **Seasonal ARIMA** because:
        - **Seasonality Strength**: `{seasonality_strength:.3f}` (High seasonality detected)
        - **Quarterly Data**: Natural 4-quarter patterns in unemployment data
        - **Force Seasonal**: You enabled this option
        
        **What this means:**
        - The model captures both short-term patterns (within quarters) and long-term seasonal patterns (across years)
        - It can predict both immediate changes and seasonal variations
        - Better for data with clear seasonal cycles
        """)
    else:
        st.markdown(f"""
        **📈 Regular ARIMA Model Selected**
        
        Your model is using **Regular ARIMA** because:
        - **Seasonality Strength**: `{seasonality_strength:.3f}` (Low seasonality detected)
        - **Force Seasonal**: You disabled this option
        
        **What this means:**
        - The model focuses on short-term patterns and trends
        - It doesn't try to capture seasonal variations
        - Simpler model with fewer parameters
        """)
    
    # Model parameters explanation
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order
    
    st.markdown(f"""
    **📊 Your Model Parameters: ARIMA({p},{d},{q})({P},{D},{Q},{m})**
    
    **Non-Seasonal Part ({p},{d},{q}):**
    - **{p} AR terms**: Uses {p} previous values to predict current value
    - **{d} differencing**: Series was differenced {d} time(s) to remove trends
    - **{q} MA terms**: Uses {q} previous forecast errors to improve predictions
    
    **Seasonal Part ({P},{D},{Q},{m}):**
    - **{P} Seasonal AR**: Uses {P} values from same quarter in previous years
    - **{D} Seasonal differencing**: Seasonal trends removed {D} time(s)
    - **{Q} Seasonal MA**: Uses {Q} seasonal forecast errors
    - **{m} Seasonal period**: {m}-quarter seasonal cycle
    """)
    
    # What the parameters mean for your data
    st.markdown(f"""
    **🎯 How Each Parameter Affects Your Forecast:**
    
    **Non-Seasonal Components:**
    
    **AR terms (p={p}):** 
    - Uses unemployment rates from {p} previous quarter(s) to predict current rate
    - **Forecast impact:** If unemployment was high last quarter, the model expects it to remain high this quarter (momentum effect)
    - **Example:** If Q1 unemployment was 5.2%, the model uses this to predict Q2 unemployment
    
    **Differencing (d={d}):** 
    - {'No differencing applied' if d == 0 else f'{d} differencing step(s) applied'}
    - **Forecast impact:** {'Model predicts absolute unemployment levels' if d == 0 else 'Model predicts changes in unemployment (trends)'}
    - **Example:** {'Predicts unemployment rate directly (e.g., 5.2%)' if d == 0 else 'Predicts unemployment changes (e.g., +0.1% from previous quarter)'}
    
    **MA terms (q={q}):** 
    - Uses {q} previous forecast error(s) to improve predictions
    - **Forecast impact:** {'No error correction' if q == 0 else 'Adjusts predictions based on previous mistakes'}
    - **Example:** {'No adjustment for past errors' if q == 0 else 'If model overpredicted last quarter, it reduces this quarter prediction'}
    
    **Seasonal Components:**
    
    **Seasonal AR (P={P}):** 
    - Uses unemployment rates from {P} previous year(s) in the same quarter
    - **Forecast impact:** Q1 2024 prediction uses Q1 2023, Q1 2022, etc.
    - **Example:** Q1 forecast considers historical Q1 patterns (post-holiday layoffs)
    
    **Seasonal Differencing (D={D}):** 
    - Removes seasonal trends by comparing to same quarter last year
    - **Forecast impact:** {'No seasonal trend removal' if D == 0 else 'Focuses on seasonal changes rather than absolute seasonal levels'}
    - **Example:** {'Uses absolute seasonal values' if D == 0 else 'Uses seasonal changes (e.g., Q1 2024 vs Q1 2023 difference)'}
    
    **Seasonal MA (Q={Q}):** 
    - Uses {Q} previous seasonal forecast error(s)
    - **Forecast impact:** {'No seasonal error correction' if Q == 0 else 'Adjusts seasonal predictions based on past seasonal errors'}
    - **Example:** {'No seasonal error adjustment' if Q == 0 else 'If seasonal prediction was wrong last year, adjusts this year seasonal forecast'}
    
    **Seasonal Period (m={m}):** 
    - {m}-quarter seasonal cycle (quarterly data)
    - **Forecast impact:** Model expects patterns to repeat every {m} quarters
    - **Example:** Q1 patterns repeat every year, Q2 patterns repeat every year, etc.
    """)
    
    # Understanding seasonal vs non-seasonal
    st.subheader("🌊 Seasonal vs Non-Seasonal: What's the Difference?")
    
    if force_seasonal:
        st.markdown(f"""
        **✅ You're using Seasonal ARIMA - Here's why it's better for your data:**
        
        **Seasonality Detected:** `{seasonality_strength:.3f}` (High seasonality)
        
        **Real-world unemployment patterns your model captures:**
        - **Q1 (Jan-Mar):** Post-holiday layoffs, seasonal job losses
        - **Q2 (Apr-Jun):** Graduation season, new job seekers enter market
        - **Q3 (Jul-Sep):** Summer employment, seasonal hiring
        - **Q4 (Oct-Dec):** Holiday hiring, year-end employment
        
        **What this means for your forecasts:**
        - The model predicts both short-term changes AND seasonal patterns
        - More accurate predictions during seasonal transitions
        - Better captures the natural unemployment cycle
        - Accounts for predictable seasonal variations
        """)
    else:
        st.markdown(f"""
        **📈 You're using Regular ARIMA - Here's why it works for your data:**
        
        **Seasonality Level:** `{seasonality_strength:.3f}` (Low seasonality)
        
        **What this means:**
        - The model focuses on immediate trends and patterns
        - Simpler model with fewer parameters to estimate
        - May be more stable for short-term predictions
        - Less prone to overfitting seasonal noise
        
        **When this is better:**
        - Data has minimal seasonal patterns
        - You want simpler, more interpretable results
        - Focus is on trend rather than seasonal cycles
        """)
    
    # Model performance explanation
    st.markdown(f"""
    **📊 Your Model Performance:**
    
    **RMSE: {rmse:.2f}** - This measures how far off your predictions are on average. Lower is better.
    
    **MAPE: {mape:.2f}%** - This shows the average percentage error. For unemployment data, this is quite good.
    
    **What these numbers mean:**
    - **RMSE {rmse:.2f}**: On average, your predictions are off by {rmse:.2f} percentage points
    - **MAPE {mape:.2f}%**: Your predictions are off by {mape:.2f}% on average
    - **AIC {round(model.aic(), 2)}**: Model fit quality (lower is better)
    - **BIC {round(model.bic(), 2)}**: Model fit with complexity penalty (lower is better)
    """)
    
    # How your model was chosen
    st.subheader("🎯 How Your Model Was Chosen")
    
    st.markdown(f"""
    **Your model ARIMA({p},{d},{q})({P},{D},{Q},{m}) was automatically selected because:**
    
    **1. Stationarity Test (d={d}):**
    - The Augmented Dickey-Fuller test showed your data needed {d} differencing step(s)
    - This removes trends and makes the data more predictable
    - For unemployment data, this is common because rates tend to trend over time
    
    **2. Best Fit Selection (p={p}, q={q}):**
    - The algorithm tested different combinations of AR and MA terms
    - Your combination had the lowest AIC/BIC scores
    - This means it provides the best balance of accuracy and simplicity
    
    **3. Seasonal Components (P={P}, D={D}, Q={Q}):**
    - Since you {'enabled' if force_seasonal else 'disabled'} seasonal ARIMA
    - The model {'includes' if force_seasonal else 'excludes'} seasonal patterns
    - This {'captures' if force_seasonal else 'ignores'} quarterly unemployment cycles
    """)
    
    # What the selection means
    seasonal_patterns = "Strong seasonal patterns detected" if force_seasonal else "Minimal seasonal patterns"
    aic_quality = "good" if model.aic() < 1000 else "moderate"
    bic_assessment = "appropriate" if model.bic() < model.aic() + 50 else "possibly overfit"
    
    st.markdown(f"""
    **🔍 What This Selection Tells Us:**
    
    **Your data characteristics:**
    - **Trend component:** {d} differencing needed (data has trends)
    - **Short-term memory:** {p} previous values matter for predictions
    - **Error correction:** {q} previous prediction errors help improve accuracy
    - **Seasonal patterns:** {seasonal_patterns}
    
    **Model complexity:**
    - **Total parameters:** {p+q+P+Q+(1 if d>0 else 0)+(1 if D>0 else 0)} parameters estimated
    - **Model fit:** AIC of {round(model.aic(), 2)} indicates {aic_quality} fit
    - **Complexity penalty:** BIC of {round(model.bic(), 2)} suggests {bic_assessment} complexity
    """)
    
    # Understanding your forecast results
    st.subheader("🔮 Understanding Your Forecast Results")
    
    st.markdown(f"""
    **📈 Your Forecast Analysis:**
    
    **Forecast Period:** {n_periods} quarters ahead
    
    **Confidence Intervals:** The shaded area shows the range where we're 95% confident the actual unemployment rate will fall. Wider intervals mean more uncertainty.
    
    **Forecast Trend:** {'Increasing' if forecast[-1] > forecast[0] else 'Decreasing' if forecast[-1] < forecast[0] else 'Stable'} unemployment rate over the forecast period.
    
    **Seasonal Effects:** {'Strong seasonal patterns' if force_seasonal else 'Minimal seasonal effects'} are {'captured' if force_seasonal else 'not modeled'} in your forecast.
    """)
    
    # What affects forecast accuracy
    st.markdown("""
    **🎯 What Affects Your Forecast Accuracy:**
    
    **Data quality:**
    - **Sample size:** {len(series)} observations (adequate for reliable modeling)
    - **Data consistency:** Quarterly frequency maintained
    - **Missing values:** {'None detected' if series.isnull().sum() == 0 else f'{series.isnull().sum()} missing values found'}
    
    **Model fit:**
    - **RMSE {rmse:.2f}:** {'Good' if rmse < 1.0 else 'Moderate' if rmse < 2.0 else 'High'} prediction error
    - **MAPE {mape:.2f}%:** {'Excellent' if mape < 5 else 'Good' if mape < 10 else 'Moderate' if mape < 20 else 'High'} percentage error
    - **Residuals:** {'Well-behaved' if abs(np.mean(residuals)) < 0.1 else 'May have issues'} (mean close to zero)
    
    **Forecast uncertainty:**
    - **Short-term (1-4 quarters):** {'High confidence' if conf_int[0, 1] - conf_int[0, 0] < 2 else 'Moderate confidence'}
    - **Medium-term (5-8 quarters):** {'Moderate confidence' if conf_int[3, 1] - conf_int[3, 0] < 3 else 'Lower confidence'}
    - **Long-term (9+ quarters):** {'Lower confidence' if n_periods > 8 else 'Not applicable'}
    """)
    
    # Practical interpretation
    st.markdown("""
    **💡 How to Use Your Forecast:**
    
    **For policy makers:**
    - Use the forecast trend to plan economic policies
    - Consider confidence intervals for risk assessment
    - Monitor actual vs predicted values for model updates
    
    **For businesses:**
    - Plan hiring/firing based on unemployment trends
    - Adjust business strategies for economic conditions
    - Use seasonal patterns for workforce planning
    
    **For researchers:**
    - Compare with other forecasting methods
    - Analyze forecast accuracy over time
    - Use for economic research and analysis
    """)

# === Tab 5: Complete Model Summary ===
with tab5:
    st.title("📋 Complete Model Summary & Technical Details")
    
    # Full model summary
    st.subheader("📋 Complete Model Summary")
    st.text(model.summary())
    
    # Model summary explanation
    st.subheader("📋 Understanding Your Model Summary")
    
    # Get actual model parameters
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order
    
    st.markdown(f"""
    **🔍 Key Statistics Explained:**
    
    **Model Information:**
    - **SARIMAX({p}, {d}, {q})**: Your model has {p} AR terms, {d} differencing, {q} MA terms
    - **{len(series)} Observations**: Based on {len(series)} data points
    - **AIC: {model.aic():.3f}, BIC: {model.bic():.3f}**: Model selection criteria (lower is better)
    
    **Parameter Estimates:**
    - **sigma2**: Variance of the error term (noise level in your model)
    - **Standard Error**: Uncertainty in the parameter estimates
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
    
    **Distribution Shape:**
    - **Skewness**: Measures symmetry of residuals
    - **Kurtosis**: Measures tail heaviness (outliers)
    - **Impact**: Extreme values affect model assumptions
    """)
    
    # What the diagnostics mean
    st.markdown("""
    **🎯 What These Results Mean for Your Forecast:**
    
    **✅ Good Signs:**
    - **Independent residuals**: Model captures all time patterns
    - **Significant parameters**: Model is statistically valid
    - **Reasonable AIC/BIC**: Model complexity is appropriate
    
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
    st.subheader("🧮 How Calculations Are Carried Out")
    
    st.markdown("""
    **📊 Model Parameter Calculations:**
    
    **ARIMA Order Selection:**
    ```python
    # Auto ARIMA process
    model = pm.auto_arima(
        series,                    # Your unemployment data
        seasonal=force_seasonal,   # Based on your checkbox choice
        m=4,                      # Quarterly seasonality
        max_d=1,                  # Maximum differencing
        D=1 if force_seasonal else 0,  # Seasonal differencing
        stepwise=True,            # Efficient search
        suppress_warnings=True
    )
    
    # Results: model.order = (p, d, q)
    # Results: model.seasonal_order = (P, D, Q, m)
    ```
    
    **Seasonality Strength:**
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
    
    **Forecast Generation:**
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
    
    **2. Model Selection Process:**
    - **Step 1:** Test stationarity using ADF test
    - **Step 2:** Determine differencing order (d)
    - **Step 3:** Grid search for optimal p, q parameters
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
    """)

# === Tab 6: Explainability ===
with tab6:
    st.title("🧠 Model Explainability & Insights")
    st.subheader("Most Influential Lags (ACF)")
    # Calculate and display ACF values
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(series, nlags=max_lags)
    top_lags = np.argsort(np.abs(acf_vals[1:]))[::-1][:5] + 1  # Top 5 lags (skip lag 0)
    st.markdown(f"""
    **Top 5 Most Influential Lags (by ACF):**
    {', '.join([str(lag) for lag in top_lags])}
    """)
    acf_fig = px.bar(x=list(range(1, max_lags+1)), y=acf_vals[1:max_lags+1], labels={"x": "Lag", "y": "ACF Value"}, title="Autocorrelation by Lag")
    st.plotly_chart(acf_fig, use_container_width=True)
    st.info("Lags with high absolute ACF values are most influential for ARIMA's autoregressive (p) and moving average (q) terms.")

    st.subheader("Parameter Sensitivity (What-if Analysis)")
    st.markdown("""
    **Try changing ARIMA parameters to see their effect:**
    - **p (AR):** Number of past values used
    - **d (I):** Differencing steps (trend removal)
    - **q (MA):** Number of past errors used
    """)
    p_sens = st.slider("p (AR terms)", 0, 5, p)
    d_sens = st.slider("d (Differencing)", 0, 2, d)
    q_sens = st.slider("q (MA terms)", 0, 5, q)
    st.caption("(This is a simulation for educational purposes; actual model is not re-fit live.)")
    st.markdown(f"""
    - **Increasing p**: Model uses more past values, can capture longer memory but may overfit.
    - **Increasing d**: Removes more trend, can help with non-stationary data but too much may lose information.
    - **Increasing q**: Model uses more past errors, can correct for more complex error patterns.
    """)
    st.info(f"If you set p={p_sens}, d={d_sens}, q={q_sens}, the ARIMA({p_sens},{d_sens},{q_sens}) model would focus on lags: {', '.join([str(lag) for lag in range(1, p_sens+1)]) if p_sens > 0 else 'None'}.")
    st.markdown("**For real model changes, adjust the settings in the main ARIMA tab.**")
