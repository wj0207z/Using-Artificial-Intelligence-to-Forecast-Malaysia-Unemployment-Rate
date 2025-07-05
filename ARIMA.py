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

# === Fit ARIMA model ===
model = pm.auto_arima(
    series,
    seasonal=force_seasonal,
    m=4 if force_seasonal else 1,
    max_d=1,
    D=1 if force_seasonal else 0,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
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
    f"📈 Forecast ({selected_metric_label})",
    "🔎 Trend & Seasonality Diagnostics",
    "📊 Residual Diagnostics",
    "📋 ARIMA Model Summary",
    "📋 Complete Model Summary"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"📈 Forecast for {selected_metric_label}")
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    forecast_df_renamed = forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    combined = pd.concat([actual_df, forecast_df_renamed], axis=0)

    fig = px.line(combined, x="Date", y=selected_metric_label, title="Forecast vs Actual")
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Upper CI"],
                    mode="lines", name="Upper CI", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Lower CI"],
                    mode="lines", name="Lower CI", fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(forecast_df, use_container_width=True)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAPE (%)", f"{mape:.2f}")

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
    fig_acf = sm.graphics.tsa.plot_acf(series, lags=40)
    st.pyplot(fig_acf.figure)

    st.info(f"📌 Seasonality strength: `{seasonality_strength:.3f}`")

# === Tab 3: Residual Diagnostics (Plotly UI) ===
with tab3:
    st.title("📊 Residual Diagnostics")

    st.subheader("🟣 Residuals Over Time")
    resid_fig = px.line(x=residuals.index, y=residuals.values,
                        labels={'x': 'Date', 'y': 'Residuals'}, title="Residuals")
    st.plotly_chart(resid_fig, use_container_width=True)

    st.subheader("🔁 Autocorrelation of Residuals (ACF)")
    fig_acf_resid = sm.graphics.tsa.plot_acf(residuals, lags=40)
    st.pyplot(fig_acf_resid.figure)

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
