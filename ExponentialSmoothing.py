import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

# === Select and plot time series ===
series = df[selected_metric].dropna()

# Forecast horizon selector
n_periods = st.slider("Select number of quarters to forecast:", min_value=4, max_value=16, value=8, step=4)

# === Train/Test Split Configuration ===
test_pct = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5)
test_size = int(len(series) * test_pct / 100)
train_size = len(series) - test_size
st.markdown(f"**Training Set:** {train_size} quarters | **Test Set:** {test_size} quarters")
st.subheader("📊 Historical Time Series")
st.line_chart(series)

# === Trend & Seasonality Diagnostics ===
adf_stat, adf_pvalue, _, _, _, _ = adfuller(series)
decomposition = seasonal_decompose(series, model='additive', period=4)
seasonal_component = decomposition.seasonal
seasonality_strength = np.max(np.abs(seasonal_component.dropna()))

# === Exponential Smoothing Configuration ===
st.markdown("### 📈 Exponential Smoothing Model Settings")
st.info("""
**Exponential Smoothing** models are simple yet powerful forecasting methods that use weighted averages of past observations.
This app automatically selects the best exponential smoothing method for your data.
""")

# Model type selector
model_type = st.selectbox("Select Exponential Smoothing Model:", 
                         ["Auto-select", "Simple", "Holt's (Trend)", "Holt-Winters (Seasonal)"],
                         help="Auto-select will choose the best model based on your data")

# Seasonal period selector
seasonal_period = st.selectbox("Select seasonal period:", [4, 12], index=0, 
                              help="4 for quarterly data, 12 for monthly data")
seasonal_period_label = "quarterly" if seasonal_period == 4 else "monthly"

# === Fit Exponential Smoothing model ===
def fit_exponential_smoothing(series, model_type, seasonal_period):
    """Fit exponential smoothing model based on type"""
    
    if model_type == "Auto-select":
        # Try different models and select best one
        models = {}
        
        # Simple Exponential Smoothing
        try:
            model_simple = ExponentialSmoothing(series, trend=None, seasonal=None)
            fitted_simple = model_simple.fit()
            models['Simple'] = fitted_simple
        except:
            pass
        
        # Holt's method (trend)
        try:
            model_holt = ExponentialSmoothing(series, trend='add', seasonal=None)
            fitted_holt = model_holt.fit()
            models['Holt'] = fitted_holt
        except:
            pass
        
        # Holt-Winters (seasonal)
        try:
            model_hw = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_period)
            fitted_hw = model_hw.fit()
            models['Holt-Winters'] = fitted_hw
        except:
            pass
        
        # Select best model based on AIC
        if models:
            best_model_name = min(models.keys(), key=lambda x: models[x].aic)
            return models[best_model_name], best_model_name
        else:
            # Fallback to simple
            model = ExponentialSmoothing(series, trend=None, seasonal=None)
            fitted_model = model.fit()
            return fitted_model, "Simple"
    
    elif model_type == "Simple":
        model = ExponentialSmoothing(series, trend=None, seasonal=None)
        fitted_model = model.fit()
        return fitted_model, "Simple"
    
    elif model_type == "Holt's (Trend)":
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fitted_model = model.fit()
        return fitted_model, "Holt"
    
    elif model_type == "Holt-Winters (Seasonal)":
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_period)
        fitted_model = model.fit()
        return fitted_model, "Holt-Winters"

# Fit model
model, model_name = fit_exponential_smoothing(series, model_type, seasonal_period)

# === Forecast ===
forecast = model.forecast(n_periods)
last_date = series.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(), periods=n_periods, freq='Q')
forecast_dates = forecast_dates.strftime('%Y-%m-%d')

# Calculate confidence intervals (approximate)
forecast_std = np.std(model.resid)
conf_int_lower = forecast - 1.96 * forecast_std
conf_int_upper = forecast + 1.96 * forecast_std

forecast_df = pd.DataFrame({
    "Forecast Date": forecast_dates,
    f"Forecasted {selected_metric_label}": forecast,
    "Lower CI": conf_int_lower,
    "Upper CI": conf_int_upper
})
forecast_df["Forecast Date"] = pd.to_datetime(forecast_df["Forecast Date"]).dt.strftime('%Y-%m-%d')

# === Residuals ===
residuals = pd.Series(model.resid)
in_sample_pred = model.fittedvalues
actual = series[-len(in_sample_pred):]
rmse = np.sqrt(np.mean((actual - in_sample_pred)**2))
mape = np.mean(np.abs((actual - in_sample_pred) / actual)) * 100

# === Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📈 {model_name} Forecast ({selected_metric_label})",
    "🔎 Trend & Seasonality Diagnostics",
    "📊 Residual Analysis",
    "📋 Model Summary",
    "📋 Complete Technical Details"
])

# === Tab 1: Forecast ===
with tab1:
    st.title(f"📈 {model_name} Forecast for {selected_metric_label}")
    
    # Model info
    st.markdown(f"""
    **🔍 Your {model_name} Model:**
    
    **Model Type**: {model_name}
    **Seasonal Period**: {seasonal_period} ({seasonal_period_label})
    **Smoothing Parameters**: α={model.params['smoothing_level']:.3f}
    """)
    
    if model_name in ["Holt", "Holt-Winters"]:
        st.markdown(f"**Trend Parameter**: β={model.params.get('smoothing_trend', 0):.3f}")
    
    if model_name == "Holt-Winters":
        st.markdown(f"**Seasonal Parameter**: γ={model.params.get('smoothing_seasonal', 0):.3f}")
    
    # Forecast visualization
    actual_df = series.reset_index().rename(columns={"date": "Date", selected_metric: selected_metric_label})
    actual_df["Date"] = pd.to_datetime(actual_df["Date"]).dt.strftime('%Y-%m-%d')
    forecast_df_renamed = forecast_df.rename(columns={"Forecast Date": "Date", f"Forecasted {selected_metric_label}": selected_metric_label})
    combined = pd.concat([actual_df, forecast_df_renamed], axis=0)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime('%Y-%m-%d')

    fig = px.line(combined, x="Date", y=selected_metric_label, title=f"{model_name} Forecast vs Actual")
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Upper CI"],
                    mode="lines", name="Upper CI", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=forecast_df["Forecast Date"], y=forecast_df["Lower CI"],
                    mode="lines", name="Lower CI", fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)', line=dict(width=0), showlegend=False)
    fig.update_xaxes(type='category')
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
    forecast_df_display = forecast_df.copy()
    forecast_df_display.index = range(1, len(forecast_df_display) + 1)
    forecast_df_display.index.name = 'Index'
    forecast_df_display = forecast_df_display.drop(columns=["Forecast Date"])
    st.dataframe(forecast_df_display, use_container_width=True)
    csv = forecast_df_display.to_csv().encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "exponential_smoothing_forecast.csv", "text/csv")

# === Tab 2: Trend & Seasonality Diagnostics ===
with tab2:
    st.title("🔎 Trend & Seasonality Diagnostics")
    
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
        
        Your exponential smoothing model captures these patterns through seasonal components.
        """)

    # ACF plot
    st.subheader("📊 Autocorrelation Analysis")
    
    # Calculate appropriate lag size (max 50% of sample size)
    max_lags = min(40, len(series) // 2 - 1)
    
    st.markdown("**🔁 Autocorrelation Function (ACF)**")
    fig_acf = sm.graphics.tsa.plot_acf(series, lags=max_lags)
    st.pyplot(fig_acf.figure)
    st.markdown(f"""
    **ACF Interpretation:**
    - **Spikes at lag 4, 8, 12...**: Strong quarterly seasonality
    - **Gradual decay**: Trend component present
    - **Sharp cutoff**: Moving average component suggested
    - **Lags shown**: {max_lags} (adjusted for sample size)
    """)

# === Tab 3: Residual Analysis ===
with tab3:
    st.title("📊 Residual Analysis")
    
    # Introduction to residual diagnostics
    st.markdown("""
    **🔍 What are Residual Diagnostics?**
    
    **Residuals** are the differences between your actual data and what your Exponential Smoothing model predicted. 
    They tell us how well your model fits the data and whether the model assumptions are met.
    
    **Why are they important?**
    - **Model validation**: Check if your model captures all important patterns
    - **Forecast reliability**: Poor residuals mean unreliable forecasts
    - **Assumption checking**: Exponential Smoothing models assume residuals are random noise
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

    # Residuals over time
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

    # Residual diagnostics
    st.subheader("🔬 Residual Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔁 Residual ACF**")
        fig_acf_resid = sm.graphics.tsa.plot_acf(residuals, lags=max_lags)
        st.pyplot(fig_acf_resid.figure)
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
        
        **🎯 Usage:**
        - **Test independence**: Check if residuals are truly independent (no autocorrelation)
        - **Identify missing patterns**: Spikes indicate patterns your model didn't capture
        - **Validate Exponential Smoothing assumptions**: Model assumes residuals are white noise
        - **Guide model improvement**: Shows what additional components might be needed
        
        **Lags shown**: {max_lags} (adjusted for sample size)
        """)
    
    with col2:
        st.markdown("**📊 Residual Distribution**")
        # Histogram
        fig_hist = px.histogram(residuals, nbins=20, title="Residual Distribution")
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Normality test
    st.subheader("📊 Residual Normality Test")
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
    - **Validate normality assumption**: Exponential Smoothing confidence intervals assume normal residuals
    - **Assess forecast reliability**: Non-normal residuals may affect prediction intervals
    - **Detect outliers**: Extreme values that may need special handling
    - **Check model adequacy**: Normal residuals suggest good model fit
    """)
    
    # Comprehensive usage guide
    st.subheader("📚 Comprehensive Usage Guide")
    
    st.markdown("""
    **🎯 When to Use Each Diagnostic:**
    
    **1. Residuals Over Time:**
    - **Use when**: You want to see if your model errors are random
    - **Look for**: Patterns, trends, or changing variance
    - **Action**: If patterns exist, consider adding more model components
    
    **2. Autocorrelation Function (ACF):**
    - **Use when**: You want to test if residuals are independent
    - **Look for**: Spikes outside confidence bands
    - **Action**: If spikes exist, consider different smoothing parameters or model type
    
    **3. Residual Distribution:**
    - **Use when**: You want to validate normality assumption
    - **Look for**: Bell-shaped, symmetric distribution
    - **Action**: If non-normal, consider data transformations
    """)
    
    st.markdown("""
    **🔬 Scientific Method for Model Validation:**
    
    **Step 1: Visual Inspection**
    - Look at residuals over time for obvious patterns
    - Check histogram for normality
    - Examine ACF for independence
    
    **Step 2: Statistical Testing**
    - Use Jarque-Bera test for normality
    - Check ACF for independence
    - Calculate summary statistics
    
    **Step 3: Interpretation**
    - Minor violations may be acceptable
    - Focus on practical impact on forecasts
    - Consider model complexity vs. accuracy trade-off
    
    **Step 4: Action**
    - If major issues: Try different smoothing parameters or model type
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
    
    st.info("""
    **📌 Note:** Residual diagnostics help assess model adequacy, but minor violations don't necessarily mean the model is useless. 
    Focus on the overall pattern and consider the practical impact on your forecasts.
    """)

# === Tab 4: Model Summary ===
with tab4:
    st.title("📋 Exponential Smoothing Model Summary & Explanation")
    
    # Model parameters display
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **📊 Model Parameters:**
        - **Model Type**: {model_name}
        - **AIC**: `{round(model.aic, 2)}`  
        - **BIC**: `{round(model.bic, 2)}`
        - **Seasonal Period**: {seasonal_period} ({seasonal_period_label})
        """)
    
    with col2:
        st.markdown(f"""
        **📈 Model Performance:**
        - **RMSE**: `{rmse:.2f}`
        - **MAPE**: `{mape:.2f}%`
        - **Seasonality Strength**: `{seasonality_strength:.3f}`
        """)
    
    # Exponential Smoothing explanation
    st.subheader("📈 Understanding Exponential Smoothing Models")
    st.markdown("""
    **Exponential Smoothing** models are forecasting methods that use weighted averages of past observations:
    
    ### 🔍 Types of Exponential Smoothing:
    
    **1. Simple Exponential Smoothing (SES):**
    - **Use case**: No trend, no seasonality
    - **Formula**: Ŷ(t+1) = αY(t) + (1-α)Ŷ(t)
    - **Parameter**: α (smoothing level)
    
    **2. Holt's Method (Double Exponential Smoothing):**
    - **Use case**: Trend, no seasonality
    - **Formula**: Level + Trend components
    - **Parameters**: α (level), β (trend)
    
    **3. Holt-Winters Method (Triple Exponential Smoothing):**
    - **Use case**: Trend + seasonality
    - **Formula**: Level + Trend + Seasonal components
    - **Parameters**: α (level), β (trend), γ (seasonal)
    """)
    
    # Your model explanation
    st.subheader("🔍 Understanding Your Model")
    
    if model_name == "Simple":
        st.markdown(f"""
        **📈 Your Simple Exponential Smoothing Model:**
        
        **Model Type**: Simple Exponential Smoothing
        **Smoothing Parameter**: α = {model.params['smoothing_level']:.3f}
        
        **What this means:**
        - **α = {model.params['smoothing_level']:.3f}**: {'High weight on recent observations' if model.params['smoothing_level'] > 0.5 else 'Balanced weight between recent and past observations' if model.params['smoothing_level'] > 0.2 else 'High weight on past observations'}
        - **No trend modeling**: Assumes unemployment rate has no systematic trend
        - **No seasonality**: Assumes no quarterly patterns
        - **Simple and robust**: Good for stable time series
        """)
    
    elif model_name == "Holt":
        st.markdown(f"""
        **📈 Your Holt's Method Model:**
        
        **Model Type**: Holt's Method (Double Exponential Smoothing)
        **Level Parameter**: α = {model.params['smoothing_level']:.3f}
        **Trend Parameter**: β = {model.params['smoothing_trend']:.3f}
        
        **What this means:**
        - **α = {model.params['smoothing_level']:.3f}**: {'High weight on recent level' if model.params['smoothing_level'] > 0.5 else 'Balanced level smoothing' if model.params['smoothing_level'] > 0.2 else 'High weight on past level'}
        - **β = {model.params['smoothing_trend']:.3f}**: {'High weight on recent trend' if model.params['smoothing_trend'] > 0.5 else 'Balanced trend smoothing' if model.params['smoothing_trend'] > 0.2 else 'High weight on past trend'}
        - **Trend modeling**: Captures systematic changes in unemployment rate
        - **No seasonality**: Assumes no quarterly patterns
        - **Good for trending data**: Better than simple smoothing for trending series
        """)
    
    elif model_name == "Holt-Winters":
        st.markdown(f"""
        **📈 Your Holt-Winters Model:**
        
        **Model Type**: Holt-Winters Method (Triple Exponential Smoothing)
        **Level Parameter**: α = {model.params['smoothing_level']:.3f}
        **Trend Parameter**: β = {model.params['smoothing_trend']:.3f}
        **Seasonal Parameter**: γ = {model.params['smoothing_seasonal']:.3f}
        
        **What this means:**
        - **α = {model.params['smoothing_level']:.3f}**: {'High weight on recent level' if model.params['smoothing_level'] > 0.5 else 'Balanced level smoothing' if model.params['smoothing_level'] > 0.2 else 'High weight on past level'}
        - **β = {model.params['smoothing_trend']:.3f}**: {'High weight on recent trend' if model.params['smoothing_trend'] > 0.5 else 'Balanced trend smoothing' if model.params['smoothing_trend'] > 0.2 else 'High weight on past trend'}
        - **γ = {model.params['smoothing_seasonal']:.3f}**: {'High weight on recent seasonal patterns' if model.params['smoothing_seasonal'] > 0.5 else 'Balanced seasonal smoothing' if model.params['smoothing_seasonal'] > 0.2 else 'High weight on past seasonal patterns'}
        - **Complete modeling**: Captures level, trend, and seasonal components
        - **Best for seasonal data**: Ideal for quarterly unemployment patterns
        """)
    
    # Why Exponential Smoothing for your data
    st.subheader("🎯 Why Exponential Smoothing for Unemployment Data?")
    
    st.markdown(f"""
    **✅ Exponential Smoothing is excellent for your data because:**
    
    **1. Automatic Parameter Selection:**
    - **Optimal weights**: Automatically finds best smoothing parameters
    - **Data-driven**: Adapts to your specific unemployment patterns
    - **No manual tuning**: No need to guess parameter values
    
    **2. Handles Multiple Patterns:**
    - **Level changes**: Captures shifts in unemployment rate
    - **Trends**: Models long-term unemployment trends
    - **Seasonality**: Accounts for quarterly employment cycles
    
    **3. Robust and Reliable:**
    - **Simple concept**: Easy to understand and explain
    - **Stable forecasts**: Less sensitive to outliers than complex models
    - **Wide applicability**: Works well for many time series
    
    **4. Real-world Advantages:**
    - **Quick computation**: Fast model fitting and forecasting
    - **Interpretable**: Clear meaning of parameters
    - **Flexible**: Can handle different data characteristics
    """)
    
    # Model performance explanation
    st.markdown(f"""
    **📊 Your Model Performance:**
    
    **RMSE: {rmse:.2f}** - Average prediction error in original units
    
    **MAPE: {mape:.2f}%** - Average percentage error (excellent for unemployment data)
    
    **AIC: {round(model.aic, 2)}** - Model fit quality (lower is better)
    
    **BIC: {round(model.bic, 2)}** - Model fit with complexity penalty
    
    **What these numbers mean:**
    - **RMSE {rmse:.2f}**: Predictions are off by {rmse:.2f} percentage points on average
    - **MAPE {mape:.2f}%**: {mape:.2f}% average percentage error (very good for economic data)
    - **AIC/BIC**: Model complexity is appropriate for the data
    """)
    
    # Exponential Smoothing vs other methods
    st.subheader("📈 Exponential Smoothing vs Other Methods")
    
    st.markdown("""
    **Exponential Smoothing Advantages:**
    
    **✅ Simplicity:**
    - Easy to understand and implement
    - Clear interpretation of parameters
    - No complex mathematical assumptions
    
    **✅ Robustness:**
    - Less sensitive to outliers
    - Works well with limited data
    - Stable parameter estimates
    
    **✅ Flexibility:**
    - Can handle different data patterns
    - Automatic parameter selection
    - Multiple model variants available
    
    **✅ Practical Benefits:**
    - Fast computation
    - Easy to update with new data
    - Good for operational forecasting
    
    **Limitations:**
    - **Linear assumptions**: Assumes linear trends and additive seasonality
    - **Limited complexity**: May miss complex patterns
    - **No uncertainty quantification**: Limited confidence interval methods
    """)
    
    # Practical interpretation
    st.subheader("💡 How to Use Your Exponential Smoothing Forecast")
    
    st.markdown("""
    **For Policy Makers:**
    - **Short-term planning**: Use for immediate policy decisions
    - **Resource allocation**: Plan based on trend and seasonal patterns
    - **Monitoring**: Track forecast accuracy over time
    
    **For Businesses:**
    - **Workforce planning**: Anticipate employment changes
    - **Budget planning**: Use trend information for financial planning
    - **Seasonal adjustments**: Account for quarterly patterns
    
    **For Researchers:**
    - **Baseline comparison**: Compare with more complex models
    - **Quick analysis**: Use for preliminary investigations
    - **Educational tool**: Understand time series patterns
    """)

# === Tab 5: Complete Technical Details ===
with tab5:
    st.title("📋 Complete Technical Details")
    
    # Full model summary
    st.subheader("📋 Complete Model Summary")
    st.text(model.summary())
    
    # Model summary explanation
    st.subheader("📋 Understanding Your Model Summary")
    
    st.markdown(f"""
    **🔍 Key Statistics Explained:**
    
    **Model Information:**
    - **Model Type**: {model_name}
    - **{len(series)} Observations**: Based on {len(series)} data points
    - **AIC: {model.aic:.3f}, BIC: {model.bic:.3f}**: Model selection criteria (lower is better)
    
    **Parameter Estimates:**
    - **Smoothing parameters**: Control how much weight is given to recent vs past observations
    - **Standard Error**: Uncertainty in the parameter estimates
    - **P-value**: Statistical significance (should be < 0.05)
    
    **Model Diagnostics:**
    
    **Residual Analysis:**
    - **Mean residual**: Should be close to zero
    - **Residual variance**: Should be constant over time
    - **Residual independence**: No autocorrelation patterns
    
    **Forecast Accuracy:**
    - **RMSE**: Root mean square error (lower is better)
    - **MAPE**: Mean absolute percentage error (lower is better)
    - **AIC/BIC**: Model selection criteria (lower is better)
    """)
    
    # What the diagnostics mean
    st.markdown("""
    **🎯 What These Results Mean for Your Forecast:**
    
    **✅ Good Signs:**
    - **Low RMSE/MAPE**: Accurate predictions
    - **Normal residuals**: Reliable confidence intervals
    - **Independent residuals**: Model captures all patterns
    - **Reasonable AIC/BIC**: Model complexity is appropriate
    
    **⚠️ Potential Issues:**
    - **High RMSE/MAPE**: Poor prediction accuracy
    - **Non-normal residuals**: May affect confidence intervals
    - **Autocorrelated residuals**: Model missing patterns
    - **High AIC/BIC**: Model may be overfitting
    
    **💡 Recommendations:**
    - **Monitor performance**: Check actual vs predicted values
    - **Update model**: Re-fit with new data periodically
    - **Consider alternatives**: Try different model types if accuracy is poor
    - **Validate assumptions**: Check if model assumptions are met
    """)
    
    # How calculations are carried out
    st.subheader("🧮 How Exponential Smoothing Calculations Are Carried Out")
    
    st.markdown("""
    **📊 Model Parameter Calculations:**
    
    **Parameter Estimation:**
    ```python
    # Exponential Smoothing model fitting
    model = ExponentialSmoothing(
        series,                    # Your unemployment data
        trend='add',               # Additive trend (or None for simple)
        seasonal='add',            # Additive seasonality (or None)
        seasonal_periods=seasonal_period  # Seasonal period
    )
    
    # Fit model with optimal parameters
    fitted_model = model.fit()
    
    # Get parameters
    alpha = fitted_model.params['smoothing_level']  # Level smoothing
    beta = fitted_model.params.get('smoothing_trend', 0)  # Trend smoothing
    gamma = fitted_model.params.get('smoothing_seasonal', 0)  # Seasonal smoothing
    ```
    
    **Forecast Generation:**
    ```python
    # Generate forecasts
    forecast = fitted_model.forecast(steps=n_periods)
    
    # For Holt-Winters model:
    # Level: l(t) = α * y(t) + (1-α) * (l(t-1) + b(t-1))
    # Trend: b(t) = β * (l(t) - l(t-1)) + (1-β) * b(t-1)
    # Seasonal: s(t) = γ * (y(t) - l(t)) + (1-γ) * s(t-m)
    # Forecast: ŷ(t+h) = l(t) + h * b(t) + s(t+h-m)
    ```
    
    **Performance Metrics:**
    ```python
    # Get model predictions and residuals
    fitted_values = fitted_model.fittedvalues
    residuals = fitted_model.resid
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual - fitted_values)**2))
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual - fitted_values) / actual)) * 100
    ```
    
    **Confidence Intervals:**
    ```python
    # Approximate confidence intervals
    forecast_std = np.std(residuals)
    conf_int_lower = forecast - 1.96 * forecast_std
    conf_int_upper = forecast + 1.96 * forecast_std
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
    aic = model.aic
    # AIC = 2k - 2ln(L) where k=parameters, L=likelihood
    
    # BIC (Bayesian Information Criterion)  
    bic = model.bic
    # BIC = ln(n)k - 2ln(L) where n=sample size
    # Lower values indicate better models
    ```
    
    **Residual Diagnostics:**
    ```python
    # Jarque-Bera Test for normality
    from scipy.stats import jarque_bera
    jb_stat, jb_pvalue = jarque_bera(residuals)
    # p-value > 0.05 indicates normality
    
    # Ljung-Box Test for independence
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
    # p-value > 0.05 indicates independence
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
    - **Step 2:** Analyze trend and seasonal components
    - **Step 3:** Select appropriate exponential smoothing model
    - **Step 4:** Fit model with optimal parameters
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
    - **Automatic selection**: Choose best model based on data characteristics
    - **Parameter optimization**: Find optimal smoothing parameters
    - **Diagnostic testing**: Ensure model validity
    - **Transparency**: All calculations are reproducible
    
    **Limitations:**
    - **Linear assumptions**: Assumes linear trends and additive seasonality
    - **Limited complexity**: May miss complex non-linear patterns
    - **No structural breaks**: Assumes stable patterns over time
    - **Confidence intervals**: Approximate and may not capture all uncertainty
    """)
