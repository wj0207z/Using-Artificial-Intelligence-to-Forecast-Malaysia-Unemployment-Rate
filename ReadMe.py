import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Read Me - Methodology & Models",
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 Read Me - Methodology & Models")
st.markdown("---")

# Table of Contents
st.sidebar.title("📋 Table of Contents")
st.sidebar.markdown("""
- [📊 Data Overview](#data-overview)
- [🔬 Methodology](#methodology)
- [📈 Data Preprocessing](#data-preprocessing)
- [🤖 Model Training Process](#model-training-process)
- [📊 Statistical Models](#statistical-models)
- [🧠 Deep Learning Models](#deep-learning-models)
- [📊 Model Comparison](#model-comparison)
- [🔍 Evaluation Metrics](#evaluation-metrics)
- [💡 Best Practices](#best-practices)
- [📚 References](#references)
""")

# Data Overview Section
st.header("📊 Data Overview")
st.markdown("""
### Dataset Description
This application uses **Malaysia Quarterly Labour Force Survey Data** containing comprehensive employment statistics from 2010 to 2024.

### Available Metrics
The dataset includes the following key indicators:
- **Labour Force (lf)**: Total number of people available for work (in thousands)
- **Employed (lf_employed)**: Number of people currently employed (in thousands)
- **Unemployed (lf_unemployed)**: Number of people actively seeking employment (in thousands)
- **Outside Labour Force (lf_outside)**: People not in the labour force (in thousands)
- **Participation Rate (%) (p_rate)**: Percentage of working-age population in labour force
- **Employment to Population Ratio (%) (ep_ratio)**: Percentage of working-age population employed
- **Unemployment Rate (%) (u_rate)**: Percentage of labour force that is unemployed

### Data Characteristics
- **Frequency**: Quarterly (4 observations per year)
- **Time Span**: 2010-2024 (approximately 60 quarters)
- **Data Points**: 60 quarterly observations
- **Seasonality**: Strong quarterly patterns due to economic cycles
- **Trend**: Long-term economic trends and structural changes
- **Units**: Labour force numbers in thousands, rates in percentages
""")

# Methodology Section
st.header("🔬 Methodology")
st.markdown("""
### Forecasting Approach
This application implements a **comprehensive forecasting methodology** that combines statistical and machine learning approaches to provide robust unemployment rate predictions.

### Key Principles
1. **Multi-Model Ensemble**: Using multiple models to capture different aspects of the data
2. **Train-Test Split**: Consistent 80-20 split for model validation
3. **Cross-Validation**: Ensuring model reliability through proper validation
4. **Residual Analysis**: Comprehensive diagnostic testing for model adequacy
5. **Explainability**: Providing insights into model decisions and feature importance

### Methodology Flow
```
Data Collection → Preprocessing → Model Selection → Training → Validation → Forecasting → Evaluation
```
""")

# Data Preprocessing Section
st.header("📈 Data Preprocessing")
st.markdown("""
### Step 1: Data Loading and Cleaning
```python
# Load dataset
df = pd.read_csv("MalaysiaQuarterlyLabourForce.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# Handle missing values
series = df[selected_metric].dropna()
```

### Step 2: Data Quality Assessment
- **Missing Values**: Identified and removed any incomplete observations
- **Outliers**: Detected and handled extreme values
- **Consistency**: Ensured data format consistency across time periods
- **Stationarity**: Tested for stationarity using Augmented Dickey-Fuller test

### Step 3: Feature Engineering
- **Lag Features**: Created lagged variables for time series models
- **Seasonal Decomposition**: Extracted trend, seasonal, and residual components
- **Scaling**: Applied Min-Max scaling for neural network models
- **Window Creation**: Generated sliding windows for deep learning models

### Step 4: Train-Test Split
```python
# Consistent 80-20 split across all models
test_pct = 20  # 20% for testing
test_size = int(len(series) * test_pct / 100)
train_size = len(series) - test_size

train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]
```
""")

# Model Training Process Section
st.header("🤖 Model Training Process")
st.markdown("""
### Training Strategy
Each model follows a systematic training process designed to ensure optimal performance and prevent overfitting.

### Common Training Steps
1. **Parameter Initialization**: Set initial hyperparameters based on data characteristics
2. **Model Fitting**: Train the model on the training dataset
3. **Validation**: Use validation set to monitor performance
4. **Hyperparameter Tuning**: Optimize parameters based on validation performance
5. **Final Training**: Retrain with optimal parameters on full training set
6. **Testing**: Evaluate on held-out test set

### Model-Specific Training Processes

#### Statistical Models (ARIMA, SARIMA, Exponential Smoothing)
```python
# Auto ARIMA - Automatic parameter selection
model = pm.auto_arima(
    train_series,
    seasonal=force_seasonal,
    m=4,  # Quarterly seasonality
    max_d=1,  # Maximum differencing
    stepwise=True,
    suppress_warnings=True
)

# Exponential Smoothing - Automatic method selection
model = ExponentialSmoothing(
    series, 
    trend='add', 
    seasonal='add', 
    seasonal_periods=4
)
fitted_model = model.fit()
```

#### Deep Learning Models (LSTM, GRU, RNN, CNN)
```python
# Data preparation for neural networks
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

# Create lagged sequences
def create_lagged_sequences(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Model architecture
model = Sequential([
    LSTM(n_units, input_shape=(n_lags, 1)),
    Dropout(dropout),
    Dense(1)
])

# Training with early stopping
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=val_split,
    callbacks=[es],
    verbose=0
)
```

### Validation Strategy
- **Time Series Cross-Validation**: Forward chaining validation
- **Early Stopping**: Prevent overfitting in neural networks
- **Residual Analysis**: Comprehensive diagnostic testing
- **Out-of-Sample Testing**: Evaluate on unseen data
""")

# Statistical Models Section
st.header("📊 Statistical Models")

## ARIMA Model
st.subheader("📈 ARIMA (AutoRegressive Integrated Moving Average)")
st.markdown("""
### Model Description
ARIMA models capture temporal dependencies through autoregressive (AR) and moving average (MA) components, with differencing (I) to achieve stationarity.

### Mathematical Formulation
**ARIMA(p,d,q) Model:**
```
(1 - φ₁B - φ₂B² - ... - φₚBᵖ)(1 - B)ᵈYₜ = (1 + θ₁B + θ₂B² + ... + θₚBᵖ)εₜ
```

Where:
- **p**: AR order (number of past values)
- **d**: Differencing order (trend removal)
- **q**: MA order (number of past errors)
- **φ**: AR coefficients
- **θ**: MA coefficients
- **B**: Backshift operator
- **εₜ**: White noise error

### Training Process
1. **Stationarity Test**: Augmented Dickey-Fuller test
2. **Differencing**: Apply if non-stationary
3. **Parameter Selection**: Auto ARIMA with AIC/BIC optimization
4. **Model Fitting**: Maximum likelihood estimation
5. **Diagnostic Testing**: Residual analysis

### Key Features
- ✅ Automatic parameter selection
- ✅ Handles trends and seasonality
- ✅ Interpretable parameters
- ✅ Confidence intervals
- ⚠️ Assumes linear relationships
- ⚠️ Requires stationarity
""")

## SARIMA Model
st.subheader("🌊 SARIMA (Seasonal ARIMA)")
st.markdown("""
### Model Description
SARIMA extends ARIMA to handle seasonal patterns by including seasonal autoregressive, differencing, and moving average components.

### Mathematical Formulation
**SARIMA(p,d,q)(P,D,Q,m) Model:**
```
Φ(Bᵐ)(1 - φ₁B - φ₂B² - ... - φₚBᵖ)(1 - B)ᵈ(1 - Bᵐ)ᴰYₜ = Θ(Bᵐ)(1 + θ₁B + θ₂B² + ... + θₚBᵖ)εₜ
```

Where:
- **P**: Seasonal AR order
- **D**: Seasonal differencing order
- **Q**: Seasonal MA order
- **m**: Seasonal period (4 for quarterly data)
- **Φ**: Seasonal AR coefficients
- **Θ**: Seasonal MA coefficients

### Training Process
1. **Seasonality Detection**: Seasonal decomposition
2. **Seasonal Differencing**: Remove seasonal trends
3. **Parameter Selection**: Grid search for optimal parameters
4. **Model Fitting**: Maximum likelihood estimation
5. **Validation**: Out-of-sample testing

### Key Features
- ✅ Captures seasonal patterns
- ✅ Handles multiple time scales
- ✅ Automatic seasonal detection
- ✅ Robust for seasonal data
- ⚠️ More complex than ARIMA
- ⚠️ Requires sufficient seasonal data
""")

## Exponential Smoothing
st.subheader("📈 Exponential Smoothing")
st.markdown("""
### Model Description
Exponential smoothing uses weighted averages of past observations, with weights decreasing exponentially for older observations.

### Mathematical Formulation
**Holt-Winters Model:**
```
Level: lₜ = αyₜ + (1-α)(lₜ₋₁ + bₜ₋₁)
Trend: bₜ = β(lₜ - lₜ₋₁) + (1-β)bₜ₋₁
Seasonal: sₜ = γ(yₜ - lₜ) + (1-γ)sₜ₋ₘ
Forecast: ŷₜ₊ₕ = lₜ + h·bₜ + sₜ₊ₕ₋ₘ
```

Where:
- **α**: Level smoothing parameter
- **β**: Trend smoothing parameter
- **γ**: Seasonal smoothing parameter
- **m**: Seasonal period

### Training Process
1. **Method Selection**: Auto-select best method (Simple/Holt/Holt-Winters)
2. **Parameter Optimization**: Minimize AIC/BIC
3. **Model Fitting**: Maximum likelihood estimation
4. **Validation**: Cross-validation

### Key Features
- ✅ Simple and interpretable
- ✅ Automatic method selection
- ✅ Handles trends and seasonality
- ✅ Robust to outliers
- ⚠️ Assumes additive seasonality
- ⚠️ Limited to linear trends
""")

# Deep Learning Models Section
st.header("🧠 Deep Learning Models")

## LSTM Model
st.subheader("🔮 LSTM (Long Short-Term Memory)")
st.markdown("""
### Model Description
LSTM networks are designed to capture long-term dependencies in sequential data through specialized memory cells and gating mechanisms.

### Architecture Components
```
Input Gate: iₜ = σ(Wᵢ[hₜ₋₁, xₜ] + bᵢ)
Forget Gate: fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)
Output Gate: oₜ = σ(Wₒ[hₜ₋₁, xₜ] + bₒ)
Cell State: C̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc)
Memory Update: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
Hidden State: hₜ = oₜ ⊙ tanh(Cₜ)
```

### Training Process
1. **Data Preparation**: Create lagged sequences
2. **Scaling**: Min-Max normalization
3. **Architecture**: LSTM + Dropout + Dense layers
4. **Training**: Adam optimizer with early stopping
5. **Validation**: Monitor validation loss

### Key Features
- ✅ Captures long-term dependencies
- ✅ Handles vanishing gradient problem
- ✅ Non-linear pattern recognition
- ✅ Robust to noise
- ⚠️ Requires more data
- ⚠️ Computationally intensive
""")

## GRU Model
st.subheader("🤖 GRU (Gated Recurrent Unit)")
st.markdown("""
### Model Description
GRU is a simplified version of LSTM with fewer parameters but similar performance, using update and reset gates.

### Architecture Components
```
Update Gate: zₜ = σ(Wz[hₜ₋₁, xₜ] + bz)
Reset Gate: rₜ = σ(Wr[hₜ₋₁, xₜ] + br)
Candidate: h̃ₜ = tanh(Wh[rₜ ⊙ hₜ₋₁, xₜ] + bh)
Hidden State: hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

### Training Process
1. **Data Preparation**: Same as LSTM
2. **Architecture**: GRU + Dropout + Dense layers
3. **Training**: Adam optimizer with early stopping
4. **Validation**: Monitor validation loss

### Key Features
- ✅ Parameter efficient
- ✅ Faster training than LSTM
- ✅ Similar performance to LSTM
- ✅ Simpler architecture
- ⚠️ May struggle with very long sequences
- ⚠️ Less interpretable than LSTM
""")

## RNN Model
st.subheader("🔄 RNN (Recurrent Neural Network)")
st.markdown("""
### Model Description
SimpleRNN is the basic recurrent neural network that processes sequential data through recurrent connections.

### Architecture Components
```
Hidden State: hₜ = tanh(Whhₜ₋₁ + Wxₜ + b)
Output: yₜ = Wyhₜ + by
```

### Training Process
1. **Data Preparation**: Create lagged sequences
2. **Architecture**: SimpleRNN + Dropout + Dense layers
3. **Training**: Adam optimizer
4. **Validation**: Monitor training/validation loss

### Key Features
- ✅ Simple architecture
- ✅ Fast training
- ✅ Good for short sequences
- ✅ Easy to understand
- ⚠️ Vanishing gradient problem
- ⚠️ Limited long-term memory
""")

## CNN Model
st.subheader("🖼️ CNN (Convolutional Neural Network)")
st.markdown("""
### Model Description
CNN uses convolutional layers to capture local temporal patterns in time series data.

### Architecture Components
```
Convolution: (f * x)[i] = Σⱼ f[j]x[i+j]
Pooling: Reduces dimensionality
Flatten: Converts to 1D for dense layers
Dense: Final prediction layer
```

### Training Process
1. **Data Preparation**: Create lagged sequences
2. **Architecture**: Conv1D + Dropout + Flatten + Dense
3. **Training**: Adam optimizer
4. **Validation**: Monitor training/validation loss

### Key Features
- ✅ Captures local patterns
- ✅ Translation invariant
- ✅ Parameter sharing
- ✅ Good for pattern recognition
- ⚠️ May miss long-term dependencies
- ⚠️ Requires appropriate kernel size
""")

# Model Comparison Section
st.header("📊 Model Comparison")
st.markdown("""
### Model Selection Guidelines

| Model Type | Best For | Strengths | Limitations |
|------------|----------|-----------|-------------|
| **ARIMA** | Linear trends, short-term forecasting | Interpretable, robust, automatic selection | Linear assumptions, limited complexity |
| **SARIMA** | Seasonal data, quarterly patterns | Captures seasonality, handles trends | More complex, requires seasonal data |
| **Exponential Smoothing** | Simple patterns, quick forecasts | Simple, interpretable, automatic selection | Limited to linear trends |
| **LSTM** | Complex patterns, long-term dependencies | Non-linear, long memory, robust | Requires more data, computationally intensive |
| **GRU** | Efficiency-focused applications | Parameter efficient, fast training | May struggle with very long sequences |
| **RNN** | Simple temporal patterns | Simple, fast, interpretable | Vanishing gradient, limited memory |
| **CNN** | Local pattern recognition | Good for local patterns, parameter sharing | May miss long-term dependencies |

### Performance Comparison Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **R²**: Coefficient of determination (higher is better)
- **AIC/BIC**: Model selection criteria (lower is better)

### Model Selection Strategy
1. **Start Simple**: Begin with ARIMA/Exponential Smoothing
2. **Check Seasonality**: Use SARIMA if strong seasonality detected
3. **Consider Complexity**: Use LSTM/GRU for complex patterns
4. **Validate Performance**: Compare models using multiple metrics
5. **Ensemble Approach**: Combine multiple models for robust forecasts
""")

# Evaluation Metrics Section
st.header("🔍 Evaluation Metrics")
st.markdown("""
### Accuracy Metrics

#### RMSE (Root Mean Square Error)
```python
RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
```
- **Interpretation**: Average prediction error in original units
- **Use Case**: Primary accuracy metric
- **Range**: 0 to ∞ (lower is better)

#### MAE (Mean Absolute Error)
```python
MAE = Σ|yᵢ - ŷᵢ| / n
```
- **Interpretation**: Average absolute prediction error
- **Use Case**: Robust to outliers
- **Range**: 0 to ∞ (lower is better)

#### MAPE (Mean Absolute Percentage Error)
```python
MAPE = (Σ|yᵢ - ŷᵢ| / |yᵢ|) × 100 / n
```
- **Interpretation**: Average percentage error
- **Use Case**: Scale-independent comparison
- **Range**: 0% to ∞% (lower is better)

#### R² (Coefficient of Determination)
```python
R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
```
- **Interpretation**: Proportion of variance explained
- **Use Case**: Model fit assessment
- **Range**: 0 to 1 (higher is better)

### Model Selection Criteria

#### AIC (Akaike Information Criterion)
```python
AIC = 2k - 2ln(L)
```
- **Interpretation**: Balance between fit and complexity
- **Use Case**: Model comparison
- **Range**: -∞ to ∞ (lower is better)

#### BIC (Bayesian Information Criterion)
```python
BIC = ln(n)k - 2ln(L)
```
- **Interpretation**: Penalizes complexity more than AIC
- **Use Case**: Model comparison
- **Range**: -∞ to ∞ (lower is better)

### Diagnostic Tests

#### Residual Analysis
- **Normality Test**: Jarque-Bera test
- **Independence Test**: Ljung-Box test
- **Heteroskedasticity Test**: Breusch-Pagan test
- **Stationarity Test**: Augmented Dickey-Fuller test
""")

# Best Practices Section
st.header("💡 Best Practices")
st.markdown("""
### Data Preparation
1. **Handle Missing Values**: Remove or impute missing data appropriately
2. **Check for Outliers**: Identify and handle extreme values
3. **Ensure Consistency**: Maintain consistent data format and frequency
4. **Validate Data Quality**: Check for data integrity issues

### Model Selection
1. **Start Simple**: Begin with simpler models (ARIMA, Exponential Smoothing)
2. **Consider Data Characteristics**: Choose models based on data patterns
3. **Validate Assumptions**: Check model assumptions before proceeding
4. **Compare Multiple Models**: Use ensemble approaches when possible

### Training Process
1. **Proper Train-Test Split**: Use time-based splitting for time series
2. **Cross-Validation**: Implement appropriate validation strategies
3. **Hyperparameter Tuning**: Optimize parameters systematically
4. **Early Stopping**: Prevent overfitting in neural networks

### Evaluation
1. **Multiple Metrics**: Use various metrics for comprehensive evaluation
2. **Out-of-Sample Testing**: Always test on unseen data
3. **Residual Analysis**: Conduct thorough diagnostic testing
4. **Confidence Intervals**: Provide uncertainty quantification

### Deployment
1. **Model Monitoring**: Track model performance over time
2. **Regular Updates**: Retrain models with new data
3. **Documentation**: Maintain comprehensive model documentation
4. **Version Control**: Track model versions and changes
""")

# References Section
st.header("📚 References")
st.markdown("""
### Academic Papers
1. **ARIMA/SARIMA**: Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control.
2. **Exponential Smoothing**: Holt, C. C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages.
3. **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
4. **GRU**: Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation.
5. **CNN for Time Series**: LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning.

### Software Libraries
- **Streamlit**: For web application framework
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Scikit-learn**: For machine learning utilities
- **TensorFlow/Keras**: For deep learning models
- **Statsmodels**: For statistical models
- **PMDARIMA**: For automatic ARIMA selection
- **Plotly**: For interactive visualizations

### Data Sources
- **Malaysia Department of Statistics**: Quarterly Labour Force Survey Data
- **World Bank**: Economic indicators and employment statistics
- **International Labour Organization**: Global employment trends

### Additional Resources
- **Time Series Analysis**: Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.
- **Deep Learning**: Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning.
- **Statistical Learning**: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 Comprehensive Methodology Documentation | Malaysia Unemployment Forecasting</p>
    <p>Last updated: """ + datetime.now().strftime("%B %d, %Y") + """</p>
</div>
""", unsafe_allow_html=True) 