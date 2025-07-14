import streamlit as st

pages = {
    "Home": [
        st.Page("MalaysiaAnnualUnemploymentRate.py", title="Malaysia Annual Unemployment Rate"),
        st.Page("MalaysiaQuater.py", title="Malaysia Quater Unemployment Rate"),
    ],
    
    "Time Series Models": [
        st.Page("ARIMA.py", title="ARIMA"),
        st.Page("SARIMA.py", title="SARIMA"),
        st.Page("ExponentialSmoothing.py", title="Exponential Smoothing"),
    ],

    "Deep Learning Models": [
        st.Page("LSTM.py", title="Long Short-Term Memory Networks"),
        st.Page("GRU.py", title="Gated Recurrent Unit Networks "),
        st.Page("CNN.py", title="Convolutional Neural Networks  "),
        st.Page("RNN.py", title="Recurrent Neural Networks ")


    ],

    "Model Comparison": [
        st.Page("ModelComparison.py", title="Model Performance Comparison"),
    ],

    "About": [
        st.Page("ReadMe.py", title="📚 Read Me - Methodology & Models"),
        st.Page("Contact.py", title="Contact")
    ]
}

pg = st.navigation(pages)
pg.run()
