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

    "Machine Learning Models": [
        st.Page("RandonForest.py", title="Random Forest"),
    ],

    "About": [
        st.Page("Contact.py", title="Contact")
    ]
}

pg = st.navigation(pages)
pg.run()
