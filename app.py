import streamlit as st

pages = {
    "Home": [
        st.Page("MalaysiaAnnualUnemploymentRate.py", title="Malaysia Annual Unemployment Rate"),
        st.Page("MalaysiaQuater.py", title="Malaysia Quater Unemployment Rate"),
    ],
    
    "Models": [
        st.Page("ARIMA.py", title="ARIMA")
    ],
    
    "About": [
        st.Page("Contact.py", title="Contact")
    ]
}

pg = st.navigation(pages)
pg.run()
