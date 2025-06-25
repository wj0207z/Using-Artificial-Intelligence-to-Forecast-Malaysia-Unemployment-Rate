import streamlit as st

pages = {
    "Home": [
        st.Page("MalaysiaAnnualUnemploymentRate.py", title="Malaysia Annual Unemployment Rate"),
        st.Page("MalaysiaQuater.py", title="Manage your account"),
    ],
    "About": [
        st.Page("Contact.py", title="Contact")
    ]
}

pg = st.navigation(pages)
pg.run()
