import streamlit as st
from streamlit_lottie import st_lottie
import requests

# === Utility function to load Lottie animation ===
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_contact = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_tno6cg2w.json")  # or any other contact animation

# Page settings
st.set_page_config(page_title="Contact", page_icon="📞")
st.title("📞 Contact Me")

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["👤 About Me", "📬 Contact", "🔗 Socials"])

# === Tab 1: About Me ===
with tab1:
    st.subheader("Hi, I'm Wei Jian 👋")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("profile.jpg", width=120)  # Optional profile image
    with col2:
        st.write("""
        - 🎓 Final-year student passionate about data, design & interactivity  
        - 📈 Exploring Malaysian labor force trends through data  
        - 🛠️ Always improving — always building
        """)

    st.divider()
    st.subheader("📍 Location")
    st.components.v1.iframe("https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d15955.155197893058!2d101.6929093!3d3.1390038!2m3!1f0!2f0!3f0", height=300)

# === Tab 2: Contact Info ===
with tab2:
    st.subheader("✉️ Get in Touch")

    st.markdown("""
If you'd like to get in touch, feel free to reach out through any of the platforms below:

- 📘 [Facebook](https://www.facebook.com/chongapps.jainwei/)
- 📸 [Instagram](https://www.instagram.com/_.weijian/)
- 💬 [WhatsApp](https://wa.me/60168302603)

---

**Email:** weijian0207@gmail.com  
**Phone:** +60 16-8302602
    """, unsafe_allow_html=True)

    st.info("Click the links above to connect with me on social media!")

    st.divider()
    st.subheader("📽️ Here's a little animation:")
    if lottie_contact:
        st_lottie(lottie_contact, height=250)
    else:
        st.error("⚠️ Lottie animation failed to load.")

# === Tab 3: Social Media Buttons ===
with tab3:
    st.subheader("📱 Connect with Me")

    st.markdown("""
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/chongapps.jainwei/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/_.weijian/)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://wa.me/60168302603)
    """, unsafe_allow_html=True)

    st.info("Click any badge above to reach out instantly!")
