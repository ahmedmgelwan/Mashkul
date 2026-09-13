import streamlit as st
from src.config import model_info, PathConfig
import requests

API_URL = PathConfig.api_url

st.set_page_config(page_title="Mashkul - مَشْكُول", page_icon="✒️")
st.title("✒️ Mashkul - مَشْكُول")
text = st.text_area("اكتب النص من غير تشكيل:", height=150, placeholder="ذهب الطالب الى المدرسة")

if st.button("شكّل النص", type="primary"):
    if not text.strip():
        st.warning("الرجاء إدخال نص لتشكيله.")
    else:
        with st.spinner("جاري تشكيل النص..."):
            try:
                response = requests.post(f"{API_URL}/diacritize", json={"text": text}, timeout=30)
                response.raise_for_status()
                result = response.json()["diacritized_text"]
                st.success("تم التشكيل:")
                st.markdown(f"### {result}")
            except requests.exceptions.RequestException as e:
                st.error(f"فيه مشكلة في الاتصال بالـ API: {e}")