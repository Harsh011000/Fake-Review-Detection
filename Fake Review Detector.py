import streamlit as st
import requests

st.title("📝 Fake Review Detection")
st.write("Enter a review below to check whether it's Fake or Real.")

user_input = st.text_area("Enter a review:")

if st.button("Analyze Review"):
    if user_input:
        response = requests.post("https://harsh-p-tset.hf.space/predict", json={"review": user_input})
        result = response.json()
        st.subheader(result["prediction"])
        st.write(f"Confidence Score: {result['score']:.4f}")
    else:
        st.warning("Please enter a review.")
