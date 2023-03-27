import streamlit as st
from PIL import Image

def home():
    image = Image.open('./image/home.jpg')
    st.image(image, use_column_width=True)
    st.markdown("## Application for Molecular Property Prediction")