import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import create_model as cm
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./Background/Background.jpg')    

original_title = '<p style="font-family:Franklin Gothic Medium;text-align:left;color:Yellow; font-size: 70px;">Chest X-ray Captioning</p>'
st.markdown(original_title, unsafe_allow_html=True)
Author_Name = '<p style="font-family:Franklin Gothic Medium;text-align:left;color:Magenta;font-weight: bold; font-size: 20px;">by Santhosh Kurnapally</p>'
st.markdown(Author_Name,unsafe_allow_html=True)
st.markdown("[<medium>Github</medium>](https://github.com/skurnapally/Medical_Image_Captioning_on_Chest_X-Rays)",
unsafe_allow_html=True)
text = '<p style="font-family:Arial;text-align:left;color:White; font-size: 20px;">\nThis app will generate Findings from the X-ray report.\nYou can upload 2   X-rays that are either front or lateral view of chest of the same individual.</p>'
st.markdown(text,unsafe_allow_html=True)
note = '<p style="font-family:Arial;text-align:left;color:White; font-size: 20px;">The 2nd X-ray is optional.</p>'
st.markdown(note,unsafe_allow_html=True)
col1,col2 = st.beta_columns(2)
predict_button = col1.button('Predict on uploaded files')
test_data = col2.button('Predict on sample data')
image_1 = None
image_2 = None
if predict_button:
	firstxray = '<p style="font-family:Arial;text-align:center;color:red; font-size: 20px;">Choose First X-Ray</p>'
	st.markdown(firstxray,unsafe_allow_html=True)
	image_1 = st.file_uploader(label = "",type=['png','jpg','jpeg'])
	image_2 = None
	if image_1:
		secondxray = '<p style="font-family:Arial;text-align:center;color:red; font-size: 20px;">Choose Second X-Ray (Optional)</p>'
		st.markdown(secondxray,unsafe_allow_html=True)
		image_2 = st.file_uploader(label = " ",type=['png','jpg','jpeg'])
	predict_uploaded = st.button('Predict')
