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
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./Background/Background.png')    

original_title = '<p style="font-family:Franklin Gothic Medium;text-align:left;color:Yellow; font-size: 70px;">Chest X-ray Captioning</p>'
st.markdown(original_title, unsafe_allow_html=True)
Author_Name = '<p style="font-family:Franklin Gothic Medium;text-align:left;color:Magenta;font-weight: bold; font-size: 25px;">by Santhosh Kurnapally</p>'
st.markdown(Author_Name,unsafe_allow_html=True)
st.markdown("[<medium>Find Code here (Github)</medium>](https://github.com/skurnapally/Medical_Image_Captioning_on_Chest_X-Rays)",
unsafe_allow_html=True)
st.markdown("[<medium>Find my blog on (Medium)</medium>](https://medium.com/@skurnapally/medical-image-captioning-using-attention-based-encoder-decoder-recurrent-neural-networks-b90f65907f49)",unsafe_allow_html=True)
text = '<p style="font-family:Arial;text-align:left;color:White; font-size: 20px;">\nThis app will generate Findings from the X-ray report.\nYou can upload 2 X-rays that are either front or lateral view of chest of the same individual.</p>'
st.markdown(text,unsafe_allow_html=True)
note = '<p style="font-family:Arial;text-align:left;color:Snow; font-size: 20px;">The 2nd X-ray is optional.</p>'
st.markdown(note,unsafe_allow_html=True)
col1,col2 = st.beta_columns(2)
note1 = '<p style="font-family:Arial;text-align:left;color:Snow; font-size: 20px;">Upload First X-Ray</p>'
col1.markdown(note1,unsafe_allow_html=True)
image_1 = col1.file_uploader("X-ray 1",type=['png','jpg','jpeg'])
image_2 = None
if image_1:
	note2 = '<p style="font-family:Arial;text-align:left;color:Snow; font-size: 20px;">Upload Second X-Ray (Optional)</p>'
	col2.markdown(note2,unsafe_allow_html=True)
	image_2 = col2.file_uploader("X-ray 2 (optional)",type=['png','jpg','jpeg'])
col1,col2 = st.beta_columns(2)
predict_button = col1.button('Predict on uploaded files')
test_data = col2.button('Predict on sample data')

@st.cache
def create_model():
    model_tokenizer = cm.create_model()
    return model_tokenizer


def predict(image_1,image_2,model_tokenizer,predict_button = predict_button):
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
            image_1 = np.array(image_1)/255
            if image_2 is None:
                image_2 = image_1
            else:
                image_2 = Image.open(image_2).convert("RGB") #converting to 3 channels
                image_2 = np.array(image_2)/255
            st.image([image_1,image_2],width=300)
            caption = cm.function1([image_1],[image_2],model_tokenizer)
            findings_text = '<p style="font-family:Arial;text-align:left;color:red; font-size: 20px;">Findings</p>'
            st.markdown(findings_text,unsafe_allow_html=True)
            findings = '<p style="font-family:Arial;text-align:left;color:white; font-size: 20px;">'+caption[0]+'</p>'
            st.markdown(findings,unsafe_allow_html=True)
            timetook = '<p style="font-family:Arial;text-align:left;color:white; font-size: 20px;">'+"Time Took for prediction: "+str(round(time.process_time()-start,2)) + ' seconds</p>'
            st.markdown(timetook,unsafe_allow_html=True)
            del image_1,image_2
        else:
            st.markdown("## Upload an Image")

def predict_sample(model_tokenizer,folder = './test_images'):
    no_files = len(os.listdir(folder))
    file = np.random.randint(1,no_files)
    file_path = os.path.join(folder,str(file))
    if len(os.listdir(file_path))==2:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = os.path.join(file_path,os.listdir(file_path)[1])
        print(file_path)
    else:
        image_1 = os.path.join(file_path,os.listdir(file_path)[0])
        image_2 = image_1
    predict(image_1,image_2,model_tokenizer,True)
    

model_tokenizer = create_model()



if test_data:
    predict_sample(model_tokenizer)
else:
    predict(image_1,image_2,model_tokenizer)





