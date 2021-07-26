# Dependencies and Libraries
from fastai.vision.all import *
from fastai.vision.widgets import *
from io import BytesIO, StringIO

import os
import sys
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title=None,
    page_icon=None, 
    layout='wide', 
    initial_sidebar_state='auto')

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

# Load model
path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)

class FileUpload(object):
    def __init__(self):
        self.fileTypes = ['png', 'jpg']

    def app():  
        st.write("""
        # Video Conference Glitch Detection
        Task Item #5 Project - Convolutional Neural Network
        """)

        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload image file", type=['png', 'jpg'])
        show_file = st.empty()

        if not file:
            show_file.info("Please upload an image file : {}".format(' '.join(['png', 'jpg'])))
            return
        content = file.getvalue()

        if isinstance(file, BytesIO):
            img = PILImage.create(file)
            pred,pred_idx,probs = learn_inf.predict(img)
            prediction = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
            st.write(prediction)
            show_file.image(file)
            


        else:
            df = pd.read_csv(file)
            st.dataframe(df.head(2))
        
        file.close()

if __name__ == "__main__":
    app = FileUpload()
    FileUpload.app()