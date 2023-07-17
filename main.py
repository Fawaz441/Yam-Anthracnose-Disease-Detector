"""
Simple StreamLit app fro plant classification

>> streamlit run main.py
"""
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow import keras
model = keras.saving.load_model('anthracnose-model.keras')

def process_image(
    img_path: str
):
    if not img_path:
        return

    img = Image.open(img_path)
    st.image(img)
    test_image = keras.utils.load_img(img_path,target_size=(200,200))
    test_image = keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)
    if(result>=0.5):
        result_perc = result[0][0] * 100
        text = "This leaf probably has the anthracnose disease, probability is {0:.2f}%".format(result_perc)
    else:
        text = "This leaf does not have the anthracnose disease"
    st.write(text)


st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header(" YAM Anthracnose Disease Detection")
img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg'])

# run the app
process_image(img_file)