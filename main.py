#!/usr/bin/env python3

import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
from fastbook import *
from nbdev.export import update_baseurl
from numpy.lib.function_base import _update_dim_sizes

path = Path('./')
df = pd.read_csv(path / 'train.csv', index_col='image')

types = {
    1: 'Cargo',
    2: 'Military',
    3: 'Carrier',
    4: 'Cruise',
    5: 'Tankers'
}

def get_vessels(path):
    return L([Path(path /  i) for i in df.index])

def get_image_cat(p):
    return df.loc[p.name]['category']

learn_inf = load_learner('export.pkl')

st.title("Vessel classifier")
st.markdown("""A [fast.ai](https://www.fast.ai/) vessel classifier based on the [Game of Deep Learning: Ship datasets](https://www.kaggle.com/arpitjain007/game-of-deep-learning-ship-datasets) dataset.""")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    img = PILImage.create(uploaded_file)

    st.image(img, caption=uploaded_file.name)
    pred,pred_idx,probs = learn_inf.predict(img)

    st.text( f'Prediction: {types[int(pred)]}; Probability: {probs[pred_idx]:.04f}')
