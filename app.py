import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Bed , Couch va Table larni klassifikatsiya qiluvchi dastur")

file = st.file_uploader("Rasm yuklash" , type=['png' , 'jpeg' , 'gif' , 'svg' , 'jfif'])
# PIL Convert
if file:
    st.image(file)
    img  = PILImage.create(file)
    # model
    model = load_learner("Transport_Model.pkl")
    
    pred , pred_id , probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probabilty: {probs[pred_id]}")

    fig  = px.bar(x = probs , y= model.dls.vocab)   
    st.plotly_chart(fig)