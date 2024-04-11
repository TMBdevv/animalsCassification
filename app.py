import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pandas as pd
import numpy as np
import pydeck as pdk
import pathlib
import webbrowser

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(
    page_title="HK modeli",
    page_icon="paw (3).png",
    layout="wide"
)

names = {
    "Camel": "Tuya 🐫",
    "Deer": "Kiyik 🦌",
    "Elephant": "Fil 🐘",
    "Giraffe": "Jiraf 🦒",
    "Goat": "Echki 🐐",
    "Horse": "Ot 🐎",
    "Kangaroo": "Kenguru 🦘",
    "Koala": "Koala 🐨",
    "Monkey": "Maymun 🙉",
    "Rabbit": "Quyon 🐰",
    "Rhinoceros": "Karkidon 🦏",
    "Sheep": "Qo'y 🐏",
    "Zebra": "Zebra 🦓"
}

st.title("Hayvonlarni klassifikatsiya qilish modeli (HK)")

st.subheader('', divider='rainbow')

st.caption('Salom bu :blue[HK modeli] :sunglasses: Model siz joylagan rasmlaringizni klassifikatsiya qilish uchun mo‘ljallangan bo‘lib quyidagi 13 ta hayvoni taniy oladi :sparkles: :underline[Tuya, Jiraf, Karkidon, Echki, Ot, Kenguru, Koala, Quyon, Qo‘y, Zebra, Maymun, Kiyik, Fil]:blue ')
st.markdown(":red[Me: ] [Telegram](https://t.me/tojiddinov_muhammad) :red[and] [Instagram](https://instagram.com/tojiddinov_muhammad__)")

filec_option = st.checkbox("Kamerani ishlatish", value=False)
if filec_option:
    filec = st.camera_input("Suratga oling")

file = st.file_uploader("Rasm yuklash", type=['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file or (filec_option and filec):
    if file:
        img = PILImage.create(file)
        st.image(file)

    elif filec_option:
        img = PILImage.create(filec)
        
    model = load_learner("animals_model.pkl")

    pred, pred_id, probs = model.predict(img)

    if probs[pred_id]*100 > 70:
        st.success(f"Bashorat: {names.get(pred)}")
        st.info(f"Ehtimoliligi: {probs[pred_id]*100:.0f}%-ni tashkil etadi 📈")
        fig = px.bar(x=names, y=probs*100)
        st.plotly_chart(fig)
    else:
        st.info(f"🆙 Rasmdagi jonivorni tasniflay olmadim. Bu noyob tur yoki hozirgi mashg'ulot ma'lumotlarimdan tashqari biror narsa bo'lishi mumkin 😔")
        error_message = st.text_area("Xato xabarini yozing")

        if st.button("Send"):
            st.error("🚨 Xatolik! Iltimos, rasmingizni tekshirib qayta yuklang yoki kamerani ishlatib suratga oling.")
            # Send error message and image (if available) to the specified email address
            email = "tajiddinovmuhammaddiyor8@gmail.com"
            st.write(f"Xatolikni aytganingiz uchun rahmat. Xato xabari {email} ga jo'natildi.")
            if error_message:
                st.write("")
