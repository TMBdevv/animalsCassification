import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import webbrowser
import pathlib
import platform
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

plt = platform.system()
if plt == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath

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

filec_option = st.checkbox("Kamerani ishlatish", value=False)
if filec_option:
    filec = st.camera_input("Suratga oling")

file = st.file_uploader("Rasm yuklash", type=['png', 'jpeg', 'gif', 'svg', 'jpg'])

button_clicked = st.sidebar.button("Telegram", help="Mening Telegram hisobim")
if button_clicked:
    webbrowser.open_new_tab("https://t.me/tojiddinov_muhammad")  

button_clicked = st.sidebar.button("Instagram", help="Mening Instagram hisobim")
if button_clicked:
    webbrowser.open_new_tab("https://instagram.com/tojiddinov_muhammad__")  

if file or (filec_option and filec):
    if file:
        img = PILImage.create(file)
        st.image(file)

    elif filec_option:
        img = PILImage.create(filec)
        
    model = load_learner("animals_model.pkl")

    try:
        pred, pred_id, probs = model.predict(img)

        if probs[pred_id]*100 > 70:
            st.success(f"Bashorat: {names.get(pred)}")
            st.info(f"Ehtimoliligi: {probs[pred_id]*100:.0f}%-ni tashkil etadi 📈")
            fig = px.bar(x=names, y=probs*100)
            st.plotly_chart(fig)
        else:
            st.info(f"🆙 Rasmdagi jonivorni tasniflay olmadim. Bu noyob tur yoki hozirgi mashg'ulot ma'lumotlarimdan tashqari biror narsa bo'lishi mumkin 😔")
    except Exception as e:
        st.error(f"❌ Xatolik yuz berdi: {e}")
        msg = MIMEMultipart()
        msg['From'] = 'tmbtojiddinov@gmail.com'
        msg['To'] = 'tajiddinovmuhammaddiyor8@gmail.com'
        msg['Subject'] = 'HK Model Xatolik'
        body = f"Xatolik: {e}"
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('tmbtojiddinov@gmail.com', 'TMBB1974')
        text = msg.as_string()
        server.sendmail('your_email@gmail.com', 'tajiddinovmuhammaddiyor8@gmail.com', text)
        server.quit()
