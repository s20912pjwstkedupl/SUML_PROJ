import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Wczytaj model i encodery
model = joblib.load("model/model.pkl")
traits_bin, health_bin, fur_bin, eye_bin, origin_enc = joblib.load("model/encoders.pkl")

st.title("🐶 Klasyfikator Rasy Psa")

# Interfejs użytkownika
height = st.slider("Wzrost psa (cale)", 5, 40, 20)
longevity = st.slider("Długość życia (lata)", 5, 25, 12)

traits_input = st.multiselect("Cechy charakteru", traits_bin.classes_)
health_input = st.multiselect("Problemy zdrowotne", health_bin.classes_)
fur_input = st.multiselect("Kolor sierści", fur_bin.classes_)
eye_input = st.multiselect("Kolor oczu", eye_bin.classes_)
origin_input = st.selectbox("Kraj pochodzenia", origin_enc.categories_[0])

# Predykcja
if st.button("Przewiduj rasę"):
    traits_vec = pd.DataFrame(traits_bin.transform([traits_input]), columns=traits_bin.classes_)
    health_vec = pd.DataFrame(health_bin.transform([health_input]), columns=health_bin.classes_)
    fur_vec = pd.DataFrame(fur_bin.transform([fur_input]), columns=fur_bin.classes_)
    eye_vec = pd.DataFrame(eye_bin.transform([eye_input]), columns=eye_bin.classes_)
    origin_vec = pd.DataFrame(origin_enc.transform([[origin_input]]), columns=origin_enc.get_feature_names_out(["Country of Origin"]))

    df_pred = pd.concat([
        pd.DataFrame([[height, longevity]], columns=["Height", "Longevity"]),
        traits_vec, health_vec, fur_vec, eye_vec, origin_vec
    ], axis=1)

    prediction = model.predict(df_pred)[0]
    st.success(f"🔍 Przewidywana rasa psa: **{prediction}**")

# Macierz pomyłek
st.header("📊 Macierz pomyłek")
image = Image.open("model/confusion_matrix.png")
st.image(image, caption="Confusion Matrix")
