# 🐶 Streamlit + Docker: Klasyfikator Rasy Psa

Autorzy: 
s20410 - Dawid Gołaszewski
s22862 - Mateusz Tomczak
S20912 - Jakub Wójtowicz

Ten projekt klasyfikuje rasę psa na podstawie jego cech (wzrost, długość życia, charakter, problemy zdrowotne) za pomocą modelu ML. Aplikacja działa w Streamlit i jest gotowa do uruchomienia w Dockerze.

## 📂 Struktura projektu

- `data/dog_breeds.csv` – dane wejściowe (CSV z cechami i rasą psa)
- `train_model.py` – trening modelu i zapis confusion matrix
- `app.py` – aplikacja Streamlit do przewidywania rasy
- `model/` – zapisany model, encodery, confusion_matrix.png
- `Dockerfile` – konteneryzacja projektu
- `requirements.txt` – zależności
- `README.md` – instrukcja

## 🧪 Uruchomienie lokalnie

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
### LUB
# Kontener
```bash
docker compose up 
```

### Streamlit: 
https://sumlprojgit-xegbbgowcep2n7ljlmmqsa.streamlit.app/