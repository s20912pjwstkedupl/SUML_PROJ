import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Wczytaj dane
df = pd.read_csv("data/dog_breeds.csv")

# Przetwórz liczby
df["Height"] = df["Height (in)"].str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1)
df["Longevity"] = df["Longevity (yrs)"].str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1)

# Przetwarzanie kolumn tekstowych
def split_and_clean(col):
    return df[col].fillna("").str.lower().str.replace(" ", "").str.split(",")

traits_bin = MultiLabelBinarizer()
traits = traits_bin.fit_transform(split_and_clean("Character Traits"))
traits_df = pd.DataFrame(traits, columns=traits_bin.classes_)

health_bin = MultiLabelBinarizer()
health = health_bin.fit_transform(split_and_clean("Common Health Problems"))
health_df = pd.DataFrame(health, columns=health_bin.classes_)

fur_bin = MultiLabelBinarizer()
fur = fur_bin.fit_transform(split_and_clean("Fur Color"))
fur_df = pd.DataFrame(fur, columns=fur_bin.classes_)

eye_bin = MultiLabelBinarizer()
eye = eye_bin.fit_transform(split_and_clean("Color of Eyes"))
eye_df = pd.DataFrame(eye, columns=eye_bin.classes_)

origin_enc = OneHotEncoder(sparse_output=False)
origin = origin_enc.fit_transform(df[["Country of Origin"]])
origin_df = pd.DataFrame(origin, columns=origin_enc.get_feature_names_out(["Country of Origin"]))


# Przygotowanie danych
X = pd.concat([
    df[["Height", "Longevity"]],
    traits_df,
    health_df,
    fur_df,
    eye_df,
    origin_df
], axis=1)
y = df["Breed"]

# Trening
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Zapis modelu
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/model.pkl")
joblib.dump((traits_bin, health_bin, fur_bin, eye_bin, origin_enc), "model/encoders.pkl")

# Macierz
cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(xticks_rotation=90)
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
