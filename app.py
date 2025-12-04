import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.write("A simple ML app deployed on Render using Streamlit + scikit-learn.")

# ---------------- LOAD DATA & TRAIN MODEL (NO PICKLE) ----------------
@st.cache_resource
def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, iris

model, iris = train_model()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = iris.target_names[prediction]

    st.success(f"✅ Predicted Iris Species: **{species.upper()}**")
