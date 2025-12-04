import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

# Load the saved model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
iris = load_iris()

st.title("Iris Flower Classifier 🌸")
st.write("A simple ML app deployed on Render using Streamlit + scikit-learn.")

st.sidebar.header("Input Features")

# Sidebar inputs
sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    # Prepare features for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    predicted_class = iris.target_names[prediction]

    st.subheader("Prediction")
    st.write(f"🌼 The predicted Iris species is **{predicted_class}**")
