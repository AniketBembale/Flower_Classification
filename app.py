
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# --- Styling ---
st.set_page_config(page_title="🌸 Iris Classifier", layout="centered")
st.markdown(
    """
    <style>
        .main { background-color: #f5f7fa; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
         }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Data ---
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(df.drop('species', axis=1), df['species'])

# --- Page Title ---
st.title("🌸 Iris Flower Species Classifier")


# --- Sidebar Inputs ---
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# --- Prediction ---
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=df.columns[:-1])
prediction = model.predict(input_data)[0]

# --- Layout: Two Columns ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌿 Prediction")
    st.success(f"The predicted species is: **{target_names[prediction]}**")

with col2:
    if prediction == 0:
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/86/Iris_setosa.JPG", caption="Iris Setosa")
    elif prediction == 1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/57/Iris_versicolor_1787.jpg", caption="Iris Versicolor")
    elif prediction == 2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/f8/Iris_virginica_2.jpg", caption="Iris Virginica")


