"""Titanic passenger survival-classification demonstration.

Author: Moe Htet Ar Kar (Phoe Cho)

This is an educational analysis of a historical dataset, not a real-world
safety prediction tool. The selected dataset may need an internet connection.
"""

import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title = "Titanic Survival Study", page_icon = "🚢", layout = "wide")

st.title("🚢 Titanic Passenger Survival Study")
st.caption("Historical dataset · Supervised binary classification")
st.info("The result is a model estimate based on a historical dataset. It is not a statement about any person's worth or current safety.")

FEATURES = ["pclass", "sex", "age", "fare"]


@st.cache_resource
def load_and_train_titanic():
    """Load the public Seaborn sample and fit a simple classification pipeline."""
    data = sns.load_dataset("titanic")[ ["survived", "pclass", "sex", "age", "fare"] ].dropna().copy()
    data["sex"] = data["sex"].map({"male": 0, "female": 1}).astype(int)
    features = data[FEATURES]
    target = data["survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size = 0.2, random_state = 42, stratify = target
    )
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(max_iter = 1000)),
        ]
    )
    classifier.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, classifier.predict(x_test))
    return classifier, float(accuracy), len(data)


try:
    model, test_accuracy, row_count = load_and_train_titanic()
except Exception as exc:
    st.error("Could not load or train from the Seaborn Titanic dataset.")
    st.info("Check your internet connection, then restart the app. Seaborn may need to download the dataset on the first run.")
    st.exception(exc)
    st.stop()

st.sidebar.header("Passenger profile")
pclass = st.sidebar.selectbox("Passenger class", [1, 2, 3], index = 1, format_func = lambda value: {1: "1st class", 2: "2nd class", 3: "3rd class"}[value])
sex = st.sidebar.selectbox("Sex recorded in dataset", ["Female", "Male"])
age = st.sidebar.slider("Age (years)", 1, 80, 28)
fare = st.sidebar.slider("Ticket fare ($)", 0.0, 500.0, 32.0)

col1, col2 = st.columns([1.0, 1.0])
with col1:
    st.subheader("Profile summary")
    st.metric("Passenger class", f"{pclass}{'st' if pclass == 1 else 'nd' if pclass == 2 else 'rd'}")
    st.metric("Age", f"{age} years")
    st.metric("Ticket fare", f"${fare:.2f}")
    st.metric("Rows used after dropping missing values", f"{row_count:,}")
with col2:
    st.subheader("Prediction report")
    if st.button("Estimate historical survival class", use_container_width = True):
        sex_value = 1 if sex == "Female" else 0
        input_row = pd.DataFrame([[pclass, sex_value, age, fare]], columns = FEATURES)
        prediction = int(model.predict(input_row)[0])
        survival_score = float(model.predict_proba(input_row)[0, 1])
        st.metric("Model survival-class score", f"{survival_score * 100:.2f}%")
        if prediction == 1:
            st.success("Model class: survived in the historical dataset.")
        else:
            st.warning("Model class: did not survive in the historical dataset.")
        st.caption("This is a model output, not a reliable claim about an individual outcome.")
    else:
        st.info("Set the profile inputs and run the estimate to view model output.")
    st.metric("Held-out test accuracy", f"{test_accuracy * 100:.2f}%")
    st.caption("Accuracy is measured on a random test split; it does not guarantee future performance.")

st.divider()
st.caption("Educational classification project · Seaborn Titanic sample · Not for consequential decisions")
