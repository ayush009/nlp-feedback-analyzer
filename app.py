# app.py
import os, base64, pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="36 Questions of Love", layout="wide")

# ---------- helpers ----------
def set_background(image_file: str = "image.jpg"):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded});
            background-size: cover;
        }}
        .main {{ background: rgba(255,255,255,.55); padding: 20px; border-radius: 10px; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def plot_confusion_matrix(cm, labels, title="Performance"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def classification_report_to_df(report_text: str) -> pd.DataFrame:
    rows = []
    for line in report_text.splitlines()[2:-3]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({
                "class": parts[0],
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1-score": float(parts[3]),
                "support": int(parts[4]),
            })
    return pd.DataFrame(rows)

# cache models
@st.cache_resource
def load_artifacts():
    req = [
        "models/random_forest_model.pkl",
        "models/naive_bayes_model.pkl",
        "models/vectorizer.pkl",
        "models/top_features.pkl",
    ]
    miss = [p for p in req if not os.path.exists(p)]
    if miss:
        st.error("Missing files:\n" + "\n".join(f"• {m}" for m in miss))
        st.stop()
    with open(req[0], "rb") as f: rf_model = pickle.load(f)
    with open(req[1], "rb") as f: nb_model = pickle.load(f)
    with open(req[2], "rb") as f: vectorizer = pickle.load(f)
    with open(req[3], "rb") as f: top_features = pickle.load(f)
    return rf_model, nb_model, vectorizer, top_features

def display_most_used_words(person_label, top_features):
    if isinstance(top_features, dict) and person_label in top_features:
        st.subheader(f"Most Used Words for {person_label}:")
        for word, score in top_features[person_label]:
            st.write(f"{word}: {score:.4f}")

# ---------- UI ----------
set_background("image.jpg")

st.title("36 Questions of Love")

tab = st.sidebar.radio("Navigation", ["Predict", "Model Metrics"])

rf_model, nb_model, vectorizer, top_features = load_artifacts()

if tab == "Predict":
    st.header("Prediction")
    st.write("Enter a response to one of the 36 Questions to Love")

    response = st.text_area("Your response:")
    model_type = st.selectbox("Select model for prediction:", ["Naive Bayes", "Random Forest"])

    if st.button("Predict"):
        if not response.strip():
            st.warning("Please enter some text.")
        else:
            X = vectorizer.transform([response])
            if model_type == "Random Forest":
                prediction = rf_model.predict(X)[0]
            else:
                prediction = nb_model.predict(X)[0]
            st.success(f"Predicted Person: {prediction}")
            display_most_used_words(prediction, top_features)

else:
    st.header("Model Test Metrics")

    uploaded_file = st.file_uploader("Upload custom test CSV (columns: response, person)", type="csv")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
    else:
        test_df = pd.read_csv("test_data.csv")

    # normalize column names
    cols = {c.lower(): c for c in test_df.columns}
    if not {"response", "person"}.issubset(cols):
        st.error("CSV must contain 'response' and 'person' columns.")
        st.stop()
    test_df = test_df.rename(columns={cols["response"]: "response", cols["person"]: "person"})

    if st.button("Evaluate Models"):
        X_test = vectorizer.transform(test_df["response"])
        y_test = test_df["person"]
        labels_sorted = sorted(y_test.unique())

        for name, model in [("Random Forest", rf_model), ("Naive Bayes", nb_model)]:
            st.subheader(f"{name} Metrics")
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc*100:.2f}%")

            report_df = classification_report_to_df(
                classification_report(y_test, y_pred, zero_division=0)
            )
            st.dataframe(report_df, use_container_width=True)

            cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
            plot_confusion_matrix(cm, labels_sorted, title=f"{name} — Confusion Matrix")
