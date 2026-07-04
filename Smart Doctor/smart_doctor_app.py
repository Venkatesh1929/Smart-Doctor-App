# --------------------------------------------
# SMART DOCTOR - STREAMLIT CHATBOT APPLICATION
# --------------------------------------------
# ----------------------------------------------------
# SMART DOCTOR - STANDALONE STREAMLIT APP (NO DATASET)
# ----------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spacy
from difflib import get_close_matches
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

# --------------------------------------------
# 1️⃣ Define a mini synthetic dataset manually
# --------------------------------------------
# Symptoms (10 example features)
symptoms = [
    "fever", "cough", "fatigue", "headache",
    "sore throat", "body pain", "vomiting", 
    "nausea", "chest pain", "dizziness"
]

# Example diseases and symptom combinations
data = [
    # fever, cough, fatigue, headache, sore throat, body pain, vomiting, nausea, chest pain, dizziness
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, "Viral Fever"],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, "Common Cold"],
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, "Typhoid"],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, "Dengue"],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, "Migraine"],
    [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, "Heart Problem"]
]

df = pd.DataFrame(data, columns=symptoms + ["diseases"])

# --------------------------------------------
# 2️⃣ Train a small RandomForest model
# --------------------------------------------
X = df[symptoms]
y = df["diseases"]

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# --------------------------------------------
# 3️⃣ Prescriptions dictionary
# --------------------------------------------
prescriptions = {
    "Viral Fever": {
        "medicine": "Paracetamol 500mg every 6 hours",
        "advice": "Drink fluids, take rest, avoid cold food",
        "precautions": "Monitor temperature, stay hydrated"
    },
    "Common Cold": {
        "medicine": "Cetirizine or steam inhalation",
        "advice": "Drink hot water, avoid cold drinks",
        "precautions": "Keep warm, rest well"
    },
    "Typhoid": {
        "medicine": "Antibiotics (as prescribed)",
        "advice": "Eat soft food, drink boiled water",
        "precautions": "Avoid spicy food, complete medication"
    },
    "Dengue": {
        "medicine": "Paracetamol only (no aspirin)",
        "advice": "Rest, drink coconut water, monitor platelets",
        "precautions": "Avoid mosquito bites"
    },
    "Migraine": {
        "medicine": "Pain relievers (as prescribed)",
        "advice": "Avoid bright lights, stay calm",
        "precautions": "Avoid stress and loud noise"
    },
    "Heart Problem": {
        "medicine": "Consult cardiologist immediately",
        "advice": "Avoid stress and heavy work",
        "precautions": "Monitor BP and heart rate regularly"
    }
}

# --------------------------------------------
# 4️⃣ Streamlit Web UI
# --------------------------------------------
st.set_page_config(page_title="Smart Doctor - AI Health Assistant", page_icon="🩺")

st.title("🩺 Smart Doctor - AI Health Assistant")
st.markdown("Talk to the Smart Doctor and get predictions based on your symptoms!")

st.write("---")

# Extract symptoms from user input using NLP
def extract_symptoms(user_input):
    doc = nlp(user_input.lower())
    found = []
    for token in doc:
        match = get_close_matches(token.text, symptoms, n=1, cutoff=0.7)
        if match:
            found.append(match[0])
    return list(set(found))

user_input = st.text_input("Describe your symptoms (e.g., I have fever and headache):")

if st.button("🔍 Predict Disease"):
    if not user_input.strip():
        st.warning("Please enter your symptoms.")
    else:
        detected = extract_symptoms(user_input)
        if not detected:
            st.error("No valid symptoms detected! Try again.")
        else:
            st.success(f"Extracted Symptoms: {', '.join(detected)}")

            # Create feature vector
            input_vector = [1 if s in detected else 0 for s in symptoms]

            # Predict probabilities
            probs = model.predict_proba([input_vector])[0]
            classes = model.classes_

            # Top predictions
            top_idx = probs.argsort()[-3:][::-1]
            top_preds = [(classes[i], round(probs[i] * 100, 2)) for i in top_idx]

            st.subheader("🎯 Prediction Results:")
            for disease, prob in top_preds:
                st.write(f"**{disease}** — Confidence: {prob}%")

            # Plot bar chart
            st.write("📊 Confidence Visualization")
            fig, ax = plt.subplots()
            ax.bar([x[0] for x in top_preds], [x[1] for x in top_preds], color=["#36a2eb", "#ffcd56", "#ff6384"])
            ax.set_ylabel("Confidence (%)")
            st.pyplot(fig)

            # Prescription for top disease
            top_disease = top_preds[0][0]
            st.subheader("💊 Prescription & Advice")
            if top_disease in prescriptions:
                advice = prescriptions[top_disease]
                st.write(f"**Medicine:** {advice['medicine']}")
                st.write(f"**Advice:** {advice['advice']}")
                st.write(f"**Precautions:** {advice['precautions']}")
            else:
                st.info("No prescription data available. Please consult a doctor.")


