from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# === Base paths ===
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "Datasets")
MODEL_PATH = os.path.join(BASE_PATH, "models", "svc.pkl")

# === Load Model ===
with open(MODEL_PATH, "rb") as file:
    svc = pickle.load(file)

# === Load Datasets (RELATIVE PATHS) ===
training_df = pd.read_csv(os.path.join(DATASET_PATH, "Training.csv"))
precautions = pd.read_csv(os.path.join(DATASET_PATH, "precautions_df.csv"))
workout = pd.read_csv(os.path.join(DATASET_PATH, "workout_df.csv"))
description = pd.read_csv(os.path.join(DATASET_PATH, "description.csv"))
medications = pd.read_csv(os.path.join(DATASET_PATH, "medications.csv"))
diets = pd.read_csv(os.path.join(DATASET_PATH, "diets.csv"))

# === Get list of features used during training ===
if "Disease" in training_df.columns:
    all_symptoms = [col for col in training_df.columns if col != "Disease"]
else:
    all_symptoms = list(training_df.columns)

# === ROUTES ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get('symptom', '').strip().lower()

    if not user_input:
        return render_template('index.html', message="⚠️ Please enter at least one symptom.")

    input_symptoms = [s.strip() for s in user_input.split(',')]
    input_data = np.zeros(len(all_symptoms))

    for symptom in input_symptoms:
        if symptom in all_symptoms:
            input_data[all_symptoms.index(symptom)] = 1

    # Feature check
    if len(input_data) != svc.n_features_in_:
        return render_template(
            'index.html',
            message="⚠️ Feature mismatch. Training data and model are not aligned."
        )

    prediction = svc.predict([input_data])[0]

    desc = description.loc[description['Disease'] == prediction, 'Description']
    desc = desc.values[0] if not desc.empty else "Description not available."

    def fetch_list(df):
        row = df.loc[df['Disease'] == prediction]
        if row.empty:
            return "Data not available"
        return ", ".join([str(i) for i in row.values[0][1:] if pd.notna(i)])

    return render_template(
        'index.html',
        disease=prediction,
        description=desc,
        precautions=fetch_list(precautions),
        medications=fetch_list(medications),
        workouts=fetch_list(workout),
        diets=fetch_list(diets)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
