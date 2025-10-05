import pandas as pd
import os
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pickle

# Directory containing the CSV files
data_dir = 'responses/'  # Update this with your actual directory path

# Load data from CSV files and remove punctuation
data = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        person_class = file_name.split('.')[0]
        df = pd.read_csv(os.path.join(data_dir, file_name),encoding='utf-8',encoding_errors='ignore')
        if 'response' not in df.columns:
            raise ValueError(f"File {file_name} does not have 'response' column.")
        for response in df['response']:
            response = response.translate(str.maketrans('', '', string.punctuation))
            data.append({'Person': person_class, 'Response': response})

df = pd.DataFrame(data)

# Feature extraction with TfidfVectorizer for Naive Bayes
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b|n't|\b\w+'\w\b", stop_words='english')
X = vectorizer.fit_transform(df['Response'])
y = df['Person']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Identify important features for Naive Bayes
feature_names = vectorizer.get_feature_names_out()
log_probabilities = nb_model.feature_log_prob_

# Dictionary to store top features for each person
top_features = {'PersonA': [], 'PersonB': [],'PersonC': [], 'PersonD': []}

for i, person_label in enumerate(['PersonA', 'PersonB','PersonC','PersonD']):
    top_words = np.argsort(log_probabilities[i])[-10:]
    for idx in top_words:
        top_features[person_label].append((feature_names[idx], log_probabilities[i][idx]))

# Identify important features for Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]
for idx in indices:
    top_features[person_label].append((feature_names[idx], importances[idx]))

print("Top Features:", top_features)

# Save the trained models, vectorizer, and top features
os.makedirs('models', exist_ok=True)

# Save the Random Forest model
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save the Naive Bayes model
with open('models/naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

# Save the TfidfVectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the top features
with open('models/top_features.pkl', 'wb') as f:
    pickle.dump(top_features, f)

print("Models, vectorizer, and top features saved successfully.")
