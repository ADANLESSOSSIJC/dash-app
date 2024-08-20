import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Chemin vers le fichier de données
file_path = 'clean_data.csv'  

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv(file_path)
    return data

# Entraînement du modèle (exemple simplifié)
@st.cache_resource
def train_model(data):
    x = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = data['TARGET']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(x_train, y_train)
    return pipeline

# Prédire le score pour un client donné
def predict_score(model, client_data):
    score = model.predict_proba(client_data)[:, 1]
    return score[0]

# Chargement des données
data = load_data()

# Entraînement du modèle
model = train_model(data)

# Interface utilisateur Streamlit
st.title("Tableau de Bord Interactif des Scores Clients")

# Sélection du client
client_id = st.selectbox("Sélectionnez un client", data['SK_ID_CURR'].unique())
client_data = data[data['SK_ID_CURR'] == client_id].drop(['TARGET', 'SK_ID_CURR'], axis=1)

# Prédiction du score
score = predict_score(model, client_data)
st.write(f"Score pour le client {client_id}: {score:.2f}")

# Interprétation du score
if score > 0.5:
    st.write("Interprétation : Risque élevé")
else:
    st.write("Interprétation : Risque faible")

# Informations descriptives relatives au client
st.subheader("Informations descriptives du client")
st.write(client_data)

# Comparaison avec l'ensemble des clients
st.subheader("Comparaison avec l'ensemble des clients")
comparison_option = st.selectbox("Comparer avec", ["Tous les clients", "Clients similaires"])

if comparison_option == "Tous les clients":
    st.write(data.drop(['TARGET', 'SK_ID_CURR'], axis=1).describe())
else:
    # Comparaison avec un groupe de clients ayant un score similaire
    similar_clients = []
    for idx, row in data.iterrows():
        other_client_data = row.drop(['TARGET', 'SK_ID_CURR'])
        other_score = predict_score(model, [other_client_data.values])
        if abs(other_score - score) < 0.1:  # Si les scores sont similaires
            similar_clients.append(row)

    if similar_clients:
        similar_clients_df = pd.DataFrame(similar_clients)
        st.write(similar_clients_df.describe())
    else:
        st.write("Aucun client similaire trouvé.")
