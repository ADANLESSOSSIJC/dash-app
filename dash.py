import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import shap

# Chemin vers le fichier de données
file_path = 'clean_data.csv'

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv(file_path)
    return data

# Entraînement du modèle avec optimisation des hyperparamètres et SMOTE
@st.cache_resource
def train_model(data):
    x = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = data['TARGET']

    # Split des données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Définir le pipeline avec SMOTE, StandardScaler et GradientBoostingClassifier
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    
    # Définir la grille de paramètres pour GridSearchCV
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_depth': [3, 5]
    }

    # Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_

# Prédire le score pour un client donné
def predict_score(model, client_data):
    score = model.predict_proba(client_data)[:, 1]
    return score[0]

# Fonction pour obtenir l'importance des caractéristiques spécifique à un client donné
def get_shap_values(model, data, client_data):
    explainer = shap.Explainer(model.named_steps['clf'], data)
    shap_values = explainer(client_data)
    return shap_values

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
    st.write("Interprétation : Risque élevé - Crédit réfusé")
else:
    st.write("Interprétation : Risque faible - Crédit accepté")

# Informations descriptives relatives au client
st.subheader("Informations descriptives du client")
st.write(client_data)

# Comparaison du revenu du client au revenu moyen de tous les clients
st.subheader("Comparaison du Revenu du Client avec le Revenu Moyen des Clients")
client_income = client_data['AMT_INCOME_TOTAL'].values[0]
average_income = data['AMT_INCOME_TOTAL'].mean()

# Affichage des revenus en graphique à barres
fig, ax = plt.subplots()
ax.bar(['Revenu du Client', 'Revenu Moyen'], [client_income, average_income], color=['blue', 'green'])
ax.set_ylabel('Revenu')
ax.set_title('Comparaison du Revenu')
st.pyplot(fig)  # Afficher la figure dans Streamlit

# Comparaison du ratio revenu par rapport au crédit du client au ratio min, moyen et max de tous les clients
st.subheader("Comparaison du Ratio Revenu/Crédit du Client avec le Min, Moyenne et Max de Tous les Clients")

# Calcul du ratio revenu/crédit pour le client
client_income_credit_perc = client_data['AMT_INCOME_TOTAL'].values[0] / client_data['AMT_CREDIT'].values[0]

# Calcul des statistiques du ratio revenu/crédit pour tous les clients
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
min_income_credit_perc = data['INCOME_CREDIT_PERC'].min()
mean_income_credit_perc = data['INCOME_CREDIT_PERC'].mean()
max_income_credit_perc = data['INCOME_CREDIT_PERC'].max()

# Affichage des ratios en graphique à barres
fig, ax = plt.subplots()
bar_labels = ['Client', 'Min', 'Moyenne', 'Max']
bar_values = [client_income_credit_perc, min_income_credit_perc, mean_income_credit_perc, max_income_credit_perc]
bar_colors = ['blue', 'red', 'green', 'orange']

# Créer les barres avec les étiquettes et la légende
bars = ax.bar(bar_labels, bar_values, color=bar_colors)
ax.set_ylabel('Ratio Revenu/Crédit')
ax.set_title('Comparaison du Ratio Revenu/Crédit')
ax.legend(bars, ['Client Sélectionné', 'Min Tous Clients', 'Moyenne Tous Clients', 'Max Tous Clients'])

# Ajouter les étiquettes de valeur sur chaque barre
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

st.pyplot(fig)  # Afficher la figure dans Streamlit

# Afficher les variables importantes pour le client choisi
st.subheader("Variables importantes pour le client choisi")

# Obtenir les valeurs SHAP pour le client sélectionné
shap_values = get_shap_values(model, data.drop(['TARGET', 'SK_ID_CURR'], axis=1), client_data)

# Afficher les 10 principales caractéristiques
importance_df = pd.DataFrame({
    'Feature': client_data.columns,
    'Importance': shap_values.values[0]
}).sort_values(by='Importance', ascending=False)

st.write(importance_df.head(10))

# Visualiser les importances des caractéristiques
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 des caractéristiques importantes pour le client')
plt.gca().invert_yaxis()  # Inverser l'axe y pour avoir les plus importantes en haut

plt.tight_layout()  # Ajuster la mise en page pour éviter les chevauchements
st.pyplot(plt.gcf())  # Afficher la figure dans Streamlit
plt.close()  # Fermer la figure pour libérer de la mémoire

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
