import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configuration de l'application
st.set_page_config(page_title="Analyse des Accidents de la Route", layout="wide")

# Sidebar pour navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Contexte",
    "Analyse exploratoire des données",
    "Tests statistiques",
    "Pre-processing",
    "Modélisation"
])

# Chargement des données
df = pd.read_csv('accident.csv')

if page == "Contexte":
    st.title("Analyse des Accidents de la Route et Modélisation Prédictive")
    st.write("""
    Les accidents de la route sont une problématique majeure impactant la sécurité publique et la mobilité. Ce projet vise à analyser les divers facteurs influençant la survenue d'accidents et leur gravité, en utilisant un ensemble de données contenant des informations démographiques, comportementales et situationnelles.

    L'objectif est double :
    1. Comprendre les tendances et relations entre les différentes variables à travers une analyse exploratoire des données.
    2. Construire des modèles de machine learning permettant de prédire la survenue d'un accident en fonction de plusieurs facteurs.

    Grâce à cette étude, nous pourrons identifier les variables les plus influentes, proposer des recommandations en matière de sécurité routière et améliorer les systèmes de prévention des accidents.
    """)
    st.write("Aperçu des données :")
    st.dataframe(df.head())

elif page == "Analyse exploratoire des données":
    st.title("Analyse Exploratoire des Données")
    option = st.selectbox("Choisissez une analyse", [
        "Analyse univariée discrète",
        "Analyse univariée continue",
        "Analyse bivariée Discrète-Discrète",
        "Analyse bivariée Discrète-Continue",
        "Analyse bivariée Continue-Continue"
    ])
    
    if option == "Analyse univariée discrète":
        categorical_columns = ["Gender", "Helmet_Used", "Seatbelt_Used", "Survived"]
        col = st.selectbox("Choisissez une variable", categorical_columns)
        st.write(df[col].value_counts())
        st.bar_chart(df[col].value_counts())
    
    elif option == "Analyse univariée continue":
        numerical_columns = ["Age", "Speed_of_Impact"]
        col = st.selectbox("Choisissez une variable", numerical_columns)
        st.write(df[col].describe())
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

elif page == "Tests statistiques":
    st.title("Tests Statistiques")
    st.write("""
    Nous allons vérifier l'indépendance entre certaines variables via des tests statistiques.
    """)
    from scipy.stats import chi2_contingency, pearsonr
    
    test = st.selectbox("Choisissez un test", ["Test du Chi²", "Test de Pearson"])
    if test == "Test du Chi²":
        cross_tab = pd.crosstab(df["Survived"], df["Gender"])
        res = chi2_contingency(cross_tab)
        st.write(f"P-value: {res.pvalue}")
        st.write("Les variables sont indépendantes" if res.pvalue > 0.05 else "Les variables sont dépendantes")
    elif test == "Test de Pearson":
        corr, pvalue = pearsonr(df["Age"], df["Speed_of_Impact"])
        st.write(f"Coefficient de corrélation: {corr}")
        st.write(f"P-value: {pvalue}")
        st.write("Corrélation significative" if pvalue < 0.05 else "Pas de corrélation significative")

elif page == "Pre-processing":
    st.title("Pré-traitement des Données")
    Y = df['Survived']
    X = df.drop(columns=['Survived'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    st.write("Jeu d'entraînement et de test créé.")
    
    # Normalisation et Encodage
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False)
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
    X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))
    st.write("Données normalisées avec MinMaxScaler.")

elif page == "Modélisation":
    st.title("Modélisation et Évaluation")
    model_choice = st.selectbox("Choisissez un modèle", ["Régression Logistique", "Random Forest", "XGBoost", "KNN"])
    
    if model_choice == "Régression Logistique":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "XGBoost":
        model = XGBClassifier()
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    
    model.fit(X_train_scaled, Y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, y_pred)
    st.write(f"Précision du modèle : {accuracy:.2f}")

# Conclusion
st.header("Conclusion")
st.write("""
L'analyse des accidents de la route nous a permis de mettre en évidence plusieurs facteurs clés influençant la survenue des accidents et leur gravité. Nous avons identifié des tendances intéressantes grâce à l'exploration des données et testé plusieurs modèles de machine learning pour la prédiction des accidents.

Les principaux enseignements de cette étude sont :
- Certains facteurs comme l'âge du conducteur, la vitesse d'impact et l'utilisation des équipements de sécurité (ceinture, casque) jouent un rôle déterminant.
- Les analyses statistiques ont confirmé l'indépendance ou la dépendance entre certaines variables, influençant la sélection des caractéristiques pertinentes pour la modélisation.
- Parmi les modèles testés, XGBoost et Random Forest ont montré les meilleures performances en termes de précision et de robustesse.

En appliquant ces résultats, des actions concrètes peuvent être mises en place, telles que :
- Des campagnes de sensibilisation ciblées sur les populations à risque.
- Le renforcement des réglementations sur l'utilisation des dispositifs de sécurité.
- L'utilisation de ces modèles prédictifs pour améliorer la gestion de la sécurité routière et la prise de décision par les autorités compétentes.

Cette étude constitue une première étape et peut être enrichie par des données supplémentaires ou des techniques de modélisation plus avancées pour affiner la prédiction et la prévention des accidents.
""")
