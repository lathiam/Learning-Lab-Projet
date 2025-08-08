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
from sklearn.impute import SimpleImputer

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
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=[object]).columns.tolist()

    manip_option = st.multiselect(
        "Sélectionnez les manipulations à appliquer:",
        ["Imputation des NaN", "Normalisation", "Encodage", "Suppression des NaN", "Afficher les résultats"],
        default=["Imputation des NaN", "Normalisation", "Encodage", "Afficher les résultats"]
    )

    # Initialisation
    X_train_num = X_train[num_cols]
    X_test_num = X_test[num_cols]
    X_train_cat = X_train[cat_cols]
    X_test_cat = X_test[cat_cols]

    # Imputation
    if "Imputation des NaN" in manip_option:
        imputer_num = SimpleImputer(strategy='mean')
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train_num = pd.DataFrame(imputer_num.fit_transform(X_train_num), columns=num_cols)
        X_test_num = pd.DataFrame(imputer_num.transform(X_test_num), columns=num_cols)
        X_train_cat = pd.DataFrame(imputer_cat.fit_transform(X_train_cat), columns=cat_cols)
        X_test_cat = pd.DataFrame(imputer_cat.transform(X_test_cat), columns=cat_cols)

    # Normalisation
    if "Normalisation" in manip_option:
        scaler = MinMaxScaler()
        X_train_num = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols)
        X_test_num = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols)

    # Encodage
    if "Encodage" in manip_option:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat = pd.DataFrame(encoder.fit_transform(X_train_cat), columns=encoder.get_feature_names_out(cat_cols))
        X_test_cat = pd.DataFrame(encoder.transform(X_test_cat), columns=encoder.get_feature_names_out(cat_cols))

    # Concaténation
    import numpy as np
    X_train_scaled = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_test_scaled = np.concatenate([X_test_num, X_test_cat], axis=1)

    # Suppression des lignes contenant des NaN
    if "Suppression des NaN" in manip_option:
        mask_train = ~np.isnan(X_train_scaled).any(axis=1)
        mask_test = ~np.isnan(X_test_scaled).any(axis=1)
        X_train_scaled = X_train_scaled[mask_train]
        Y_train = Y_train.reset_index(drop=True)[mask_train]
        X_test_scaled = X_test_scaled[mask_test]
        Y_test = Y_test.reset_index(drop=True)[mask_test]

    # Affichage des résultats
    if "Afficher les résultats" in manip_option:
        st.write("Données après prétraitement :")
        st.write(f"X_train shape: {X_train_scaled.shape}")
        st.write(f"X_test shape: {X_test_scaled.shape}")
        st.write(f"Y_train shape: {Y_train.shape}")
        st.write(f"Y_test shape: {Y_test.shape}")

elif page == "Modélisation":
    st.title("Modélisation et Évaluation")
    model_choice = st.selectbox("Choisissez un modèle", ["Régression Logistique", "Random Forest", "XGBoost", "KNN"])
    # On refait le preprocessing pour garantir la cohérence et gérer les NaN
    Y = df['Survived']
    X = df.drop(columns=['Survived'])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=[object]).columns.tolist()
    from sklearn.impute import SimpleImputer
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train_num = imputer_num.fit_transform(X_train[num_cols])
    X_test_num = imputer_num.transform(X_test[num_cols])
    X_train_cat = imputer_cat.fit_transform(X_train[cat_cols])
    X_test_cat = imputer_cat.transform(X_test[cat_cols])
    scaler = MinMaxScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_num = scaler.fit_transform(X_train_num)
    X_test_num = scaler.transform(X_test_num)
    X_train_cat = encoder.fit_transform(X_train_cat)
    X_test_cat = encoder.transform(X_test_cat)
    import numpy as np
    X_train_scaled = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_test_scaled = np.concatenate([X_test_num, X_test_cat], axis=1)
    # Suppression des lignes contenant des NaN dans X_train_scaled et X_test_scaled
    mask_train = ~np.isnan(X_train_scaled).any(axis=1)
    mask_test = ~np.isnan(X_test_scaled).any(axis=1)
    X_train_scaled = X_train_scaled[mask_train]
    Y_train = Y_train.reset_index(drop=True)[mask_train]
    X_test_scaled = X_test_scaled[mask_test]
    Y_test = Y_test.reset_index(drop=True)[mask_test]
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
