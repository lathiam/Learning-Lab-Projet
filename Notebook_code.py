import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# Importation dataset
df = pd.read_csv('accident.csv')
df = pd.DataFrame(df)
print(df.head(20))

# EDA dataset
df.info()

# Analyse Univariée variables Continues : Speed_of_Impact, Age
print(df[["Age", "Speed_of_Impact"]].describe().round(2))
sns.histplot(df["Age"])
plt.show()
sns.boxplot(df["Age"])
plt.show()
sns.histplot(df["Speed_of_Impact"])
plt.show()
sns.boxplot(df["Age"])
plt.show()

# Analyse Bivariée variables Continues : Speed_of_Impact, Age
sns.scatterplot(x="Age", y="Speed_of_Impact", data=df)
plt.show()

# Analyses Univariée variables discrètes:
print(df["Gender"].value_counts())
print(df["Gender"].value_counts()*100/199)
df["Gender"].value_counts().plot(kind="bar")
plt.show()
print(df["Helmet_Used"].value_counts())
print(df["Helmet_Used"].value_counts()*100/199)
df["Helmet_Used"].value_counts().plot(kind="bar")
plt.show()
print(df["Seatbelt_Used"].value_counts())
print(df["Seatbelt_Used"].value_counts()*100/199)
df["Seatbelt_Used"].value_counts().plot(kind="bar")
plt.show()
print(df["Survived"].value_counts())
print(df["Survived"].value_counts()*100/199)
df["Survived"].value_counts().plot(kind="bar")
plt.show()

# Analyse Bivariée variables discrètes
cross_tab_Survived_Gender = pd.crosstab(df["Survived"], df["Gender"])
print(cross_tab_Survived_Gender)
sns.heatmap(cross_tab_Survived_Gender, annot=True)
plt.show()
cross_tab_Survived_Seatbelt_Used = pd.crosstab(df["Survived"], df["Seatbelt_Used"])
print(cross_tab_Survived_Seatbelt_Used)
sns.heatmap(cross_tab_Survived_Seatbelt_Used, annot=True)
plt.show()
cross_tab_Survived_Helmet_Used = pd.crosstab(df["Survived"], df["Helmet_Used"])
print(cross_tab_Survived_Helmet_Used)
sns.heatmap(cross_tab_Survived_Helmet_Used, annot=True)
plt.show()

# Analyse Bivariée variables discrètes-continues
print(df.groupby("Survived")["Age"].describe().round(2))
print(df.groupby("Survived")["Speed_of_Impact"].describe().round(2))

# Tests statistiques
from scipy.stats import chi2_contingency
res = chi2_contingency(cross_tab_Survived_Gender)
print(f"La valeur de pvalue est {res.pvalue}")
print(f"La valeur de pvalue {res.pvalue} étant superieur à 0.05, on ne rejette pas l'hypothèse nulle ")
print("Les variables 'Survived' et 'Age' sont indépendantes")
res = chi2_contingency(cross_tab_Survived_Seatbelt_Used)
print(f"La valeur de pvalue est {res.pvalue}")
print(f"La valeur de pvalue {res.pvalue} étant superieur à 0.05, on ne rejette pas l'hypothèse nulle ")
print("Les variables 'Survived' et 'Seatbelt_Used' sont indépendantes")
res = chi2_contingency(cross_tab_Survived_Helmet_Used)
print(f"La valeur de pvalue est {res.pvalue}")
print(f"La valeur de pvalue {res.pvalue} étant superieur à 0.05, on ne rejette pas l'hypothèse nulle ")
print("Les variables 'Survived' et 'Helmet_Used' sont indépendantes")

# Test Analyse Bivariée variables continues
df_copy = df.copy()
df_copy = df_copy.drop(columns=["Helmet_Used"])
df_copy.dropna(inplace=True)
print(df_copy.isnull().sum())
from scipy.stats import pearsonr
corr = pearsonr(df_copy["Age"], df_copy["Speed_of_Impact"])
print("L'hypothèse H0: IL n'y a pas de corrélation entre l'age et la vitesse d'impact")
print("L'hypothèse H1: IL y a une corrélation entre l'age et la vitesse d'impact")
print(f"La valeur de pvalue est {corr.pvalue}")
print(f"La valeur de pvalue {corr.pvalue} étant superieur à 0.05, on ne rejettera pas l'hypothèse H0 ")

# Pre-processing
Y = df_copy['Survived']
X = df_copy.drop(columns=['Survived'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

# Encodage
X_train_num = X_train.select_dtypes(include=[np.number])
X_train_cat = X_train.select_dtypes(include=[object])
scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(X_train_num)
onehot_encoder = OneHotEncoder(sparse_output=False)
X_train_cat_encoded = onehot_encoder.fit_transform(X_train_cat)
X_train_cat_encoded_df = pd.DataFrame(
    X_train_cat_encoded, 
    columns=onehot_encoder.get_feature_names_out(X_train_cat.columns)
)
X_train_combined = pd.concat([pd.DataFrame(X_train_num, columns=X_train.select_dtypes(include=[np.number]).columns), X_train_cat_encoded_df], axis=1)

# Transformion X_test
X_test_num = X_test.select_dtypes(include=[np.number])
X_test_cat = X_test.select_dtypes(include=[object])
X_test_num[["Age", "Speed_of_Impact"]] = scaler.transform(X_test_num[["Age", "Speed_of_Impact"]])
X_test_cat_encoded = onehot_encoder.transform(X_test_cat)
X_test_cat_encoded_df = pd.DataFrame(
    X_test_cat_encoded, 
    columns=onehot_encoder.get_feature_names_out(X_test_cat.columns)
)
X_test_combined = pd.concat([X_test_num.reset_index(drop=True), X_test_cat_encoded_df.reset_index(drop=True)], axis=1)

# Entrainement du modèle
model = LogisticRegression()
model.fit(X_train_combined, Y_train)

# Test du modèle
# prediction
y_pred = model.predict(X_test_combined)
print(y_pred)
# Evaluation
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)

# Randomforest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_combined, Y_train)
y_pred_rf = model_rf.predict(X_test_combined)
print(y_pred_rf)
accuracy_rf = accuracy_score(Y_test, y_pred_rf)
print(accuracy_rf)

# XGBOOST
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train_combined, Y_train)
y_pred_xgb = model_xgb.predict(X_test_combined)
print(y_pred_xgb)
accuracy_xgb = accuracy_score(Y_test, y_pred_xgb)
print(accuracy_xgb)

# KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier()
model_knn.fit(X_train_combined, Y_train)
y_pred_knn = model_knn.predict(X_test_combined)
print(y_pred_knn)
accuracy_knn = accuracy_score(Y_test, y_pred_knn)
print(accuracy_knn)
