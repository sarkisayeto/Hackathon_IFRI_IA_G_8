import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("D:/Hackathon_IFRI_IA_G_8/data/processed/df_clean.csv")

X = df.drop(columns=["stade_irc"])  
y = df["stade_irc"]

# Enregistrer les colonnes de X avant prétraitement
X_columns = X.columns.tolist()

# Encodage des variables catégorielles
categorical_cols = X.select_dtypes(include=["object"]).columns
encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)], remainder="passthrough")
X_encoded = encoder.fit_transform(X)

# Normalisation des valeurs numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Séparation des données en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Dictionnaire des modèles à tester
models = {
    "Arbre de décision": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Entraîner et tester chaque modèle
best_model = None
best_accuracy = 0
for name, model in models.items():
    print(f"🔹 Entraînement du modèle: {name}")
    model.fit(X_train, y_train)
    
    # Prédiction et évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Précision: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarder le meilleur modèle
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Sauvegarder le modèle et les informations
info = {
    'model': best_model,
    'encoder': encoder,
    'scaler': scaler,
    'features': X_columns,  # Utiliser les colonnes sauvegardées
    'depth': best_model.get_depth() if hasattr(best_model, 'get_depth') else None,
    'leaves': best_model.get_n_leaves() if hasattr(best_model, 'get_n_leaves') else None,
    'score_train': best_model.score(X_train, y_train)
}

joblib.dump(info, 'arbre_complet.pkl')


print(f"\n✅ {best_model} est le Modèle enregistré sous 'arbre_complet.pkl'")

from sklearn.metrics import accuracy_score

# Prédictions sur les ensembles d'entraînement et de test
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calcul des précisions
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"Accuracy sur Train: {train_acc:.2f}")
print(f"Accuracy sur Test: {test_acc:.2f}")

# Vérification de l'écart
res = train_acc - test_acc
if res > 0.1:  # Plus de 10% de différence
    print("⚠️ Overfitting détecté !", res)

# Charger les infos sauvegardées
info = joblib.load("arbre_complet.pkl")
model = info['model']

# Vérifier les colonnes
print("Profondeur :", info['depth'])
print("Feuilles :", info['leaves'])
print("Score train :", info['score_train']*100)
print("Score test :", model.score(X_test, y_test)*100)