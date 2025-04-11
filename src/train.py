import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Charger les données
df = pd.read_csv("../data/processed/df_clean.csv")
X = df.drop(columns=["stade_irc"])  
y = df["stade_irc"]
X_columns = X.columns.tolist()

# Encodage et normalisation
categorical_cols = X.select_dtypes(include=["object"]).columns
encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)], remainder="passthrough")
X_encoded = encoder.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Grilles d'hyperparamètres
param_grids = {
    "Arbre de décision": {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 7, 10],
        'criterion': ['gini', 'entropy']
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Modèles
base_models = {
    "Arbre de décision": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

best_model = None
best_accuracy = 0

for name, model in base_models.items():
    print(f"\n GridSearch pour: {name}")
    grid = GridSearchCV(model, param_grids[name], cv=6, n_jobs=-1, scoring='accuracy')
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f" Meilleure précision pour {name} : {acc:.2f}")
    print(" Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f" Meilleurs paramètres : {grid.best_params_}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = best

# Sauvegarde
info = {
    'model': best_model,
    'encoder': encoder,
    'scaler': scaler,
    'features': X_columns,
    'depth': best_model.get_depth() if hasattr(best_model, 'get_depth') else None,
    'leaves': best_model.get_n_leaves() if hasattr(best_model, 'get_n_leaves') else None,
    'score_train': best_model.score(X_train, y_train)
}
joblib.dump(info, 'arbre_complet.pkl')
print(f"\n {best_model} est le modèle enregistré sous 'arbre_complet.pkl'")

# Évaluation finale
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

precision = precision_score(y_test, test_preds, average='weighted')
recall = recall_score(y_test, test_preds, average='weighted')
f1 = f1_score(y_test, test_preds, average='weighted')

print(f"\n Précision : {precision:.2f}")
print(f" Rappel : {recall:.2f}")
print(f" F1-score : {f1:.2f}")

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
print(f" Accuracy sur Train: {train_acc}")
print(f" Accuracy sur Test: {test_acc}")

res = train_acc - test_acc
if res > 0.1:
    print(" Overfitting détecté !", res)
else:
    print(" Pas d'overfitting détecté.")

# Résumé
info = joblib.load("arbre_complet.pkl")
model = info['model']

print("\n Résumé du meilleur modèle:")
print("Profondeur :", info['depth'])
print("Feuilles :", info['leaves'])
print("Score train :", info['score_train']*100)
print("Score test :", model.score(X_test, y_test)*100)

# Matrice de confusion
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show()
