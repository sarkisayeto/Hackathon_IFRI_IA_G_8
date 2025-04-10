# Hackathon_IFRI_IA_G_8

Ce projet a été développé dans le cadre du Hackathon IFRI IA G8. Il contient une API Flask, des notebooks d'exploration, des scripts pour entraîner des modèles d'apprentissage automatique, et une interface utilisateur avec Streamlit.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

- [Python 3.8+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

## Installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/votre-utilisateur/Hackathon_IFRI_IA_G_8.git
   cd Hackathon_IFRI_IA_G_8

2. Créez un environnement virtuel Python :
    python -m venv .env

3. Activez l'environnement virtuel :
    Sur Windows : .env\Scripts\activate
    Sur macOS/Linux : source .env/bin/activate

4. Installez les dépendances :
    pip install -r requirements.txt


## Entraînement du modèle

1. Accédez au dossier contenant les scripts d'entraînement :
cd src

2. Lancez le script d'entraînement :
python train_model.py

Ce script entraînera le modèle et sauvegardera les fichiers nécessaires (par exemple, le modèle entraîné) dans le dossier models/.


## Lancement de l'API Flask

1. Accédez au dossier de l'API Flask :
cd Flask_API

2. Lancez 
python app.py

3. L'API sera disponible à l'adresse suivante : http://127.0.0.1:5000.


## Test avec Streamlit

1. Ouvrez un nouveau Shell pour le test avec streamlit :
2. Accédez au dossier de l'API Flask apres avoir avoir activé l'environnement :
cd Flask_API

3. Lancez l'application Streamlit :
streamlit run Frontend.py

4. Ouvrez votre navigateur et accédez à l'adresse suivante : http://localhost:8501.


## Structure du projet
Flask_API/ : Contient les fichiers pour l'API Flask.
notebooks/ : Contient les notebooks Jupyter pour l'exploration des données.
src/ : Contient les scripts pour l'entraînement des modèles.
data/ : Contient les données brutes et traitées.





