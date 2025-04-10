import streamlit as st
import requests
import json

# URL de ton API Flask locale
API_URL = "http://127.0.0.1:5000/predict"

# Correspondance entre les valeurs prédictives et les stades
stade_mapping = {
    0.0: ("Stade 1", "#A0D995"),
    1.0: ("Stade 2", "#D9E95D"),
    2.0: ("Stade 3a", "#FFDD57"),
    3.0: ("Stade 3b", "#FFA94D"),
    4.0: ("Stade 4", "#FF6B6B"),
    5.0: ("Stade 5", "#C92A2A"),
}

# Titre de l'application
st.set_page_config(page_title="Prédiction des stades de MRC", page_icon="🩺")
st.title("🩺 Prédiction du Stade de la Maladie Rénale Chronique")
st.markdown("Chargez un fichier **.json** contenant les caractéristiques du patient pour obtenir une estimation du **stade de la MRC**.")

# Uploader le fichier JSON
uploaded_file = st.file_uploader("📁 Uploader un fichier JSON", type=["json"])

# Si un fichier est uploadé
if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)

        # Afficher les données du fichier
        with st.expander("📄 Voir les données importées"):
            st.json(data)

        # Appel à l'API Flask
        with st.spinner("⏳ Prédiction en cours..."):
            response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            prediction = response.json().get("prediction", [None])[0]
            stade, color = stade_mapping.get(prediction, ("Inconnu", "#666666"))

            # Afficher le résultat joliment
            st.markdown(f"""
                <div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white; font-size:24px;'>
                    🧬 Résultat : <strong>{stade}</strong>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"❌ Erreur API : {response.json().get('error', 'Erreur inconnue')}")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la lecture ou de l'envoi : {str(e)}")
