from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Chargement des objets enregistrés
model_info = joblib.load("D:/Hackathon_IFRI_IA_G_8/src/arbre_complet.pkl")
model = model_info['model']
expected_features = model_info['features']
scaler = model_info['scaler']
encoder = model_info['encoder']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'Aucune donnée JSON reçue ou format invalide'}), 400

        input_data = pd.DataFrame([json_data])
        
        # Vérifier que toutes les colonnes attendues sont là
        if not all(col in input_data.columns for col in expected_features):
            return jsonify({
                'error': 'Les colonnes du fichier ne correspondent pas aux colonnes attendues',
                'attendues': expected_features,
                'reçues': input_data.columns.tolist()
            }), 400

        input_data = input_data[expected_features]

        encoded = encoder.transform(input_data)
        print("✅ Encodage terminé.")
        scaled = scaler.transform(encoded)
        prediction = model.predict(scaled)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
