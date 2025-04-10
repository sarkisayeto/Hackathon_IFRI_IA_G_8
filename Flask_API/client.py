import requests

url = 'http://127.0.0.1:5000/predict'

# Exemple de données (doivent correspondre aux colonnes attendues)
data = {
 "contour_regulier_rein_gauche": "non",
  "contour_regulier_rein_droit": "oui",
  "calcul_renal": "non",
  "kyste": "oui",
  "differenciation_des_reins": "différencié",
  "echogenicite": "normale",
  "personnels_medicaux_diabete_1": 0,
  "personnels_medicaux_diabete_2": 1,
  "personnels_familiaux_diabete": 1,
  "duree_diabete_2_mois": 15,
  "pathologies_retinopathie_diabetique": 0,
  "causes_majeure_apres_diagnostic_diabete": 1,
  "personnels_medicaux_hta": 1,
  "personnels_familiaux_hta": 1,
  "duree_hta_mois": 36,
  "pathologies_retinopathie_hypertensive": 1,
  "causes_majeure_apres_diagnostic_hta": 0,
  "pathologies_glaucome": 0,
  "personnels_medicaux_pathologies_virales_hb_hc_hiv": 0,
  "enquete_sociale_phytotherapie_traditionnelle": 1,
  "enquete_sociale_pec_oui": 0,
  "enquete_sociale_pec_non": 1,
  "creatinine_mg_l": 22.5,
  "age": 100,
  "symptomes_fievre": 1,
  "symptomes_cephalees": 0,
  "symptomes_douleur_lombaire": 1,
  "symptomes_dysurie": 0,
  "symptomes_oligurie": 1,
  "symptomes_diarrhee": 0,
  "symptomes_douleur_thoracique": 1,
  "na_meq_l": 138,
  "k_meq_l": 4.2,
  "ca_meq_l": 2.3,
  "cl_meq_l": 102,
  "p_meq_l": 1.1,
  "hb_g_dl": 12.5,
  "hte_pourcent": 40,
  "vgm_fl": 90,
  "tcmh_pg": 50,
  "ccmh_pourcent": 33,
  "plaquettes_g_l": 250,
  "poul_bpm": 75,
  "poids_kg": 90,
  "personnels_medicaux_irc": 1
}
# Envoie en JSON (Content-Type = application/json automatiquement)
response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Résultat:", response.json())
