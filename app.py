# Load librairies
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json
from sklearn.neighbors import NearestNeighbors
import shap

# Importation des modèles:
model = joblib.load('loan_scorer.pkl')
threshold = joblib.load('threshold_model.pkl')

# Importation des données:
X_test = joblib.load('X_test_sample_v2.csv')
X_train = joblib.load('X_train_sample_v2.csv')
y_test = joblib.load('y_test_sample_v2.csv')
y_train = joblib.load('y_train_sample_v2.csv')

x_train_sample = X_train.iloc[0: 100]
y_train_sample = y_train.iloc[0: 100]

X = pd.concat([X_test, X_train], axis=0)
y = pd.concat([y_test, y_train], axis=0)
preproc_cols = X_train.columns

# Instantiation de l'objet app de Flask:
app = Flask(__name__)

@app.route("/")
def index():
    return "Loan API is running correctly"


# Récupérer tous les ids des utilisateurs:
@app.route('/app/id/', methods=["GET"])
def ids_list() -> dict:
    """
    Cette route permet de retourner la liste des IDs des utilisateurs de notre dataset.

    Returns:
        - Elle retourne la liste des ids sous forme d'un json.
    """
    try:
        # Extraction de la liste des "SK_ID_CURR" depuis le X_test:
        customers_id_list = pd.Series(list(X.index.sort_values()))
        # Convertir la Série en JSON:
        customers_id_list_json = json.loads(customers_id_list.to_json())
        # Retourner le résultat:
        return jsonify({'status': 'ok',
                        'data': customers_id_list_json})
    except Exception as exp:
        return {
            "status": "ko",
            "error": f"Erreur: {exp}"
        }


@app.route('/app/data_cust/')
def selected_cust_data() -> dict:
    """
    Cette route retourne les données d'un utilisateur par son ID.

    Arguments:
        - l'id de l'utilisateur.
    
    Returns:
        - Les données de l'utilisateur sous format JSON.
    """
    try:
        id_utilisateur = int(request.args.get('SK_ID_CURR'))
        # Récupération des données d'INPUT:
        x_cust = X.loc[id_utilisateur: id_utilisateur]
        # Récupération des données d'OUTPUT:
        y_cust = y.loc[id_utilisateur: id_utilisateur]
        # Conversion de la série pandas en JSON:
        data_x_json = json.loads(x_cust.to_json())
        y_cust_json = json.loads(y_cust.to_json())
        # Retourner le résultat sous format JSON:
        return jsonify({'status': 'ok',
                        'y_cust': y_cust_json,
                        'data': data_x_json})
    except Exception as exep:
        return {
            "status": "ko",
            "error": f"Erreur: {exep}"
        }


def get_df_neigh(id_utilisateur: int) -> tuple:
    """
    Cette fonction permet de retrouver les 20 plus proches voisins d'un utilisateur par son ID.

    Arguments:
        - id_utilisateur: l'Id de l'utilisateur.
    
    Returns:
        - Les données X et y des 20 plus proches voisins sous format pandas.
    """
    # Création et entrainement du modèle NearestNeighbors:
    NN = NearestNeighbors(n_neighbors=20)
    NN.fit(X_train)
    X_cust = X.loc[id_utilisateur: id_utilisateur]
    idx = NN.kneighbors(X=X_cust,
                        n_neighbors=20,
                        return_distance=False).ravel()
    nearest_cust_idx = list(X_train.iloc[idx].index)
    # Les données X et le Y des 20 plus proches voisins:
    x_neigh = X_train.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]

    return x_neigh, y_neigh


@app.route('/app/neigh_cust/')
def neigh_cust() -> dict:
    """
    Cette route permet de retourner pour un utilisateur donné (id) ses 20 plus proches voisins.

    Arguments:
        - l'id de l'utilisateur.
    Returns:
        - les données X, y des 20 plus proches voisins sous format JSON.
    """
    try:
        # Récupération de l'id utilisateur:
        id_utilisateur = int(request.args.get('SK_ID_CURR'))
        # Execution du KNN:
        data_neigh, y_neigh = get_df_neigh(id_utilisateur)
        # Transformation en JSON:
        data_neigh_json = json.loads(data_neigh.to_json())
        y_neigh_json = json.loads(y_neigh.to_json())
        # Retourner les données sous format JSON avec le statut:
        return jsonify({'status': 'ok',
                        'y_neigh':  y_neigh_json,
                        'data_neigh': data_neigh_json},  # 'x_cust': x_cust_json},
                    )
    except Exception as exep:
        return {
            "status": "ok",
            "error": f"Erreur: {exep}"
        }


# Trouver les 500 plus proches voisins:
def get_df_fivehundred_neigh(id_utilisateur):
    """
    Cette fonction permet de retrouver les 500 plus proches voisins d'un utilisateur par son ID.

    Arguments:
        - id_utilisateur: l'Id de l'utilisateur.
    
    Returns:
        - Les données X et y des 500 plus proches voisins sous format pandas.
    """
    # fit nearest neighbors among the selection
    fivehundred_nn = NearestNeighbors(n_neighbors=500)  # len(X_train)
    fivehundred_nn.fit(X_train)  # X_train_NN
    X_cust = X.loc[id_utilisateur: id_utilisateur]   # X_test
    idx = fivehundred_nn.kneighbors(X=X_cust,
                                 n_neighbors=500,  # len(X_train)
                                 return_distance=False).ravel()
    nearest_cust_idx = list(X_train.iloc[idx].index)
    # data and target of neighbors
    # ----------------------------
    x_fivehundred_neigh = X_train.loc[nearest_cust_idx, :]
    y_fivehundred_neigh = y_train.loc[nearest_cust_idx]
    return x_fivehundred_neigh, y_fivehundred_neigh, X_cust


@app.route('/app/fivehundred_neigh/')
def fivehundred_neigh() -> dict:
    """
    Cette route retourne les 500 plus proches voisins d'un utilisateur par son ID.

    Arguments:
        - l'id de l'utilisateur.
    Returns:
        - les données X et Y des 500 plus proches voisins de l'utilisateur.
    """
    try:
        # Récupérer l'id de l'utilisateur:
        id_utilisateur = int(request.args.get('SK_ID_CURR'))
        # return the nearest neighbors
        x_fivehundred_neigh, y_fivehundred_neigh, x_customer = get_df_fivehundred_neigh(id_utilisateur)
        # Converting the pd.Series to JSON
        x_fivehundred_neigh_json = json.loads(x_fivehundred_neigh.to_json())
        y_fivehundred_neigh_json = json.loads(y_fivehundred_neigh.to_json())
        x_customer_json = json.loads(x_customer.to_json())
        # Returning the processed data
        return jsonify({'status': 'ok',
                        'X_fivehundred_neigh': x_fivehundred_neigh_json,
                        'X_custom': x_customer_json,
                        'y_fivehundred_neigh': y_fivehundred_neigh_json})
    except Exception as exep:
        return {
            "status": "ko",
            "error": f"Error: {exep}"
        }


@app.route('/app/shap_val/')
def shap_value() -> dict:
    """
    Cette route permet de calculer les valeurs SHAP pour un utilisateur et ses 20 plus proches voisins.

    Arguments:
        - Id de l'utilisateur dans les args.
    
    Returns:
        - Les valeurs SHAP.
    """
    try:
        # Récupération de l'id de l'utilisateur:
        id_utilisateur = int(request.args.get('SK_ID_CURR'))
        # Récupération des 20 plus proches voisins:
        X_neigh, y_neigh = get_df_neigh(id_utilisateur)
        # L'utilisateur actuel:
        X_cust = X.loc[id_utilisateur: id_utilisateur]
        # Préparation des valeurs SHAP pour les voisins et l'utilisateur:
        shap.initjs()
        # Création d'un arbre d'explication:
        explainer = shap.TreeExplainer(model)
        # Valeurs attendues (Expected Values):
        expected_vals = pd.Series(list(explainer.expected_value))
        # Calcul des valeurs SHAP de l'utilisateur sélectionné:
        shap_vals_cust = pd.Series(list(explainer.shap_values(X_cust)[1]))
        # Calcul des valeurs SHAP des voisins:
        shap_val_neigh_ = pd.Series(list(explainer.shap_values(X_neigh)[1]))
        # Conversion à un formation JSON:
        X_neigh_json = json.loads(X_neigh.to_json())
        expected_vals_json = json.loads(expected_vals.to_json())
        shap_val_neigh_json = json.loads(shap_val_neigh_.to_json())
        shap_vals_cust_json = json.loads(shap_vals_cust.to_json())
        # Retourner le résultat sous format JSON:
        return jsonify({'status': 'ok',
                        'X_neigh_': X_neigh_json,
                        'shap_val_neigh': shap_val_neigh_json,  # double liste
                        'expected_vals': expected_vals_json,  # liste
                        'shap_val_cust': shap_vals_cust_json})  # double liste
    except Exception as exep:
        return {
            "status": "ko",
            "error": f"Error: {exep}"
        }


@app.route('/app/customer_score/')
def scoring_cust() -> dict:
    """
    Cette route permet de faire le scoring (prédiction) pour un utilisateur donné.

    Arguments:
        - ID de l'utilsateur.

    Returns:
        - Le score pour cette utilisateur avec le seuil.
    """
    # Récupération de l'id de l'utilisateur:
    customer_id = int(request.args.get('SK_ID_CURR'))
    # Récupération des données X de l'utilisateur:
    X_cust = X.loc[customer_id:customer_id]
    # Scoring en faisant un predict:
    score_cust = model.predict_proba(X_cust)[:, 1][0]
    # Retour du score:
    return jsonify({'status': 'ok',
                    'SK_ID_CURR': customer_id,
                    'score': score_cust,
                    'thresh': threshold})


@app.route('/app/feat/')
def features() -> dict:
    """
    Cette route permet de retourner la liste des features de notre dataset.

    Returns:
        - Liste des features sous format JSON.
    """
    try:
        feat = X_test.columns
        f = pd.Series(feat)
        # Convertir les données en format JSON:
        feat_json = json.loads(f.to_json())
        # Retourner les données traitées:
        return jsonify({'status': 'ok',
                        'data': feat_json})
    except Exception as exep:
        return {
            "status": "ko",
            "error": f"Error: {exep}"
        }


@app.route('/app/feat_imp/')
def send_feat_imp():
    """
    Cette route permet de retourner l'importance des variables (feature importance) de notre modèle.

    Returns:
        - Les feature importance sous format JSON.
    """
    try:
        feat_imp = pd.Series(model.feature_importances_,
                            index=X_test.columns).sort_values(ascending=False)
        # Conversion depuis le format pandas Series à JSON:
        feat_imp_json = json.loads(feat_imp.to_json())
        # Retourner les résultats sous format JSON avec un statut:
        return jsonify({'status': 'ok',
                        'data': feat_imp_json})
    except Exception as exep:
        return {
            "status": "ko",
            "error": f"Error: {exep}"
        }


if __name__ == "__main__":
    app.run()
