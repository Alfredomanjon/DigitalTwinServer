from flask import Blueprint, render_template, request, make_response
from flask import render_template, url_for
import tensorflow as tf
from keras.models import load_model
from prophet.serialize import model_to_json, model_from_json
from IPython.display import HTML
import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import requests


bp = Blueprint("api_predictions", __name__, url_prefix="/api-predictions")


@bp.route("/", methods=("GET", "POST"))
def predict():
    if request.method == "GET":
        return "Bienvenido a la API de Digital Twin BA"


@bp.route("/lstm", methods=("GET", "POST"))
def lstmPredict():
    if request.method == "GET":
        return "GET A lstm"
    if request.method == "POST":
        lista_datos = []

        mlflow.set_tracking_uri("http://54.87.203.158:5000")

        try:
            experiment_id_path = requests.get(
                "http://54.87.203.158:5001/experiment/model-lstm/best/path"
            )
            response = experiment_id_path.json()
            loaded_model = mlflow.pyfunc.load_model(response["path"])
        except Exception as e:
            print("[ERROR] carga modelo MLFlow: ", e)
            loaded_model = load_model("models/LSTM")
            
        json_data = request.get_json(force=True)
        for item in range(len(json_data)):
            valores = json_data["tiempo_" + str(item + 1)]
            datos_tiempo = {
                "carga_radar_base": valores["carga_radar_base"],
                "carga_radar_1": valores["carga_radar_1"],
                "distancia_radar_1": valores["distancia_radar_1"],
                "duracion_radar_1": valores["duracion_radar_1"],
                "carga_radar_2": valores["carga_radar_2"],
                "distancia_radar_2": valores["distancia_radar_2"],
                "duracion_radar_2": valores["duracion_radar_2"],
                "carga_radar_3": valores["carga_radar_3"],
                "distancia_radar_3": valores["distancia_radar_3"],
                "duracion_radar_3": valores["duracion_radar_3"],
                "carga_radar_4": valores["carga_radar_4"],
                "distancia_radar_4": valores["distancia_radar_4"],
                "duracion_radar_4": valores["duracion_radar_4"],
                "carga_radar_5": valores["carga_radar_5"],
                "distancia_radar_5": valores["distancia_radar_5"],
                "duracion_radar_5": valores["duracion_radar_5"],
            }
            lista_datos.append(datos_tiempo)
        df = pd.DataFrame(lista_datos)
        df = df.astype(float)
        Input = [
            np.array([df[["carga_radar_base"]].values]),
            np.array(
                [df[["carga_radar_1", "distancia_radar_1", "duracion_radar_1"]].values]
            ),
            np.array(
                [df[["carga_radar_2", "distancia_radar_2", "duracion_radar_2"]].values]
            ),
            np.array(
                [df[["carga_radar_3", "distancia_radar_3", "duracion_radar_3"]].values]
            ),
            np.array(
                [df[["carga_radar_4", "distancia_radar_4", "duracion_radar_4"]].values]
            ),
            np.array(
                [df[["carga_radar_5", "distancia_radar_5", "duracion_radar_5"]].values]
            ),
        ]
        predict_res = loaded_model.predict(Input)
        return str(predict_res[0])


@bp.route("/prophet", methods=("GET", "POST"))
def prophetPredict():
    if request.method == "POST":
        dates = []

        mlflow.set_tracking_uri("http://54.87.203.158:5000")
        try:
            experiment_id_path = requests.get(
                "http://54.87.203.158:5001/experiment/model-prophet/best/path"
            )
            response = experiment_id_path.json()
            model = mlflow.sklearn.load_model(response["path"])
            loaded_model = model_from_json(json.dumps(model))
        except Exception as e:
            print("[ERROR] carga modelo MLFlow: ", e)
            with open("./models/Prophet/prophet_model.json", "r") as fin:
                loaded_model = model_from_json(fin.read())

        json_data = request.get_json(force=True)
        for item in range(len(json_data)):
            dates.append(json_data["fecha_" + str(item + 1)])

        fechas = pd.DataFrame({"ds": list(dates)})
        predicciones = loaded_model.predict(fechas)
        result = predicciones.to_json(orient="split")
        parsed = json.loads(str(result))
        return json.dumps(parsed, indent=4)


@bp.route("/lasso", methods=("GET", "POST"))
def lassoPredict():
    if request.method == "POST":
        # row = [4315, 1, 883, 6.0, 0.0, 0, 13, 16, 6, 7957]
        response = ''
        mlflow.set_tracking_uri("http://54.87.203.158:5000")

        try:
            experiment_id_path = requests.get(
                "http://54.87.203.158:5001/experiment/model-lasso/best/path"
            )
            response = experiment_id_path.json()
            print(response)
            loaded_model = mlflow.sklearn.load_model(response["path"])
            print(load_model)
        except Exception as e:
            print("[ERROR] carga modelo MLFlow: ", e)
            with open("models/Lasso/lasso_model.pkl", "rb") as f:
                loaded_model = pickle.load(f)

        json_data = request.get_json(force=True)    

        predicciones = loaded_model.predict([json_data])
        parsed = json.loads(str(predicciones))
        return json.dumps(parsed, indent=4)


@bp.route("/stgnn", methods=("GET", "POST"))
def stgnnPredict():
        loaded_model = load_model("models/STGNN")
        json_data = request.get_json(force=True)
        prediction = loaded_model.predict(json_data)
        parsed = json.loads(str(prediction))
        return json.dumps(parsed, indent=4)