from flask import Blueprint, render_template, request, make_response
from flask import render_template, url_for
import tensorflow as tf
from keras.models import load_model
from prophet.serialize import model_to_json, model_from_json
from IPython.display import HTML
import numpy as np
import pandas as pd
import json
import mlflow

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
        model = load_model("models/LSTM")
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
        predict_res = model.predict(Input)
        return str(predict_res[0])


@bp.route("/prophet", methods=("GET", "POST"))
def prophetPredict():
    if request.method == "POST":
        with open("models/Prophet/prophet_model.json", "r") as fin:
            loaded_model = model_from_json(fin.read())  # Load model

        json_data = request.get_json(force=True)
        dates = []

        for item in range(len(json_data)):
            dates.append(json_data["fecha_" + str(item + 1)])
        fechas = pd.DataFrame({"ds": list(dates)})
        predicciones = loaded_model.predict(fechas)
        result = predicciones.to_json(orient="split")
        parsed = json.loads(result)
        return json.dumps(parsed, indent=4)
