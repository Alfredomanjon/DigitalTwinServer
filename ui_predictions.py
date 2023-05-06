from flask import Blueprint, render_template, request, make_response
from flask import render_template, url_for
import tensorflow as tf
from keras.models import load_model
from prophet.serialize import model_to_json, model_from_json
from IPython.display import HTML
import numpy as np
import pandas as pd

bp = Blueprint("ui_predictions", __name__, url_prefix="/ui-predictions")

mi_lista = []
prophet_dates = []


@bp.route("/", methods=("GET", "POST"))
def predict():
    if request.method == "GET":
        return render_template("base.html")


@bp.route("/lstm-predict", methods=("GET", "POST"))
def lstmPredict():
    if request.method == "GET":
        print("ENTRO EN GET")
        if len(mi_lista) == 4:
            model = load_model("models/LSTM")
            df = pd.DataFrame.from_dict(mi_lista)
            df = df.astype(float)
            Input = [
                np.array([df[["carga_radar_base"]].values]),
                np.array(
                    [
                        df[
                            ["carga_radar_1", "distancia_radar_1", "duracion_radar_1"]
                        ].values
                    ]
                ),
                np.array(
                    [
                        df[
                            ["carga_radar_2", "distancia_radar_2", "duracion_radar_2"]
                        ].values
                    ]
                ),
                np.array(
                    [
                        df[
                            ["carga_radar_3", "distancia_radar_3", "duracion_radar_3"]
                        ].values
                    ]
                ),
                np.array(
                    [
                        df[
                            ["carga_radar_4", "distancia_radar_4", "duracion_radar_4"]
                        ].values
                    ]
                ),
                np.array(
                    [
                        df[
                            ["carga_radar_5", "distancia_radar_5", "duracion_radar_5"]
                        ].values
                    ]
                ),
            ]
            print(Input)
            predict_res = model.predict(Input)
            mi_lista.pop(0)
            return render_template(
                "lstm-predict-response.html", data=str(round(predict_res[0][0], 2))
            )
        else:
            return render_template("lstm-predict.html", lista=mi_lista)
    if request.method == "POST":
        print("ENTRO EN POST")
        new_weypoint = {
            "carga_radar_base": request.form["carga_radar_base"],
            "carga_radar_1": request.form["carga_radar_1"],
            "distancia_radar_1": request.form["distancia_radar_1"],
            "duracion_radar_1": request.form["duracion_radar_1"],
            "carga_radar_2": request.form["carga_radar_2"],
            "distancia_radar_2": request.form["distancia_radar_2"],
            "duracion_radar_2": request.form["duracion_radar_2"],
            "carga_radar_3": request.form["carga_radar_3"],
            "distancia_radar_3": request.form["distancia_radar_3"],
            "duracion_radar_3": request.form["duracion_radar_3"],
            "carga_radar_4": request.form["carga_radar_4"],
            "distancia_radar_4": request.form["distancia_radar_4"],
            "duracion_radar_4": request.form["duracion_radar_4"],
            "carga_radar_5": request.form["carga_radar_5"],
            "distancia_radar_5": request.form["distancia_radar_5"],
            "duracion_radar_5": request.form["duracion_radar_5"],
        }
        mi_lista.append(new_weypoint)
        return render_template("lstm-predict.html", lista=mi_lista)
    return render_template("lstm-predict.html", lista=mi_lista)


@bp.route("/prophet-predict", methods=("GET", "POST"))
def prophetPredict():
    if request.method == "GET":
        if len(prophet_dates) >= 1:
            print("ENTRO")
            fechas = pd.DataFrame({"ds": prophet_dates})
            with open("./models/Prophet/prophet_model.json", "r") as fin:
                m = model_from_json(fin.read())
            predicciones = m.predict(fechas)
            while prophet_dates:
                prophet_dates.pop()
            return predicciones.to_html(classes="table table-stripped")
        else:
            return render_template("prophet-predict.html")
    if request.method == "POST":
        prophet_dates.append(request.form["date_input_1"])
        return render_template("prophet-predict.html", dateList=prophet_dates)
