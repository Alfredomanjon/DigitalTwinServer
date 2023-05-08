# DigitalTwinBA API
En este proyecto se encuentra la implementación del servidor flask de DigitalTwinBA, con distintos modelos de predicción para calcular la carga de tráfico en la ciudad de Madrid.

Para poder utilizar la api en local, es necesario seguir los siguientes pasos:
- Instalar las librerías necesarias utilizando el comando `pip install -t requirements.txt`
- Con las librerías instaladas, ejecutar el servidor utilizando el comando `flask run`

## Arquitectura Base
La arquitectura base de la API incluye un servidor flask que permite realizar peticiones REST a los distintos modelos. Estos modelos se encuentran almacenados en MLFlow, que alamcena todas las versiones de los modelos. 

![Estructura de la API](https://drive.google.com/uc?export=view&id=14qo2FPlibDAHGNAZU1LamvnAa0gsxwt0)

## Infraestructura Escalable
Para que el servidor sea escalable, ...

## ¿Que incluye esta API?
