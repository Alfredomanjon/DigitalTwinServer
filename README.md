# DigitalTwinBA API
En este proyecto se encuentra la implementación del servidor flask de DigitalTwinBA, con distintos modelos de predicción para calcular la carga de tráfico en la ciudad de Madrid.

Para poder utilizar la api en local, es necesario seguir los siguientes pasos:
- Instalar las librerías necesarias utilizando el comando `pip install -t requirements.txt`
- Con las librerías instaladas, ejecutar el servidor utilizando el comando `flask run`

## ¿Que incluye esta API?
![Funcionalidades de la API](https://drive.google.com/uc?export=view&id=1ecq2c8OHuP_hIBmL7cbYot6dLABLI21A)

Esta API incluye dos productos diferenciados:
- API Predictions: Se trata de una API genérica donde se pueden realizar peticiones REST para utilizar los distintos modelos. Cuenta con swagger una libería donde se pueden probar los distintos modelos a modo de documentación y playground con ejemplos de prueba. Los modelos disponibles para utilizar en API Predictions son LSTM, Prophet, Lasso y STGNN.
- UI Predictions: Proporciona la capacidad de utilizar algunos modelos mediante una interfaz gráfica basada en 'templates' donde no es necesario tener conocimientos técnicos para poder utilizar los distintos modelos. Los modelos disponibles para utilizar en UI Predictions son LSTM y Prophet. 

## Infraestructura Escalable
La arquitectura base de la API incluye un servidor flask que permite realizar peticiones REST a los distintos modelos. Estos modelos se encuentran almacenados en MLFlow, que alamcena todas las versiones de los modelos. 

![Estructura de la API](https://drive.google.com/uc?export=view&id=1D3vAkVDn3dfeXUbbWjWJ94H-VovRioV6)

