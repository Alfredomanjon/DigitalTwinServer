import os

from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint
import api_predictions
import ui_predictions

# create and configure the app
app = Flask(__name__)

# config flask app instance
app.config.from_mapping(
    SECRET_KEY="dev",
    DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
)

# swagger specific
SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={"app_name": "Seans-Python-Flask-REST-Boilerplate"}
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass


@app.route("/")
def main():
    return "Hello, World!"


app.register_blueprint(api_predictions.bp)
app.register_blueprint(ui_predictions.bp)

if __name__ == "__main__":
    app.run(debug=True)
