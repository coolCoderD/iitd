from flask import Flask, jsonify
from flask_cors import CORS
from src.routes.report_route import report_bp
from src.routes.patientChat import patient_chat_bp
from src.routes.doctorChat import doctor_chat_bp
from werkzeug.middleware.proxy_fix import ProxyFix

import sys
print(sys.path)


# Initialize Flask app
app = Flask(__name__)

# Middleware setup
CORS(app)

# Static files
app.wsgi_app = ProxyFix(app.wsgi_app)  # Equivalent to serving static files
app.static_folder = 'public'

#  Reports routes
app.register_blueprint(report_bp, url_prefix="/api/v1/reports")

# Patient chat routes
app.register_blueprint(patient_chat_bp, url_prefix="/api/v1/patientChat")

# Doctor chat routes
app.register_blueprint(doctor_chat_bp, url_prefix="/api/v1/doctorChat")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})