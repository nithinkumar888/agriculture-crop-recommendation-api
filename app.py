
import os
import hmac
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# API security key from Render environment variables
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "")

# Load the crop recommendation dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

dataset = pd.read_csv(DATASET_PATH)

# Features and target
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = dataset[FEATURES]
y = dataset["label"]

# Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=10,
    criterion="entropy",
    random_state=0
)

model.fit(X, y)

print("Crop Recommendation model loaded successfully!")


# Check API key
def check_api_key():
    provided_key = request.headers.get("X-API-Key", "")

    if not API_SECRET_KEY:
        return False

    return hmac.compare_digest(provided_key, API_SECRET_KEY)


# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Agriculture Crop Recommendation API is running"
    })


# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# Crop prediction
@app.route("/predict", methods=["POST"])
def predict():

    if not check_api_key():
        return jsonify({
            "error": "Unauthorized. Invalid API key."
        }), 401

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please provide valid JSON input."
        }), 400

    try:
        values = []

        for feature in FEATURES:
            if feature not in data:
                return jsonify({
                    "error": "Missing input: " + feature
                }), 400

            value = float(data[feature])

            if not np.isfinite(value):
                return jsonify({
                    "error": "Invalid input: " + feature
                }), 400

            values.append(value)

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = model.predict(input_data)[0]

        return jsonify({
            "success": True,
            "recommended_crop": str(prediction)
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "All seven inputs must be valid numbers."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "An error occurred while predicting the crop."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
