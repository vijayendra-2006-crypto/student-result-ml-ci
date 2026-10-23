
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON data is required"}), 400

    for feature in FEATURES:
        if feature not in data:
            return jsonify({
                "error": f"Missing field: {feature}"
            }), 400

    try:
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        model = joblib.load(MODEL_PATH)
        prediction_code = int(model.predict(input_data)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except FileNotFoundError:
        return jsonify({"error": "Trained model not found"}), 500

    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input values"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
