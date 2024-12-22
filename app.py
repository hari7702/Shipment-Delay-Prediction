from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
from flask_cors import CORS

try:
    with open("random_forest_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

app = Flask(__name__, static_folder="static")
CORS(app)

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        input_df[["Distance (km)", "Delivery Delay"]] = scaler.transform(
            input_df[["Distance (km)", "Delivery Delay"]]
        )
        prediction = model.predict(input_df)
        prediction_label = "Delayed" if prediction[0] == 1 else "On Time"
        return jsonify({"prediction": prediction_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)