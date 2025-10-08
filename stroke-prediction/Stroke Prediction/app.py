from flask import Flask, request, jsonify, render_template
import pandas as pd
from joblib import load
from flask_cors import CORS

# Load the trained model
model = load('./stroke_prediction_model.joblib')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        prediction = model.predict(df)[0]

        print(f"Prediction: {prediction}")

        return jsonify({"stroke": int(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
