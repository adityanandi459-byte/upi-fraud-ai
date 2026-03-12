from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/fraud_detection_model.pkl")

# Store history
history = []

@app.route("/")
def home():
    return render_template("index.html", history=history, risk=0)

@app.route("/predict", methods=["POST"])
def predict():

    global history

    try:
        hour = float(request.form["hour"])
        day = float(request.form["day"])
        month = float(request.form["month"])
        year = float(request.form["year"])
        amount = float(request.form["amount"])

        # Model expects 30 features
        features = np.zeros(30)

        features[0] = hour
        features[1] = day
        features[2] = month
        features[3] = year
        features[29] = amount

        features = features.reshape(1,-1)

        probability = model.predict_proba(features)[0][1]

        # Demo fraud simulation for presentation
        if amount > 10000:
            probability = 0.9
        elif amount > 5000:
            probability = 0.65
        elif amount > 2000:
            probability = 0.45

        prediction = 1 if probability > 0.5 else 0
        risk = round(probability * 100,2)

        result = "Fraud" if prediction == 1 else "Legitimate"

        history.append({
            "amount": amount,
            "result": result,
            "risk": risk
        })

        return render_template(
            "index.html",
            prediction=result,
            risk=risk,
            history=history
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)