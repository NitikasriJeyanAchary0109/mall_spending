from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans_model.pkl")

# Cluster to spending category
cluster_mapping = {
    0: "Very Low Spender",
    1: "Low Spender",
    2: "Moderate Spender",
    3: "High Spender",
    4: "Premium Spender"
}

# HTML + CSS in one file
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Spending Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(120deg, #ff9a9e, #fad0c4);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            width: 350px;
            text-align: center;
            box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
        }
        input {
            padding: 10px;
            width: 90%;
            margin: 10px 0;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        button {
            padding: 10px 20px;
            border: none;
            background: #ff758c;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #ff5e78;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            font-weight: bold;
            color: #444;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Spending Predictor</h1>
        <form method="POST">
            <input type="number" name="income" placeholder="Annual Income (k$)" required>
            <input type="number" name="score" placeholder="Spending Score (1-100)" required>
            <br>
            <button type="submit">Predict</button>
        </form>

        {% if result %}
            <div class="result">
                {{ result }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        income = float(request.form["income"])
        score = float(request.form["score"])

        # Scale input
        data = scaler.transform([[income, score]])

        # Predict cluster
        cluster = model.predict(data)[0]

        # Convert to category
        category = cluster_mapping.get(cluster, "Unknown")

        result = f"Customer is a {category}"

    return render_template_string(html_template, result=result)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

