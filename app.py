from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

print("Saving model to:", os.getcwd())
pickle.dump(model, open(r"D:\ML-Project-2\model.pkl", "wb"))
print("Model saved successfully!")

# Load dataset for graphs and statistics
df = pd.read_csv("data.csv")

# Compute basic statistics
avg_studytime = round(df["studytime"].mean(), 2)
avg_price = round(df["price"].mean(), 2)
corr_value = round(df["studytime"].corr(df["price"]), 2)

def generate_plot():
    plt.figure()
    plt.scatter(df["studytime"], df["price"])
    plt.xlabel("Study Time")
    plt.ylabel("House Price")
    plt.title("Study Time vs Price")

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches="tight")
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    plot_img = generate_plot()

    if request.method == "POST":
        sqft = float(request.form["sqft"])
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])

        pred = model.predict([[sqft, bedrooms, bathrooms]])
        prediction = int(pred[0])

    return render_template(
        "index.html",
        prediction=prediction,
        avg_studytime=avg_studytime,
        avg_price=avg_price,
        corr_value=corr_value,
        graph=plot_img
    )

if __name__ == "__main__":
    app.run(debug=True)
