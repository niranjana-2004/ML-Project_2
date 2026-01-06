from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
df = pd.read_csv("data.csv")

# Basic statistics
avg_area = round(df["sqft_living"].mean(), 2)
avg_bedrooms = round(df["bedrooms"].mean(), 2)
avg_bathrooms = round(df["bathrooms"].mean(), 2)
avg_price = round(df["price"].mean(), 2)

corr_area_price = round(df["sqft_living"].corr(df["price"]), 2)
corr_bed_price = round(df["bedrooms"].corr(df["price"]), 2)
corr_bath_price = round(df["bathrooms"].corr(df["price"]), 2)

def create_graph(x_col, y_col, title, xlabel):
    plt.figure()
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(xlabel)
    plt.ylabel("Price")
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_str


# Generate images
graph_area = create_graph(
    "sqft_living",
    "price",
    "Living Area vs Price",
    "Living Area (sqft)"
)

graph_bed = create_graph(
    "bedrooms",
    "price",
    "Bedrooms vs Price",
    "Bedrooms"
)

graph_bath = create_graph(
    "bathrooms",
    "price",
    "Bathrooms vs Price",
    "Bathrooms"
)
html_code = """
<!DOCTYPE html>
<html>
<head>
    <title>House Performance Analysis</title>

    <style>
    body {
        font-family: Arial;
        background: #f0f2f5;
    }

    .container {
        width: 700px;
        margin: 40px auto;
        background: white;
        padding: 20px;
        border-radius: 10px;
    }

    input {
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
    }

    button {
        padding: 10px;
        width: 100%;
        background: #007BFF;
        color: white;
        border: none;
    }

    .graphs img {
        width: 100%;
        margin-bottom: 20px;
    }

    .stats {
        background: #e7f3ff;
        padding: 10px;
        margin-bottom: 20px;
    }

    .popup {
        display:none;
        position: fixed;
        top: 30%;
        left: 35%;
        width: 400px;
        background: white;
        padding: 20px;
        border: 2px solid #007BFF;
    }
    </style>

</head>

<body>

<div class="container">

<h2>Predict Final Grade & Performance</h2>

<form method="post">

<label>Area (sqft)</label>
<input type="number" name="sqft_living" required>

<label>Bedrooms</label>
<input type="number" name="bedrooms" required>

<label>Bathrooms</label>
<input type="number" name="bathrooms" required>

<button type="submit">Predict Final Grade</button>

</form>

{% if prediction %}
<script>
alert("Predicted Result: ₹{{ prediction }}");
</script>
{% endif %}

<h3>Statistics Summary</h3>

<div class="stats">
Average Area: {{ avg_area }}<br>
Average Bedrooms: {{ avg_bedrooms }}<br>
Average Bathrooms: {{ avg_bathrooms }}<br>
Average Price: ₹{{ avg_price }}<br><br>

Correlation (Area vs Price): {{ corr_area }}<br>
Correlation (Bedrooms vs Price): {{ corr_bed }}<br>
Correlation (Bathrooms vs Price): {{ corr_bath }}
</div>

<div class="graphs">
<h3>Graphs</h3>

<div class="graphs">
<h4>Area vs Price</h4>
<img src="data:image/png;base64,{{ graph_area }}">
</div>

<div class="graphs">
<h4>Bedrooms vs Price</h4>
<img src="data:image/png;base64,{{ graph_bed }}">
</div>

<div class="graphs">
<h4>Bathrooms vs Price</h4>
<img src="data:image/png;base64,{{ graph_bath }}">
</div>

</div>

</div>

</body>
</html>
"""
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        sqft = float(request.form["sqft_living"])
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])

        result = model.predict([[sqft, bedrooms, bathrooms]])
        prediction = int(result[0])

    return render_template_string(
        html_code,
        prediction=prediction,
        avg_area=avg_area,
        avg_bedrooms=avg_bedrooms,
        avg_bathrooms=avg_bathrooms,
        avg_price=avg_price,
        corr_area=corr_area_price,
        corr_bed=corr_bed_price,
        corr_bath=corr_bath_price,
        graph_area=graph_area,
        graph_bed=graph_bed,
        graph_bath=graph_bath
    )
if __name__ == "__main__":
    app.run(debug=True)
