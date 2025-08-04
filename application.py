from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/Ridge.pkl','rb'))
std_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")  # <-- Render the main HTML page

# ...existing code...

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        classes = float(request.form['Classes'])
        region = float(request.form['Region'])

        # Use the scaler instance, not the class
        new_data_scaled = std_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template("predction_form.html", result=result[0])
    else:
        return render_template("predction_form.html")

if __name__ == '__main__':
    app.run(debug=True)