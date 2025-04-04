
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            edu = int(request.form['education'])
            cg = int(request.form['capital_gain'])
            wh = int(request.form['hours_per_week'])

            new_data = [[age, edu, cg, wh]]
            transformed = scaler.transform(new_data)
            result = model.predict(transformed)[0]
            prediction = "Above 50K" if result == 1 else "50K or Below"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
