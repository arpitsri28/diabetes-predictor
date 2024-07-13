from flask import Flask, request, render_template, url_for
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__, template_folder="templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True


model = pickle.load(open('diabetes_model.pkl', 'rb'))


scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
   return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   try:
       input_features = [float(x) for x in request.form.values()]
       features_value = np.array(input_features)
       features_value = features_value.reshape(1, -1)
       features_value = scaler.transform(features_value)
       prediction = model.predict(features_value)
       output = prediction[0]
       prediction_text = 'Diabetes Positive' if output == 1 else 'Diabetes Negative'
       return render_template('index.html', output=prediction_text)
   except Exception as e:
       return render_template('index.html', output=str(e))




if __name__ == "__main__":
   app.run(debug=True)
  
