from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("gold_model.pkl")

@app.route('/api/gold_price', methods=['POST'])
def gold():
    
    open = float(request.form.get('open'))
    high = float(request.form.get('high'))
    low = float(request.form.get('low'))
    adjustclose = float(request.form.get('adjclose'))
    close = float(request.form.get('close'))
    volume = float(request.form.get('volume'))
    
    # Prepare the input for the model
    x = np.array([[open, high, low, adjustclose, close,volume]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result
    return {'price': round(prediction[0], 2)}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)