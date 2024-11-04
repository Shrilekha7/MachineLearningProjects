from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('house_price1.pkl', 'rb') as file:
    lr = pickle.load(file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form submission
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    sqft_living = int(request.form['sqft_living'])
    sqft_lot = int(request.form['sqft_lot'])
    floors = int(request.form['floors'])
    waterfront = int(request.form['waterfront'])
    view = int(request.form['view'])
    condition = int(request.form['condition'])

    # Prepare the data for prediction
    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition]])
    
    # Make a prediction
    predicted_price = lr.predict(features)[0]
    

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
