from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the cleaned dataset
dataset = pd.read_csv('carData.csv')

cols = ['name', 'fuel', 'seller_type', 'gear_box', 'owner', 'torque','production_year', 'km_driven', 'mileage', 'engine', 'max_power','seats']
# Extract unique values for form dropdowns
car_name = dataset['name'].unique()
production_year = dataset['production_year'].unique()
fuel = dataset['fuel'].unique()
gearbox = dataset['gear_box'].unique()
km_driven = dataset['km_driven'].unique()
mileage = dataset['mileage'].unique()
engine = dataset['engine'].unique()
max_power = dataset['max_power'].unique()
seats = dataset['seats'].unique()
seller_type = dataset['seller_type'].unique()
owner = dataset['owner'].unique()
torque = dataset['torque'].unique()

# Load the pre-trained model
model = pickle.load(open('ensemble_dev.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        try:
            # Convert data types
            data['production_year'] = int(data['production_year'])
            data['km_driven'] = float(data['km_driven'])
            data['mileage'] = float(data['mileage'])
            data['engine'] = float(data['engine'])
            data['max_power'] = float(data['max_power'])
            data['seats'] = float(data['seats'])

            # Extract features for prediction
            features = pd.DataFrame({
                'name': [data['name']],
                'fuel': [data['fuel']],
                'seller_type': [data['seller_type']],
                'gear_box': [data['gear_box']],
                'owner': [data['owner']],
                'torque': [data['torque']],
                'production_year': [data['production_year']],
                'km_driven': [data['km_driven']],
                'mileage': [data['mileage']],
                'engine': [data['engine']],
                'max_power': [data['max_power']],
                'seats': [data['seats']]
            })

             # One-hot encode categorical features
            features = pd.get_dummies(features)

            # Make prediction
            prediction = model.predict(features)
            output = round(prediction[0], 2)
            return jsonify({'predicted_price': output})
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

    return render_template('predict.html', car_names=car_name, fuels=fuel, sellers=seller_type, gearboxes=gearbox, owners=owner, torque=torque, prod_year=production_year, kmdriven=km_driven, mileage=mileage, engine=engine, maxpower=max_power, seat=seats)




if __name__ == '__main__':
    app.run(debug=True)


