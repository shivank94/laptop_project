import numpy as np
from flask import Flask, render_template, request
import pickle
import pandas as pd
import dropdown_options
app = Flask(__name__, template_folder= 'template')

# Load the trained model
with open('xgboost.pkl', 'rb') as file:
    model = pickle.load(file)

#Load our encoder
with open('encoders.pkl', 'rb') as file2:
    encoders = pickle.load(file2)

# Define lists for categorical dropdowns (replace with actual unique values from your dataset)
company = dropdown_options.company_list
typenames = dropdown_options.typenames_list
opsys_list = dropdown_options.opsys_lists
memory_types = dropdown_options.memory_types_list
#gpu_brand = ['Intel', 'Nvidia', 'AMD']
#cpu_brand = ['AMD', 'Intel']
ram = dropdown_options.ram_list
memory_1_sto = dropdown_options.memory_1_sto_list
inches = dropdown_options.inches_list
resolution = dropdown_options.resolution_list
cpu_options = dropdown_options.cpu_options_list
gpu_options = dropdown_options.gpu_options_list

default_input = {
    'company': 'Dell',
    'product': 'Inspiron',
    'typename': 'Notebook',
    'inches': 15.6,
    'cpu': 'Intel Core i5 7200U',
    'ram(GB)': 8,
    'gpu': 'Intel UHD Graphics 620',
    'opsys': 'Windows 10',
    'weight(kg)': 2.0,
    'resolution': '1920x1080',
    'screentype': 'IPS',
    'touchscreen': 0,
    'cpu_freq(GHz)': 2.5,
    'memory_1_sto(GB)': 256,
    'memory_1_type': 'SSD',
    'memory_2_sto(GB)': 0,
    'memory_2_type': 'None',
    'cpu_brand': 'Intel',
    'gpu_brand': 'Intel'
}

categorical_columns = ['company', 'typename', 'cpu', 'gpu', 'opsys', 'resolution',
                       'memory_1_type', 'cpu_brand', 'gpu_brand']
feature_order = ['company', 'product', 'typename', 'inches', 'cpu', 'ram(GB)', 'gpu',
       'opsys', 'weight(kg)', 'resolution', 'screentype',
       'touchscreen', 'cpu_freq(GHz)', 'memory_1_sto(GB)', 'memory_1_type',
       'memory_2_sto(GB)', 'memory_2_type', 'cpu_brand', 'gpu_brand']

def preprocess_input(user_input):
    # Fill missing inputs
    complete_input = default_input.copy()
    complete_input.update(user_input)

    # Encode categorical columns
    for col in feature_order:
        if col in encoders:
            le = encoders[col]
            val = complete_input[col]
            if val in le.classes_:
                complete_input[col] = le.transform([val])[0]
            else:
                # handle unknown category
                complete_input[col] = np.nan  # or use most common class, etc.

    return pd.DataFrame([[complete_input[col] for col in feature_order]], columns=feature_order)

@app.route('/')
def home():
    return render_template('index.html', **get_common_context(), prediction_text=None)

def get_common_context():
    return dict(
        company=company,
        typenames=typenames,
        opsys_list=opsys_list,
        inches=inches,
        ram=ram,
        memory_types=memory_types,
        memory_1_sto=memory_1_sto,
        resolution=resolution,
        gpu_options=gpu_options,
        cpu_options=cpu_options
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = {
        'company': request.form['company'],
        'typename': request.form['typename'],
        'inches': float(request.form['inches']),
        'cpu': request.form['cpu'],
        'ram(GB)': int(request.form['ram']),
        'gpu': request.form['gpu'],
        'opsys': request.form['opsys'],
        'resolution': request.form['resolution'],
        'touchscreen': 1 if 'touchscreen' in request.form else 0,
        'memory_1_sto(GB)': int(request.form['memory_1_sto(GB)']),
        'memory_1_type': request.form['memory_1_type'],
        'cpu_brand': request.form['cpu_brand'],
        'gpu_brand': request.form['gpu_brand'],
    }

    input_df = preprocess_input(input_data)
    with open('input_df.pkl', 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(input_df, f)

    # Predict
    log_prediction = model.predict(input_df)[0]
    prediction = np.exp(log_prediction)
    output = round(float(prediction*97.11), 2)

    if output < 0:
        message = "Predicted Price is negative, values entered not reasonable"
    else:
        message = f"Predicted Price of the laptop is: ₹{output}"

    return render_template('index.html', **get_common_context(), prediction_text=message)

if __name__ == '__main__':
    app.run(debug=True)
