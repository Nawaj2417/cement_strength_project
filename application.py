from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application


# Route for the homepage
@app.route('/')
def index():
    return render_template("index.html")

# Route for single data point prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('form.html')
        

    else:
            data = CustomData(  
            # Extract form data and ensure correct data types
            Cement = float(request.form.get('Cement')),
            Blast_Furnace_Slag = float(request.form.get('Blast Furnace Slag')),
            Fly_Ash = float(request.form.get('Fly Ash')),
            Water = float(request.form.get('Water')),
            Superplasticizer = float(request.form.get('Superplasticizer')),
            Coarse_Aggregate = float(request.form.get('Coarse Aggregate')),
            Fine_Aggregate = float(request.form.get('Fine Aggregate')),
            Age_day = float(request.form.get('Age (day)'))
            )

            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)

            results = round(pred[0],2)

            return render_template("form.html",final_result = results)
    
if __name__=="__main__":
     app.run(host='0.0.0.0', debug=True)    


