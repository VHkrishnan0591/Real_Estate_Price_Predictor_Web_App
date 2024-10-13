from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from real_estate_price_predictor.pipeline.stage_06_prediction import PredictPipeline

application=Flask(__name__)

app=application

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data_input ={
        'MSSubClass' : [int(request.form.get('MSSubClass'))],
        'LotArea' : [int(request.form.get('LotArea'))],
        'Neighborhood' : [request.form.get('Neighborhood')],
        'OverallQual' : [int(request.form.get('OverallQual'))],
        'OverallCond'  : [int(request.form.get('OverallCond'))],
        'YearBuilt' : [int(request.form.get('YearBuilt'))],
        'YearRemodAdd' : [int(request.form.get('YearRemodAdd'))],
        'BsmtFinType1' : [request.form.get('BsmtFinType1')],
        'BsmtFinSF1' : [int(request.form.get('BsmtFinSF1'))],
        'BsmtUnfSF' : [int(request.form.get('BsmtUnfSF'))],
        'TotalBsmtSF' : [int(request.form.get('TotalBsmtSF'))],
        'HeatingQC' : [request.form.get('HeatingQC')],
        '1stFlrSF' : [int(request.form.get('1stFlrSF'))],
        'GrLivArea' : [int(request.form.get('GrLivArea'))],
        'BsmtFullBath' : [int(request.form.get('BsmtFullBath'))],
        'KitchenQual' : [request.form.get('KitchenQual')],
        'TotRmsAbvGrd' : [int(request.form.get('TotRmsAbvGrd'))],
        'Fireplaces' : [int(request.form.get('Fireplaces'))],
        'FireplaceQu' : [request.form.get('FireplaceQu')],
        'GarageYrBlt' : [int(request.form.get('GarageYrBlt'))],
        'GarageCars' : [int(request.form.get('GarageCars'))],
        'WoodDeckSF' : [int(request.form.get('WoodDeckSF'))],
        'OpenPorchSF' : [int(request.form.get('OpenPorchSF'))],
        'YrSold': [int(request.form.get('YrSold'))]
        }

        dataframe = pd.DataFrame(data_input)
        obj = PredictPipeline()
        return render_template('index.html',results=obj.main(dataframe))
if __name__=="__main__":
    app.run(host="0.0.0.0")  
