from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#Load model_prediction
mul_reg = open("multiple_regression_model.pkl", "rb")
ml_model = joblib.load(mul_reg)


@app.route('/') #@ denotes decorator which allows function property of class it preceeds to be dynamically altered.It tells flask what URL user has to browse in order for function need to be called
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        print(request.form.get('NewYork'))
        try:
            NewYork = float(request.form['NewYork'])
            California = float(request.form['California'])
            Florida = float(request.form['Florida'])
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])
            pred_args = [NewYork,California,Florida,RnD_Spend,Admin_Spend,Market_Spend]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            #mul_model = ("multiple_linear_model.pkl","rb")
            #mul_model = joblib.load(mul_model)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except valueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ =="__main__":
    app.run()
