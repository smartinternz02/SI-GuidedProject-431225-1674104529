import numpy as np 
import pandas as pd
from flask import Flask, render_template, request
import pickle
model = pickle.load(open(r'D:\HDI\Flask\HDI.pkl','rb')) 
app = Flask(__name__) 

@app.route('/')
def home():
    return render_template('home.html') 
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    

    input_features = [float(x) for x in request.form.values()]
    print(input_features)
    features_value = [np.array(input_features)]
    
    features_name = ['Country','Life expectancy','Mean years of schooling','Gross national income (GNI) per capita','Internet Users']
    
    df = pd.DataFrame(features_value, columns=features_name)
    model = pickle.load(open(r'D:\HDI\Flask\HDI.pkl', 'rb')) 
    output = model.predict(df)
    print(round(output[0][0],2))
   
    y_pred =round(output[0][0],2)
    if(y_pred >= 0.3 and y_pred <= 0.4) :
        return render_template("resultnew.html",prediction_text = 'Low HDI'+ str(y_pred))
    elif(y_pred >= 0.4 and y_pred <= 0.7) :
        return render_template("resultnew.html",prediction_text = 'Medium HDI '+str(y_pred))
    elif(y_pred >= 0.7 and y_pred <= 0.8) :
        return render_template("resultnew.html",prediction_text = 'High HDI'+str(y_pred))
    elif(y_pred >= 0.8 and y_pred <= 0.94) :
        return render_template("resultnew.html",prediction_text = 'Very High HDI'+str(y_pred))
    else :
        return render_template("resultnew.html",prediction_text = 'The given values do not match the range of values of the model.Try giving the values in the mnetioned range'+str(y_pred))
    
    
    
    return render_template('resultnew.html', prediction_text=y_pred)


if __name__ == '__main__':  
    app.run(debug=False,port=5000)
