from flask import Flask, request, render_template
import pickle
import pandas as pd
import joblib

app=Flask(__name__)
model=pickle.load(open('delhiweathernew.pkl','rb'))
scaler = joblib.load('scalernew.save')

@app.route('/')
def home():
    return render_template('weatheroptimized.html')

@app.route('/classify',methods=['POST'])
def classify():
    if request.method == 'POST':

        #Maximum Temperature    
        Maximum_Temperature=float(request.form['Maximum Temperature'])

        #Minimum Temperature
        Minimum_Temperature=float(request.form['Minimum Temperature'])

        #Wind Speed
        Wind_Speed=float(request.form['Wind Speed'])

        #Wind Direction
        Wind_Direction=float(request.form['Wind Direction'])

        #Visibility
        Visibility=float(request.form['Visibility'])

        #Cloud Cover
        Cloud_Cover=float(request.form['Cloud Cover'])

        #Relative Humidity
        Relative_Humidity=float(request.form['Relative Humidity'])

        #Sea Level Pressure
        Sea_Level_Pressure=float(request.form['Sea Level Pressure'])

        #Dew Point
        Dew_Point=float(request.form['Dew Point'])



        X=[[Minimum_Temperature,
            Maximum_Temperature,
            Wind_Speed,
            Wind_Direction,
            Cloud_Cover,
            Relative_Humidity,
            Visibility,
            Sea_Level_Pressure,
            Dew_Point
        ]]
        
        X_scaled=scaler.transform(X)


        prediction=model.predict(X_scaled)
        output=''
        if prediction==1:
            output='Clear'
        elif prediction==2:
            output='Partially Cloudy'
        elif prediction==3:
            output='Overcast' 
        else:
            output='Rain'

    return render_template('weatheroptimized.html',prediction_text='Predicted Weather Conditions {}'.format(output))
    
if __name__=="__main__":
    app.run(debug=True)