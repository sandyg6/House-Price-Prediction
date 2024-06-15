from flask import Flask, render_template,request
from housepriceprediction import prediction
import pickle
import numpy as np

app=Flask(__name__)

app = Flask(__name__, template_folder="template")

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def login():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    param1 = request.form.get('param1')
    param2 = request.form.get('param2')
    param3 = request.form.get('param3')
    param4 = request.form.get('param4')
    param5 = request.form.get('param5')
    param6 = request.form.get('param6')
    param7 = request.form.get('param7')
    param8 = request.form.get('param8')
    param9 = request.form.get('param9')
    print(param1,param2,param3,param4,param5,param6,param7,param8,param9)
    result=model.predict([[param1,param2,param3,param4,param5,param6,param7,param8,param9]])
    #return render_template('test.html',result=prediction(float(param1),float(param2),float(param3),float(param4),float(param5),float(param6),float(param7),float(param8),float(param9)))
    return render_template('index.html',result=f"The predicted price is {result[0]}")

if __name__=='__main__':
    app.run(debug=True)



