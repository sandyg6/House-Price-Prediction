import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
import warnings

def prediction(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, parking, prefarea):
    data=pd.read_csv("Housing.csv")

    columns_to_convert=['mainroad','guestroom','basement','prefarea']

    le = LabelEncoder()

    for col in columns_to_convert:
        data[col] = le.fit_transform(data[col])

    data = data[pd.to_numeric(data['area'], errors='coerce').notnull()]

    data['area'].astype('float')

    feature_data = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'parking',
                    'prefarea']

    X = np.asarray(feature_data)
    y = np.asarray(data['price'])

    X = data.drop('price', axis=1)
    y = data['price']

    X=X.values
    y=y.values
    train_X = X[:80]
    train_y = y[:80]
    test_X = X[80:]
    test_y = y[80:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    #mse = mean_squared_error(y_test, y_pred)
    #print(mse)
    #metrics.r2_score(y_test,y_pred)


    pickle.dump(model,open('model.pkl','wb'))
    model_loaded =pickle.load(open('model.pkl','rb'))
    hpp = [[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, parking, prefarea]]
# y_pred = model.predict(X_test)
    y_pred = model.predict(hpp)
    print(y_pred)

    #print(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, parking, prefarea)
prediction(8960,4,4,4,1,0,0,3,0)



