import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    'bedrooms':[2,4,4,2,5,3,4,6,2],
    'bathrooms':[1,2,2,1,3,2,2,4,1],
    'balcony':[0,1,1,0,2,1,1,3,0],
    'servant_room':[0,1,1,0,1,1,1,2,0],
    'Area_sqft':[500,800,1000,1200,1500,1800,2000,2200,2500],
    'price':[200000,220000,300000,320000,400000,450000,500000,550000,600000]
}

df = pd.DataFrame(data)
print(df.head())

x=df.drop('price', axis=1)
y=df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Rmse:",np.sqrt(mean_squared_error(y_test, y_pred)))

house=np.array([[3,2,1,1,1500]])    
price=model.predict(house)
print("Predicted price for house with 3 bedrooms, 2 bathrooms, 1 balcony, 1 servant room and 1500 sqft area is:", price[0])
