import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data={
    'Area_sqft': [500,800,1000,1200,1500,1800,2000,2200,2500,3000,3200,3500,4000,4500,5000],
    'price': [200000,220000,300000,320000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000]
}

df=pd.DataFrame(data)
x=df[['Area_sqft']]
y=df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))