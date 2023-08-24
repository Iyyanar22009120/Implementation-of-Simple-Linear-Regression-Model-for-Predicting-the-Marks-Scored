# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1,Use the standard libraries in python.
2.Set variables for assigning dataset values.
3.Import LinearRegression from the sklearn.
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of graph.
6.Compare the graphs and hence we obtain the LinearRegression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: IYYANAR S
RegisterNumber: 212222240036

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

X = df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

# splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_train,Y_train,color="violet")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show() 
*/
```

## Output:
## X output
![1ml2](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/43a6d7cf-8d85-43de-a85f-93decc96890b)

## Y output
![1ml22](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/a5970ea9-eb92-4ab4-be47-531b0eb92022)

## Y pred output
![1ml23](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/963bfe0c-06f4-45d8-860e-be0045fc2d68)

## Y test output
![1ml23](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/963bfe0c-06f4-45d8-860e-be0045fc2d68)


## Training set 1 output
![1ml24](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/2765e3ec-2276-4997-bb05-a400fac0bba9)


## Training set 2 
![1ml25](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/f6a5c582-0bcf-4f96-83bd-ae870a866e52)


## errors output
![1ml26](https://github.com/Iyyanar22009120/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118680259/e2fecb84-d830-4b0b-bd1a-44e9d3f88bab)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
