import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("hours.csv")
X=dataset.iloc[:,:-1].values  #iloc-integer location all rows and columns except the last column
y=dataset.iloc[:,1].values
print(dataset)
print(X)
print(y)

#Applying linear regression to dataset  
regressor=LinearRegression()
regressor.fit(X,y)   #fit() used for tranining the Linear regeression Model.

print("coef : ",regressor.coef_)
print("intercept : ",regressor.intercept_)
print('accuracy of LR model is : ',regressor.score(X,y)*100)
print("equation of line : ","Y = "+str(regressor.coef_[0])+" x +"+str(regressor.intercept_))

#Plotting Line of Fit and dataset
plt.xlabel('height')
plt.ylabel('weight')
plt.plot(X, regressor.predict(X), color = 'red')
plt.scatter(X,y,color='blue')
plt.legend()
plt.show()


