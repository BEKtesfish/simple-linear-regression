# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('house_price.csv')
x= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values


#split the data set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#training our data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#the prediction result
predic=regressor.predict(x_test)

#Displaying predictions and actual values
result_df= pd.DataFrame( { 'Actual' : y_test,'Predicted': predic}) 
print(result_df)

#visualization

# for the training set
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Square Foot Area vs Price(Training set)')
plt.xlabel('Years Of Experience')
plt.ylabel('PRice')

plt.show()


#for the test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color='blue')  
plt.title('Square Foot Area Vs Price(Test set)')
plt.xlabel('Square Foot Area ')
plt.ylabel('Price')

# Display the second plot
plt.show()

# Displaying predictions and actual values
result_df= pd.DataFrame( { 'Actual' : y_test,'Predicted': predic}) 