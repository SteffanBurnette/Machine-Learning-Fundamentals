import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Downloads and prepare the data
#Stores the raw link to the data in a variable to be later converted to a csv
data_root = "https://github.com/ageron/data/raw/main/"
#Stores the data from the previous link and data from the lifesatisfaction csv into one dataframe
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
#Sets the X value as the CDP per capita
X = lifesat[["GDP per capita (USD)"]].values
#Sets the y variable as the life satisfcation rating
y = lifesat[["Life satisfaction"]].values

# Visualize the data
#Plots the data so that we can have a better understanding of what we are working with
lifesat.plot(kind='scatter', grid=True,
x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
#Creates an instance of the Linear Regression Model to perform  our regression analysis
model = LinearRegression()

# Trains the model on the data  (note that we did not split the data into testing and trainings sets)
# (Also did not normalize the data)
model.fit(X, y)

#After training the model we used Cyprus GDP to predict the life satisfacto\ion
# Make a prediction for Cyprus
X_new = [[37_655.2]] # Cyprus' GDP per capita in 2020
#Outputting said prediction
print(model.predict(X_new)) # output: [[6.30165767]]