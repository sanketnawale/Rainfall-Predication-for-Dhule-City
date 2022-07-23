
# Rainfall-Predication-for-Dhule-City

Rainfall-Predication-for-Dhule-City using CSV data

To create a rainfall prediction model to predeict the rainfall in certian region
for effective use of water resourses and planning water structures

with the help of linear regression I have tried to predict the rainfall
we can predict the inches of rainfall we can expect.





## Step 1

Get raw weather data in csv format in my case I have a file named as Dhule_Raw_weather_data

Now the data in this file is not in the structured form, it has invalid fields with missing information or additional unnecessary values

So to structyure it type the code as follows

make a python code file in my case i have name it as Predict.python

and write the following code of lines to clean the raw data

# Structuring the data
#import Libraries

import pandas as pd  //can pipinstall pandas in pip folder
import numpy as np  //can pipinstall numpy in pip folder

# read data in pandas 
data = pd.read_csv("Dhule_Raw_weather_data.csv")

# deleting the unnecessary columns in the data
data = data.drop(
    ['Events', 'Date', 'SeaLevelPressureHighInches', 'SeaLevelPressureLowInches'], axis=1)

data = data.replace('T', 0.0)  // replace occurance of T with 0
data = data.replace('-', 0.0)  //replace occurance of - with 0

# This Saves the data in a csv file with the name we give 
data.to_csv('filtered_data.csv') //clean filtered file with structure data

after running this code You will find an new file with filtered_data




## Step 2 
So now we need to import some library
# import the libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt //pip install all this

# Reading the structured data
data = pd.read_csv("filtered_data.csv")

X = data.drop(['PrecipitationSumInches'], axis=1)

Y = data['PrecipitationSumInches'] //output label
Y = Y.values.reshape(-1, 1) //2d vector

day_index = 798 //random day in data plot we will be observing the day
days = [i for i in range(Y.size)]

clf = LinearRegression() //initializing the regression
clf.fit(X, Y) //classed the data

inp = np.array([[94], [44], [45], [67], [40], [43], [33], [45],
                [57], [29.68], [10], [9], [2], [0], [20], [4], [31]]) //sample data in 2d vector to check its random sample input

inp = inp.reshape(1, -1)

# Printing output
print('The precipitation in inches for the input is:', clf.predict(inp))

print('The precipitation trend graph: ')
plt.scatter(days, Y, color='g')
plt.scatter(days[day_index], Y[day_index], color='r')
plt.title('Precipitation level')
plt.xlabel('Days')
plt.ylabel('Precipitation in inches')

# Plot a graph of precipitation levels vs no of days
plt.show()

After this you can see the graph below with precipitation levels

with one day which we have chosen

## Screenshots

![App Screenshot](https://drive.google.com/file/d/1Z0yC_VUbRrrPBfHHl6wvqxHHp_ZlVtSx/view?usp=sharing)


## Step 3

Now we will make the graph with more vraibles 

x_f = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                'WindAvgMPH'], axis=1)
print('Preciptiation Vs Selected Attributes Graph: ')
for i in range(x_f.columns.size):
    plt.subplot(3, 2, i+1)
    plt.scatter(days, x_f[x_f.columns.values[i][:100]], color='g')
    plt.scatter(days[day_index], x_f[x_f.columns.values[i]]
                [day_index], color='r')
    plt.title(x_f.columns.values[i])

# plot a graph with a few features vs precipitation to observe the trends
plt.show()

now we can observe different levels 

1st we will have precipitation graph 

then after colsing it we wil have a detail graph with TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                'WindAvgMPH parameters'




## Screenshots

![App Screenshot](https://drive.google.com/file/d/1Z0yC_VUbRrrPBfHHl6wvqxHHp_ZlVtSx/view?usp=sharing)

![App Screenshot](https://drive.google.com/file/d/1floIzpob5dx_UyAE9T4mNY0K_YupFj9M/view?usp=sharing)


