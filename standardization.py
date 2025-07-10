# What is standardization ?
# the process of converting the data to a common format or range
# standardization gives us an accurate model and improve the performance of model
# standardization transforms mean to be 0 and std as 1

import pandas as pd
import sklearn.datasets as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# lets dive into an example

dataset = dt.load_breast_cancer()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(df)

# It is evident in the output that the values of a column has a wide range , so algorithms might be biased to larger scale features . For ex salary of one person is 1000 and the other is 90000k ,
X = df
Y = dataset.target

#Splitting data into training and testing data

X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size=0.2,random_state=3)
# print(X_train)

# lets see if the original data's STD, i.e if it is in the same range.
print(dataset.data.std())

# The std in output is 228 . so lets standardize it

scaler = StandardScaler()

scaler.fit(X_train)

x_train_standardised = scaler.transform(X_train)

print(x_train_standardised)

# the above command transformed the data in to smaller values but the values have same impact and significance as how they were previously in number.

print(x_train_standardised.std())

# the above command print the standard deviation of the new transformed dataset which is 1.

