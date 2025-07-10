import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


sonar_data = pd.read_csv('./sonar_data.csv',header=None)
# lets first see how big the data is by using the below function
print(sonar_data.shape)

# lets describe the data to know how the central tendencies range

print(sonar_data.describe())

# let's see if there are any empty values in the data and fill them.
print(sonar_data.isnull().sum())

# all the rows have the data and none of them are empty

# so nows lets see that the target is equally distributed if not the model might make a mistake

print(sonar_data[60].value_counts())

# it is also convincing that the Mine and rock output are almost similar and none of them surpass another , so there is no need to balance the dataset

# lets now Train our model and segregate the features and output
X = sonar_data.drop(columns=60)
Y = sonar_data[60]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y ,test_size=0.2 , stratify=Y ,random_state=2)

model =  LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
X_training_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Accuracy for X_train is :",X_training_accuracy)

X_test_prediction = model.predict(X_test)
X_testing_accuracy = accuracy_score(X_test_prediction,Y_test)

print("Accuracy for X_test is :",X_testing_accuracy)

# now that we have seen the accuracy for both trained and tested datasets lets test them.
# we also pass the data from our dataset just to know how accurate is it to cross verify
input_data = (0.0291,0.0400,0.0771,0.0809,0.0521,0.1051,0.0145,0.0674,0.1294,0.1146,0.0942,0.0794,0.0252,0.1191,0.1045,0.2050,0.1556,0.2690,0.3784,0.4024,0.3470,0.1395,0.1208,0.2827,0.1500,0.2626,0.4468,0.7520,0.9036,0.7812,0.4766,0.2483,0.5372,0.6279,0.3647,0.4572,0.6359,0.6474,0.5520,0.3253,0.2292,0.0653,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0056,0.0237,0.0204,0.0050,0.0137,0.0164,0.0081,0.0139,0.0111)
input_data_array = np.asarray(input_data).reshape(1,-1) # reshape means if you give 1,-1 means it will fit data to 1 row and 'n' number of columns , -1,1 means 'n' rows and 1 column
predict_input_data = model.predict(input_data_array)
print(predict_input_data)

# wohoo! we have got the accurate prediction for many number of input_data