import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import accuracy_score

# loading the data set into a variable diabetes_data_set
diabetes_data_set = pd.read_csv('./diabetes.csv')
# we will also check the target , if it is  balanced or not
print(diabetes_data_set['Outcome'].value_counts())

# while it is imbalanced , but not severe lets not balance the target as it seems to be accepted and convinced
#0 - 500 ,  1 - 268  -- seems to be around  60% and 30 % , it is suggested to balance dataset when there is 85-90% of difference
# Splitting the features and outcomes
X_data = diabetes_data_set.drop('Outcome',axis=1)
Y = diabetes_data_set['Outcome']

print(X_data)
 # the above command has given the features output and it has a range among values so lets bring them all under on tree and
# that we have an accurate model and improve the performance of model
# standardization transforms mean to be 0 and std as 1
scalar = StandardScaler()

data = scalar.fit_transform(X_data)

# once the data variable acquired the data yet it doesn't have the columns as of our original dataset so lets obtain it by below command

X = pd.DataFrame(data,columns=X_data.columns)

# training and testing

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2 , stratify=Y,random_state=2)

# linear line , throws the estimated points as per their resulted outcome values , splitted by a linear line
classifier = svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)

classifier_prediction = classifier.predict(X_train)
# getting the accuracy of the model
classifier_accuracy = accuracy_score(classifier_prediction,Y_train)

print(classifier_accuracy)
# performing the same above operations on the test data
classifier_prediction_test = classifier.predict(X_test)
classifier_accuracy_test = accuracy_score(classifier_prediction_test,Y_test)

print(classifier_accuracy_test)
# now testing our model with new values
new_data = (1,146,56,0,0,29.7,0.564,29)

input_data_array = np.asarray(new_data).reshape(1,-1)

std_input_data = scalar.transform(input_data_array)
# dont you think why are we not using fit as how we have done in line 30 ?
# well the answer to this is if we give fit it again calculates the mean and all tendencies which is like peeking into the data.
# this breaks the rule of keeping the data unseen which we have trained and tested before and this can lead to inaccuracy.
new_prediction = classifier.predict(std_input_data)
print(new_prediction)

# : )
