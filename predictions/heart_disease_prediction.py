import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

heart_data=pd.read_csv('heart_disease_data.csv')
#
# print(heart_data.head())
#
# print(heart_data.isnull().sum())
#
# print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target',axis=1)
Y = heart_data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify=Y,random_state=2,test_size=0.2)


model = LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)

train_model=model.predict(X_train)

accuracy = accuracy_score(train_model,Y_train)
print("Accuracy on training data is : ",accuracy)

test_model = model.predict(X_test)

accuracy_test = accuracy_score(test_model,Y_test)

print("Accuracy on testing data is ",accuracy_test)

new_heart_input = [52,1,2,172,199,1,1,162,0,0.5,2,0,3]
new_heart_data = np.asarray(new_heart_input).reshape(1,-1)
new_prediction = model.predict(new_heart_data)


if new_prediction[0] == 0:
    print("The person is not vulnerable to heart disease")
elif new_prediction[0] ==1 :
    print("The person has a complication in heart")






