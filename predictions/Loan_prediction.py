import pandas as pd
from PIL.GimpGradientFile import linear
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

loan = pd.read_csv('loan_dataset.csv')
# print(loan.isnull().sum())
loan_data = loan.dropna()
# print(loan_data.shape)

loan_data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)
loan_data.replace({"Dependents":{'3+':4}},inplace=True)
# print(loan_data)
sns.countplot(x="Education",hue="Loan_Status",data=loan_data)
sns.countplot(x="Married",hue="Loan_Status",data=loan_data)
loan_data.replace({"Married":{'Yes':1,'No':0},"Gender":{"Male":1,"Female":0},"Education":{"Graduate":1,"Not Graduate":0},"Self_Employed":{"No":0,"Yes":1},"Property_Area":{"Rural":0,"Urban":2,"Semiurban":3}},inplace=True)
# print(loan_data.head())

X = loan_data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y= loan_data['Loan_Status']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify=Y,test_size=0.1,random_state=2)

model = svm.SVC(kernel = 'linear')
model.fit(X_train,Y_train)
train_model = model.predict(X_train)
accuracy = accuracy_score(train_model,Y_train)
print(accuracy)
test_model = model.predict(X_test)
accuracy_test = accuracy_score(test_model,Y_test)
print(accuracy_test)
