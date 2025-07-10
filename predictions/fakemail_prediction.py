import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


read_mail_data = pd.read_csv('mail_data.csv')

print(read_mail_data.isnull().sum())

# printing the shape of the data
print(read_mail_data)

# Label encoding
read_mail_data.loc[read_mail_data['Category'] == 'spam' , 'Category'] = 0
read_mail_data.loc[read_mail_data['Category'] == 'ham','Category'] = 1



X =  read_mail_data['Message']
Y = read_mail_data['Category']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify=Y,random_state=3 ,test_size=0.2)

feature_extract = TfidfVectorizer(min_df=1,lowercase=True,stop_words='english')

X_train_features     = feature_extract.fit_transform(X_train)
X_test_features = feature_extract.transform(X_test)

Y_train = Y_train.astype('int')
Y_test =Y_test.astype('int')

model = LogisticRegression()

model.fit(X_train_features ,Y_train)

train_model = model.predict(X_train_features)

accuracy = accuracy_score(Y_train,train_model)



input_email = ["Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"]
feature_extract_email = feature_extract.transform(input_email)
result  = model.predict(feature_extract_email)
print(result)
if result == [0]:
    print("Its a  fake email")
if result == [1]:
    print("Its a valid Mail")




