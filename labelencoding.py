# The main use of Label encoding is to convert the categorical data into numerical form so that it can be used by machine learning models.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('./breast_cancer_data.csv')

print(dataset['diagnosis'].value_counts())

# so the above command prints the count of total  values in the diagnosis column so it might be like Male : 270 , female : 350

# our task is to convert the male and female to 0 and 1 , so that our model can understand only numerics not categories

# lets convert the to numerics

label_encoder = LabelEncoder()

label=label_encoder.fit_transform(dataset.diagnosis)

dataset['New'] = label

print(dataset['New'].value_counts())

# now it prints the new numerics inste ad of char values. 1 : 270 , 0 : 350 , why 0 to female why not to men  ? numbers are assigned as per alphabetical order.