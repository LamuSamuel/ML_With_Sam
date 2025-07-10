import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from pandas import value_counts


credit_card_data = pd.read_csv('./credit_data.csv')


legit = credit_card_data[credit_card_data['Class']==0]
fraud = credit_card_data[credit_card_data['Class']==1]
#
# print(legit.shape)
# print(fraud.shape)

## we can clearly see the values range a lot. So let's make them even and balance , we have legit in 24k umber while fraud in 492 so lets bring legit to 492. This way the the model is accurate  .
legit_sample = legit.sample(n=492)
# print(legit_sample)
# creating a dataset with the new legit values of 492 and old fraud values of 492 into one
new_dataset = pd.concat([legit_sample,fraud],axis = 0)
print(new_dataset['Class'].value_counts())

