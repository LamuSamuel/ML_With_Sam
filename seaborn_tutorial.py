# Seaborn - Data Visualization Library
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
tip = sns.load_dataset('tips')
tip_df = pd.DataFrame(tip)
sns.relplot(data=tip_df,x='total_bill',y='tip',hue='smoker',style='sex',size='size')
plt.show()

titanic_data = sns.load_dataset('titanic')
sns.countplot(data=titanic_data,x='survived' )
plt.show()

sns.barplot(data=titanic_data,x='sex',y='survived',hue='class')
plt.show()

house = fetch_california_housing()
house_data = pd.DataFrame(house.data,columns=house.feature_names)
house_data['price']=house.target
print(max(house_data.price))


# lets see the distribution plot of the average price the house being sold
sns.displot(data=house_data,x='price')
plt.show()


# Correlation matrix
print(house_data.corr())
correlation_house_dataset=house_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_house_dataset, cbar=True, square=True , fmt = '.1f', annot = True, annot_kws={'size':8},cmap='Blues')
plt.show()

