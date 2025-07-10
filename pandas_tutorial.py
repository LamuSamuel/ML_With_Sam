# pandas  - mainly useful for data processing and Analysis
# Pandas Data frame is a 2d tabular data structure having rows and columns

# importing pandas library as pd
import pandas as pd

# importing a dataset from sklearn library datasets
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()
california_housing_df = pd.DataFrame(california_housing.data,columns = california_housing.feature_names)


diabetes_data = pd.read_csv('/Users/lamusamuel/Documents/smoke/summerbreak/ML/diabetes.csv')
diabetes_data_df = pd.DataFrame(diabetes_data)
# prints the first 5 rows
print(diabetes_data_df.head())
# prints the last 5 rows
diabetes_data_df.tail()
# print the min value
print(diabetes_data_df.min())
# print the max value
print(diabetes_data_df.max())
# print the total values per column
print(diabetes_data_df.count())
# print the mean for particular column
print(diabetes_data_df.mean())
# Count how many values for the particular column
print(diabetes_data_df.value_counts('SkinThickness'))
# Grouping the values based on Mean
print(diabetes_data_df.groupby('Outcome').mean())

# printing all the Statistical measure's for the data frame
print(diabetes_data_df.describe())


# Manipulating the dataframe
# Adding a column to a dataframe
# For the california_housing_df we have values till longitude however we need to add the Price as a column to make the dataset more informative, so lets grep the data from target column and add in our DF
california_housing_df['Price'] = california_housing.target

# Removing a row
california_housing_df.drop(index=1,axis=0,inplace=True)
print(california_housing_df)

# Removing a column
california_housing_df.drop(columns='AveBedrms',axis=1,inplace=True)
print(california_housing_df)

# print a particular row :
print(california_housing_df.iloc[3])
# printing a particular column :
print(california_housing_df.iloc[:,0])
print(california_housing_df.iloc[:,2])


# Correlation

# 1. Positive Correlation , if there are more Avgrooms then there may be high prices (+ve correlation)
# 2. Negative Correlation , If there are more population then the price in that area is less (-ve correlation)
print(california_housing_df.corr())
