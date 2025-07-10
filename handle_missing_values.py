# Two ways to handle Missing Values in a Data set
# 1. Imputation
# 2. Dropping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
### Method Imputation
placement_data = pd.read_csv('./Placement_Dataset.csv')
print(placement_data.shape)

#first find out how may rows are empty and the related columns

print(placement_data.isnull().sum())

# the above command in the output shows that salary has 67 rows of empty values so how to fill these ?

### central tendencies  - Mean ,  Median , Mode

plt.subplots(figsize=(8,8))
sns.displot(placement_data.salary)
plt.show()
'''
as we can see most of the salaries are on one side , so there is sew curve on one part of graph mostly so we will neglect the Mean

so lets replace with   median or mode

but lets first check the median of the salary column such that that value has to be replaced with the nan value
'''
print(placement_data['salary'].median())

# so its evident that the median is 265000.0 , now lets swap this value with all the null values in the salary column

placement_data['salary'].fillna(placement_data['salary'].median(),inplace=True)

# lets confirm that the null values in the dataset are non by the below command


print(placement_data.isnull().sum())



#### Method Dropping
salary_dataset = pd.read_csv('./Placement_Dataset.csv')

print(placement_data.isnull().sum())

### check the rows and columns

print(placement_data.shape)

## dropping the Missing Values

salary_dataset=salary_dataset.dropna(how='any')

# once the above command is executed , check again the rows and columns

print(placement_data.shape)

# You should be seeing the rows dropped if they were Nan.




