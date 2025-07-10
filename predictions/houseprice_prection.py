import pandas as pd
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


house_Data = pd.read_csv('BostonHousing.csv')

#
# correlation = house_Data.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation, cbar = True ,square=True,fmt = '.1f',annot=True , annot_kws={'size':8},cmap='Blues')


X = house_Data.drop(['price'   ],axis=1)

Y = house_Data['price']


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

model = XGBRegressor(n_estimators=200,
                     learning_rate=0.05,
                     max_depth=4,
                     subsample=0.9,
                     colsample_bytree=0.8,
                     random_state=42)
model.fit(X_train,Y_train)
model_predict = model.predict(X_train)
r_square  = metrics.r2_score(Y_train,model_predict)
avg_mean = metrics.mean_absolute_error(Y_train,model_predict)
print(r_square)
print(avg_mean)

model_predict_test = model.predict(X_test)
r_square_test  = metrics.r2_score(Y_test,model_predict_test)
avg_mean_test = metrics.mean_absolute_error(Y_test,model_predict_test)
print(r_square_test)
print(avg_mean_test)



## This prediction is not accurate as the accuracies differ. Lets see how to overcome them in upcoming session
