import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, linear_model, preprocessing, neighbors, tree, ensemble, svm, metrics
#datasets
x, y = datasets.fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test =model_selection.train_test_split(x, y, test_size=0.1, random_state=42)

#normalization
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#linear regression
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
y_pred_reg = lr.predict(x_test)
print("Linear Regression MSE:", metrics.mean_squared_error(y_test, y_pred_reg))

#Decision Tree
dt = tree.DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred_dec = dt.predict(x_test)
print("Decision Tree MSE:", metrics.mean_squared_error(y_test, y_pred_dec))

#Random Forest
rf = ensemble.RandomForestRegressor()
rf.fit(x_train, y_train)
y_pred_ran = rf.predict(x_test)
print("Random Forest MAE:", metrics.mean_absolute_error(y_test, y_pred_ran))

#Support Vector Regression
sv = svm.SVR()
sv.fit(x_train, y_train)
y_pred_sup = sv.predict(x_test)
print("SVM Regression MSE:", metrics.mean_squared_error(y_test, y_pred_sup))

#draw
plt.scatter(y_test, y_pred_ran, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted - Random Forest")
plt.show()