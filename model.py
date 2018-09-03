#Import Libraries and Packages
import math
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')

#Load Model Data
X = pd.read_csv("trainingSet(MLTrainingSet,e1,v1).csv")
X_abs = pd.read_csv("absorptances(MLTrainingSet,e1,v1).csv")
X_RF = pd.read_csv("analyticalAbsorptance(MLTrainingSet,e1,v1).csv")

X_NoSE = pd.read_csv("trainingSet(MLTrainingSet,NoSE).csv")
X_abs_NoSE = pd.read_csv("absorptances(MLTrainingSet,NoSE).csv")
X_RF_NoSE = pd.read_csv("analyticalAbsorptance(MLTrainingSet,NoSE).csv")


#Drop Unecessary Columns
X.drop(['ab' ,'aa' ,'Eg','T'],axis = 1, inplace=True)

#Concatenate Absorptance,e1, and RF columns to X
X = pd.concat([X, X_abs,X_RF], axis = 1)



#View snippet of data
print(X.head())

#Only extract rows with values of eb > .50
#X= X.loc[X['eb'] > .50]
#X= X.loc[X['eb'] > .30]

#Visualize Raw Data
plt.figure(1)
plt.subplot(221)
plt.scatter(X.iloc[:,0], X.iloc[:,4],  color='red',alpha =.5)
plt.ylabel(r'$\beta$')
plt.xlabel(r'$f$')

plt.subplot(222)
plt.scatter(X.iloc[:,1], X.iloc[:,4],  color='red',alpha =.5)
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\epsilon_E$')

plt.subplot(223)
plt.scatter(X.iloc[:,2], X.iloc[:,4],  color='red',alpha =.5)
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\epsilon_b$')

plt.subplot(224)
plt.scatter(X.iloc[:,3], X.iloc[:,4],  color='red',alpha =.5)
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\alpha$')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.7,
                    wspace=0.35)

plt.show()



#Extract Target Variable
y = X.iloc[:,4]

#Extract Features
X = X.iloc[:,0:4]


#Create new features
alpha_f_test = X.iloc[:,0]*(X.iloc[:,3]**(1))
alpha_1_eb_test = (1/(X.iloc[:,2]))*(X.iloc[:,3]**(1))
alpha_e1_test = (1-(X.iloc[:,1]))*(X.iloc[:,3])

#Add new features to feature matrix X
X = pd.concat([X,alpha_1_eb_test,alpha_f_test,alpha_e1_test], axis = 1)

#Drop all old features from X
X.drop(['eb','e1','f','alpha'],axis = 1, inplace=True)
print(X.head())

#Split training data into X_train and X_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.40, random_state=2)

#Import Linear Regression Object
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import make_scorer, accuracy_score

#Fit OLS to training data set
ols = linear_model.LinearRegression(fit_intercept=False)
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

#Evaluate and print OLS model performence on test set
print('Coefficients: \n', ols.coef_)

print('Intercept: \n', ols.intercept_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_ols))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred_ols))

print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, y_pred_ols))

#Convert
y_test = pd.DataFrame(y_test)
y_pred_ols = pd.DataFrame(y_pred_ols)

#Create a Pandas Excel writer using XlsxWriter as the engine and write datasets to Excel
X_test.to_excel('XTest.xls', sheet_name='Sheet1', index=False, engine='xlsxwriter')
y_test.to_excel('yTest.xls', sheet_name='Sheet1', index=False, engine='xlsxwriter')
y_pred_ols.to_excel('yPred.xls', sheet_name='Sheet1', index=False, engine='xlsxwriter')


#Visualize Model Testing Results
plt.figure(2)
plt.subplot(311)
plt.scatter(X_test.iloc[:,0], y_test,  color='red', alpha=0.5,label = 'Actual')
plt.scatter(X_test.iloc[:,0] ,y_pred_ols,  color='black',alpha=0.5,label='Predicted')
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\frac{\alpha}{\epsilon_b}$')
plt.legend()

plt.subplot(312)
plt.scatter(X_test.iloc[:,1], y_test,  color='red',alpha=0.5,label = 'Actual')
plt.scatter(X_test.iloc[:,1] ,y_pred_ols,  color='black',alpha=0.5,label='Predicted')
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\alpha\times f$')
plt.legend()

plt.subplot(313)
plt.scatter(X_test.iloc[:,2], y_test,  color='red',alpha=0.5,label = 'Actual')
plt.scatter(X_test.iloc[:,2] ,y_pred_ols,  color='black',alpha=0.5,label='Predicted')
plt.ylabel(r'$\beta$')
plt.xlabel(r'$\alpha\times (1-\epsilon_E)$')
plt.legend()

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.7,
                    wspace=0.35)
plt.show()

#Save model to disk
import pickle
filename = 'Beta_e1_Model.sav'
pickle.dump(ols, open(filename, 'wb'))
