#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:36:39 2020

@author: god
"""

import pandas as pd
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.preprocessing import binarize
from sklearn.preprocessing import normalize
df=pd.read_excel('/home/god/Downloads/Concrete_Data.xls')
df.columns





df1,df2=train_test_split(df,test_size=0.2,random_state=101)




scaler = StandardScaler()
y=scaler.fit_transform(df1)
norm=normalize(y)
bine=binarize(norm,threshold=0.0,copy=True)


df1=pd.DataFrame(bine,columns=df.columns)



scaler = StandardScaler()
y=scaler.fit_transform(df2)
norm=normalize(y)
bine=binarize(norm,threshold=0.0,copy=True)


df2=pd.DataFrame(bine,columns=df.columns)


X_train=df1[['Cement (component 1)(kg in a m^3 mixture)','Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
      'Fly Ash (component 3)(kg in a m^3 mixture)','Water  (component 4)(kg in a m^3 mixture)',
      'Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
      'Fine Aggregate (component 7)(kg in a m^3 mixture)','Age (day)']]
y_train=df1[['Concrete compressive strength(MPa, megapascals) ']]




X_test=df2[['Cement (component 1)(kg in a m^3 mixture)','Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
      'Fly Ash (component 3)(kg in a m^3 mixture)','Water  (component 4)(kg in a m^3 mixture)',
      'Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
      'Fine Aggregate (component 7)(kg in a m^3 mixture)','Age (day)']]
y_test=df2[['Concrete compressive strength(MPa, megapascals) ']]




sns.countplot(y_test['Concrete compressive strength(MPa, megapascals) '])
sns.countplot(y_train['Concrete compressive strength(MPa, megapascals) '])




from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train, y_train)
log_pred=clf.predict(X_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
g_pred=gnb.predict(X_test)

from sklearn.linear_model import SGDClassifier
sdg = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
sdg.fit(X_train, y_train)
sdg_pred=sdg.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=12) 
knn.fit(X_train, y_train) 
knn_pred=knn.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc = dtc.fit(X_train,y_train)
dtc_pred=dtc.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=101)
rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)

from sklearn.svm import SVC
svm=SVC(kernel='rbf',C=1)
svm.fit(X_train,y_train)
svm_pred=svm.predict(X_test)



from sklearn import metrics
print("logistic regression:",metrics.accuracy_score(y_test, log_pred)*100)
print("naive bias:",metrics.accuracy_score(y_test, g_pred)*100)
print("gradient:",metrics.accuracy_score(y_test, sdg_pred)*100)
print("knn:",metrics.accuracy_score(y_test, knn_pred)*100)
print("decision tree:",metrics.accuracy_score(y_test, dtc_pred)*100)
print("random forest:",metrics.accuracy_score(y_test, rfc_pred)*100)
print("support vector machine:",metrics.accuracy_score(y_test, svm_pred)*100)


metrics.confusion_matrix(y_test,log_pred)
metrics.confusion_matrix(y_test, g_pred)
metrics.confusion_matrix(y_test, sdg_pred)
metrics.confusion_matrix(y_test, knn_pred)
metrics.confusion_matrix(y_test, dtc_pred)
metrics.confusion_matrix(y_test, rfc_pred)
metrics.confusion_matrix(y_test, svm_pred)

x1=df[['Cement (component 1)(kg in a m^3 mixture)','Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
      'Fly Ash (component 3)(kg in a m^3 mixture)','Water  (component 4)(kg in a m^3 mixture)',
      'Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
      'Fine Aggregate (component 7)(kg in a m^3 mixture)','Age (day)']]

x2=df[['Concrete compressive strength(MPa, megapascals) ']]

scaler = StandardScaler()
y=scaler.fit_transform(x1)
norm=normalize(y)
bine=binarize(norm,threshold=0.0,copy=True)


x1=pd.DataFrame(bine,columns=X_train.columns)

pred_log=clf.predict(x1)
pred_gnb=gnb.predict(x1)
pred_sdg=sdg.predict(x1)
pred_knn=knn.predict(x1)
pred_dtc=dtc.predict(x1)
pred_rfc=rfc.predict(x1)
pred_svm=svm.predict(x1)

df['predict']=pred_rfc

x1=df[df['predict']==0]
x2=df[df['predict']==1]


X1=x1[['Cement (component 1)(kg in a m^3 mixture)','Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
      'Fly Ash (component 3)(kg in a m^3 mixture)','Water  (component 4)(kg in a m^3 mixture)',
      'Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
      'Fine Aggregate (component 7)(kg in a m^3 mixture)','Age (day)']]
y1=x1[['Concrete compressive strength(MPa, megapascals) ']]


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2,random_state=101)


lr = LinearRegression()
lasso = Lasso() 
ridge = Ridge() 
dtr = DecisionTreeRegressor() 
rfr = RandomForestRegressor(n_estimators=50) 


lr.fit(X_train1, y_train1) 
lasso.fit(X_train1, y_train1) 
ridge.fit(X_train1, y_train1) 
dtr.fit(X_train1, y_train1) 
rfr.fit(X_train1, y_train1) 


y_pred_lr = lr.predict(X_test1) 
y_pred_lasso = lasso.predict(X_test1) 
y_pred_ridge = ridge.predict(X_test1) 
y_pred_dtr = dtr.predict(X_test1)
y_pred_rfr = rfr.predict(X_test1)


def autolabel(rects): 
   for rect in rects: 
      height = rect.get_height() 
      ax.annotate('{:.2f}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom') 
      
models = [lr, lasso, ridge, dtr, rfr] 
names = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor"] 
rmses = [] 

for model in models: 
   rmses.append(np.sqrt(mean_squared_error(y_test1, model.predict(X_test1)))) 
   
x = np.arange(len(names)) 
width = 0.3 
fig, ax = plt.subplots(figsize=(10,7)) 
rects = ax.bar(x, rmses, width) 
ax.set_ylabel('RMSE') 
ax.set_xlabel('Models') 
ax.set_title('RMSE with Different Algorithms') 
ax.set_xticks(x) 
ax.set_xticklabels(names, rotation=45) 
autolabel(rects) 
fig.tight_layout() 
plt.show()


plt.scatter(y_test1,y_pred_rfr)
r2_score(y_test1,y_pred_rfr)


X2=x2[['Cement (component 1)(kg in a m^3 mixture)','Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
      'Fly Ash (component 3)(kg in a m^3 mixture)','Water  (component 4)(kg in a m^3 mixture)',
      'Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
      'Fine Aggregate (component 7)(kg in a m^3 mixture)','Age (day)']]
y2=x2[['Concrete compressive strength(MPa, megapascals) ']]


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2,random_state=101)


lr1 = LinearRegression()
lasso1 = Lasso() 
ridge1 = Ridge() 
dtr1 = DecisionTreeRegressor() 
rfr1 = RandomForestRegressor(n_estimators=100) 


lr1.fit(X_train2, y_train2) 
lasso1.fit(X_train2, y_train2) 
ridge1.fit(X_train2, y_train2) 
dtr1.fit(X_train2, y_train2) 
rfr1.fit(X_train2, y_train2) 


y_pred_lr1 = lr1.predict(X_test2) 
y_pred_lasso1 = lasso1.predict(X_test2) 
y_pred_ridge1 = ridge1.predict(X_test2) 
y_pred_dtr1 = dtr1.predict(X_test2)
y_pred_rfr1 = rfr1.predict(X_test2)


models1 = [lr1, lasso1, ridge1, dtr1, rfr1] 
names1 = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor"] 
rmses1 = [] 

for model in models1: 
   rmses1.append(np.sqrt(mean_squared_error(y_test2, model.predict(X_test2)))) 
   
x = np.arange(len(names1)) 
width = 0.3 
fig, ax = plt.subplots(figsize=(10,7)) 
rects = ax.bar(x, rmses1, width) 
ax.set_ylabel('RMSE') 
ax.set_xlabel('Models') 
ax.set_title('RMSE with Different Algorithms') 
ax.set_xticks(x) 
ax.set_xticklabels(names1, rotation=45) 
autolabel(rects) 
fig.tight_layout() 
plt.show()

plt.scatter(y_test2,y_pred_rfr1)
r2_score(y_test2,y_pred_rfr1)

y_pred_rfr1=pd.DataFrame(y_pred_rfr1,columns=y_test.columns)
y_test2.reset_index(drop=True,inplace=True)

percentage1=[]
for i in range(len(y_test2)):
    percentage1.append(((y_test2['Concrete compressive strength(MPa, megapascals) '][i]-y_pred_rfr1['Concrete compressive strength(MPa, megapascals) '][i])/y_test2['Concrete compressive strength(MPa, megapascals) '][i])*100)
percentage1=[abs(i) for i in percentage1]
plt.scatter(percentage1,range(len(percentage1)))




y_pred_rfr=pd.DataFrame(y_pred_rfr,columns=y_test.columns)
y_test1.reset_index(drop=True,inplace=True)
percentage=[]
for i in range(len(y_test1)):
    percentage.append(((y_test1['Concrete compressive strength(MPa, megapascals) '][i]-y_pred_rfr['Concrete compressive strength(MPa, megapascals) '][i])/y_test1['Concrete compressive strength(MPa, megapascals) '][i])*100)
percentage=[abs(i) for i in percentage]   
plt.scatter(percentage,range(len(percentage)))






sns.boxplot(percentage)
sns.boxplot(percentage1)


count=0
for i in range(0,len(percentage)):
    if percentage[i]<=40:
        count+=1
        
count1=0
for i in range(0,len(percentage1)):
    if percentage1[i]<=40:
        count1+=1

print((count/len(percentage))*100)
print((count1/len(percentage1))*100)

print(sum(percentage)/len(percentage))
print(sum(percentage1)/len(percentage1))


rfr.feature_importances_
rfr1.feature_importances_
