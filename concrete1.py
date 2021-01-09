#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 07:55:13 2020

@author: god
"""
import pandas as pd
import seaborn as sns
df=pd.read_excel('/home/god/Downloads/Concrete_Data.xls')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import binarize
from sklearn.preprocessing import normalize

scaler = StandardScaler()
y=scaler.fit_transform(df)
norm=normalize(y)
bine=binarize(norm,threshold=0.0,copy=True)


df1=pd.DataFrame(bine,columns=df.columns)



t1=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==0)]

t2=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==0)]

t3=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==0)]

t4=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==0)]

t5=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==1)]

t6=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==1)]

t7=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==1)]

t8=df[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==1)]

t1_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==0)]

t2_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==0)]

t3_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==0)]

t4_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==0)]

t5_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==1)]

t6_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==0) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==1)]

t7_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==0) &
       (df1['Age (day)']==1)]

t8_1=df1[(df1['Cement (component 1)(kg in a m^3 mixture)']==1) & 
       (df1['Concrete compressive strength(MPa, megapascals) ']==1) &
       (df1['Age (day)']==1)]

sns.boxplot(t1['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t1['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t1['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t1['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t1['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t1['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t1['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t1['Age (day)'])
sns.boxplot(t1['Concrete compressive strength(MPa, megapascals) '])

sns.countplot(t1_1['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
t1_1['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t1_1['Fly Ash (component 3)(kg in a m^3 mixture)'])
t1_1['Fly Ash (component 3)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t1_1['Water  (component 4)(kg in a m^3 mixture)'])
t1_1['Water  (component 4)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t1_1['Superplasticizer (component 5)(kg in a m^3 mixture)'])
t1_1['Superplasticizer (component 5)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t1_1['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
t1_1['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t1_1['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
t1_1['Fine Aggregate (component 7)(kg in a m^3 mixture)'].value_counts()


sns.boxplot(t2['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t2['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t2['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t2['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t2['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t2['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t2['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t2['Age (day)'])
sns.boxplot(t2['Concrete compressive strength(MPa, megapascals) '])

sns.countplot(t2_1['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
t2_1['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t2_1['Fly Ash (component 3)(kg in a m^3 mixture)'])
t2_1['Fly Ash (component 3)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t2_1['Water  (component 4)(kg in a m^3 mixture)'])
t2_1['Water  (component 4)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t2_1['Superplasticizer (component 5)(kg in a m^3 mixture)'])
t2_1['Superplasticizer (component 5)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t2_1['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
t2_1['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'].value_counts()
sns.countplot(t2_1['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
t2_1['Fine Aggregate (component 7)(kg in a m^3 mixture)'].value_counts()

sns.boxplot(t3['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t3['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t3['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t3['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t3['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t3['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t3['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t3['Age (day)'])
sns.boxplot(t3['Concrete compressive strength(MPa, megapascals) '])

sns.boxplot(t4['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t4['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t4['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t4['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t4['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t4['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t4['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t4['Age (day)'])
sns.boxplot(t4['Concrete compressive strength(MPa, megapascals) '])

sns.boxplot(t5['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t5['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t5['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t5['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t5['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t5['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t5['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t5['Age (day)'])
sns.boxplot(t5['Concrete compressive strength(MPa, megapascals) '])

sns.boxplot(t6['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t6['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t6['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t6['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t6['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t6['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t6['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t6['Age (day)'])
sns.boxplot(t6['Concrete compressive strength(MPa, megapascals) '])

sns.boxplot(t7['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t7['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t7['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t7['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t7['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t7['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t7['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t7['Age (day)'])
sns.boxplot(t7['Concrete compressive strength(MPa, megapascals) '])

sns.boxplot(t8['Cement (component 1)(kg in a m^3 mixture)'])
sns.boxplot(t8['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'])
sns.boxplot(t8['Fly Ash (component 3)(kg in a m^3 mixture)'])
sns.boxplot(t8['Water  (component 4)(kg in a m^3 mixture)'])
sns.boxplot(t8['Superplasticizer (component 5)(kg in a m^3 mixture)'])
sns.boxplot(t8['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'])
sns.boxplot(t8['Fine Aggregate (component 7)(kg in a m^3 mixture)'])
sns.boxplot(t8['Age (day)'])
sns.boxplot(t8['Concrete compressive strength(MPa, megapascals) '])

p1=df1[df1['Concrete compressive strength(MPa, megapascals) ']==0]
p2=df1[df1['Concrete compressive strength(MPa, megapascals) ']==1]

p1_c1=p1[p1['Cement (component 1)(kg in a m^3 mixture)']==0]
p1_c2=p1[p1['Cement (component 1)(kg in a m^3 mixture)']==1]
p2_c1=p2[p2['Cement (component 1)(kg in a m^3 mixture)']==0]
p2_c2=p2[p2['Cement (component 1)(kg in a m^3 mixture)']==1]

p1_a1=p1_c1[p1_c1['Age (day)']==0]
p1_a2=p1_c1[p1_c1['Age (day)']==1]
p1_a11=p1_c2[p1_c2['Age (day)']==0]
p1_a22=p1_c2[p1_c2['Age (day)']==1]

p2_a1=p2_c1[p2_c1['Age (day)']==0]
p2_a2=p2_c1[p2_c1['Age (day)']==1]
p2_a11=p2_c2[p2_c2['Age (day)']==0]
p2_a22=p2_c2[p2_c2['Age (day)']==1]