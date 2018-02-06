#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:41:59 2018

@author: Arthur Grigoryan
"""
    
#%% #1

import numpy as np
import urllib.request

k = 3 
n = 100

X = np.random.randn(n,k)
Y = X.dot(np.array([1,2,3]))+np.random.randn(n)

 
class OLS(object): 
    def __init__(self, y, X): 
        y.shape = (n, 1) 
        X.shape = (n, k) 
        self.y = y 
        self.X = X 
        XTX = (X.T @ X) 
        XTX_1 = np.linalg.inv(XTX) 
        self.b = XTX_1 @ X.T @ y 
        sigma_2 = 1/(n-k) * (y - X @ self.b).T @ (y - X @ self.b) 
        self.sigma = sigma_2**(1/2) 
        V_b = sigma_2 * XTX_1 
        self.V = V_b 
    def predict(self, x): 
            yp = x.T @ self.b 
            Var_y = self.sigma * (1 + x.T @ np.linalg.inv(self.X.T @ self.X)
            @ x) 
            return (yp[0], Var_y[0][0]) 

model = OLS(Y, X) 

model.b 
model.V
model.predict(np.array([1,0,1]))


#%% #2
import matplotlib.pylab as plt



#plt.plot(x, y, 'o', label='Data')
#plt.legend(loc='best')
#plt.show()
#(1)(2)(3)
n=200
k=1
beta = np.random.rand(11)
x=np.random.uniform(-5,5,n)
u=10*np.random.randn(200)
#N(m,var)=var^(1/2)+m
#m=0

#(4)

def fact(n):
    if n ==0:
        return 1
    return n*fact(n-1)

y=np.zeros((200,1))

for i in range(200):
    y[i] += u[i]
    for j in range(11):
        y[i]+=beta[j]*(x[i]**j/fact(j))
#(5)

plt.scatter(x,y)

#K=1
model_k1=OLS(y,x)
print(model_k1.b)

yk1=[]
yk2=[]
yk3=[]
yk4=[]
for i in range(n):
         y_new = model_k1.predict(x[i])[0]
         yk1 = np.append(yk1,[y_new])  

plt.plot(x,yk1, label = 'K=1')




#%% #3

import scipy as sc 
import numpy as np 
#import math 
from scipy import stats 
from scipy.stats import t 


t = sc.stats.t.ppf(0.95, 99) 
matrx = np.random.normal(size=(100, 100)).astype('int')
Colmean = np.mean(matrx, axis=0) 
#Rowmean = np.mean(matrx, axis=1) 
Colstd = np.std(matrx, axis=0) 

#Rowmean = np.std(matrx, axis=1) 
print (t) 

left = Colmean - t * Colstd / np.sqrt(len(matrx)) 
right = Colmean + t * Colstd / np.sqrt(len(matrx)) 


dataframe = [] 
count = 0 
for i in range(len(right)): 
    if left[i] < 0 < right[i]: 
        dataframe.append(True) 
        count += 1 
    else: 
        dataframe.append(False) 

 
for i in range (len(dataframe)): 
    print(dataframe[i])



#%% #4
import pandas as pd
#import csv
#import numpy
#(1)
df = pd.DataFrame.from_csv('/Users/macbookpro/goalies-2014-2016.csv', sep = ';')
#df=pd.read_csv('goalies-2014-2016.csv', sep = ';')
data4 = df.head(n=5)
data4.iloc[:,0:6]

#(2)

d1=df.saves/df.shots_against
d1=round(d1,3)

avgd1=pd.Series.sum(d1)/len(d1)
absdev=d1-avgd1

pd.Series.max(absdev)

#(3)

pd.DataFrame.mean(df.games_played)
pd.DataFrame.mean(df.goals_against)
pd.DataFrame.mean(df.save_percentage)

pd.DataFrame.std(df.games_played)
pd.DataFrame.std(df.goals_against)
pd.DataFrame.std(df.save_percentage)

#(4)

df1=df[df.season=='2016-17']
df1=df1[df1.games_played > 40]
#pd.DataFrame.sort_values(by= ['df1.save_percentage'])
df1=df1.sort_values('save_percentage', ascending = False)
#pd.DataFrame.max(df1.save_percentage)
df1.player[:1]

#(5)
data5=[]
data5=df.season
df1617=df[df.season == '2016-17']
df1617=df1617[['season','player', 'saves' ]]
#df1617=df1617.drop(['team','position','games_played', 'games_started', 'wins', 'losses',
                    #'overtime_losses','shots_against','saves','goals_against','save_percentage'], axis=1)
df1516=df[df.season == '2015-16']
df1516=df1516[['season','player','saves']]  #sort by columns
df1415=df[df.season == '2014-15']
df1415=df1415[['season','player','saves']]


#def Savesmax(dataframe):
 #   for i in dataframe.season:
  #      if i == '2016-17':
            
df1617.loc[df['saves'] == pd.DataFrame.max(df1617.saves)]
df1516.loc[df['saves'] == pd.DataFrame.max(df1516.saves)]
df1415.loc[df['saves'] == pd.DataFrame.max(df1415.saves)]


#(6)

winsdata=df[df.wins >=30]
winsdata=winsdata[['season', 'player', 'wins']]

#def best(a):
 #   for i in winsdata.player:
  #      if count(a[i])==3:
   #         return a[i]
    

data6=winsdata.groupby('player').player.count()
data6=data6.sort_values('index', ascending = False)
"""Just 6 players had 3 wins in a row"""

print(data6)

    