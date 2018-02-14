#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:38:09 2018

@author: khankishialiev
"""


#%%
def fact(n):
    if n==0:
        return 1
    if n==1:
        return 1
    return n*fact(n-1)
#%%
#Task 1
import numpy as np
class OLS(object):
    def __init__(self, y, X):
        self.y = y
        self.X = X
        #self.beta = (np.linalg.inv((X.T).dot(X))).dot((X.T).dot(y))
    
    def beta(self):
        bet = (np.linalg.inv((self.X.T).dot(self.X))).dot((self.X.T).dot(self.y))
        return bet
    
    def V(self):
        bet = (np.linalg.inv((self.X.T).dot(self.X))).dot((self.X.T).dot(self.y))
        n = self.X.shape[0]
        k = self.X.shape[1]
        return (1.0/(n - k) * ((self.y - self.X.dot(bet)).T).dot(self.y - self.X.dot(bet))) * np.linalg.inv((self.X.T).dot(self.X))
    
    def sigma(self):
        bet = (np.linalg.inv((self.X.T).dot(self.X))).dot((self.X.T).dot(self.y))
        n = self.X.shape[0]
        k = self.X.shape[1]
        return (1.0/(n - k) * ((self.y - self.X.dot(bet)).T).dot(self.y - self.X.dot(bet)))**(1/2)
    
    def predict(self, xx):
        bet = (np.linalg.inv((self.X.T).dot(self.X))).dot((self.X.T).dot(self.y))
        yy = (xx.T).dot(bet)
        n = self.X.shape[0]
        k = self.X.shape[1]
        VV = 1.0/(n - k) * ((self.y - self.X.dot(bet)).T).dot(self.y - self.X.dot(bet)) * (1 + (xx.T).dot(np.linalg.inv((self.X.T).dot(self.X)).dot(xx)))
        otvet = (yy,VV)
        return otvet
        #return print('(',yy,', ',VV,')')
        
    #def __repr__(self):
     #   person = 'Person: %s\n' % self.name
      #  person += 'Age: %s\n' % self.age
       # return person
#%%
import numpy as np
X = np.random.randn(100,3)
y = X.dot(np.array([1,2,3]))+np.random.randn(100)
model = OLS(y, X)
model.beta()
model.V()
model.predict(np.array([1,0,1]))


#%%
import numpy as np
import matplotlib as mp
#Task 2
##subtask 1
beta = np.random.rand(11)
x = np.random.uniform(-5,5,200)
x = np.sort(x)
#u = np.array([np.random.randn()*10 for i in range(200)])
u = np.random.randn(200)*10
#u = np.random.normal(0,10,200)
y = np.array([0 for i in range(200)])
for i in range(200): 
   for k in range(11):
        y[i] = y[i] + (beta[k]*x[i]**k)/fact(k)
   y[i] = y[i] + u[i]
#mp.pyplot.scatter(x,y)

odin = np.array([1 for i in range(200)])
#K=1
X1 = np.array([odin, x]).T
model1 = OLS(y, X1)
K1 = np.array([model1.predict(np.array([1,x[i]]))[0] for i in range(200)])

#mp.pyplot.plot(x, K1)
#K=2
X2 = np.array([odin, x, x**2]).T
model2 = OLS(y, X2)
K2 = np.array([model2.predict(np.array([1,x[i],x[i]**2]))[0] for i in range(200)])

#mp.pyplot.plot(x, K2)

#K=3
X3 = np.array([odin, x, x**2, x**3]).T
model3 = OLS(y, X3)
K3 = np.array([model3.predict(np.array([1,x[i],x[i]**2,x[i]**3]))[0] for i in range(200)])

#mp.pyplot.plot(x, K3)

#K=4
X4 = np.array([odin, x, x**2, x**3, x**4]).T
model4 = OLS(y, X4)
K4 = np.array([model4.predict(np.array([1,x[i],x[i]**2,x[i]**3,x[i]**4]))[0] for i in range(200)])
K41 = [0 for i in range(200)]
for i in range(200):
    K41[i] = K4[i] + 1.65*model4.sigma()
K42 = [0 for i in range(200)]
for i in range(200):
    K42[i] = K4[i] - 1.65*model4.sigma()
#mp.pyplot.plot(x, K4)

fig, ax = mp.pyplot.subplots()
ax.scatter(x,y)
ax.plot(x, K1, label='K=1')
ax.plot(x, K2, label='K=2')
ax.plot(x, K3, label='K=3')
ax.plot(x, K4, label='K=4')
legend = ax.legend(loc='upper left')

fig1, ax1 = mp.pyplot.subplots()
ax1.scatter(x,y)
ax1.plot(x, K4, label='K=4')
ax1.fill_between(x, K41, K42, facecolor='grey', alpha = 0.4)
legend = ax1.legend(loc='upper left')

#%%
#Task3
A = np.random.randn(100,100)
sigmaA = A.std(axis = 0)
EA = A.mean(axis=0)

for i in range(100):
    print(i+1,'-й столбец: ','(',EA[i] - 1.65*sigmaA[i]/(100**(1/2)),', ',EA[i] + 1.65*sigmaA[i]/(100**(1/2)),')')

logic = [True if (0>EA[i] - 1.65*sigmaA[i]/(100**(1/2))) and (0<EA[i] + 1.65*sigmaA[i]/(100**(1/2))) else False for i in range(100)]

print(logic)

print(np.sum(logic))

#строки
sigmaA1 = A.std(axis = 1)
EA1 = A.mean(axis=1)

for i in range(100):
    print(i+1,'-я строка: ','(',EA1[i] - 1.65*sigmaA1[i]/(100**(1/2)),', ',EA1[i] + 1.65*sigmaA1[i]/(100**(1/2)),')')

logic1 = [True if (0>EA1[i] - 1.65*sigmaA1[i]/(100**(1/2))) and (0<EA1[i] + 1.65*sigmaA1[i]/(100**(1/2))) else False for i in range(100)]

print(logic1)

print(np.sum(logic1))

#%%
#Task4
import pandas as pd
pd.set_option('float_format', '{:6.3f}'.format)
path = './goalies-2014-2016.csv'
df = pd.read_csv(path, sep=';', header=0)
df.iloc[:5,:6]
#%%
perc = [np.absolute(float('{:.3f}'.format(df['saves'][i]/df['shots_against'][i])) - float('{:.3f}'.format(df['save_percentage'][i]))) for i in range(len(df['n']))]
perc1 = [float('{:.3f}'.format(df['saves'][i]/df['shots_against'][i])) for i in range(len(df['n']))]

print(max(perc))
#%%
gp_mean = df['games_played'].mean(axis=0)
ga_mean = df['goals_against'].mean(axis=0)
sp_mean = df['save_percentage'].mean(axis=0)

gp_std = df['games_played'].std(axis=0)
ga_std = df['goals_against'].std(axis=0)
sp_std = df['save_percentage'].std(axis=0)

print('games_played mean, std: ', gp_mean,'  ', gp_std)
print('goals_against mean, std: ', ga_mean,'  ', ga_std)
print('save_percentage mean, std: ', sp_mean,'  ', sp_std)
#%%
df1 = df[df['save_percentage'] == max(df[(df['season'] == '2016-17') & (df['games_played']>40)]['save_percentage'])]
df1[['player','save_percentage']]
#%%

df2 = df[df['season'] == '2016-17']
id1 = df2['saves'].idxmax()

df3 = df[df['season'] == '2015-16']
id2 = df3['saves'].idxmax()

df4 = df[df['season'] == '2014-15']
id3 = df4['saves'].idxmax()

df5 = df.iloc[[id1,id2,id3]][['season', 'player', 'saves']]
#%%

