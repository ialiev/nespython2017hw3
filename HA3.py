import numpy as np 
import pandas as pd  
import math 
  #%%
#Задача #1  
class OLS(object):    
    
    def __init__(self, y, X):
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        eps = y - X.dot(self.beta)
        sigma_hat = eps.T.dot(eps)/(np.shape(X)[0]-np.shape(X)[1])
        self.V = sigma_hat*np.linalg.inv(X.T.dot(X))*np.shape(X)[0]
        
    def predict(self, param): 
        return param.dot(self.beta)
    
    def predict_var(self, param):
        eps = y - X.dot(self.beta)
        sigma_hat = eps.T.dot(eps)/(np.shape(X)[0]-np.shape(X)[1])
        return sigma_hat*(1+param.T.dot(np.linalg.inv(X.T.dot(X))).dot(param))

N = 100000
X = np.random.randn(N,3)
y = X.dot(np.array([1,2,3]))+np.random.randn(N) 

model = OLS(y, X)
print(model.beta)
print(model.V)
print(model.predict(np.array([1,0,1])))
print(model.predict_var(np.array([1,0,1])))
#%%
#Задача #4
df1 = pd.read_csv("goalies-2014-2016.csv", sep =';', header = 0, index_col=0, engine='python')
print(df1.iloc[:5,:6])
#%%
print((abs(df1["saves"]/df1["shots_against"])-abs(df1["save_percentage"])).max())
#%%
print(df1.games_played.mean())
print(df1.goals_against.mean())
print(df1.save_percentage.mean())
print(df1.games_played.std())
print(df1.goals_against.std())
print(df1.save_percentage.std())
df1.dtypes
#%%
df2 = df1[df1['season'] == '2016-17']
df2 = df2[df2['games_played'] > 40] 
print(df2[df1['save_percentage'] == df2['save_percentage'].max()].iloc[:,:1]),df2[df1['save_percentage'] == df2['save_percentage'].max()].iloc[:,12:13]
#%%
df3 = df1[df1['season'] == '2016-17']
print(df3[df3['saves'] == df3['saves'].max()].iloc[:,:1],df3[df3['saves'] == df3['saves'].max()].iloc[:,10:11])
df3 = df1[df1['season'] == '2015-16']
print(df3[df3['saves'] == df3['saves'].max()].iloc[:,:1],df3[df3['saves'] == df3['saves'].max()].iloc[:,10:11])
df3 = df1[df1['season'] == '2014-15']
print(df3[df3['saves'] == df3['saves'].max()].iloc[:,:1],df3[df3['saves'] == df3['saves'].max()].iloc[:,10:11])
#%%
df1 = pd.read_csv("goalies-2014-2016.csv", sep =';', header = 0, index_col=0, engine='python')
df4 = df1
df4['player1'] = df4[(df1['wins'] > 30)].iloc[:,:1]
df4['player2'] = df4[(df1['season'] == '2015-16') & (df1['wins']>30)].iloc[:,:1]
df4['player3'] = df4[(df1['season'] == '2014-15') & (df1['wins']>30)].iloc[:,:1]
df4

