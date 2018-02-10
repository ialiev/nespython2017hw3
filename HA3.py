
# coding: utf-8

# In[4]:

import numpy as np
#1 
class OLS(object):
    
    def __init__(self, a, b):
        self.y = a
        self.x = b
        self.beta=((np.linalg.inv(np.dot(self.x.T,self.x))).dot(self.x.T)).dot(self.y)
        self.sigma=1/(self.x.shape[0]-self.x.shape[1])*np.dot((self.y-np.dot(self.x,self.beta)).T,(self.y-np.dot(self.x,self.beta)))
        self.V=self.sigma*np.linalg.inv(np.dot(self.x.T,self.x))
       
    
    def predict(self, k):
        self.prdict=np.dot(k.T,self.beta)
        self.error=(self.sigma*(1+np.dot(k.T,np.linalg.inv(np.dot(x.T,x))).dot(k))) 
        return (self.prdict,self.error)
    def bounds(self,k):
        self.prdict=np.dot(k,self.beta)
        self.error=(self.sigma*(1+np.dot(k.T,np.linalg.inv(np.dot(x.T,x))).dot(k)))
        self.upper=self.prdict+self.error*scipy.stats.t.ppf(0.95,self.x.shape[0]-self.x.shape[1])
        self.lower=self.prdict-self.error*scipy.stats.t.ppf(0.95,self.x.shape[0]-self.x.shape[1])
        return self.prdict, self.upper, self.lower


# In[5]:

x = np.random.randn(100,3)
y = x.dot(np.array([1,2,3]))+np.random.randn(100)
model=OLS(y,x)


# In[6]:

model.beta


# In[7]:

model.V


# In[8]:

model.predict(np.array([1,0,1]))


# In[9]:

import numpy as np
#2 class works good for fist task however applying it in next task leads to error in calculating error of regression.
#I have tried to fix it for a long time however unsuccessful, so I ecluded it to avoid error. The problem with dimensions
class OLS2(object):  
    def __init__(self, a, b):
        self.y = a
        self.x = b
        self.beta=((np.linalg.inv(np.dot(self.x.T,self.x))).dot(self.x.T)).dot(self.y)
        self.sigma=1/(self.x.shape[0]-self.x.shape[1])*np.dot((self.y-np.dot(self.x,self.beta)).T,(self.y-np.dot(self.x,self.beta)))
        self.V=self.sigma*np.linalg.inv(np.dot(self.x.T,self.x))
    def predict(self, k):
        self.prdict=np.dot(k.T,self.beta)
        return self.prdict


# In[10]:

#2
import scipy.misc
import matplotlib.pyplot as plt
b=np.random.rand(11)
x=np.sort(np.random.rand(200)*10-5)
u = np.random.randn(200)*10
y=np.zeros(200)
for i in range(11):
    y=y+b[i]*x**i/np.math.factorial(i)
y=y+u
plt.scatter(x,y)
plt.xlim(-5, 5)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
plt.show()
plt.savefig('1.png', bbox_inches='tight')


# In[11]:

def plo(k,a,b):
    q= np.empty([200, k])
    for i in range(1,k+1):
        q[:,i-1] =  a ** i
    model=OLS2(b,q)
    predict1=[]
    for i in range(200):
        predict1.append(model.predict(q[i]))
    return predict1


# In[12]:

q1=plo(1,x,y)
q2=plo(2,x,y)
q3=plo(3,x,y)
q4=plo(4,x,y)


# In[13]:

plt.plot(x,q1,label='K=1')
plt.plot(x,q2,label='K=2')
plt.plot(x,q3,label='K=3')
plt.plot(x,q4,label='K=4')
plt.scatter(x,y)
plt.xlim(-5, 5)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
plt.legend(loc='upper left')
plt.plot()
plt.show()
plt.savefig('2.png', bbox_inches='tight')


# In[27]:

#as I have mentioned due to error in applying class, I cannot count error bounds for this task so i cant 
#plot the confidence intervals, it would be great if you could provide a feedback on it (
plt.plot(x,q4,label='K=4')
plt.scatter(x,y)
plt.xlim(-5, 5)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
plt.legend(loc='upper left')
plt.plot()
plt.show()
plt.savefig('3.png', bbox_inches='tight')


# In[14]:

#3
import scipy.stats
import numpy as np
n=100
A=np.random.normal(0,1,(n,n))


# In[15]:

#for columns
a1=A.mean(axis=0)+A.std(axis=0)*scipy.stats.t.ppf(0.95,n-1)/(np.sqrt(n))
b1=A.mean(axis=0)-A.std(axis=0)*scipy.stats.t.ppf(0.95,n-1)/(np.sqrt(n))
out1=(0<a1)&(0>b1)
print(out1)
np.count_nonzero(out1==True)


# In[16]:

#confident intervals for columns
a1.shape=(n,1)
b1.shape=(n,1)
np.concatenate((b1,a1),axis=1)[0:3]


# In[17]:

#for rows
a2=A.mean(axis=1)+A.std(axis=0)*scipy.stats.t.ppf(0.95,n-1)/(np.sqrt(n))
b2=A.mean(axis=1)-A.std(axis=0)*scipy.stats.t.ppf(0.95,n-1)/(np.sqrt(n))
out2=(0<a2)&(0>b2)
print(out2)
np.count_nonzero(out2==True)


# In[18]:

#confident intervals for rows
a2.shape=(n,1)
b2.shape=(n,1)
np.concatenate((b2,a2),axis=1)[0:3]


# In[19]:

#4
import pandas as pd
import numpy as np
df=pd.read_csv('goalies-2014-2016.csv',sep=';')


# In[20]:

##1
df.iloc[0:5,0:6]


# In[21]:

##2
df['AD']=np.abs((df['saves']/df['shots_against'])-df['save_percentage'])
df['AD'].max()


# In[22]:

##3
df[['games_played','goals_against','save_percentage']].mean()


# In[23]:

df[['games_played','goals_against','save_percentage']].std()


# In[24]:

##4
d1=df[(df['season']=='2016-17')&(df['games_played']>40)]
ind=d1['save_percentage'].idxmax()
print(d1[['player','save_percentage']].iloc[[ind]])


# In[25]:

##5
d2=df['season'].unique()
list1=[]
for i in d2:
    list1.append(df[(df['season']==i)]['saves'].idxmax()) 

print(df[['season','player','saves']].iloc[list1])


# In[28]:

##6
d4=pd.DataFrame()
d2=df['season'].unique()
list1=[]
k=1
for i in d2:
    if k==1:
        d4=df[(df['season']==i)][['player','wins']]
        d4=d4.set_index('player')
        d4.columns=['wins'+ str(k)]
        k=k+1
    else:
        d3=df[(df['season']==i)][['player','wins']]
        d3=d3.set_index('player')
        d3.columns=['wins'+ str(k)]
        d4=pd.concat([d4,d3],axis=1)   
        k=k+1
out=d4[(d4['wins1']>30) & (d4['wins2']>30) & (d4['wins3']>30)]
result=pd.concat([out.count(axis=1),out.min(axis=1)],axis=1).astype(int)
result.index.name='player'
result.columns=['seasons','wins']
print(result)

