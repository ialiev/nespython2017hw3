import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as ss
import pandas as pd

###############################################################################
################################### TASK 1 ####################################
###############################################################################

class OLS():
    
    def __init__(self, y, X):
        '''Ordinary least squares class. Estimates the regression, predicts 
        values.
        
        Parameters
        ----------
        y : ndarray (n x 1)
            Array of endogenous values
        X : ndarray (n x k)
            Array of exogenous values
        '''
        self.y = y
        self.X = X
        
        self.fit()
        
    def fit(self):
        '''Fit linear regression.
        '''
        self.beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T.dot(self.y))
        
        self.errors = self.y - self.X.dot(self.beta)
        self.sigma_sq = 1./(self.X.shape[0] - self.X.shape[1]) * \
                        self.errors.T.dot(self.errors)
        self.V = self.sigma_sq * np.linalg.inv(self.X.T.dot(self.X))
        
        self.fitted_values = self.X.dot(self.beta)
        
    def predict(self, X):
        '''Predict y based on passed X.
        
        Parameters
        ----------
        X : ndarray (n x k)
            Array of exogenous values
            
        Returns
        -------
        predicted_values : ndarray (n x 1)
            Predicted values
        predicted_variance : ndarray (n x 1)
            Estimated variance of predicted values
        '''
        if len(X.shape) < 2:
            X = X.reshape(X.shape[0],1)
            
        predicted_values, predicted_variance = np.array([]), np.array([])
        for x in X:
            predicted_values = np.append(predicted_values, x.T.dot(self.beta))
            predicted_variance = np.append(predicted_variance, self.sigma_sq*\
                    (1+x.T.dot(np.linalg.inv(self.X.T.dot(self.X)).dot(x))))
    
        return predicted_values, predicted_variance

###############################################################################
################################### TASK 2 ####################################
###############################################################################

# Generating pairs
np.random.seed(40)
n = 200 
k = 11
betas = np.random.rand(k)
X = (np.random.rand(n)-0.5)*10
X = X.reshape(X.shape[0],1)
u = np.random.randn(n)*10

y = np.array([sum([betas[j]*(X[i]**j)/np.prod(range(1,j)) for j in range(k)]) + \
              u[i] for i in range(n)])

# Plotting generated pairs
plt.figure(figsize=(10,7))
plt.scatter(X, y)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
plt.show()

# Fitting models
model = [OLS(y, np.hstack([X**j for j in range(i+1)])) for i in range(1,5)]

base_linspace = np.linspace(-5,5,100)
X_predict = [np.vstack([base_linspace**j for j in range(i+1)]).T
             for i in range(1,5)]

# Getting predictions for linspaces
prediction = [model[i].predict(X_predict[i]) for i in range(len(model))]

# Plotting predictions
plt.figure(figsize=(10,7))
plt.scatter(X, y)
for i in range(len(model)):
    plt.plot(base_linspace, prediction[i][0], 
             label='K = '+str(i+1))
plt.legend(loc=2)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
plt.show()

# Plotting confidence interval for last model predictions
plt.figure(figsize=(10,7))
plt.scatter(X, y)
plt.plot(base_linspace, prediction[3][0], label='K = 4')
plt.fill_between(base_linspace, 
                 prediction[3][0]+prediction[3][1]**0.5*ss.norm.ppf(0.05),
                 prediction[3][0]+prediction[3][1]**0.5*ss.norm.ppf(0.95),
                 alpha=0.1, color='b') 
plt.legend(loc=2)
plt.xlabel('$x_i$')
plt.ylabel('$y_i$')
plt.show()

###############################################################################
################################### TASK 3 ####################################
###############################################################################

A = np.random.randn(100,100)

# Calculating result for culumns
conf_int = np.vstack([A.mean(axis=0) + A.std(axis=0) * ss.t.ppf(0.05, A.shape[0]-1),
                      A.mean(axis=0) + A.std(axis=0) * ss.t.ppf(0.95, A.shape[0]-1)]).T
zero_in_ci = [ci[0] < 0 < ci[1] for ci in conf_int]
print(zero_in_ci)
print(sum(zero_in_ci))

# Calculating result for rows
conf_int = np.vstack([A.mean(axis=1) + A.std(axis=1) * ss.t.ppf(0.05, A.shape[1]-1),
                      A.mean(axis=1) + A.std(axis=1) * ss.t.ppf(0.95, A.shape[1]-1)]).T
zero_in_ci = [ci[0] < 0 < ci[1] for ci in conf_int]
print(zero_in_ci)
print(sum(zero_in_ci))

###############################################################################
################################### TASK 4 ####################################
###############################################################################

# Reading data
data = pd.read_csv('goalies-2014-2016.csv', sep=';')
print(data.iloc[:5,:6])

# Max deviation of save percentage
dev_save_percentage = abs(data['saves'] / data['shots_against'] - 
                          data['save_percentage'])
print(max(dev_save_percentage))

# Mean and standard deviations for games played, goals against and save 
# percentage
print(data[['games_played', 'goals_against', 'save_percentage']].mean())
print()
print(data[['games_played', 'goals_against', 'save_percentage']].std())

# Player with max save percentage among who played over 40 games in 2016-17
print(data.loc[data[(data['season']=='2016-17') & 
                    (data['games_played'] > 40)]['save_percentage'].idxmax(),
               ['player', 'save_percentage']])
    
# Player, having max saves in season
print(data.loc[data.groupby(['season'])['saves'].idxmax()].sort_values(by='n')\
               [['season','player','saves']])

# Players, who won 30 or over games in all 3 seasons
data['over_30_wins'] = data['wins'] >= 30
over_30_wins = (data.groupby(['player'])['over_30_wins'].sum() == 3)
print(over_30_wins[over_30_wins])