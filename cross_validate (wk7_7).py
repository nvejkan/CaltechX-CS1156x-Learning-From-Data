from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error

#For information: http://scikit-learn.org/stable/modules/cross_validation.html

p_list = [(np.sqrt(np.sqrt(3)+4)),(np.sqrt(np.sqrt(3)-1)),(np.sqrt(9+4*np.sqrt(6))),(np.sqrt(9-np.sqrt(6)))]

model = DummyRegressor(strategy='median')
lin = LinearRegression()

count = 1
for p in p_list:
    X = np.array([[-1,1,p]])
    X = X.T #convert to vector
    Y = np.array([[0,0,1]])
    Y = Y.T #convert to vector

    
    #use the constant model given X,Y
    #cv = k = 3 means using 3-folds cross valiadation
    #scoring method of mean_squared_error
    scores_const = cross_val_score(model, X, Y, cv=3, scoring='neg_mean_squared_error')
    err_const = np.mean(scores_const)

    print("\n\n Using choice",count,"\n")
    print("constant; p =",p,"error =",err_const)

    #use the linear regression model given X,Y
    scores_lin = cross_val_score(lin, X, Y, cv=len(X), scoring='neg_mean_squared_error')
    err_lin = np.mean(scores_lin)
    print("regression; p =",p,"error =",err_lin)

    count = count + 1
    
#answer is C.
