from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Perceptron

def create_fx():
    lin = LinearRegression()
    points = np.random.uniform(-1, 1, (2, 2))

    X = np.array([points[:, 0]]).T
    Y = np.array([points[:, 1]]).T

    lin.fit(X,Y)

    #from y = mx + c 
    w0 = lin.intercept_[0] # c = y-intercept
    w1 = lin.coef_[0][0] # m = slope

    return (w0,w1)

def create_random_points_label(N,w0,w1):
    #from y1-mx1-c = 0 then the point is on the line
    #if y1-mx1-c >= 0 then the point is above the line -> label = 1
    #if y1-mx1-c < 0 then the point is below the line -> label = -1

    points = np.random.uniform(-1,1,(N,2))
    X1 = np.array([points[:, 0]]).T
    X2 = np.array([points[:, 1]]).T #x2 is y in the 2d space

    Y = ( X2 - w1*X1 - w0 >= 0 )
    Y = np.where(Y,1,-1)

    return (points,Y)
def error(Y_predict,Y_true):
    error = (Y_predict != Y_true).astype(int)
    return np.mean(error)

w0,w1 = create_fx()
while(True):
    X,Y = create_random_points_label(10,w0,w1)
    X_test,Y_test = create_random_points_label(100,w0,w1)
    if -1 in Y and 1 in Y and -1 in Y_test and 1 in Y_test :
        break

model = Perceptron()
model.fit(X,Y)
Y_predict = np.array([model.predict(X)]).T
Y_test_predict =  np.array([model.predict(X_test)]).T

print("Ein =",error(Y_predict,Y))
print("Eout =",error(Y_test_predict,Y_test))
