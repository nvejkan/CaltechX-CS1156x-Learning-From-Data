import numpy as np
from sklearn.linear_model import Ridge
import re


def read_data(input_data):
    N = len(input_data)
    # x = 1 x1 x2
    X = np.ones((N,3))
    Y = np.ones((N,1))
    for i in range(0,N):
        X[i,1] = eval(input_data[i][0])
        X[i,2] = eval(input_data[i][1])

        Y[i,0] = eval(input_data[i][2])

    return X,Y
def transform(X):
    N = len(X)
    Z = np.ones((N,8))
    Z[:,1] = X[:,1]
    Z[:,2] = X[:,2]
    Z[:,3] = X[:,1]**2
    Z[:,4] = X[:,2]**2
    Z[:,5] = X[:,1] * X[:,2]
    Z[:,6] = np.abs(X[:,1] - X[:,2])
    Z[:,7] = np.abs(X[:,1] + X[:,2])
    return Z

def error(Y_predict,Y_true):
    error = (Y_predict != Y_true).astype(int)
    return np.mean(error)

#input file
train = 'in.dta'
infile = open(train)
train_input = [ re.split(r'\s*',i.strip()) for i in infile]

X,Y = read_data(train_input)
Z = transform(X)

#output file
test = 'out.dta'
infile = open(test)
test_input = [ re.split(r'\s*',i.strip()) for i in infile]

X_out,Y_out = read_data(test_input)
Z_out = transform(X_out)

#train Prob3
regu = Ridge(alpha=10**-3) #k=-3; alpha = lambda from Tikhonov regularization
regu.fit(Z, Y)

#predict
Y_in_predict = np.sign(regu.predict(Z))
Y_out_predict = np.sign(regu.predict(Z_out))

#print("Mean squared error:", np.mean(( Y_in_predict - Y ) ** 2 ))
print("Prob 3.")
print("Ein:", error(Y_in_predict,Y))
print("Eout:", error(Y_out_predict,Y_out))


#train Prob4
regu = Ridge(alpha=10**3) #k=3; alpha = lambda from Tikhonov regularization
regu.fit(Z, Y)

#predict
Y_in_predict = np.sign(regu.predict(Z))
Y_out_predict = np.sign(regu.predict(Z_out))

#print
print("Prob 4.")
print("Ein:", error(Y_in_predict,Y))
print("Eout:", error(Y_out_predict,Y_out))

#prob5
for k in [2,1,0,-1,2]:
    regu = Ridge(alpha=10**k) #k=3; alpha = lambda from Tikhonov regularization
    regu.fit(Z, Y)

    #predict
    Y_in_predict = np.sign(regu.predict(Z))
    Y_out_predict = np.sign(regu.predict(Z_out))

    #print
    print("\nProb 5. k =",k)
    print("Ein:", error(Y_in_predict,Y))
    print("Eout:", error(Y_out_predict,Y_out))
