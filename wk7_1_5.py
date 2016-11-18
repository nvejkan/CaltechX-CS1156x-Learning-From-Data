import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
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

def train_and_validate(model,k,Z_train,Y_train,Z_val,Y_val,Z_out,Y_out):
    #train
    model.fit(Z_train[:,:k+1], Y_train) # pass Z for every rows but for the first k columns

    #predict
    Y_in_predict = np.sign(model.predict(Z_train[:,:k+1]))
    Y_val_predict = np.sign(model.predict(Z_val[:,:k+1]))
    Y_out_predict = np.sign(model.predict(Z_out[:,:k+1]))
    
    #print("Mean squared error:", np.mean(( Y_in_predict - Y ) ** 2 ))
    
    print("Ein:", error(Y_in_predict,Y_train))
    print("Eval:", error(Y_val_predict,Y_val))
    print("Eout:", error(Y_out_predict,Y_out))
    
def run_test(param_list):
    Z_train,Y_train,Z_val,Y_val,Z_out,Y_out = param_list
    '''
    for i in range(3,8):
        model = Ridge(alpha=10**-1)
        print("\nModel : linear regression with regularization; k =",i)
        train_and_validate(model,i,Z_train,Y_train,Z_val,Y_val,Z_out,Y_out)
    '''
    for i in range(3,8):  
        model = LinearRegression()
        print("\nModel : linear regression; k =",i)
        train_and_validate(model,i,Z_train,Y_train,Z_val,Y_val,Z_out,Y_out)
if __name__ == '__main__':
    
    #input file
    train = 'in.dta'
    infile = open(train)
    train_input = [ re.split(r'\s*',i.strip()) for i in infile]

    X,Y = read_data(train_input)

    #output file
    test = 'out.dta'
    infile = open(test)
    test_input = [ re.split(r'\s*',i.strip()) for i in infile]

    X_out,Y_out = read_data(test_input)
    Z_out = transform(X_out)

    params_probs = [] # lists of parameters for each problem [[p1_params],[p1_params]]
    
    #split training and validation set
    #problem 1,2
    X_train,Y_train = X[0:25,:],Y[0:25,:]
    X_val,Y_val = X[-10:,:],Y[-10:,:]

    Z_train = transform(X_train) 
    Z_val = transform(X_val)
    
    params_probs.append([Z_train.copy(),Y_train.copy(),Z_val.copy(),Y_val.copy(),Z_out.copy(),Y_out.copy()])

    #problem 3,4
    X_val,Y_val = X[0:25,:],Y[0:25,:] # just swap them around
    X_train,Y_train = X[-10:,:],Y[-10:,:]

    Z_train = transform(X_train) 
    Z_val = transform(X_val)
    
    params_probs.append([Z_train.copy(),Y_train.copy(),Z_val.copy(),Y_val.copy(),Z_out.copy(),Y_out.copy()])
    count = 1
    for p in params_probs:
        if count == 1:
            print("\n\n------{0}------\n\n".format("Problem 1,2"))
        else:
            print("\n\n------{0}------\n\n".format("Problem 3,4"))
        run_test(p)
        count = count + 1
