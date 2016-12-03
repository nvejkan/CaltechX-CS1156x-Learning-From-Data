from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score, ShuffleSplit
#readme
#http://scikit-learn.org/stable/modules/svm.html#svm-kernels


def n_vs_all(Y,n):
    converted_Y = np.copy(Y)
    converted_Y[ Y == n ] = 1
    converted_Y[ Y != n ] = -1
    c, r = converted_Y.shape
    converted_Y = converted_Y.reshape(c,)
    return converted_Y

def n1_vs_n2(X,Y,n1,n2):
    #return new X and Y for training
    #n1 label will be 1
    #n2 label will be -1
    #the rests(X and Y) will be deleted
    converted_Y = np.copy(Y)
    converted_X = np.copy(X)
    del_indexes = np.where( (converted_Y != n1) & (converted_Y != n2) )
    
    #delete others
    converted_Y = np.delete(converted_Y,del_indexes,0) #axis 0 will sustain the current shape
    converted_X = np.delete(converted_X,del_indexes,0)

    #convert the labels
    converted_Y[ converted_Y == n1 ] = 1
    converted_Y[ converted_Y == n2 ] = -1

    #!!!NEED TO RESHAPE THE Y FROM (c,1) to (c,)
    #the cross_val_score requires this shape!
    c, r = converted_Y.shape
    converted_Y = converted_Y.reshape(c,)

    return converted_X,converted_Y

def error(Y_predict,Y_true):
    error = np.copy((Y_predict != Y_true).astype(int))
    return np.mean(error)

def question_2_3_4():
    for i in range(0,10):
        Yi = n_vs_all(train_Y,i)
        model = SVC(C=0.01,kernel='poly',degree=2,coef0=1,gamma=1)

        #train
        model.fit(train_X,Yi)
        
        #predict
        #Y_train_predict = np.array([model.predict(train_X)]).T
        #Ein1 = error(Y_train_predict,Yi)
        
        #get Ein from the model itself
        Ein2 = 1 - model.score(train_X,Yi)
        n_sv = np.size(model.support_)
        #print(i,Ein1,Ein2)
        print(i,Ein2,n_sv)

def question_5():
    # 1 vs 5
    for cx in [0.001,0.01,0.1,1]:
        Xi, Yi = n1_vs_n2(train_X,train_Y,1,5)
        model = SVC(C=cx,kernel='poly',degree=2,coef0=1,gamma=1)

        #train
        model.fit(Xi,Yi)
        
        #predict
        #Y_train_predict = np.array([model.predict(train_X)]).T
        #Ein1 = error(Y_train_predict,Yi)
        
        #get Ein from the model itself
        Ein = 1 - model.score(Xi,Yi)
        n_sv = np.size(model.support_)
        print(cx,Ein,n_sv)

def question_6():
    # 1 vs 5
    for cx in [0.0001,0.001,0.01,1]:
        for qx in [2,5]:
            
            Xi, Yi = n1_vs_n2(train_X,train_Y,1,5)
            
            model = SVC(C=cx,kernel='poly',degree=qx,coef0=1,gamma=1) # degree param is Q

            #train
            model.fit(Xi,Yi)
            
            #predict
            #Y_train_predict = np.array([model.predict(train_X)]).T
            #Ein1 = error(Y_train_predict,Yi)
            
            #get Ein from the model itself
            Ein = 1 - model.score(Xi,Yi)
            n_sv = np.size(model.support_)
            print("C = {0}, Q = {1}, Ein = {2}, #SV= {3}".format(cx,qx,Ein,n_sv))         

     
def question_7():
    from numpy.random import randint
    # 1 vs 5
    win_dict = {}
    win_dict[0.0001] = 0
    win_dict[0.001] = 0
    win_dict[0.01] = 0
    win_dict[0.1] = 0
    win_dict[1] = 0
    
    Xi, Yi = n1_vs_n2(train_X,train_Y,1,5)
    
    for t in range(0,100):
    
        #run for 100 times

        #init
        min_error = 1000
        winner = None #if someone beat the min_error change this
        
        for cx in [0.0001,0.001,0.01,0.1,1]:
            
            model = SVC(C=cx,kernel='poly',degree=2,coef0=1.0,gamma=1.0) # degree param is Q
            #use the constant model given X,Y
            #cv = k = 10 means using 10-folds cross valiadation
            #scoring method of mean_squared_error

            rs = randint(2147483647)
            cv = ShuffleSplit(n_splits = 10 ,random_state=rs)
            #scores = cross_val_score(model, Xi, Yi, cv=cv, scoring='f1') #f1 = for binary targets
            scores = cross_val_score(model, Xi, Yi, cv=cv)
            Ein = 1- np.mean(scores)

            if Ein < min_error:
                min_error = Ein
                winner = cx
        
        #end of each round
        win_dict[winner] = win_dict.get(winner) + 1

    #end all rounds
    print(win_dict)

def question_8():
    from numpy.random import randint
    # 1 vs 5
    
    Xi, Yi = n1_vs_n2(train_X,train_Y,1,5)
    
    Ein_list = []
    for t in range(0,100):
        
        cx = 0.001
            
        model = SVC(C=cx,kernel='poly',degree=2,coef0=1.0,gamma=1.0) # degree param is Q
        #use the constant model given X,Y
        #cv = k = 10 means using 10-folds cross valiadation
        #scoring method of mean_squared_error

        rs = randint(2147483647)
        cv = ShuffleSplit(n_splits = 10 ,random_state=rs)
        #scores = cross_val_score(model, Xi, Yi, cv=cv, scoring='f1') #f1 = for binary targets
        scores = cross_val_score(model, Xi, Yi, cv=cv)
        Ein = 1- np.mean(scores)

        Ein_list.append(Ein)
    #end all rounds
    print("C = {0} Ecv = {1}".format(cx,np.mean(Ein_list)))

def question_9_10():
    # 1 vs 5
    for cx in [0.01,1,100,10**4,10**6]:
        Xi, Yi = n1_vs_n2(train_X,train_Y,1,5)
        Xi_test, Yi_test = n1_vs_n2(test_X,test_Y,1,5)
        
        model = SVC(C=cx,kernel='rbf',gamma=1.0)

        #train
        model.fit(Xi,Yi)
        
        #predict
        #Y_train_predict = np.array([model.predict(train_X)]).T
        #Ein1 = error(Y_train_predict,Yi)
        
        #get Ein from the model itself
        Ein = np.round(1 - model.score(Xi,Yi),5)
        Eout = np.round(1 - model.score(Xi_test,Yi_test),5)
        n_sv = np.size(model.support_)
        print("C = {0:10}, Ein = {1}, #SV= {2},Eout = {3}".format(cx,Ein,n_sv,Eout))

if __name__ == "__main__":
    #load training set
    train = np.genfromtxt('features.train.txt',dtype=np.float)
    train_Y = np.array([train[:,0]]).T
    train_X = train[:,1:]

    #load test set
    test = np.genfromtxt('features.test.txt',dtype=np.float)
    test_Y = np.array([test[:,0]]).T
    test_X = test[:,1:]

    #QUESTIONS
    print("\nquestion 2,3,4")
    question_2_3_4()
    print("\nquestion 5")
    question_5()
    print("\nquestion 6")
    question_6()
    print("\nquestion 7")
    question_7()
    print("\nquestion 8")
    question_8()
    print("\nquestion 9-10")
    question_9_10()
