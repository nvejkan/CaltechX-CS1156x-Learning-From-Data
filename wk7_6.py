import numpy as np

e1_mean = []
e2_mean = []
e_mean = []
for t in range(0,100): #run the test 100 times
    #random
    e1 = np.random.uniform(0,1,1000)
    e2 = np.random.uniform(0,1,1000)
    
    #assign e from selecting min of e1 and e2 for each element
    e = np.minimum(e1,e2)

    #find mean and keep it for t times
    e1_mean.append(np.mean(e1))
    e2_mean.append(np.mean(e2))
    e_mean.append(np.mean(e))

#summary
print("e1 mean: ",np.mean(e1_mean))
print("e2 mean: ",np.mean(e2_mean))
print("e mean: ",np.mean(e_mean))
