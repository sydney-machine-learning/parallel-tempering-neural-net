import numpy as np 


traindata = np.genfromtxt('rtrain.txt')
testdata = np.genfromtxt('rtest.txt')

traindata = np.c_[traindata, np.ones((traindata.shape[0],1))]
testdata = np.c_[testdata, np.ones((testdata.shape[0],1))]

for index in range(traindata.shape[0]):
    
    if traindata[index, -2] == 0:
        traindata[index, -1] = 1
    
    elif traindata[index, -2] == 1:
        traindata[index, -1] = 0

for index in range(testdata.shape[0]):
    
    if testdata[index, -2] == 0:
        testdata[index, -1] = 1
    
    elif testdata[index, -2] == 1:
        testdata[index, -1] = 0


np.savetxt('ftest.csv', testdata, delimiter=',')
np.savetxt('ftrain.csv', traindata, delimiter=',')


print traindata

print testdata
