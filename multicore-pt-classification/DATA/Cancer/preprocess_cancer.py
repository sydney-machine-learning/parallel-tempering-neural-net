import numpy as np


testdata = np.genfromtxt('rtrain.txt')
traindata = np.genfromtxt('rtest.txt')

testdataf = np.zeros((testdata.shape[0], testdata.shape[1]+1))
traindataf = np.zeros((traindata.shape[0], traindata.shape[1]+1))


traindataf[:, :-1] = traindata[:, :]
testdataf[: , :-1] = testdata[:, :]

for index in range(testdata.shape[0]):
    if testdata[index, -1] == 1:
        testdataf[index, -1] = 0
    else:
        testdataf[index, -1] = 1

for index in range(traindata.shape[0]):
    if traindata[index, -1] == 1:
        traindataf[index, -1] = 0
    else:
        traindataf[index, -1] = 1



np.savetxt('ftest.txt', testdataf)
np.savetxt('ftrain.txt', traindataf)