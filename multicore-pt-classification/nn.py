import sklearn
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def main():

	for i in range(3, 9) : 

		problem = i
		
		separate_flag = False
		
		if problem == 1: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-red.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-red"
			hidden = 50
			ip = 11 #input
			output = 10
			NumSample = 50000 
		if problem == 3: #IRIS
			data  = np.genfromtxt('DATA/iris.csv',delimiter=';')
			classes = data[:,4].reshape(data.shape[0],1)-1
			features = data[:,0:4]
 
			separate_flag = True
			name = "iris"
			hidden = 12
			ip = 4 #input
			output = 3
			NumSample = 50000 
		if problem == 2: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-white.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-white"
			hidden = 50
			ip = 11 #input
			output = 10
			NumSample = 50000 
		if problem == 4: #Ionosphere
			traindata = np.genfromtxt('DATA/Ions/Ions/ftrain.csv',delimiter=',')[:,:-1]
			testdata = np.genfromtxt('DATA/Ions/Ions/ftest.csv',delimiter=',')[:,:-1]
			name = "Ionosphere"
			hidden = 50
			ip = 34 #input
			output = 2
			NumSample =50000 
		if problem == 5: #Cancer
			traindata = np.genfromtxt('DATA/Cancer/ftrain.txt',delimiter=' ')[:,:-1]
			testdata = np.genfromtxt('DATA/Cancer/ftest.txt',delimiter=' ')[:,:-1]
			name = "Cancer"
			hidden = 12
			ip = 9 #input
			output = 2
			NumSample =50000
	
		if problem == 6: #Bank additional
			data = np.genfromtxt('DATA/Bank/bank-processed.csv',delimiter=';')
			classes = data[:,20].reshape(data.shape[0],1)
			features = data[:,0:20]
			separate_flag = True
			name = "bank-additional"
			hidden = 50
			ip = 20 #input
			output = 2
			NumSample = 50000 
		if problem == 7: #PenDigit
			traindata = np.genfromtxt('DATA/PenDigit/train.csv',delimiter=',')
			testdata = np.genfromtxt('DATA/PenDigit/test.csv',delimiter=',')
			name = "PenDigit"
			for k in range(16):
				mean_train = np.mean(traindata[:,k])
				dev_train = np.std(traindata[:,k]) 
				traindata[:,k] = (traindata[:,k]-mean_train)/dev_train
				mean_test = np.mean(testdata[:,k])
				dev_test = np.std(testdata[:,k]) 
				testdata[:,k] = (testdata[:,k]-mean_test)/dev_test
			ip = 16
			hidden = 30
			output = 10

			NumSample = 50000 
		if problem == 8: #Chess
			data  = np.genfromtxt('DATA/chess.csv',delimiter=';')
			classes = data[:,6].reshape(data.shape[0],1)
			features = data[:,0:6]
			separate_flag = True
			name = "chess"
			hidden = 25
			ip = 6 #input
			output = 18

			NumSample = 50000
	
		if separate_flag is True:
			#Normalizing Data
			for k in range(ip):
				mean = np.mean(features[:,k])
				dev = np.std(features[:,k])
				features[:,k] = (features[:,k]-mean)/dev
			train_ratio = 0.7 #Choosable
			indices = np.random.permutation(features.shape[0])
			x_train = features[indices[:np.int(train_ratio*features.shape[0])],:]
			y_train = classes[indices[:np.int(train_ratio*features.shape[0])],:].ravel()
			x_test = features[indices[np.int(train_ratio*features.shape[0])]:,:]
			y_test = classes[indices[np.int(train_ratio*features.shape[0])]:,:].ravel()
		adam_train = []
		adam_test =[]
		sgd_tr = []
		sgd_te = []
		res = open('result_comparison.txt','a+')
		for l in range(30):
			mlp_adam = MLPClassifier(hidden_layer_sizes=(hidden, ), activation='relu', solver='adam', alpha=0.1,max_iter=100000, tol=0)
			mlp_adam.fit(x_train,y_train)
			train_acc = mlp_adam.score(x_train, y_train)
			test_acc = mlp_adam.score(x_test, y_test)
			print('ADAM', name,train_acc, test_acc)
			print('ADAM', name,train_acc, test_acc, file = res)
			adam_train.append(train_acc)
			adam_test.append(test_acc)
			#np.savetxt(res, np.asarray([1, problem,train_acc, test_acc]), fmt='%1.2f')
			
			mlp_sgd = MLPClassifier(hidden_layer_sizes=(hidden, ), activation='relu', solver='sgd', alpha=0.1,max_iter=100000, tol=0)
			mlp_sgd.fit(x_train,y_train)
			train_acc = mlp_sgd.score(x_train, y_train)
			test_acc = mlp_sgd.score(x_test, y_test)
			print('SGD', name,train_acc, test_acc)
			print('SGD', name,train_acc, test_acc , file = res)
			sgd_tr.append(train_acc)
			sgd_te.append(test_acc)
			#np.savetxt(res, (2, problem,train_acc, test_acc), fmt='%1.2f')
			
			# rf = RandomForestClassifier()
			# rf.fit(x_train,y_train)
			# train_acc = rf.score(x_train,y_train)
			# test_acc = rf.score(x_test, y_test)
			# print('RF', name,train_acc, test_acc, file = res)
			# #np.savetxt(res, (3, problem,train_acc, test_acc), fmt='%1.2f')
		
		print('ADAM net train', name, np.mean(adam_train[:]), np.std(adam_train[:]), np.max(adam_train[:]), file=res)
		print('ADAM net test', name, np.mean(adam_test[:]), np.std(adam_test[:]), max(adam_test[:]), file=res)
		print('SGD net train', name, np.mean(sgd_tr[:]), np.std(sgd_tr[:]), max(sgd_tr[:]), file=res)
		print('SGD net test', name, np.mean(sgd_te[:]), np.std(sgd_te[:]), max(sgd_te[:]), file=res)
		res.close()
if __name__ == "__main__": main()