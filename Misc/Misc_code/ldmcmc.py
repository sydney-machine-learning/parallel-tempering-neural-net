# !/usr/bin/python


# MCMC Random Walk for Feedforward Neural Network for One-Step-Ahead Chaotic Time Series Prediction

# Data (Sunspot and Lazer). Taken' Theorem used for Data Reconstruction (Dimension = 4, Timelag = 2).
# Data procesing file is included.

# RMSE (Root Mean Squared Error)

# based on: https://github.com/rohitash-chandra/FNN_TimeSeries
# based on: https://github.com/rohitash-chandra/mcmc-randomwalk


# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2017 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra



# Reference for publication for this code
# [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
# (to be addeded on https://www.researchgate.net/profile/Rohitash_Chandra)





import matplotlib.pyplot as plt
import numpy as np
import random
import time
from scipy.stats import multivariate_normal
from scipy.stats import norm
import math
import os


# An example of a class
class Network:
	def __init__(self, Topo, Train, Test, learn_rate):
		self.Top = Topo  # NN topology [input, hidden, output]
		self.TrainData = Train
		self.TestData = Test
		np.random.seed()
		self.lrate = learn_rate

		self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
		self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
		self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
		self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

		self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
		self.out = np.zeros((1, self.Top[2]))  # output last layer

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sampleEr(self, actualout):
		error = np.subtract(self.out, actualout)
		sqerror = np.sum(np.square(error)) / self.Top[2]
		return sqerror

	def ForwardPass(self, X):
		z1 = X.dot(self.W1) - self.B1
		self.hidout = self.sigmoid(z1)  # output of first hidden layer
		z2 = self.hidout.dot(self.W2) - self.B2
		self.out = self.sigmoid(z2)  # output second hidden layer

	def BackwardPass(self, Input, desired):
		out_delta = (desired - self.out) * (self.out * (1 - self.out))
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

		#self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
		#self.B2 += (-1 * self.lrate * out_delta)
		#self.W1 += (Input.T.dot(hid_delta) * self.lrate)
		#self.B1 += (-1 * self.lrate * hid_delta)

		layer = 1  # hidden to output
		for x in range(0, self.Top[layer]):
			for y in range(0, self.Top[layer + 1]):
				self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
		for y in range(0, self.Top[layer + 1]):
			self.B2[y] += -1 * self.lrate * out_delta[y]

		layer = 0  # Input to Hidden
		for x in range(0, self.Top[layer]):
			for y in range(0, self.Top[layer + 1]):
				self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
		for y in range(0, self.Top[layer + 1]):
			self.B1[y] += -1 * self.lrate * hid_delta[y]

	def decode(self, w):
		w_layer1size = self.Top[0] * self.Top[1]
		w_layer2size = self.Top[1] * self.Top[2]

		w_layer1 = w[0:w_layer1size]
		self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

		w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
		self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
		self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
		self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


	def encode(self):
		w1 = self.W1.ravel()
		w2 = self.W2.ravel()
		w = np.concatenate([w1, w2, self.B1, self.B2])
		return w

	def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for i in range(0, depth):
			for j in range(0, size):
				pat = j
				Input = data[pat, 0:self.Top[0]]
				Desired = data[pat, self.Top[0]:]
				self.ForwardPass(Input)
				self.BackwardPass(Input, Desired)

		w_updated = self.encode()

		return  w_updated

	def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for i in range(0, size):  # to see what fx is produced by your current weight update
			Input = data[i, 0:self.Top[0]]
			self.ForwardPass(Input)
			fx[i] = self.out

		return fx



# --------------------------------------------------------------------------

# -------------------------------------------------------------------


class MCMC:
	def __init__(self, samples, traindata, testdata, topology):
		self.samples = samples  # NN topology [input, hidden, output]
		self.topology = topology  # max epocs
		self.traindata = traindata  #
		self.testdata = testdata
		self.num_param = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
		self.path = "mcmcresults"
		# ----------------

	def rmse(self, predictions, targets):
		return np.sqrt(((predictions - targets) ** 2).mean())

	def likelihood_func(self, neuralnet, data, w, tausq):
		y = data[:, self.topology[0]]
		fx = neuralnet.evaluate_proposal(data, w)
		rmse = self.rmse(fx, y)
		loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
		return [np.sum(loss), fx, rmse]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
		h = self.topology[1]  # number hidden neurons
		d = self.topology[0]  # number input neurons
		part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
		log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss
	def plot_figure(self, list, title): 

		list_points =  list

		fname = self.path
		width = 9 

		font = 9

		fig = plt.figure(figsize=(10, 12))
		ax = fig.add_subplot(111)
 

		slen = np.arange(0,len(list),1) 
		 
		fig = plt.figure(figsize=(10,12))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)
	
		ax1 = fig.add_subplot(211) 

		n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', normed=False)  
 
  
		color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel('Parameter values', size= font+1)
	
		ax2 = fig.add_subplot(212)

		list_points = np.asarray(np.split(list_points,  1 ))
 

 

		ax2.set_facecolor('#f2f2f3') 
		ax2.plot( list_points.T , label=None)
		ax2.set_title(r'Trace plot',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel('Parameter values', size= font+1) 

		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		 
 
		plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()
	def sampler(self, w_limit, tau_limit):

		# ------------------- initialize MCMC
		testsize = self.testdata.shape[0]
		trainsize = self.traindata.shape[0]
		samples = self.samples


		self.sgd_depth = 1

		x_test = np.linspace(0, 1, num=testsize)
		x_train = np.linspace(0, 1, num=trainsize)

		netw = self.topology  # [input, hidden, output]
		y_test = self.testdata[:, netw[0]]
		y_train = self.traindata[:, netw[0]]

		w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

		pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
		pos_tau = np.ones((samples, 1))

		fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
		fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
		rmse_train = np.zeros(samples)
		rmse_test = np.zeros(samples)

		w = np.random.randn(w_size)
		w_proposal = np.random.randn(w_size)

		#step_w = 0.05;  # defines how much variation you need in changes to w
		#step_eta = 0.2; # exp 0


		step_w = w_limit  # defines how much variation you need in changes to w
		step_eta = tau_limit #exp 1
		# --------------------- Declare FNN and initialize
		learn_rate = 0.5

		neuralnet = Network(self.topology, self.traindata, self.testdata, learn_rate)
		#print 'evaluate Initial w'

		pred_train = neuralnet.evaluate_proposal(self.traindata, w)
		pred_test = neuralnet.evaluate_proposal(self.testdata, w)

		eta = np.log(np.var(pred_train - y_train))
		tau_pro = np.exp(eta)

		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0
 

		sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
		np.fill_diagonal(sigma_diagmat, step_w)

		delta_likelihood = 0.5 # an arbitrary position


		prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

		[likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
		[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

		#print likelihood

		naccept = 0
		#print 'begin sampling using mcmc random walk'
		plt.plot(x_train, y_train)
		plt.plot(x_train, pred_train)
		plt.title("Plot of Data vs Initial Fx")
		plt.savefig('mcmcresults/begin.png')
		plt.clf()

		plt.plot(x_train, y_train)


		for i in range(samples - 1):
			timer_0 = time.time()

			w_gd = neuralnet.langevin_gradient(self.traindata, w.copy(), self.sgd_depth) # Eq 8

			w_proposal = w_gd  + np.random.normal(0, step_w, w_size) # Eq 7

			w_prop_gd = neuralnet.langevin_gradient(self.traindata, w_proposal.copy(), self.sgd_depth)

			diff_prop =  np.log(multivariate_normal.pdf(w, w_prop_gd, sigma_diagmat)  - np.log(multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat)))

			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = math.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
																				tau_pro)
			[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
																			tau_pro)

			# likelihood_ignore  refers to parameter that will not be used in the alg.

			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
											   tau_pro)  # takes care of the gradients


			diff_prior = prior_prop - prior_current

			diff_likelihood = likelihood_proposal - likelihood

			mh_prob = min(1, math.exp(diff_likelihood + diff_prior + diff_prop))



			u = random.uniform(0, 1)

			if u < mh_prob:
				# Update position
				#print    i, ' is accepted sample'
				naccept += 1
				likelihood = likelihood_proposal
				prior_current = prior_prop
				w = w_proposal
				eta = eta_pro
				print(i)
				#print  likelihood, prior_current, diff_prop, rmsetrain, rmsetest, w, 'accepted'
				#print w_proposal, 'w_proposal'
				#print w_gd, 'w_gd'

				#print w_prop_gd, 'w_prop_gd'

				pos_w[i + 1,] = w_proposal
				pos_tau[i + 1,] = tau_pro
				fxtrain_samples[i + 1,] = pred_train
				fxtest_samples[i + 1,] = pred_test
				rmse_train[i + 1,] = rmsetrain
				rmse_test[i + 1,] = rmsetest

				plt.plot(x_train, pred_train)


			else:
				pos_w[i + 1,] = pos_w[i,]
				pos_tau[i + 1,] = pos_tau[i,]
				fxtrain_samples[i + 1,] = fxtrain_samples[i,]
				fxtest_samples[i + 1,] = fxtest_samples[i,]
				rmse_train[i + 1,] = rmse_train[i,]
				rmse_test[i + 1,] = rmse_test[i,]

				# print i, 'rejected and retained'
		pos_w = pos_w.T.reshape(self.num_param,-1)
		print (naccept, ' num accepted')
		print (naccept / (samples * 1.0), '% was accepted')
		accept_ratio = naccept / (samples * 1.0) * 100
		for s in range(self.num_param):  
			self.plot_figure(pos_w[s,:], 'pos_distri_'+str(s))
		plt.title("Plot of Accepted Proposals")
		plt.savefig('mcmcresults/proposals.png')
		plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
		plt.clf()

		return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio)


def main():
	for problem in range(2,3):

		path = 'mcmcresults'
		try:
			os.makedirs(path)
		except OSError:
			if not os.path.isdir(path):
				raise
		outres = open(path+'/resultspriors.txt', 'w')
		outpos_w = open(path+'/pos_w.txt', 'w')

		hidden = 5
		input = 4  #
		output = 1
 
		x = 3
 
		if x == 3:
			w_limit =  0.02
			tau_limit = 0.2
		#if x == 4:
			#w_limit =  0.02
			#tau_limit = 0.1  



		if problem == 1:
			traindata = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Lazer/test.txt")  #
		if problem == 2:
			traindata = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt")  #
		if problem == 3:
			traindata = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Mackey/test.txt")  #
		if problem == 4:
			traindata = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Lorenz/test.txt")  #
		if problem == 5:
			traindata = np.loadtxt("Data_OneStepAhead/Rossler/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Rossler/test.txt")  #
		if problem == 6:
			traindata = np.loadtxt("Data_OneStepAhead/Henon/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/Henon/test.txt")  #
		if problem == 7:
			traindata = np.loadtxt("Data_OneStepAhead/ACFinance/train.txt")
			testdata = np.loadtxt("Data_OneStepAhead/ACFinance/test.txt")  #

		

		topology = [input, hidden, output]
		t = time.time()

		numSamples = 500000   # need to decide yourself

		mcmc = MCMC(numSamples, traindata, testdata, topology)  # declare class

		[pos_w, pos_tau, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_ratio] = mcmc.sampler(w_limit, tau_limit)
		print ('sucessfully sampled')
		print (time.time() - t)
		burnin = 0.1 * numSamples  # use post burn in samples

		pos_w = pos_w[int(burnin):, ]
		pos_tau = pos_tau[int(burnin):, ]

		fx_mu = fx_test.mean(axis=0)
		fx_high = np.percentile(fx_test, 95, axis=0)
		fx_low = np.percentile(fx_test, 5, axis=0)

		fx_mu_tr = fx_train.mean(axis=0)
		fx_high_tr = np.percentile(fx_train, 95, axis=0)
		fx_low_tr = np.percentile(fx_train, 5, axis=0)

		pos_w_mean = pos_w.mean(axis=0)
		np.savetxt(outpos_w, pos_w_mean, fmt='%1.5f')






		rmse_tr = np.mean(rmse_train[int(burnin):])
		rmsetr_std = np.std(rmse_train[int(burnin):])
		rmse_tes = np.mean(rmse_test[int(burnin):])
		rmsetest_std = np.std(rmse_test[int(burnin):])
		print (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
		np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_ratio), fmt='%1.5f')

		ytestdata = testdata[:, input]
		ytraindata = traindata[:, input]

		plt.plot(x_test, ytestdata, label='actual')
		plt.plot(x_test, fx_mu, label='pred. (mean)')
		plt.plot(x_test, fx_low, label='pred.(5th percen.)')
		plt.plot(x_test, fx_high, label='pred.(95th percen.)')
		plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Test Data vs MCMC Uncertainty ")
		plt.savefig(path+'/mcmcrestest.png')
		plt.savefig(path+'/mcmcrestest.svg', format='svg', dpi=600)
		plt.clf()
		# -----------------------------------------
		plt.plot(x_train, ytraindata, label='actual')
		plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
		plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
		plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
		plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Train Data vs MCMC Uncertainty ")
		plt.savefig(path+'/mcmcrestrain.png')
		plt.savefig(path+'/mcmcrestrain.svg', format='svg', dpi=600)
		plt.clf()

		# mpl_fig = plt.figure()
		# ax = mpl_fig.add_subplot(111)

		# ax.boxplot(pos_w)

		# ax.set_xlabel('[W1] [B1] [W2] [B2]')
		# ax.set_ylabel('Posterior')

		# plt.legend(loc='upper right')

		# plt.title("Boxplot of Posterior W (weights and biases)")
		# plt.savefig(path+'/w_pos.png')
		# plt.savefig(path+'/w_pos.svg', format='svg', dpi=600)

		# plt.clf()


if __name__ == "__main__": main()
