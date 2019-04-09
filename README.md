# Parallel Tempering for Neural Networks for Chaotic Time Series Prediction

Parallel Tempering is a sampling technique for bayesian inference. It is an extension of the Monte Carlo Markov Chains Method.

## Running PT Classification 

1. The main code: https://github.com/sydney-machine-learning/parallel-tempering-neural-net/blob/master/multicore-pt-classification/pt_classifier_hybridpos.py. The langevin grad gives better results and you can turn it to True/False in main. 

2. You need to use subline text editor and run it straight from the terminal. eg. go to the folder using terminal and then: "python pt_classifier_hybridpos.py" would run it. 

3. Also, note that there are two places the code saves the results. one in Desktop - so you need to change the path: line 1036: problemfolder = '/home/rohit/Desktop/PT/PT_hybridrw_langevingrad/' # change this to your directory for results output - produces large datasets 


## SOME CONCLUSIONS FROM THE EXPERIMENTATION:

 1. Increasing Number of Chains with constant maximum temperature and number of samples increase the RMSE values. This is probably due to less number of samples per chain and therefore low training.

 2. Increasing Maximum Temperature with constant number of chains and samples, we find that the optimum temperature is 15 for linear spacing. This is a problem dependent hyperparameter.

 3. Swap ratio's optimum value was found to be 0.125

 4. Using geometric spacing in the temperature ladder, and setting maximum temperature as infinity we get poor RMSE values for increasing number of chains because of the high acceptance rate of chains with higher temperatures.

 5. Using exponential increase in maximum temperature for increasing number of chains while using geometric spacing we still get increasing RMSE values. 

 6. Increasing Maximum Temperature in proportion to the increasing number of chains while using geometric spacing poses almost constant RMSE Values. This is a significant result.

 7. Constant Maximum Temperature for increasing number of chains also gives almost constant RMSE Values. This is a surprising result.

 8. Keeping the Maximum Temperature and number of chains constant respectively at 15 and 10, we increased the number of samples from 8000 per chain to 40000 per chain and found the Parallel Tempering Random Walk reaches the RMSE values of MCMC Random Walk at about 36000 samples per chain (compared to 80000 per chain of MCMC).
