from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        '''
            Args : 
            nb_features : Number of features
            max_iteration : maximum iterations. You algorithm should terminate after this
            many iterations even if it is not converged 
            margin is the min value, we use this instead of comparing with 0 in the algorithm
        '''
        
        self.nb_features = nb_features
        # self.w = w0 ... wn is an 1-D array shape (nb.features+1,)
        self.w = [0 for i in range(0,nb_features+1)]
        self.w = np.array(self.w)
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            labels : label of each feature [-1,1]
            
            Returns : 
                True/ False : return True if the algorithm converges else False. 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and should update 
        # to correct weights w. Note that w[0] is the bias term. and first term is 
        # expected to be 1 --- accounting for the bias
        ############################################################################
        
        # x has add the bias
        # x_bias shape is (100,3)
        # y shape is (100,) label{-1,1}
        x_bias = np.array(features)
        y = np.array(labels)
        
        nb_train = x_bias.shape[0]
        
        # normalize x
        a = np.linalg.norm(x_bias,axis=1)
        x_bias_nml = x_bias/a.reshape(a.shape[0],1)
        
        # change all NaN value into 0
        x_bias_nml[np.isnan(x_bias_nml)] = 0
        
        # only update weights if the sample point makes a mistake
        # update rule is w <- w + yi*xi, xi is normalized
        # mistake means that the prediction is not the same as real label
        # in other words, mistake: yi*w^T*xi < = margin (or use 0)
        # self.w shape (nb.features+1, )
        for n in range(0,self.max_iteration):
            for i in range(0,nb_train):
                # xi must be normalized, xi shape(nb.features+1, )
                xi = x_bias_nml[i]
                yi = y[i]
                mistake = yi * np.dot(self.w,xi)
            
                if mistake <= self.margin:
                    self.w = self.w + yi * xi
                
        # check convergence: return True if converges else False. 
        # mis is number of mislabelled samples
        mislabel = 0 
        for j in range(0,nb_train):
            xj = x_bias_nml[j]
            yj = y[j]
            prod = yj * np.dot(self.w,xj)
            
            if prod <= self.margin:
                mislabel = mislabel + 1
        
        
        if mislabel == 0:
            return True
        else:
            return False

        #raise NotImplementedError
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        self.w = np.array(self.w)
        
    def predict(self, features: List[List[float]]) -> List[int]:
        '''
            Args  : 
            features : List of features. First element of each feature vector is 1 
            to account for bias
            
            Returns : 
                labels : List of integers of [-1,1] 
        '''
        ############################################################################
        # TODO : complete this function. 
        # This should take a list of features and labels [-1,1] and use the learned 
        # weights to predict the label
        ############################################################################
        x_test = np.array(features)
        nb_test = x_test.shape[0]
        
        y_test = []
        
        # z = w^T*xi
        for j in range(0,nb_test):
            xj = x_test[j]
            z = np.dot(self.w,xj)
            if z >= 0:
                y_test.append(1)
            else:
                y_test.append(-1)
        
        return y_test
  
        # raise NotImplementedError

    def get_weights(self) -> List[float]:
        weights = self.w.tolist()
        return weights
    