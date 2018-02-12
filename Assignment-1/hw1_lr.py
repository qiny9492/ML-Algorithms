from __future__ import division, print_function

from typing import List

import numpy as np
import scipy
import matplotlib.pyplot as plt


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.w_bias = np.zeros((self.nb_features+1, 1))
        
        # 10 is a random number to simulate the number of traing samples
        
        
    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        # features are x, x=[x1,x2, ... xn] of one sample, where n is nb_features
        # number of samples = len(features)
        # values are y, we need to find w that y=w*x+b
        # number of features is max to 10
        # values are real values
        np_x = np.array(features)
        np_y = np.array(values)
        
        
        nb_samples = len(values)
        y = np_y.reshape(nb_samples,1)
        
        # set x0=1, add x0 to features and get features_bias
        all_one = np.ones((nb_samples,1))
        x_bias = np.hstack((all_one,np_x))
        
        # w = (x^T*x)^(-1)*x^T*x      
        xt_dot_x = np.dot(x_bias.transpose(),x_bias)
        xt_dot_y = np.dot(x_bias.transpose(),y)
        self.w_bias = np.dot(np.linalg.inv(xt_dot_x),xt_dot_y)
        #print(self.w_bias.shape)
        

        # raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        # h is the prediction h = x_bias * w_bias
        np_x = np.array(features)
        
        # set x0=1, add x0 to features and get features_bias
        nb_samples = np_x.shape[0]
        all_one = np.ones((nb_samples,1))
        x_bias = np.hstack((all_one,np_x))
        
        np_h = np.dot(x_bias,self.w_bias)
        h = np_h.tolist()
        return h
        

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        weights = self.w_bias.tolist()
        return weights
        


class LinearRegressionWithL2Loss:
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features

    def train(self, features: List[List[float]], values: List[float]):
        """TODO : Complete this function"""
        
        np_x = np.array(features)
        np_y = np.array(values)
        
        
        nb_samples = len(values)
        y = np_y.reshape(nb_samples,1)
        
        # set x0=1, add x0 to features and get features_bias
        all_one = np.ones((nb_samples,1))
        x_bias = np.hstack((all_one,np_x))
        
        # w = (x^T*x + lambda * I)^(-1)*x^T*x   
        i_no_rows = x_bias.shape[1]
        i = np.identity(i_no_rows) 
        
        # x^T*x
        xt_dot_x = np.dot(x_bias.transpose(),x_bias)
        # x^T*x + lambda * I
        xt_dot_x_plus = xt_dot_x + self.alpha * i
        
        xt_dot_y = np.dot(x_bias.transpose(),y)
        
        self.w_bias = np.dot(np.linalg.inv(xt_dot_x_plus),xt_dot_y)
        # raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[float]:
        """TODO : Complete this function"""
        # h is the prediction h = x_bias * w_bias
        np_x = np.array(features)
        
        # set x0=1, add x0 to features and get features_bias
        nb_samples = np_x.shape[0]
        all_one = np.ones((nb_samples,1))
        x_bias = np.hstack((all_one,np_x))
        
        np_h = np.dot(x_bias,self.w_bias)
        h = np_h.tolist()
        return h
    
        # raise NotImplementedError

    def get_weights(self) -> List[float]:
        """TODO : Complete this function"""
        """
        for a model y = 1 + 3 * x_0 - 2 * x_1,
        the return value should be [1, 3, -2].
        """
        # w should include w0
        weights = self.w_bias.tolist()
        return weights
    
        #raise NotImplementedError


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
    #print(sklearn.__version__)
