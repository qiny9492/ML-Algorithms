from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
import sklearn


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function
        
        self.x_train = np.zeros((10,10))
        self.y_train = np.zeros((10,))
        self.nb_trainpoints = 0

    def train(self, features: List[List[float]], labels: List[int]):
        # just store the features 
        # features shape: (no.samples,no.features) is 2-D
        # labels shape: (no.samples,) is 1-D 
        # self.y_train is 1-D array shape:(np.trainpoints,)
        self.nb_trainpoints = len(labels)
        np_x = np.array(features)
        np_y = np.array(labels)
        self.x_train = np_x
        self.y_train = np_y
        
 
        # raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        # features shape: (no.samples,no.features) is 2-D
        pred_x = np.array(features)
        nb_testpoints = pred_x.shape[0]
        nb_features = pred_x.shape[1]
        
        # for each test point, calculate its distance to all training points
        # x_test shape: (no.features,) is 1-D
        # x_tr shape: (no.features,) is 1-D
        # dis is a list:length is no.trainpoints
        y_test = []
        for i in range(0,nb_testpoints):
            x_test = pred_x[i]
            dis = []
            for j in range(0,self.nb_trainpoints):
                x_tr = self.x_train[j]
                one_dis = self.distance_function(x_test.tolist(),x_tr.tolist())
                dis.append(one_dis)
            
            # find indices of k-smallest elements in distance array 
            # idx is 1-D array, index start from 0
            # k_idx stores indices of k nearest neighbors
            # knn_labels stores labels of k nearest neightbor
            idx = np.argpartition(np.array(dis),self.k)
            k_idx = idx[:self.k]
            knn_labels = self.y_train[k_idx]
            mode,count = scipy.stats.mode(knn_labels)
            # mode is an 1-D array only contain one element
            # mode.tolist() is an int
            one_label = mode.tolist()
            y_test.extend(one_label)
        
        return y_test
        
  
        #raise NotImplementedError


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
    print(sklearn.__version__)
