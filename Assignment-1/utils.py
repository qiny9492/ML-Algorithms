from typing import List

import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    
    y_one = np.array(y_true)
    y = y_one.reshape(len(y_true),1)
    h = np.array(y_pred)
    mse = np.mean((y-h)**2)
    return mse
    
    # raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    real = np.array(real_labels)
    pred = np.array(predicted_labels)
    
    tp = np.sum(np.logical_and(pred == 1, real == 1))
    #tn = np.sum(np.logical_and(pred == 0, real == 0))
    fp = np.sum(np.logical_and(pred == 1, real == 0))
    fn = np.sum(np.logical_and(pred == 0, real == 1))
    
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
        
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * ((precision * recall)/(precision + recall))
        
    
    return f1

    # raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    # polynomial features is x = x1,...xn,x21,...,x2n,...,xk1,...xkn
    # n is no.features, k is k-th degree polynomial
    first_x = np.array(features)
    
    # initialize poly features as first order version
    poly_x = first_x

    # calculate high-order features and append after the first-order
    for degree in range(2,k+1):
        kth_x = np.power(first_x,degree)
        poly_x = np.hstack((poly_x,kth_x))
    
    # convert nparray into list
    poly = poly_x.tolist()
    return poly

    #raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    # d(x,y) = sqrt((x-y)^2)
    
    x = np.array(point1)
    y = np.array(point2)
    
    dis = np.linalg.norm(x-y)
    
    return dis
     
    # raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    # d(x,y) = <x,y>
    x = np.array(point1)
    y = np.array(point2)
    
    dis = np.inner(x,y)
    
    return dis


    # raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    # d(x,y) = exp(-0.5*||x-y||^2)
    x = np.array(point1)
    y = np.array(point2)
    x_minus_y_sqrt = (np.linalg.norm(x-y))**2
    dis = np.exp(-0.5*(x_minus_y_sqrt))
    
    return dis
    
    #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        x_all = np.array(features)
        a = np.linalg.norm(x_all,axis=1)
        x_nml = x_all/a.reshape(a.shape[0],1)
        
        # change all NaN value into 0
        x_nml[np.isnan(x_nml)] = 0
        x_list = x_nml.tolist()
        return x_list
    
        #raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        # x_scaled = (x - min)/(max - min)
        x = np.array(features)
        # nb_features = x.shape[1]
        nb_s = x.shape[0]
        # max and min of each column(feature)
        x_max = np.amax(x,axis=0)
        x_min = np.amin(x,axis=0)
        # min_mtx is the same shape as x
        min_mtx = np.tile(x_min,(nb_s,1))
        
        x_scaled = (x - min_mtx)/(x_max - x_min)
        # change all NaN value into 0
        x_scaled[np.isnan(x_scaled)] = 0
        
        features_scaled = x_scaled.tolist()
        
        return features_scaled
            

        
        #raise NotImplementedError