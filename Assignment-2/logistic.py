from __future__ import division, print_function

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from matplotlib import cm


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression

    Find the optimal parameters w and b for inputs X and y.
    Use the average of the gradients for all training examples to
    update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0


    """
    TODO: add your code here
    """
    # use batch gradient descent to find w and b
    # update rule is
    # w <- w - step * dw
    # b <- b - step * db
    # compute sigmoid function
    # compute a = w^T * x + b
    # h is our predict value, y is real value
    # h shape is (N,)
    # y shape is (N,)
    # b_vector size is (N,)
    
    for i in range(0,max_iterations):
        b_vector = np.full((N,),b)
        a = np.dot(X,w) + b_vector
        h = sigmoid(a)
    
        # gradient of b is db = 1/N * sum(h-y)
        # gradient of w is dw = 1/N * x * (h-y)^T
        db = (np.sum(h-y)) / N
        dw = (np.dot((h-y),X)) / N
        
        # use update rule to update w and b
        w = w - step_size * dw
        b = b - step_size * db
    
    

    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N) 


    """
    TODO: add your code here
    """      
    
    # compute sigmoid function
    b_vector = np.full((N,),b)
    a = np.dot(X,w) + b_vector
    h = sigmoid(a)
    
    # h < 0.5 , pred = 0
    # h >= 0.5, pred = 1
    temp = np.around(h)
    preds = temp.astype(int)
    
    
    
    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: maximum number for iterations to perform

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement a multinomial logistic regression for multiclass 
    classification. Keep in mind, that for this task you may need a 
    special (one-hot) representation of classification labels, where 
    each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i belongs to. 
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0


    """
    TODO: add your code here
    """
    # Step 1 Compute softmax function
    # In softmax function, w has bias, x has 1
    # i.e. we use w_bias and x_bias
    b_2D = b.reshape((C,1))
    w_bias = np.hstack((b_2D,w))
    
    all_one = np.ones((N,1))
    x_bias = np.hstack((all_one,X))
    
    #print(x_bias.shape)
    min_num = min(N,max_iterations)
    
    #print(min_num)
    
    for i in range(0,min_num):
        # for a single training point
        xi_bias = x_bias[i]
        class_index = y[i]
        
        # from given class number to a K-dimensional vector using 1-of-K encoding
        yi = np.zeros(C)
        yi[class_index] = 1
        
        # compute softmax function
        # compute sum(exp(w1*x)+...+exp(wK*x))
        
        exp_sum = 0
        prod = np.zeros(C)
        
        
        
        for j in range(0,C):
            wj_bias = w_bias[j]
            # use prod_bar = prod - max(prod) instead of prod for the softmax function
            #prod[j] = np.dot(wj_bias,xi_bias) - np.amax(np.dot(wj_bias,xi_bias))
            prod[j] = np.dot(wj_bias,xi_bias)
            
            exp_sum += np.exp(prod[j])
        
        soft = (np.exp(prod)) / exp_sum
        
        # Compute gradient
        # gradient = (softmax - yn)*xn
        # soft, yi, xi are all 1-D, need to reshape
        # gradient shape (C,D+1)
        err = soft - yi
        gradient = np.dot(err[:,None],xi_bias[None,:])
        
        # update rule
        w_bias = w_bias - step_size * gradient
        
        

        
    # first column of w_bias is b
    # the rest columns is w
    b = w_bias[:,0]
    w = w_bias[:,1:]

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


    

def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    """
    TODO: add your code here
    """   
    # For each test point, compute wk*xn for each class 
    # b_mtx shape is (N,C)
    # h = wk^t * x + b for each class
    # label k = argmax h
    b_mtx = np.tile(b,(N,1))
    w_t = np.transpose(w)
    h = np.dot(X,w_t) + b_mtx
    preds = h.argmax(axis=1)



    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: maximum number of iterations for gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using binary classifier and 
    one-versus-rest strategy. Recall, that the OVR classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    """
    TODO: add your code here
    """
    
    # For each class, we train a classifier
    # For a classifier to classify No.K class
    # Change the real label into binary label
    # If label != K, let label = 0
    # Else label = 1
    for i in range(0,C):
        label = np.copy(y)
        label[label != i] = -1
        label[label == i] = 1
        label[label == -1] = 0
        w[i], b[i] = binary_train(X,label)


    
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and predictions from binary
    classifier. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    
    """
    TODO: add your code here
    """
    # For each test point, compute sigmoid function for each class 
    # b_mtx shape is (N,C)
    # a = w^t * x + b
    # preds = max h(a) for each test point
    b_mtx = np.tile(b,(N,1))
    w_t = np.transpose(w)
    a = np.dot(X,w_t) + b_mtx
    h = sigmoid(a)
    preds = h.argmax(axis=1)
    
    
    
    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        