import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm

    Return:
    - train_obj: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here
    
    # X and w have already included the bias term.
    N, D = X.shape
    w_1d = np.reshape(w,(D,))
    
    
    # first_term is a scalar
    # b shape is (N,1)
    w_norm = np.linalg.norm(w_1d)
    first_term = (lamb / 2) * (w_norm ** 2)

    
    
    a = np.dot(X,w_1d)
    b = 1 - np.multiply(y,a)
    
    b = np.reshape(b,(N,1))
    all_zeros = np.zeros((N,1))
    # comp_mtx shape is (N,2)
    # max_mtx shape is (N,)
    # second_term is a scalar
    comp_mtx = np.hstack((all_zeros,b))
    max_mtx = np.amax(comp_mtx, axis=1)
    second_term = (np.sum(max_mtx)) / N
    
    #obj_value = np.amin(first_term + second_term)
    obj_value = first_term + second_term
    return obj_value


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the maximum number of iterations to update parameters

    Returns:
    - learnt w
    - train_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]
    w_t = np.reshape(w,(D,))
    train_obj = []

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch

        # you need to fill in your solution here
        x = Xtrain[A_t]
        y = ytrain[A_t]
        
        
        
        a = np.dot(x,w_t)
        product = np.multiply(y,a)
        
        # set A_t_plus
        x_plus = x[product<1]
        y_plus = y[product<1]
        
        # set learning rate
        eta = 1 / (lamb * iter)
        
        # set w_t_plushalf
        first_term = (1 - eta * lamb) * w_t
        
        num = x_plus.shape[0]
        sum_prod = 0
        for i in range(0,num):
            sum_prod = sum_prod + y_plus[i] * x_plus[i]
        
        second_term = sum_prod * eta / k
        w_t_plushalf = first_term + second_term
        
        # set w_t_plusone
        w_norm = np.linalg.norm(w_t_plushalf)  
        second = (1 / (lamb ** 0.5)) / w_norm
        
        w_t_plusone = min(1,second) * w_t_plushalf
        
        # update the weight
        w_t = w_t_plusone
        
        # calculate objective function value
        obj_value = objective_function(Xtrain, ytrain, w_t, lamb)
        train_obj.append(obj_value)
        
    w = np.reshape(w_t,(D,1))
    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w, t = 0.):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
    - t: threshold, when you get the prediction from SVM classifier, it should be real number from -1 to 1. Make all prediction less than t to -1 and otherwise make to 1 (Binarize)

    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here
    x = np.array(Xtest)
    y = np.array(ytest)
    N, D = x.shape
    w_1d = np.reshape(w,(D,))
    
    # pred shape is (N,)
    # y shape is (N,)
    pred = np.dot(x,w_1d)
    pred[pred < t] = -1
    pred[pred >= t] = 1
    comp = np.multiply(y,pred)
    pos = np.count_nonzero(comp == 1)
    neg = np.count_nonzero(comp == -1)
    test_acc = pos / (pos + neg)
    
    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
