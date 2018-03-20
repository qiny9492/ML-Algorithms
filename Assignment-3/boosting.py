import math
import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
         x = np.array(features)
         N, D = x.shape
         
         h_t_mtx = np.zeros((self.T,N))
         
         i = 0
         for classifier in self.clfs_picked:
             h_t_mtx[i] = np.array(classifier.predict(features))
             i = i + 1
         # h_all shape is (T,N)
         # beta_all shape is (T,)
         beta_t_mtx = np.array(self.betas)
         #print(beta_t_mtx.shape)
         
         # h shape is D,)
         sigma = np.dot(beta_t_mtx,h_t_mtx)
         h = np.zeros((N,))
         h[sigma > 0] = 1
         h[sigma <= 0] = -1
         
         h = h.astype(int)
         h_list = h.tolist()
         
         return h_list

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
    
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
         clfs_list = list(self.clfs)
         
         x = np.array(features)
         y = np.array(labels)
         N, D = x.shape
         w_t = np.ones((N,)) * (1/N)
         
         
         # h_collection store all the classifiers in an array
         # get all the possible classifier in the set
         # h_collection shape is (num_clfs,N)
         h_collection = np.zeros((self.num_clf,N))
         i = 0
         for classifier in self.clfs:
             h_collection[i] = np.array(classifier.predict(features))
             i = i + 1
         
         # y shape is (N,)
         # repeat y num_clf times
         # y_mtx shape is (num_clf,N)
         # indicator shape is (num_clf,N)
         y_mtx = np.tile(y,(self.num_clf,1))
         indicator = np.multiply(y_mtx,h_collection)
         
         indicator[indicator == 1] = 0
         indicator[indicator == -1] = 1
         
         
         
         
         
         
         for t in range(0,self.T):
             # Step 3: find the h minimize the error
             # find_mtx shape is (num_clf,)
             find_mtx = np.dot(indicator,w_t)
             
             index = np.argmin(find_mtx)
             
             h_t = clfs_list[index]
             self.clfs_picked.append(h_t)
             
             e_t = find_mtx[index]
             
             epsilon = 10 ** (-6)
             beta_t = math.log((1 - e_t + epsilon)/(e_t + epsilon)) * 0.5
             
#             if e_t == 1:
#                 beta_t = 0
#             else:
#                 beta_t = math.log((1-e_t)/e_t) * 0.5

             
             self.betas.append(beta_t)
             
             # Step 6 
             # indicator_t is I[y != h_t]
             indicator_t = indicator[index]
             exp_beta = np.zeros((N,))
             exp_beta[indicator_t == 0] = math.exp(-beta_t)
             exp_beta[indicator_t == 1] = math.exp(beta_t)
             
             w_t_plus = np.multiply(w_t,exp_beta)
             # Step 7: normalize
             if np.sum(w_t_plus) == 0:
                 w_t_plus = 0
             else:
                 w_t_plus = w_t_plus / np.sum(w_t_plus)
                 
                 
             w_t = w_t_plus
             
        
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
        
         clfs_list = list(self.clfs)
        
         x = np.array(features)
         y = np.array(labels)
         N, D = x.shape
         pi_t = np.ones((N,)) * 0.5
         
         # initialize f(x)
         f_t = np.zeros((N,))
         
         # h_collection store all the classifiers in an array
         # get all the possible classifier in the set
         # h_collection shape is (num_clfs,N)
         h_collection = np.zeros((self.num_clf,N))
         i = 0
         for classifier in self.clfs:
             h_collection[i] = np.array(classifier.predict(features))
             i = i + 1
         
         
         for t in range(0,self.T):
             # Step 3: compute working response
             # z_t shape is (N,)
             numerator = ((y + 1)/2) - pi_t
             denominator = pi_t - np.multiply(pi_t,pi_t)
             z_t = np.divide(numerator,denominator)
             z_t[np.isnan(z_t)] = 0
             
             # Step 4: compute weights
             w_t = denominator
             
             # Step 5: find h_t
             z_t_mtx = np.tile(z_t,(self.num_clf,1))
             a = np.square(z_t_mtx - h_collection)
             find_mtx = np.dot(a,w_t)
             index = np.argmin(find_mtx)
             
             
             h_t = clfs_list[index]
             self.clfs_picked.append(h_t)
             
             # Step 6: update f(x)
             h_t_x = h_collection[index]
             f_t_plus = f_t + (0.5 * h_t_x)
             self.betas.append(0.5)
             
             # Step 7: compute pi_t_plus
             b = np.exp((-2)*f_t_plus)
             pi_t_plus = 1/(1+b)
             
             
             f_t = f_t_plus
             pi_t = pi_t_plus
             
             
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	