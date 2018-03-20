import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
	def __init__(self, s:int, b:float, d:int):
		self.clf_name = "Decision_stump"
		self.s = s
		self.b = b
		self.d = d

	def train(self, features: List[List[float]], labels: List[int]):
		pass
		
	def predict(self, features: List[List[float]]) -> List[int]:
		##################################################
		# TODO: implement "predict"
		##################################################
         
         # x shape is (N,D)
         x = np.array(features)
         N, D = x.shape
         h = np.zeros((N,))
         
         # get the d-th features of all samples
         # x_d shape is (N,)
         x_d = x[:,self.d]
         h[x_d > self.b] = self.s
         h[x_d <= self.b] = - self.s
         
         h = h.astype(int)
         h_list = h.tolist()
         
         
         return h_list