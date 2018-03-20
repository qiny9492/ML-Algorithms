import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		# self feature_dim is number of features
		self.feature_dim = len(features[0])
		
		# number of classes of the tree
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
#		print(features)
        
		for feature in features:
            
			y_pred.append(self.root_node.predict(feature))
		
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
        
		def conditional_entropy(branches: List[List[int]]) -> float:
            
            
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			

			# number of points in each branch
			x = np.array(branches)
			C, B = x.shape
			# points_per_b shape is (B,)
			# there are total num_points points
			points_per_b = np.sum(x,axis=0)
			
			w = points_per_b / np.sum(x)
			
			# points_mtx shape is (C,B)
			points_mtx = np.tile(points_per_b,(C,1))
			div = np.divide(x,points_mtx)
			# log0 convert to log1
			div[div==0] = 1
			a = np.log2(div )
			entropy_one = np.multiply(div,a)
			entropy_branch = np.sum(entropy_one,axis=0) * (-1)
			entropy = np.dot(w,entropy_branch)
			
			return entropy
			
 
		min_entropy = 20
		min_idx = 0

		x = np.array(self.features)
		y = np.array(self.labels)
		#print('fe',self.features)
#		print('la',self.labels)
#		print('x.shape',x.shape)
		if x.shape[1] == 0:
			self.splittable = False

		
		#print('x',self.splittable)
		#print(len(self.features[0]))
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best attribute to split the node
		############################################################
			
              
			
			if self.splittable:
				# get the idx_dim feature
				# split the attribute into branches
				# feature_curr shape is (N,)
				feature_curr = x[:,idx_dim]
				#print(feature_curr)
				branch_vector = np.unique(feature_curr)
				
				B = len(branch_vector.tolist())
				
                  # for all branches of this attribute
				branches = np.zeros((self.num_cls,B))
				
				for i in range (0,B):
					index = np.where(feature_curr == branch_vector[i])
					
					cls_labels = y[index]
					#print(cls_labels)
					#cls_unique = np.unique(cls_labels)
                       #cls_li = cls_labels.tolist()
                       
					#print(cls_unique)
					for j in range (0,cls_labels.shape[0]):
						for m in range(0,self.num_cls):
							if cls_labels[j] == m:
								branches[m][i] = branches[m][i] + 1
							#print(branches)
						
						
				#print(branches)
				entropy = conditional_entropy(branches.tolist())
				if entropy < min_entropy:
					min_entropy = entropy
					min_idx = idx_dim
				
#				print(entropy)
#				print(min_idx)
				
#			else:
#				feature_curr = self.features[:,idx_dim]
#				branch_vector = np.unique(feature_curr)
				
                


		############################################################
		# TODO: split the node, add child nodes
		############################################################
		self.dim_split = min_idx
         
		
		
		if (x.shape[1]>1):
			features_split = x[:,min_idx]
#			print('x[:,min_idx]',x[:,min_idx])
		elif (x.shape[1]==1):
#			print('x[:,min_idx]',x)
			features_split = np.reshape(x,(x.shape[0],))
		else:
			features_split = np.array([])
		
		b_vect = np.unique(features_split)
#		print(b_vect)
		self.feature_uniq_split = b_vect.tolist()
		b_len = len(b_vect.tolist())
		# delete the current feature column



		if self.splittable :
			x_del = np.delete(x,min_idx,axis=1)

			for k in range(0,b_len):
				child_idx = np.where(features_split == b_vect[k])
				child_features = x_del[child_idx]
				child_labels = y[child_idx]
#				print('child labels',child_labels)

				child_num_cls = np.max(child_labels) + 1
				child_node = TreeNode(child_features.tolist(), child_labels.tolist(), child_num_cls)
				
				self.children.append(child_node)

		for child in self.children:
			if child.splittable:
				child.split()


		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



