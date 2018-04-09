import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        # Initialize miu
        # rand_k shape is (K,)
        # miu shape is (K,D)
        rand_k = np.random.choice(N, self.n_cluster, replace=False)
        miu = x[rand_k]
        
        
        
        # miu_3d shape is (K,D,N)
        # x_3d shape is (N,D,K)
        miu_3d = np.dstack([miu]*N)
        x_3d = np.dstack([x]*self.n_cluster)
        
        # both miu_3d_t and x_3d_t shape are (N,K,D)
        miu_3d_t = np.transpose(miu_3d,(2,0,1))
        x_3d_t = np.transpose(x_3d,(0,2,1))
        
        # norm_mtx shape is (N,K)
        norm_mtx = np.linalg.norm((miu_3d_t - x_3d_t),axis=2)
        dist_mtx = np.square(norm_mtx)
        
        # find min of each row, and store the position of these min values
        row = np.arange(N)
        col = np.argmin(dist_mtx,axis=1)
        
        # r_mtx shape is (N,K)
        r_mtx = np.zeros((N,self.n_cluster))
        r_mtx[row,col] = 1
        
        j_old = np.sum(np.multiply(r_mtx,dist_mtx)) / N
        
        # compute miu_k
        # r_sum shape is (K,)
        # r_x_sum shape is (K,D)
        r_sum = np.sum(r_mtx,axis=0)
        r_mtx_t = np.transpose(r_mtx)
        r_x_sum = np.dot(r_mtx_t,x)
        
        # r_sum_mtx shape is (K,D)
        r_sum_mtx = np.transpose(np.tile(r_sum,(D,1)))
        # miu shape is (K,D)
        miu = np.divide(r_x_sum,r_sum_mtx)
        
        number_of_updates = 1
        
        for i in range(self.max_iter):
            # miu_3d shape is (K,D,N)
            miu_3d = np.dstack([miu]*N)
            # both miu_3d_t and x_3d_t shape are (N,K,D)
            miu_3d_t = np.transpose(miu_3d,(2,0,1))
            # norm_mtx shape is (N,K)
            norm_mtx = np.linalg.norm((miu_3d_t - x_3d_t),axis=2)
            dist_mtx = np.square(norm_mtx)
            # find min of each row, and store the position of these min values
            row = np.arange(N)
            col = np.argmin(dist_mtx,axis=1)
            # r_mtx shape is (N,K)
            r_mtx = np.zeros((N,self.n_cluster))
            r_mtx[row,col] = 1
            
            j_new = np.sum(np.multiply(r_mtx,dist_mtx)) / N
            
            if abs(j_old - j_new) <= self.e:
                break
            
            j_old = j_new
            # compute miu_k
            # r_sum shape is (K,)
            # r_x_sum shape is (K,D)
            r_sum = np.sum(r_mtx,axis=0)
            r_mtx_t = np.transpose(r_mtx)
            r_x_sum = np.dot(r_mtx_t,x)
        
            # r_sum_mtx shape is (K,D)
            r_sum_mtx = np.transpose(np.tile(r_sum,(D,1)))
            # miu shape is (K,D)
            miu = np.divide(r_x_sum,r_sum_mtx)
            
            number_of_updates = number_of_updates + 1
            
        
        
        membership = np.argmax(r_mtx,axis=1)
        
        tup = (miu,membership,number_of_updates)
        
        return tup
        
        
        
        
        
#        raise Exception(
#            'Implement fit function in KMeans class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        k_means = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, num = k_means.fit(x)
        
        centroid_labels = np.zeros((self.n_cluster,))
        
        for i in range(self.n_cluster):
            y_i = y[membership== i].astype(int)
            counts = np.bincount(y_i)
            centroid_labels[i] = np.argmax(counts)
            
            
        
#        raise Exception(
#            'Implement fit function in KMeansClassifier class (filename: kmeans.py')

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        
        # centroids_3d shape is (K,D,N)
        # x_3d shape is (N,D,K)
        centroids_3d = np.dstack([self.centroids]*N)
        x_3d = np.dstack([x]*self.n_cluster)
        
        # both centroids_3d_t and x_3d_t shape are (N,K,D)
        centroids_3d_t = np.transpose(centroids_3d,(2,0,1))
        x_3d_t = np.transpose(x_3d,(0,2,1))
        
        # norm_mtx shape is (N,K)
        norm_mtx = np.linalg.norm((centroids_3d_t - x_3d_t),axis=2)
        # member shape is (N,), min(member) = 0, max(member) = K-1
        member = np.argmin(norm_mtx,axis=1) 
        
        #pred shape is (N,)
        pred = self.centroid_labels[member]
        
        return pred
        
#        raise Exception(
#            'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE
