import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(self.n_cluster, self.max_iter, self.e)
            self.means, membership, i = k_means.fit(x)
            # compute variance
            self.variances = np.zeros((self.n_cluster,D,D))
            # compute pi_k
            self.pi_k = np.zeros((self.n_cluster,))
            for i in range(self.n_cluster):
                x_i_mtx = x[membership==i]
                num = x_i_mtx.shape[0]
                mean_i_mtx = np.tile(self.means[i],(num,1))
                # x_minus_mean shape is (num,D)
                x_minus_mean = x_i_mtx - mean_i_mtx
                self.variances[i] = np.dot(np.transpose(x_minus_mean),x_minus_mean) / num
                self.pi_k[i] = num / N
            
            
            
#            raise Exception(
#                'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k
            
            # self.means shape is (K,D)
            self.means = np.random.rand(self.n_cluster,D)
            self.pi_k = np.ones((self.n_cluster,)) / self.n_cluster
            covar = np.identity(D)
            self.variances = np.tile(covar,(self.n_cluster,1,1))
            
            
            
            

            # DONOT MODIFY CODE ABOVE THIS LINE
#            raise Exception(
#                'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        
        # Computer log-likelihood
        l = self.compute_log_likelihood(x)
        num_iter = 0
        
        for j in range(self.max_iter):
            # calculate number of iterations
            num_iter = num_iter + 1
            #E-step: compute responsibilities
            # p_x_k shape is (K,N)
            p_x_k = np.zeros((self.n_cluster,N))
            for k in range(self.n_cluster):
                 mean_i_mtx = np.tile(self.means[k],(N,1))
                 # determinant of covariance_i
                 var_det = np.linalg.det(self.variances[k])
                 deno = (((2*np.pi)**D)*var_det)**(0.5)
                 # x_minus_mean shape is (N,D)
                 x_minus_mean = x - mean_i_mtx
                 a = np.dot(x_minus_mean,np.linalg.inv(self.variances[k]))
                 b = np.dot(a,np.transpose(x_minus_mean))
                 product = np.diagonal(b)
                 p_x_k[k] = np.exp((-0.5) * product) / deno
            
#            # p_x shape is (N,)
#            p_x = np.dot(self.pi_k,p_x_k)
            # p_x_k_t shape is (N,K)
            p_x_k_t = np.transpose(p_x_k)
            
            # pi_mtx shape is (N,K)
            pi_mtx = np.tile(self.pi_k,(N,1)) 
            p_xi = np.multiply(p_x_k_t,pi_mtx)
                        
            # sum_row shape is (N,)
            sum_row = np.sum(p_xi,axis=1)
            
            # sum_row_mtx shape is (N,K)
            sum_row_mtx = np.transpose(np.tile(sum_row,(self.n_cluster,1)))
            # r is responsibilities matrix, shape (N,K)       
            r = np.divide(p_xi,sum_row_mtx)
                
            
            # M-step
            # N_k shape is (K,)
            N_k = np.sum(r,axis=0)
            
            # Estimate means
            # r_3d shape is (D,N,K)
            # x_3d shape is (K,N,D)
            r_3d = np.tile(r,(D,1,1))
            x_3d = np.tile(x,(self.n_cluster,1,1))
            # both r_3d_t and x_3d_t shape are (K,D,N)
            r_3d_t = np.transpose(r_3d,(2,0,1))
            x_3d_t = np.transpose(x_3d,(0,2,1))
            
            # c shape is (K,D)
            c = np.sum(np.multiply(r_3d_t,x_3d_t),axis=2)
            # N_k_mtx shape is (K,D)
            N_k_mtx = np.transpose(np.tile(N_k,(D,1)))
            # update means, self.means shape is (K,D)
            self.means = np.divide(c,N_k_mtx)
            
            
            # Estimate variances, shape is (K,D,D)
            for m in range(self.n_cluster):
                # r_i_mtx shape is (D,N)
                r_i_mtx = np.tile(r[:,m],(D,1))
                # mean_i_mtx shape is (N,D)
                mean_i_mtx = np.tile(self.means[m],(N,1))
                # x_minus_mean shape is (N,D)
                x_minus_mean = x - mean_i_mtx
                # prod shape is (D,N)
                prod = np.multiply(r_i_mtx,np.transpose(x_minus_mean))
                self.variances[m] = np.dot(prod,x_minus_mean) / N_k[m]
                
            
            # Estimate pi_k
            self.pi_k = N_k / N
            
            # Compute log-likelihood l_new
            l_new = self.compute_log_likelihood(x)
            
            # Compare l and l_new
            if abs(l - l_new) <= self.e:
                break
            
            # Set l:=l_new
            l = l_new
                
             
#        print('means: ',self.means)
#        print('variances: ',self.variances)
#        print('pi_k:',self.pi_k)
        return num_iter
#        raise Exception('Implement fit function (filename: gmm.py)')             l
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples
        
        
        
        # generate N samples scattered in different K clusters according to pi_k
        k_samples = np.random.multinomial(N,self.pi_k)
        
        samples_all = np.random.multivariate_normal(self.means[0],self.variances[0],k_samples[0])
        for i in range(1,self.n_cluster):
            # in cluster i there are k_samples[i] points
            samples_cls_k =np.random.multivariate_normal(self.means[i],self.variances[i],k_samples[i])
            samples_all = np.vstack((samples_all,samples_cls_k))
        
        return samples_all
            
            
            



        # DONOT MODIFY CODE ABOVE THIS LINE
#        raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N, D = x.shape
        
        # p_x_k shape is (K,N)
        
        p_x_k = np.zeros((self.n_cluster,N))
       
        # make all covariance matrix invertible
        for i in range(self.n_cluster):
            while np.linalg.det(self.variances[i]) == 0:
                self.variances[i] = self.variances[i] + 0.001 * np.identity(D)
            
            mean_i_mtx = np.tile(self.means[i],(N,1))
            
            # determinant of covariance_i
            var_det = np.linalg.det(self.variances[i])
            deno = (((2*np.pi)**D)*var_det)**(0.5)
            
            
            # x_minus_mean shape is (N,D)
            x_minus_mean = x - mean_i_mtx
            a = np.dot(x_minus_mean,np.linalg.inv(self.variances[i]))
            b = np.dot(a,np.transpose(x_minus_mean))
            product = np.diagonal(b)
            p_x_k[i] = np.exp((-0.5) * product) / deno
        
        # p_x shape is (N,)
        p_x = np.dot(self.pi_k,p_x_k)
        ln_px = np.log(p_x)
        log_likelihood = np.sum(ln_px)
        
        return log_likelihood.tolist()
#        raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
