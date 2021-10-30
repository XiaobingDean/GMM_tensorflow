from sklearn.mixture import GaussianMixture
from gmm_tf_debug import GaussianMixture as GMM
import numpy as np 
import time 
from sklearn.metrics import homogeneity_completeness_v_measure 

np.random.seed(42)
 
n_feature = 32
n_components = 5 
sampe_size = [100, 200, 300, 200, 100]
scale = 1.1 
 
X = [] 
 
for i in range(n_components): 
    A = np.random.rand(n_feature, n_feature) 
    B = np.dot(A, A.transpose()) 
    X.append(np.random.multivariate_normal(mean = np.random.normal(scale = scale, size = n_feature), cov = B, size = sampe_size[i])) 
 
X = np.concatenate(X)

Y = np.repeat(range(n_components), sampe_size) 
 
 
it = time.time() 
gmm_sklearn = GaussianMixture(n_components=n_components, random_state=0, n_init = 50, tol=1e-3).fit(X)
ft = time.time() 
print('sklearn time:', ft - it) 
print('sklearn accuracy:', homogeneity_completeness_v_measure(Y, gmm_sklearn.predict(X))) 
 
 
gmm_tf = GMM(n_components=n_components, random_state=0, n_init = 50, eps=1e-3, batch=800)
it2 = time.time() 
gmm_tf.fit(X) 
ft2 = time.time() 
print('TF time:', ft2 - it2) 
print('TF accuracy:', homogeneity_completeness_v_measure(Y, gmm_tf.predict(X)))

