from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gmm_tf import GaussianMixture as GMM
import numpy as np
import time
from sklearn.metrics import homogeneity_completeness_v_measure

X = np.concatenate([np.random.normal(loc = np.random.normal(scale = 0.5, size = 32), size = (100, 32)), np.random.normal(loc = np.random.normal(scale = 0.5, size = 32), size = (100, 32)), np.random.normal(loc = np.random.normal(scale = 0.5, size = 32), size = (100, 32))])

Y = np.repeat(range(3), 100)

K = 3

it = time.time()
gmm_sklearn = GaussianMixture(n_components=K, random_state=0).fit(X)
ft = time.time()
print('sklearn time:', ft - it)
print('sklearn accuracy:', homogeneity_completeness_v_measure(Y, gmm_sklearn.predict(X)))


gmm_tf = GMM(n_components=K, random_state=0)
it2 = time.time()
gmm_tf.fit(X)
ft2 = time.time()
print('TF time:', ft2 - it2)
print('TF accuracy:', homogeneity_completeness_v_measure(Y, gmm_tf.predict(X)))
