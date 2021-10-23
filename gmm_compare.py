from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gmm_tf import GaussianMixture as GMM
from logger import logger
import numpy as np
import time
log = logger()
data_dict = np.load('./data_k2_1000.pkl', allow_pickle=True)
X = data_dict['xn']
Y = data_dict['zn']

K = 2

it = time.time()
gmm_sklearn = GaussianMixture(n_components=K, random_state=0).fit(X)
ft = time.time()
print('sklearn time:', ft - it)
print('sklearn accuracy:', max(accuracy_score(1 - gmm_sklearn.predict(X), Y),
                          accuracy_score(gmm_sklearn.predict(X), Y)))

gmm_tf = GMM(n_components=K, random_state=0)
it2 = time.time()
gmm_tf.fit(X)
ft2 = time.time()
print('TF time:', ft2 - it2)
print('TF accuracy:', max(accuracy_score(1 - gmm_tf.predict(X), Y),
                          accuracy_score(gmm_tf.predict(X), Y)))
