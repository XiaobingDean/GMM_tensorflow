import tensorflow as tf
import numpy as np

import numbers
from math import pi

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses tf.matmul to reduce memory footprint.
    args:
        mat_a:      tf.Tensor (n, k, 1, d)
        mat_b:      tf.Tensor (1, k, d, d)
    """
    res = np.zeros(mat_a.shape)
    mat_a = tf.cast(mat_a, tf.double)
    mat_b = tf.cast(mat_b, tf.double)
    for i in range(n_components):
        mat_a_i = tf.squeeze(mat_a[:, i, :, :], -2)
        mat_b_i = tf.squeeze(mat_b[0, i, :, :])
        res[:, i, :, :] = tf.expand_dims(tf.matmul(mat_a_i, mat_b_i), 1)

    return tf.convert_to_tensor(res)


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses tf.matmul to reduce memory footprint.
    args:
        mat_a:      tf.Tensor (n, k, 1, d)
        mat_b:      t.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return tf.reduce_sum(tf.squeeze(mat_a, -2) * tf.squeeze(mat_b, -1), axis=2, keepdims=True)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class GaussianMixture(tf.keras.models.Model):
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
    Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d),
    probabilities are shaped (n, k, 1) if they relate to an individual sample,
    or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """

    def __init__(self, n_components, covariance_type="full", eps=1.e-6, init_params="kmeans", mu_init=None,
                 var_init=None, n_init=2, max_iter=1000, tol=1e-3, random_state=None):
        """
        Initializes the model and brings all tensors into their required shape.
        The class expects data to be fed as a flat tensor in (n, d).
        The class owns:
            x:               tf.Tensor (n, 1, d)
            mu:              tf.Tensor (1, k, d)
            var:             tf.Tensor (1, k, d) or (1, k, d, d)
            pi:              tf.Tensor (1, k, 1)
            covariance_type: str
            eps:             float
            init_params:     str
            log_likelihood:  float
            n_components:    int
            n_init:          int
            max_iter:        int
            tol:             float
            random_state:    int
        args:
            n_components:    int
            n_features:      int
        options:
            mu_init:         tf.Tensor (1, k, d)
            var_init:        tf.Tensor (1, k, d) or (1, k, d, d)
            covariance_type: str
            eps:             float
            init_params:     str
            n_init:          int
            max_iter:        int
            tol:             float
            random_state:    int
        """
        super(GaussianMixture, self).__init__()

        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.n_components = n_components

        self.mu_init = mu_init
        self.var_init = var_init
        self.eps = eps

        self.log_likelihood = -np.inf

        self.covariance_type = covariance_type
        self.init_params = init_params

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]



    def _init_params(self):
        if self.mu_init is not None:
            assert self.mu_init.shape == (1, self.n_components,
                                           self.n_features), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
            self.n_components, self.n_features)
            # (1, k, d)
            self.mu = self.add_weight(name='mu',
                                     shape=self.mu_init.shape,
                                     initializer=tf.initializers.Constant(self.mu_init),
                                     dtype='float32',
                                     trainable=False,
                                     )
        else:
            self.mu = self.add_weight(name='mu', shape=(1, self.n_components, self.n_features), dtype='float32',
                                      initializer=tf.initializers.Zeros(),
                                      trainable=False)

        if self.covariance_type == "diag":
            if self.var_init is not None:
                # (1, k, d)
                assert self.var_init.shape == (1, self.n_components, self.n_features), \
                    "Input var_init does not have required tensor dimensions (1, %i, %i)" % (
                        self.n_components, self.n_features)
                self.var = self.add_weight(name='var', shape=self.var_init.shape, dtype='float32',
                                           initializer=tf.initializers.constant(self.var_init), trainable=False)
            else:
                self.var = self.add_weight(name='var', shape=(1, self.n_components, self.n_features), dtype='float32',
                                           initializer=tf.initializers.ones, trainable=False)
        elif self.covariance_type == "full":
            if self.var_init is not None:
                # (1, k, d, d)
                assert self.var_init.shape == (1, self.n_components, self.n_features,
                                                self.n_features), "Input var_init does not have required tensor dimensions (1, %i, %i, %i)" % (
                self.n_components, self.n_features, self.n_features)
                self.var = self.add_weight(name='var', shape=self.var_init.shape, dtype='float32',
                                           initializer=tf.initializers.constant(self.var_init), trainable=False)
            else:
                var_init = tf.cast(tf.tile(tf.reshape(tf.eye(self.n_features, dtype='float32'),
                                                (1, 1, self.n_features, self.n_features)),
                                     (1, self.n_components, 1, 1)), tf.float32)
                self.var = self.add_weight(name='var', shape=var_init.shape, dtype='float32',
                                           initializer=tf.initializers.constant(var_init), trainable=False)

        # (1, k, 1)
        self.pi = self.add_weight(name='pi', shape=(1, self.n_components, 1), trainable=False, dtype='double',
                                  initializer=tf.keras.initializers.constant(1. / self.n_components))

        self.params_fitted = False

    def check_size(self, x):
        if len(x.shape) == 2:
            # (n, d) --> (n, 1, d)
            x = tf.expand_dims(x, 1)

        return x

    def bic(self, x):
        """
        Bayesian information criterion for a batch of samples.
        args:
            x:      tf.Tensor (n, d) or (n, 1, d)
        returns:
            bic:    float
        """
        x = self.check_size(x)
        n = x.shape[0]

        # Free parameters for covariance, means and mixture components
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1

        bic = -2. * self.__score(x, as_average=False).mean() * n + free_params * tf.math.log(n)

        return bic

    def __check_inv(self):
        result = tf.linalg.inv(self.var)
        if tf.reduce_sum(tf.cast(tf.math.is_nan(result), tf.int32)) > 0:
            return 1
        return 0

    def fit(self, x, warm_start=False):
        """
        Fits model to the data.
        args:
            x:          tf.Tensor (n, d) or (n, k, d)
        options:
            delta:      float
            n_iter:     int
            warm_start: bool
        """
        self.n_features = x.shape[1]
        self._init_params()

        mu_best = self.mu
        var_best = self.var
        log_likelihood_best = self.log_likelihood

        for init in range(self.n_init):
            x = self.check_size(x)

            if self.init_params == "kmeans" and self.mu_init is None:
                mu = self.get_kmeans_mu(x, n_centers=self.n_components)
                self.mu = mu

            i = 0
            j = np.inf
            while (i <= self.max_iter) and (j >= self.tol):

                log_likelihood_old = self.log_likelihood
                mu_old = self.mu
                var_old = self.var

                if self.__check_inv():
                    init -= 1
                    break
                self.__em(x)

                if self.__check_inv():
                    init -= 1
                    break

                self.log_likelihood = self.__score(x)

                if tf.math.is_inf(tf.abs(self.log_likelihood)) or tf.math.is_nan(self.log_likelihood):
                    # When the log-likelihood assumes inane values, reinitialize model
                    self.__init__(self.n_components,
                                  covariance_type=self.covariance_type,
                                  mu_init=self.mu_init,
                                  var_init=self.var_init,
                                  eps=self.eps)

                    if self.init_params == "kmeans":
                        self.mu = self.get_kmeans_mu(x, n_centers=self.n_components)

                i += 1
                j = self.log_likelihood - log_likelihood_old

                if j <= self.tol:
                    # When score decreases, revert to old parameters
                    self.__update_mu(mu_old)
                    self.__update_var(var_old)

            # update weights with best performance
            if self.log_likelihood > log_likelihood_best:
                log_likelihood_best = self.log_likelihood
                mu_best = self.mu
                var_best = self.var

            self.params_fitted = True

        self.__update_mu(mu_best)
        self.__update_var(var_best)


    def predict(self, x, probs=False):
        """
        Assigns input data to one of the mixture components by evaluating the likelihood under each.
        If probs=True returns normalized probabilities of class membership.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
            probs:      bool
        returns:
            p_k:        tf.Tensor (n, k)
            (or)
            y:          tf.LongTensor (n)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + tf.math.log(self.pi)
        if probs:
            p_k = tf.exp(weighted_log_prob)
            return tf.squeeze(p_k / (tf.reduce_sum(p_k, axis=1, keepdims=True)))
        else:
            return tf.cast(tf.squeeze(tf.argmax(weighted_log_prob, 1)), tf.int32)

    def predict_proba(self, x):
        """
        Returns normalized probabilities of class membership.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            y:          tf.LongTensor (n)
        """
        return self.predict(x, probs=True)

    def score_samples(self, x):
        """
        Computes log-likelihood of samples under the current model.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
        returns:
            score:      tf.Tensor (n)
        """
        x = self.check_size(x)

        score = self.__score(x, as_average=False)
        return score

    def _estimate_log_prob(self, x):
        """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            tf.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     tf.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        if self.covariance_type == "full":
            mu = self.mu
            var = self.var
            precision = tf.linalg.inv(var)

            d = x.shape[-1]

            log_2pi = d * tf.math.log(2. * pi)
            log_2pi = tf.cast(log_2pi, tf.double)
            log_det = self._calculate_log_det(precision)
            log_det = tf.cast(log_det, tf.double)


            x = tf.cast(x, tf.double)
            mu = tf.cast(mu, tf.double)
            x_mu_T = tf.expand_dims((x - mu), -2)
            x_mu = tf.expand_dims((x - mu), -1)

            x_mu_T_precision = calculate_matmul_n_times(self.n_components, x_mu_T, precision)
            # print(x_mu_T_precision, x_mu_T, x, mu)
            x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

            return -.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

        elif self.covariance_type == "diag":
            mu = self.mu
            prec = tf.math.rsqrt(self.var)

            log_p = tf.math.reduce_sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), axis=2, keepdims=True)
            log_det = tf.math.reduce_sum(tf.math.log(prec), axis=2, keepdims=True)

            return -.5 * (self.n_features * tf.math.log(2. * pi) + log_p) + log_det

    def _calculate_log_det(self, var):
        """
        Calculate log determinant in log space, to prevent overflow errors.
        args:
            var:            torch.Tensor (1, k, d, d)
        """
        log_det = []

        for k in range(self.n_components):

            evals, evecs = tf.linalg.eig(var[0, k])

            log_det.append(tf.reduce_sum(tf.math.log(tf.math.real(evals))))
        log_det = tf.convert_to_tensor(log_det)
        return tf.expand_dims(log_det, -1)

    def _e_step(self, x):
        """
        Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
        Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
        This is the so-called expectation step of the EM-algorithm.
        args:
            x:              tf.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob_norm:  tf.Tensor (1)
            log_resp:       tf.Tensor (n, k, 1)
        """
        x = self.check_size(x)

        weighted_log_prob = self._estimate_log_prob(x) + tf.math.log(self.pi)

        log_prob_norm = tf.reduce_logsumexp(weighted_log_prob, axis=1, keepdims=True)
        log_resp = weighted_log_prob - log_prob_norm

        return tf.reduce_mean(log_prob_norm), log_resp

    def _m_step(self, x, log_resp):
        """
        From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
        args:
            x:          tf.Tensor (n, d) or (n, 1, d)
            log_resp:   tf.Tensor (n, k, 1)
        returns:
            pi:         tf.Tensor (1, k, 1)
            mu:         tf.Tensor (1, k, d)
            var:        tf.Tensor (1, k, d)
        """
        x = self.check_size(x)

        resp = tf.exp(log_resp)

        pi = tf.reduce_sum(resp, axis=0, keepdims=True) + self.eps

        resp = tf.cast(resp, tf.double)
        pi = tf.cast(pi, tf.double)
        x = tf.cast(x, tf.double)

        mu = tf.reduce_sum(resp * x, axis=0, keepdims=True) / pi

        if self.covariance_type == "full":
            eps = (tf.linalg.eye(self.n_features) * self.eps)
            eps = tf.cast(eps, tf.double)

            var = tf.reduce_sum(tf.matmul(tf.expand_dims(x - mu, -1), tf.expand_dims(x - mu, -2)) * tf.expand_dims(resp, -1), axis=0,
                            keepdims=True) / tf.expand_dims(tf.reduce_sum(resp, axis=0, keepdims=True), -1) + eps
        elif self.covariance_type == "diag":
            x2 = tf.reduce_sum((resp * x * x), axis=0, keepdims=True) / pi
            mu2 = mu * mu
            xmu = tf.reduce_sum((resp * mu * x), axis=0, keepdims=True) / pi
            var = x2 - 2 * xmu + mu2 + self.eps

        pi = pi / x.shape[0]

        return pi, mu, var

    def __em(self, x):
        """
        Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
        args:
            x:          torch.Tensor (n, 1, d)
        """

        _, log_resp = self._e_step(x)

        pi, mu, var = self._m_step(x, log_resp)

        self.__update_pi(pi)
        self.__update_mu(mu)
        self.__update_var(var)

    def __score(self, x, as_average=True):
        """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  tf.Tensor (n, 1, d)
            sum_data:           bool
        returns:
            score:              tf.Tensor (1)
            (or)
            per_sample_score:   tf.Tensor (n)
        """
        weighted_log_prob = self._estimate_log_prob(x) + tf.math.log(self.pi)
        per_sample_score = tf.reduce_logsumexp(weighted_log_prob, axis=1)

        if as_average:
            return tf.reduce_mean(per_sample_score)
        else:
            return tf.squeeze(per_sample_score)

    def __update_mu(self, mu):
        """
        Updates mean to the provided value.
        args:
            mu:         tf.FloatTensor
        """
        assert mu.shape in [(self.n_components, self.n_features), (1, self.n_components,
                                                                    self.n_features)], "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
        self.n_components, self.n_features, self.n_components, self.n_features)

        if mu.shape == (self.n_components, self.n_features):
            self.mu = tf.expand_dims(mu, 0)
        elif mu.shape == (1, self.n_components, self.n_features):
            self.mu = mu

    def __update_var(self, var):
        """
        Updates variance to the provided value.
        args:
            var:        tf.FloatTensor
        """
        if self.covariance_type == "full":
            assert var.shape in [(self.n_components, self.n_features, self.n_features), (
            1, self.n_components, self.n_features,
            self.n_features)], "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)" % (
            self.n_components, self.n_features, self.n_features, self.n_components, self.n_features, self.n_features)

            if var.shape == (self.n_components, self.n_features, self.n_features):
                self.var = tf.expand_dims(var, 0)
            elif var.shape == (1, self.n_components, self.n_features, self.n_features):
                self.var = var

        elif self.covariance_type == "diag":
            assert var.shape in [(self.n_components, self.n_features), (1, self.n_components,
                                                                         self.n_features)], "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)" % (
            self.n_components, self.n_features, self.n_components, self.n_features)

            if var.shape == (self.n_components, self.n_features):
                self.var = tf.expand_dims(var, 0)
            elif var.shape == (1, self.n_components, self.n_features):
                self.var.data = var

    def __update_pi(self, pi):
        """
        Updates pi to the provided value.
        args:
            pi:         tf.FloatTensor
        """
        assert pi.shape in [
            (1, self.n_components, 1)], "Input pi does not have required tensor dimensions (%i, %i, %i)" % (
        1, self.n_components, 1)

        self.pi.data = pi

    def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
        """
        Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
        The algorithm is repeated init_times often, after which the best centerpoint is returned.
        args:
            x:            tf.FloatTensor (n, d) or (n, 1, d)
            init_times:   init
            min_delta:    int
        """
        if len(x.shape) == 3:
            x = tf.squeeze(x, 1)
        x_min, x_max = tf.reduce_min(x), tf.reduce_max(x)
        x = (x - x_min) / (x_max - x_min)

        random_state = check_random_state(self.random_state)

        min_cost = np.inf

        for i in range(init_times):
            tmp_center = x.numpy()[random_state.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
            l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - tmp_center, ord=2, axis=2)
            l2_cls = tf.argmin(l2_dis, axis=1)

            cost = 0
            for c in range(n_centers):
                cost += tf.reduce_mean(tf.norm(x[l2_cls == c] - tmp_center[c], ord=2, axis=1))

            if cost < min_cost:
                min_cost = cost
                center = tmp_center

        delta = np.inf
        while delta > min_delta:
            l2_dis = tf.norm(tf.tile(tf.expand_dims(x, 1), (1, n_centers, 1)) - center, ord=2, axis=2)
            l2_cls = tf.argmin(l2_dis, axis=1)
            center_old = tf.convert_to_tensor(center, dtype=tf.double)

            for c in range(n_centers):
                center[c] = tf.reduce_mean(x[l2_cls == c], axis=0)

            delta = tf.reduce_max(tf.reduce_sum(tf.square(center_old - center), axis=1))

        return tf.expand_dims(center, 0) * (x_max - x_min) + x_min



if __name__ == '__main__':
    model = GaussianMixture(3)
    model.n_features = 3
    model._init_params()
    _, log = model._e_step(np.ones((9, 3)))