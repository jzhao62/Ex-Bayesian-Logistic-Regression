from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import eigvalsh
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.linear_model.logistic import _logistic_loss_and_grad
from utilities import *

class BayesianLogisticRegression(LinearClassifierMixin, BaseEstimator):
    def __init__(self, epochs):
        self.itr = epochs

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        X = self.add_bias(X)

        self.coef_ = [0] * 10
        self.sigma_ = [0] * 10
        self.intercept_ = [0] * 10

        for i in range(10):
            curr_class = self.classes_[i]
            mask = (y == curr_class)
            y_binary = np.ones(y.shape, dtype=np.float64)
            y_binary[~mask] = -1
            coef_eb, sigma_eb = self.eb_fit(X, y_binary)
            self.intercept_[i], self.coef_[i] = self._get_intercept(coef_eb)

        self.coef_ = np.asarray(self.coef_)
        return self




class EBLogisticRegression(BayesianLogisticRegression):
    '''
    Implements Bayesian Logistic Regression with type II maximum likelihood
    (sometimes it is called Empirical Bayes), uses Gaussian (Laplace) method
    for approximation of evidence function.

    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 50)
        Maximum number of iterations before termination

    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.

    solver: str, optional (DEFAULT = 'lbfgs_b')
        Optimization method that is used for finding parameters of posterior
        distribution ['lbfgs_b','newton_cg']

    n_iter_solver: int, optional (DEFAULT = 15)
        Maximum number of iterations before termination of solver

    tol_solver: float, optional (DEFAULT = 1e-3)
        Convergence threshold for solver (it is used in estimating posterior
        distribution),

    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations

    alpha: float (DEFAULT = 1e-6)
        Initial regularization parameter (precision of prior distribution)

    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model

    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    sigma_ : array, shape = (n_features, )
        eigenvalues of covariance matrix

    alpha_: float
        Precision parameter of weight distribution

    intercept_: array, shape = (n_features)
        intercept


    References:
    -----------
    [1] Pattern Recognition and Machine Learning, Bishop (2006) (pages 293 - 294)
    '''

    def __init__(self, epoch=50, epoch_lbfgs=15, tol_solver=1e-3, alpha=1e-6):
        super(EBLogisticRegression, self).__init__(epoch)
        self.epoch_solver = epoch_lbfgs
        self.tol_solver = tol_solver
        self.alpha = alpha

    def eb_fit(self, X, Y):

        alpha = self.alpha
        w0 = np.zeros(785)
        for i in range(self.itr):
            alpha0 = alpha
            #  mean & covariance
            w, d = self.posterior(X,Y, alpha, w0)
            mu_sq = np.sum(w ** 2)
            # use Iterative updates for Bayesian Logistic Regression
            # Note in Bayesian Logistic Gull-MacKay fixed point updates
            # and Expectation - Maximization algorithm give the same update
            # rule
            alpha = X.shape[1] / (mu_sq + np.sum(d))

            # check convergence
            delta_alpha = abs(alpha - alpha0)
            if delta_alpha < 1e-3: break

        # after convergence we need to find updated MAP vector of parameters
        # and covariance matrix of Laplace approximation

        coef_, sigma_ = self.posterior(X, Y, alpha, w)

        self.alpha_ = alpha
        return coef_, sigma_



    def posterior(self, X, Y, alpha0, w0):
        '''Finds MAP estimates for weights and Hessian at convergence point'''
        n_samples, n_features = X.shape

        f = lambda w: _logistic_loss_and_grad(w, X[:, :-1], Y, alpha0)

        w = fmin_l_bfgs_b(f, x0=w0, pgtol=self.tol_solver, maxiter=self.epoch_solver)[0]


        # calculate negative of Hessian at w
        xw = np.dot(X, w)
        s = sigmoid(xw)
        R = s * (1 - s)
        Hess = np.dot(X.T * R, X)
        Alpha = np.ones(n_features) * alpha0
        # if self.fit_intercept:
        Alpha[-1] = np.finfo(np.float16).eps
        np.fill_diagonal(Hess, np.diag(Hess) + Alpha)
        e = eigvalsh(Hess)
        return w, 1.0 / e
    def add_bias(self, X):
        return np.hstack((X, np.ones([X.shape[0], 1])))
    def _get_intercept(self, coef):
        return coef[-1], coef[:-1]

    # return variance for predictive distribution
    def get_variance(self, X):
        return np.asarray([np.sum(X ** 2 * s, axis=1) for s in self.sigma_])
    def try_predict(self,X):

        prediction = self.predict_proba(X);
        ret = np.argmax(prediction,1)
        return ret
    def predict_proba(self, X):
        scores = self.decision_function(X)

        X = self.add_bias(X)

        # probit approximation to predictive distribution
        sigma = self.get_variance(X)


        ks = 1.0 / (1.0 + np.pi * sigma / 8) ** 0.5
        probs = sigmoid(scores.T * ks).T

        probs /= np.reshape(np.sum(probs, axis=1), (probs.shape[0], 1))
        return probs