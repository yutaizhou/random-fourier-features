"""============================================================================
Kernel ridge regression using random Fourier features. Based on "Random 
Features for Large-Scale Kernel Machines" by Rahimi and Recht (2007).

For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
============================================================================"""

import numpy as np
from   sklearn.exceptions import NotFittedError


# -----------------------------------------------------------------------------

class RFFRidgeRegression:

    def __init__(self, rff_dim=1, alpha=1.0, sigma=1.0):
        """Kernel ridge regression using random Fourier features.

        rff_dim : Dimension of random feature.
        alpha :   Regularization strength. Should be a positive float.
        sigma :   sigma^2 is the variance.
        """
        self.rff_dim = rff_dim
        self.alpha   = alpha
        self.sigma   = sigma
        self.beta_   = None
        self.b_      = None
        self.W_      = None

    def fit(self, X, y):
        """Fit model with training data X and target y.
        """
        Z, W, b = self._get_rffs(X, return_vars=True)
        I = self.alpha * np.eye(self.rff_dim)
        self.beta_ = np.linalg.solve(Z @ Z.T + I, Z @ y)
        self.b_ = b
        self.W_ = W
        return self

    def predict(self, X):
        """Predict using fitted model and testing data X.
        """
        if self.beta_ is None or self.b_ is None or self.W_ is None:
            msg = "This instance is not fitted yet. Call 'fit' with "\
                  "appropriate arguments before using this method."
            raise NotFittedError(msg)
        Z = self._get_rffs(X, return_vars=False)
        return self.beta_ @ Z

    def _get_rffs(self, X, return_vars):
        """Return random Fourier features based on data X, as well as random
        variables W and b.
        """
        N, D = X.shape
        W, b = self._get_rvs(D)
        B    = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1./ np.sqrt(self.rff_dim)
        Z    = norm * np.sqrt(2) * np.cos(self.sigma * W @ X.T + B)
        if return_vars:
            return Z, W, b
        return Z

    def _get_rvs(self, D):
        """On first call, return random variables W and b. Else, return cached
        values.
        """
        if self.W_ is not None:
            return self.W_, self.b_
        W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
        b = np.random.uniform(0, 2*np.pi, size=self.rff_dim)
        return W, b
