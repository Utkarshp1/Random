import numpy as np
from scipy.linalg.blas import dger, dgemm

class PCA:

    def __init__(self):
        self.mean, self.std, self.eig_val, self.eig_vec = None, None, None, None

    def preprocess(self, X, std_divide=False):
        '''
            Preprocesses Data for PCA by subtracting mean. If `std_divide`
            = True, then also divides by the standard deviation.

            Arguments:
            ---------
                - X : Data matrix where rows are data points and columns are
                    features.
        '''

        X = X.copy()

        self.mean = X.mean(axis=0)

        X = X - self.mean

        if std_divide:
            self.std = np.std(X, axis=0)
            X /= self.std

        return X

    def forward(self, X, method='cov'):
        '''
            Performs PCA.

            Arguments:
            ---------
                - X : Data matrix where rows are data points and columns are
                    features.
        '''
        
        cov = np.cov(X, rowvar=False)
        # _, cov = self.compute_cov(X)
        eig_val, eig_vec = np.linalg.eigh(cov)
        sort_idx = np.argsort(eig_val)[::-1]

        self.eig_val = eig_val[sort_idx]
        self.eig_vec = eig_vec[:, sort_idx]

    def compute_cov(self, X):
        '''
            Faster way to compute covariance matrix.
            Refer: https://groups.google.com/g/scipy-user/c/FpOU4pY8W2Y
        '''
        n,p = X.shape
        m = X.mean(axis=0)
        # covariance matrix with correction for rounding error
        # S = (cx'*cx - (scx'*scx/n))/(n-1)
        # Am Stat 1983, vol 37: 242-247.
        cx = X - m
        scx = cx.sum(axis=0)
        scx_op = dger(-1.0/n,scx,scx)
        S = dgemm(1.0, cx.T, cx.T, beta=1.0,
        c=scx_op, trans_a=0, trans_b=1, overwrite_c=1)
        S[:] *= 1.0/(n-1)
        return m,S.T
    
    def get_params(self):
        params = {
            'mean' : self.mean,
            'std' : self.std,
            'eig_val' : self.eig_val,
            'eig_vec' : self.eig_vec,
        }

        return params