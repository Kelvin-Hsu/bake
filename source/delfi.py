"""
Deconditional Kernel Mean Embeddings for Likelihood-Free Inference.

Core functions for Deconditional Embedding Likelihood-Free Inference (DELFI).
"""
import numpy as np
import scipy.linalg as la
from kernels import gaussian_kernel_gramix


def transformation_matrix(y, y_tilde, ls_y, lamb):
    """
    Compute the transformation matrix for DME/TTGP, defined as $A := (L + n \lambda I)^{-1} \tilde{L}$.
    
    This code uses Gaussian kernels only.
    
    Parameters
    ----------
    y : np.ndarray [Size: (n, d_y)]
        Samples of the mediating variable from the simulation process
    y_tilde : np.ndarray [Size: (m, d_y)]
        Samples of the mediating variable from the observation process
    ls_y : float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the mediating variable(s)
    lamb : float
        Regularization parameter
    
    Returns
    -------
    np.ndarray [Size: (n, m)]
        The transformation matrix
    """
    # Size: (n, n)
    l = gaussian_kernel_gramix(y, y, ls_y)
    # Size: (n, m)
    tilde_l = gaussian_kernel_gramix(y, y_tilde, ls_y)
    # Size: (n, n)
    n = y.shape[0]
    lower = True
    l_chol = la.cholesky(l + n * lamb * np.eye(n), lower=lower)
    # Size: (n, m)
    a = la.cho_solve((l_chol, lower), tilde_l)
    return a


def dme_query_fast(t_query, t_tilde, t, x, y, ls_t, ls_x, lamb, eps):
    """
    Compute the deconditional mean embedding using the alternative form which is cubic in simulation samples but linear in prior samples.
    
    This code uses Gaussian kernels only.
    
    Parameters
    ----------
    t_query : np.ndarray [Size: (n_q, d_t)]
        The parameters to query the deconditional mean embedding at
    t_tilde: np.ndarray [Size: (m, d_t)]
        The prior parameter samples
    t : np.ndarray [Size: (n, d_t)]
        The likelihood parameter samples
    x : np.ndarray [Size: (n, d_x)]
        The likelihood statistic samples
    y : np.ndarray [Size: (d_x,)]
        The observed statistic
    ls_t : float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the parameters
    lt_x : float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the statistics
    lamb : float
        The regularization hyperparameter for the prior operator inversion
    eps: float
        The regularization hyperparameter for the evidence operator inversion
        
    Returns
    -------
    np.ndarray [Size: (n_q, 1)]
        The deconditional mean embedding evaluated at the query parameters
    """
    m = t_tilde.shape[0]
    n = t.shape[0]
    # Size: (n, m)
    a = transformation_matrix(t, t_tilde, ls_t, lamb)
    # Size: (n, n)
    k = gaussian_kernel_gramix(x, x, ls_x)
    # Size: (n, 1)
    k_y = gaussian_kernel_gramix(x, y, ls_x)
    # Size: (m, 1)
    query_weights = np.dot(np.transpose(a), la.solve(np.dot(k, np.dot(a, np.transpose(a))) + m * eps * np.eye(n), k_y))
    # Size: (m, n_q)
    l_query = gaussian_kernel_gramix(t_tilde, t_query, ls_t)
    # Size: (n_q, 1)
    q_query = np.dot(np.transpose(l_query), query_weights)
    return q_query