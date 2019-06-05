"""
Task Transformed Gaussian Process Module.

Core functions for task transformed Gaussian processes.
"""
import numpy as np
import scipy.linalg as la
from kernels import gaussian_kernel_gramix, negative_log_gaussian


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


def tgp_pred(x_q, x, z_tilde, ls_x, trans_mat, noise_cov):
    """
    Compute the posterior predictive mean and covariance of a transformed Gaussian process with a given transformation and noise covariance.
    
    This code uses Gaussian kernels only.
    
    Parameters
    ----------
    x_q : np.ndarray [Size: (n_q, d)]
        The query features
    x : np.ndarray [Size: (n, d)]
        The features
    z_tilde : np.ndarray [Size: (m,)]
        The transformed targets
    ls_x: float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the input variable(s)
    trans_mat : np.ndarray [Size: (n, m)]
        The transformation matrix
    noise_cov: np.ndarray [Size: (m, m)]
        The noise covariance
        
    Returns
    -------
    np.ndarray [Size: (n_q,)]
        The predictive mean on the query points
    np.ndarray [Size: (n_q, n_q)]
        The predictive covariance between the query points
    """
    # Size: (n, n)
    k = gaussian_kernel_gramix(x, x, ls_x)
    # Size: (m, m)
    s = np.dot(np.transpose(trans_mat), np.dot(k, trans_mat)) + noise_cov
    # Size: (m, m)
    lower = True
    s_chol = la.cholesky(s, lower=lower)
    # Size: (m, n)
    smt = la.cho_solve((s_chol, lower), np.transpose(trans_mat))
    # Size: (n, n_q)
    k_q = gaussian_kernel_gramix(x, x_q, ls_x)
    # Size: (n_q, n_q)
    k_qq = gaussian_kernel_gramix(x_q, x_q, ls_x)
    # Size: (m, n_q)
    smt_k_q = np.dot(smt, k_q)
    # Size: (n_q,)
    f_mean = np.dot(np.transpose(smt_k_q), z_tilde)
    # Size: (n_q, n_q)
    f_cov = k_qq - np.dot(np.transpose(smt_k_q), np.dot(np.transpose(trans_mat), k_q))
    # The posterior predictive mean and covariance of the latent function
    return f_mean, f_cov
           

def ttgp_pred(x_q, x, y, y_tilde, z_tilde, ls_x, ls_y, sigma, full=True):
    """
    Compute the posterior predictive mean and covariance of a task transformed Gaussian process.
    
    This code uses Gaussian kernels only.
    
    Parameters
    ----------
    x_q : np.ndarray [Size: (n_q, d_x)]
        The query features
    x : np.ndarray [Size: (n, d_x)]
        The features from the transformation set
    y : np.ndarray [Size: (n, d_y)]
        The mediators from the transformation set
    y_tilde: np.ndarray [Size: (m, d_y)]
        The mediators from the task set
    z_tilde : np.ndarray [Size: (m,)]
        The targets from the task set
    ls_x: float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the input variable(s)
    ls_y: float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the mediating variable(s)
    sigma : float
        The noise standard deviation
    full : boolean, optional
        Whether to do full Bayesian inference on g or use maximum a posteriori approximations on g
        
    Returns
    -------
    np.ndarray [Size: (n_q,)]
        The predictive mean on the query points
    np.ndarray [Size: (n_q, n_q)]
        The predictive covariance between the query points
    """
    # Size of the transformation and task datasets
    n = y.shape[0]
    m = y_tilde.shape[0]
    # The equivalent regularization parameter from noise standard deviation due to DME-TTGP equivalence
    lamb = sigma ** 2 / n
    # Size: (n, m)
    trans_mat = transformation_matrix(y, y_tilde, ls_y, lamb)
    # Compute the noise covariance depending on whether we are performing full Bayesian inference on g
    if full:
        # Size: (m, m)
        l_tt = gaussian_kernel_gramix(y_tilde, y_tilde, ls_y)
        # Size: (n, m)
        l_t = gaussian_kernel_gramix(y, y_tilde, ls_y)
        # Size: (m, m)
        noise_cov = l_tt + sigma ** 2 * np.eye(m) - np.dot(np.transpose(l_t), trans_mat)
    else:
        # Size: (m, m)
        noise_cov = sigma ** 2 * np.eye(m)
    # Once we have the transformation and noise covariance, apply the transformed Gaussian process equations for prediction
    return tgp_pred(x_q, x, z_tilde, ls_x, trans_mat, noise_cov)


def tgp_nlml(x, z_tilde, ls_x, trans_mat, noise_cov):
    """
    Compute the negative log marginal likelihood of a transformed Gaussian process.
    
    This code uses Gaussian kernels only.
    
    Parameters
    ----------
    x : np.ndarray [Size: (n, d)]
        The features
    z_tilde : np.ndarray [Size: (m,)]
        The transformed targets
    ls_x: float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the input variable(s)
    trans_mat : np.ndarray [Size: (n, m)]
        The transformation matrix
    noise_cov: np.ndarray [Size: (m, m)]
        The noise covariance
        
    Returns
    -------
    float
        The negative log marginal likelihood
    """
    k = gaussian_kernel_gramix(x, x, ls_x)
    s = np.dot(np.transpose(trans_mat), np.dot(k, trans_mat)) + noise_cov
    return negative_log_gaussian(z_tilde, 0, s)


def ttgp_nlml(x, y, y_tilde, z_tilde, ls_x, ls_y, sigma, full=True):
    """
    Compute the negative log marginal likelihood of a task transformed Gaussian process.
    
    This code uses Gaussian kernels only.
    
    Parameters
    ----------
    x : np.ndarray [Size: (n, d_x)]
        The features from the transformation set
    y : np.ndarray [Size: (n, d_y)]
        The mediators from the transformation set
    y_tilde: np.ndarray [Size: (m, d_y)]
        The mediators from the task set
    z_tilde : np.ndarray [Size: (m,)]
        The targets from the task set
    ls_x: float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the input variable(s)
    ls_y: float or np.ndarray [Size: () or (1,) for isotropic; (d_y,) for anistropic]
        The length scale(s) of the mediating variable(s)
    sigma : float
        The noise standard deviation
    full : boolean, optional
        Whether to do full Bayesian inference on g or use maximum a posteriori approximations on g
        
    Returns
    -------
    float
        The negative log marginal likelihood
    """    
    # Size of the transformation and task datasets
    n = y.shape[0]
    m = y_tilde.shape[0]
    # The equivalent regularization parameter from noise standard deviation due to DME-TTGP equivalence
    lamb = sigma ** 2 / n
    # Size: (n, m)
    trans_mat = transformation_matrix(y, y_tilde, ls_y, lamb)
    # Compute the noise covariance depending on whether we are performing full Bayesian inference on g
    if full:
        # Size: (m, m)
        l_tt = gaussian_kernel_gramix(y_tilde, y_tilde, ls_y)
        # Size: (n, m)
        l_t = gaussian_kernel_gramix(y, y_tilde, ls_y)
        # Size: (m, m)
        noise_cov = l_tt + sigma ** 2 * np.eye(m) - np.dot(np.transpose(l_t), trans_mat)
    else:
        # Size: (m, m)
        noise_cov = sigma ** 2 * np.eye(m)
    # Once we have the transformation and noise covariance, apply the transformed Gaussian process equations for learning
    return tgp_nlml(x, z_tilde, ls_x, trans_mat, noise_cov)