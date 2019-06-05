"""
Kernel Module.

Basic functions used for computing anisotropic Gaussian kernels.

Note: Matern and other kernels were removed from this version, since they were not used in the experiments in the end.
"""
import numpy as np
import scipy.linalg as la


def sqdist_gramix(x_p, x_q, length_scale):
    """
    Compute a gram matrix of euclidean distances between two datasets under an isotropic or anisotropic length scale.
    
    Parameters
    ----------
    x_p : np.ndarray [Size: (n_p, d)]
        A dataset
    x_q : np.ndarray [Size: (n_q, d)]
        A dataset
    length_scale : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)
        
    Returns
    -------
    np.ndarray [Size: (n_p, n_q)]
        The resulting gram matrix
    """
    # Size: (n_p, d)
    z_p = np.atleast_2d(x_p) / length_scale
    # Size: (n_q, d)
    z_q = np.atleast_2d(x_q) / length_scale
    # Size: (n_p, n_q)
    d_pq = np.dot(z_p, z_q.transpose())
    # Size: (n_p,)
    d_p = np.sum(z_p ** 2, axis=1)
    # Size: (n_q,)
    d_q = np.sum(z_q ** 2, axis=1)
    # Size: (n_p, n_q)
    return d_p[:, np.newaxis] - 2 * d_pq + d_q


def gaussian_kernel_gramix(x_p, x_q, length_scale):
    """
    Compute a gram matrix of Gaussian kernel values between two datasets under an isotropic or anisotropic length scale.
    
    Parameters
    ----------
    x_p : np.ndarray [Size: (n_p, d)]
        A dataset
    x_q : np.ndarray [Size: (n_q, d)]
        A dataset
    length_scale : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)
        
    Returns
    -------
    np.ndarray [Size: (n_p, n_q)]
        The resulting gram matrix
    """
    # Size: (n_p, n_q)
    return np.exp(-0.5 * sqdist_gramix(x_p, x_q, length_scale))


def convert_anisotropic(length_scale, d):
    """
    Convert isotropic length scale format to anisotropic length scale format.
    
    Parameters
    ----------
    length_scale : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The length scale(s)
    d : int
        The dimensionality of the anisotropic kernel
    
    Returns
    -------
    np.ndarray [Size: (d,)]
        The anisotropic length scale(s)
    """
    # Size: (1,) or (d,)
    length_scale_array = np.atleast_1d(length_scale)
    # Make sure it is not more than 1 dimensional
    assert length_scale_array.ndim == 1
    # Make sure it is either isotropic or anisotropic
    assert length_scale_array.shape[0] in [1, d]
    # Convert it to anistropic only if it is isotropic or keep it the same
    if length_scale_array.shape[0] == 1:
        return np.repeat(length_scale_array, d)
    else:
        return length_scale_array

    
def gaussian_density_gramix(x, mu, sigma):
    """
    Compute a gram matrix of Gaussian density values of a dataset for multiple means.
    Parameters
    ----------
    x : np.ndarray [Size: (n, d)]
        A dataset
    mu : np.ndarray [Size: (m, d)]
        A dataset
    sigma : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The standard deviations(s)
        
    Returns
    -------
    np.ndarray [Size: (n, m)]
        The resulting gram matrix
    """
    # Compute the scaling factor of a diagonal multivariate Gaussian distribution
    d = x.shape[-1]
    const = (np.sqrt(2 * np.pi) ** d) * np.prod(convert_anisotropic(sigma, d))
    # Size: (n, m)
    return gaussian_kernel_gramix(x, mu, sigma) / const


def gaussian_density_gramix_multiple(x, mu, sigma):
    """
    Compute the average gram matrix of Gaussian density values of a dataset for multiple arrays of means.
    
    Parameters
    ----------
    x : np.ndarray [Size: (n, d)]
        A dataset
    mu : np.ndarray [Size: (m, s, d)]
        A dataset where the average is to be taken over the middle dimension (s)
    sigma : float or np.ndarray [Size: () or (1,) for isotropic; (d,) for anistropic]
        The standard deviations(s)
        
    Returns
    -------
    np.ndarray [Size: (n, m)]
        The resulting gram matrix
    """    
    # Obtain and ensure the shape matches
    x = np.atleast_2d(x)
    n, d = x.shape
    m, s, d = mu.shape
    assert d == x.shape[-1]
    assert d == mu.shape[-1]
    # Size: (m * s, d)
    mu_2d = np.reshape(mu, (m * s, d))
    # Size: (n, m * s)
    gramix_2d = gaussian_density_gramix(x, mu_2d, sigma)
    # Size: (n, m, s)
    gramix_3d = np.reshape(gramix_2d, (n, m, s))
    # Size: (n, m)
    return np.mean(gramix_3d, axis=-1)


def negative_log_gaussian(x, mu, sigma):
    """
    Compute the negative log Gaussian density.
    
    Parameters
    ----------
    x : np.ndarray [Size: (d,)]
        The evaluation vector
    mu : np.ndarray [Size: (d,)]
        The mean vector
    sigma : np.ndarray [Size: (d, d)]
        The covariance matrix
        
    Returns
    -------
    float
        The negative log Gaussian density
    """
    lower = True
    # Size: (d, d)
    chol = la.cholesky(sigma, lower=lower)
    # Size: (d,)
    alpha = la.cho_solve((chol, lower), x - mu)
    # The negative log Gaussian density
    return 0.5 * np.dot(np.transpose(x - mu), alpha) + np.sum(np.log(chol.diagonal())) + 0.5 * sigma.shape[0] * np.log(2 * np.pi)