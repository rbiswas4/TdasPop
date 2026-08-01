"""
Sampling routines
"""
__all__ = ['Sample1D', 'GMMSampler']

import numpy as np
from sklearn.mixture import GaussianMixture


class Sample1D(object):
    """
    Class to implement one dimensional sampling for an arbitrary function. 


    Parameters
    ----------
    xvals : `np.ndarray`
        x values
    pdf : `np.ndarray`
        pdf values evaluated at x
    """
    def __init__(self, xvals, pdf):
        """
        Parameters
        ----------
        xvals : `np.ndarray`
            x values
        pdf : `np.ndarray`
            pdf values evaluated at x
        """
        self.xvals = xvals
        self.pdf = pdf
        self.cdf = pdf.cumsum() * (self.xvals[-1] - self.xvals[0])/len(xvals)

    def x_from_cdf(self, cdf):
        """
        Given values of the normalized CDF, return the x values corresponding to them.

        Parameters
        ----------
        cdf : `np.ndarray`
            CDF values normalized to reach 1 at x tending to infinity.

        Returns
        -------
        x : `np.ndarray`
             x values for he pdf

        """
        return np.interp(cdf, self.cdf, self.xvals)

    def sample_pdf(self, rng=None, size=100):
        """
        Samples drawn from the pdf  with a random state of requested size

        Parameters
        ----------
        rng : `np.random.RandomState` instance, defaults to `None`
            if `None`, a new `np.random.RandomState` with seed 0 is used.
        size : int, defaults to 100
            number of samples requested

        Returns
        -------
        vals : `np.ndarray`
            numpy array containing samples
        """
        if rng is None:
            rng = np.random.RandomState(0)
        cdf_sample = rng.uniform(size=size)
        vals = self.x_from_cdf(cdf_sample)
        return vals


class GMMSampler(object):
    """
    Class to draw samples from a Gaussian Mixture Model (GMM) with known
    component weights, means and covariances, using
    `sklearn.mixture.GaussianMixture`.

    Parameters
    ----------
    weights : `np.ndarray`, shape (n_components,)
        mixture weight of each component. Must sum to 1.
    means : `np.ndarray`, shape (n_components, n_features)
        mean of each Gaussian component.
    covariances : `np.ndarray`, shape (n_components, n_features, n_features)
        covariance matrix of each Gaussian component.
    """
    def __init__(self, weights, means, covariances):
        self.weights = np.asarray(weights)
        self.means = np.asarray(means)
        self.covariances = np.asarray(covariances)

        n_components = self.means.shape[0]
        self._gmm = GaussianMixture(n_components=n_components,
                                     covariance_type='full')
        # Set the mixture parameters directly instead of calling `fit`,
        # since the GMM here is already fully specified.
        self._gmm.weights_ = self.weights
        self._gmm.means_ = self.means
        self._gmm.covariances_ = self.covariances

    def sample(self, size=1, rng=None):
        """
        Draw samples from the GMM.

        Parameters
        ----------
        size : int, defaults to 1
            number of samples requested
        rng : int or `np.random.RandomState` instance, defaults to `None`
            random state used for reproducibility. If `None`, the samples
            are not reproducible between calls.

        Returns
        -------
        X : `np.ndarray`, shape (size, n_features)
            samples drawn from the GMM
        component : `np.ndarray`, shape (size,)
            index of the mixture component each sample was drawn from
        """
        self._gmm.random_state = rng
        X, component = self._gmm.sample(size)
        return X, component
