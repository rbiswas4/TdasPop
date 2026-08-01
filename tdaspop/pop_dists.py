"""
Distributions often needed for sampling populations 
"""
__all__ = ['double_gaussian', 'double_gaussian_pdf', 'double_gaussian_logpdf']

import numpy as np
from scipy.stats import norm


def double_gaussian(mode, sigmam, sigmap, size=1000,
                    rng=None):
    """Draw samples from a double gaussian distribution


    Parameters
    ----------
    mode: `np.float`
	mode of the distribution.
    sigmam: `np.float`
	standard deviation of the distribution
    sigmap: `np.float`
	standard deviation of the distribution
    size: int
	number of samples required
    rng: instance `np.random.RandomState`, defaults to `None`
	if `None`, a new `np.random.RandomState` with seed 1 is used.

    Returns
    -------
    samples: `np.ndarray`
    samples from the double gaussian distribution with given parameters.

    Notes
    -----
    This code is essentially the same as code contributed by D. Rubin for SN.
    """
    if rng is None:
        rng = np.random.RandomState(1)

    # Stick to a convention sigmam is a +ve number
    sigs = np.abs([sigmam, sigmap])
    probs = sigs/sigs.sum()
    sigsamps = rng.choice((-sigs[0], sigs[1]), size=size, replace=True, p=probs)
    samps = np.abs(rng.normal(0., 1., size=size))
    return samps * sigsamps + mode

def double_gaussian_pdf(x, mode, sigmam, sigmap):
    """
    Return the pdf of the double gaussian

    Parameters
    ----------
    x : scalar or `np.ndarray` of floats
    mode : float
        mode of the distribution
    sigmam : float
        standard deviation of lower side
    sigmap : float
        standard deviation of higher side
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    A = 2.0 / (sigmam + sigmap)

    mask = x > mode

    pdf = sigmam * norm.pdf(x, mode, sigmam)
    pdf[mask] = sigmap * norm.pdf(x[mask], mode, sigmap)
    pdf *= A

    return pdf[0] if scalar_input else pdf

def double_gaussian_logpdf(x, mode, sigmam, sigmap):
    """
    Return the log pdf of the double gaussian

    Parameters
    ----------
    x : scalar or `np.ndarray` of floats
    mode : float
        mode of the distribution
    sigmam : float
        standard deviation of lower side
    sigmap : float
        standard deviation of higher side
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)

    A = 2.0 / (sigmam + sigmap)
    logA = np.log(A)

    mask = x > mode

    logpdf = np.log(sigmam) +   norm.logpdf(x, mode, sigmam)
    logpdf[mask] = np.log(sigmap) +   norm.logpdf(x[mask], mode, sigmap)
    logpdf += logA

    return logpdf[0] if scalar_input else logpdf
