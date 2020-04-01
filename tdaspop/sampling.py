"""
Sampling routines
"""
__all__ = ['Sample1D']

import numpy as np


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

    def sample_pdf(self, rng=np.random.RandomState(0), size=100):
        """
        Samples drawn from the pdf  with a random state of requested size

        Parameters
        ----------
        rng : `np.random.RandomState` instance, defaults to `RandomState(0)`
            random state 
        size : int, defaults to 100
            number of samples requested

        Returns
        -------
        vals : `np.ndarray`
            numpy array containing samples
        """
        cdf_sample = rng.uniform(size=size)
        vals = self.x_from_cdf(cdf_sample)
        return vals
