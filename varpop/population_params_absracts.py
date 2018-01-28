from __future__ import absolute_import, print_function, division
"""
A module with the abstract classes for implementing populaions of variables and
transients.
"""
__all__ = ['RateDistributions']
from future.utils import with_metaclass
import abc
import numpy as np


class RateDistributions(with_metaclass(abc.ABCMeta, object)):
    """
    Populations of objects occur with different abundances or rates at
    different redshifts. This is a base class to describe the numbers of
    objects at different redshifts, and provide samples of redshifts.
    The redshifts should go upto a maximum redshift determined by the
    values of interest for a particular survey. The number of objects
    also depends on the cosmological volume, and usually depends on
    the cosmology. Finally, the area of the sky is also encoded in
    skyFraction.
    """
    @abc.abstractproperty
    def randomState(self):
        """
        Used to provide samples
        """
        pass

    @abc.abstractproperty
    def cosmology(self):
        """
        `~astropy.Cosmology` instance
        """
        pass

    @abc.abstractproperty
    def skyFraction(self):
        """
        The sky fraction over which the samples of the objects will be returned
        """
        pass

    @abc.abstractmethod
    def zSampleSize(self):
        """
        Given a collection of edges of redshift bins, and a skyFraction
        return a collection of expected (possibly non-integral) numbers of SN.
        Since this is an expected number, the number is a float and it is
        perfectly fine to get 3.45 SN in a redshift bin.
        """
        pass

    @abc.abstractproperty
    def zSamples(self):
        pass
