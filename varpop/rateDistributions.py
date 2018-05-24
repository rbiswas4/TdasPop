"""
"""
from __future__ import absolute_import, print_function, division
__all__ = ['PowerLawRates']
import numpy as np
from astropy.cosmology import Planck15
from . import BaseRateDistributions


class PowerLawRates(BaseRateDistributions):
    """
    This class is a concrete implementation of `RateDistributions` with the
    following properties:
    - The var rate : The variable rate is a single power law with numerical
        coefficients (alpha, beta)  passed into the instantiation. The rate is
        the number of variables at redshift z per comoving volume per unit
        observer time over the entire sky expressed in units of 
        numbers/Mpc^3/year 
    - A binning in redshift is used to perform the calculation of the expected
        number of variables.
    - It is assumed that the change of rates and volume within a redshift bin
        is negligible enough that samples to the true distribution may be drawn
        by obtaining number of variable samples of z from a uniform distribution
        within the z bin.
    - The expected number of variables in each of these redshift bins is
        computed using the rate above at the midpoint of the redshift bin, using
        a cosmology to compute the comoving volume for the redshift bin
    - The actual numbers of variables are determined by a Poisson Distribution
        using the expected number of variables in each redshift bin,
        determined with a random state passed in as an argument.
        This number must be integral.

    """

    def __init__(self,
                 rng,
                 cosmo=Planck15,
                 alpha=2.6e-5, beta=1.5,
                 zbinEdges=None,
                 zlower=1.0e-8,
                 zhigher=1.4,
                 numBins=20,
                 surveyDuration=10., # Unit of  years
                 fieldArea=None, # Unit of degree square
                 skyFraction=None):
        """
        Parameters
        ----------
        rng : instance of `np.random.RandomState`
        cosmo : Instance of `astropy.cosmology` class, optional, defaults to Planck15
            data structure specifying the cosmological parameters 
        alpha : float, optional, defaults to 2.6e-5
            constant amplitude in SN rate powerlaw
        beta : float, optional, defaults to 1.5
            constant exponent in SN rate powerlaw
        surveyDuration : float, units of years, defaults to 10.
            duration of the survey in units of years
        fieldArea : float, units of degree sq, defaults to None
            area of the field over which supernovae are being simulated in
            units of square degrees. Overridden by `SkyFraction`
        skyFraction : float, optional, defaults to None
            Alternative way of specifying fieldArea by supplying the unitless
            ratio between sky area where the supernovae are being simulated and
            accepting the default `None` for `fieldArea`. Overrides `fieldArea`
        """
        self._rng = rng
        self.cosmo = cosmo
        self.alpha = alpha
        self.beta = beta
        self.zlower = zlower
        self.zhigher = zhigher
        self.numBins = numBins
        self._zbinEdges = zbinEdges
        self.DeltaT = surveyDuration
        self.fieldArea = fieldArea
        self._skyFraction = skyFraction
        # not input
        self._numSN = None
        self._zSamples = None

    @property
    def randomState(self):
        if self._rng is None:
            raise ValueError('rng must be provided')
        return self._rng

    @property
    def cosmology(self):
        return self.cosmo

    @property
    def skyFraction(self):
        """
        fraction of the sky
        """
        if self._skyFraction is None:
            if self.fieldArea is None:
                raise ValueError('Of fieldArea, skyFraction one must be supplied')
            self._skyFraction = self.fieldArea * np.radians(1.)**2.0  / 4.0 / np.pi
        elif self.fieldArea is not None:
            raise ValueError('Both fieldArea and skyFraction are supplied')
        return self._skyFraction


    @property
    def zbinEdges(self):
        if self._zbinEdges is None:
            if any(x is None for x in (self.zlower, self.zhigher, self.numBins)):
                raise ValueError('Both zbinEdges, and'
                                 '(zlower, zhigher, numBins) cannot be None')
            if self.zlower >= self.zhigher:
                raise ValueError('zlower must be less than zhigher')
            self._zbinEdges = np.linspace(self.zlower, self.zhigher, self.numBins + 1)
        return self._zbinEdges

    def snRate(self, z):
        """
        The rate of SN at a redshift z in units of number of SN/ comoving
        volume in Mpc^3/yr in earth years according to the commonly used
        power-law expression 

        .. math:: rate(z) = \alpha (h/0.7)^3 (1.0 + z)^\beta
        
        Parameters
        ----------
        z : array-like, mandatory 
            redshifts at which the rate is evaluated

        Examples
        --------
        """
        res = self.alpha * (1.0 + z)**self.beta 
        res *= ((self.cosmo.h / 0.7) **3.)  
        return res

    def zSampleSize(self): 
        #, zbinEdges=self.zbinEdges, DeltaT=self.DeltaT,
        #            skyFraction=self.skyFraction,
        #            zlower=None, zhigher=None, numBins=None):
        """
        Parameters
        ----------
        zbinEdges : `nunpy.ndarray` of edges of zbins, defaults to None
            Should be of the form np.array([z0, z1, z2]) which will have
            zbins (z0, z1) and (z1, z2)
        skyFraction : np.float, optional, 
        fieldArea : optional, units of degrees squared
            area of sky considered.
        zlower : float, optional, defaults to None
            lower edge of z range
        zhigher : float, optional, defaults to None
            higher edge of z range
        numBins : int, optional, defaults to None
           if not None, overrides zbinEdges
        """
        DeltaT = self.DeltaT
        skyFraction = self.skyFraction
        zbinEdges = self.zbinEdges
        z_mids = 0.5 * (zbinEdges[1:] + zbinEdges[:-1])
        snpervolume = self.snRate(z_mids) 

        # Comoving volume of the univere in between zlower and zhigher
        vols = self.cosmo.comoving_volume(zbinEdges)

        # Comoving volume in each bin
        vol = vols[1:] - vols[:-1]
        vol *= skyFraction
        
        numSN = vol * snpervolume * DeltaT / (1.0 + z_mids)
        return numSN.value

    def numSN(self):
        """
        Return the number of expected supernovae in time DeltaT, with a rate snrate
        in a redshift range zlower, zhigher divided into numBins equal redshift
        bins. The variation of the rate within a bin is ignored.
    
        Parameters
        ----------
        zlower : mandatory, float
            lower limit on redshift range
        zhigher : mandatory, float
            upper limit on redshift range
        numBins : mandatory, integer
            number of bins
        cosmo : `astropy.cosmology` instance, mandatory
            cosmological parameters
        fieldArea : mandatory, units of radian square
            sky area considered
        """
        if self._numSN is None:
            lam = self.zSampleSize()
            self._numSN = self.randomState.poisson(lam=lam)
        return self._numSN


    @property
    def zSamples(self):
        # Calculate only once
        if self._zSamples is None:
            numSN = self.numSN()
            zbinEdges =  self.zbinEdges
            x = zbinEdges[:-1]
            y = zbinEdges[1:]
            arr = (self.randomState.uniform(low=xx, high=yy, size=zz).tolist()
                                           for (xx, yy, zz) in zip(x,y, numSN))
            self._zSamples = np.asarray(list(__x for __lst in arr
                                             for __x in __lst))
        return self._zSamples
