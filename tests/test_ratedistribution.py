from __future__ import print_function, division, absolute_import
import os
import pytest
import numpy as np
from astropy.cosmology import Planck15 as cosmo

from tdaspop import PowerLawRates

def test_construction():
    rng = np.random.RandomState(1)
    pl = PowerLawRates(rng, sky_area=5, zlower=1.0e-8, zhigher=1.2,
                       num_bins=200, sky_fraction=None, zbin_edges=None,
                       beta_rate=1.0)
    assert pl.zlower == 1.0e-8
    assert pl.zhigher == 1.2
    assert pl.zbin_edges.size == pl.num_bins + 1
    assert pl.zbin_edges.min() == pl.zlower
    assert pl.zbin_edges.max() == pl.zhigher

    np.testing.assert_almost_equal(pl.sky_fraction, 0.00012120342027738399)


    pl = PowerLawRates(rng, sky_area=None, zbin_edges=np.array([1.0e-8, 0.5, 1.0]),
                       num_bins=None, sky_fraction=0.5,
                       beta_rate=1.0)

    assert pl.zlower == 1.0e-8
    assert pl.zhigher == 1.0
    assert pl.zbin_edges.size - 1 == pl.num_bins
    assert pl.zbin_edges.min() == pl.zlower
    assert pl.zbin_edges.max() == pl.zhigher

    np.testing.assert_almost_equal(pl.sky_area, 20626.48062470964)

def test_limiting_case():

    rng = np.random.RandomState(1)
    pl = PowerLawRates(rng, sky_area=5, zlower=1.0e-8, zhigher=1.2,
                       num_bins=200, sky_fraction=None, zbin_edges=None,
                       beta_rate=1.0)
    np.testing.assert_allclose(pl.z_sample_sizes.sum(), 6910.968, rtol=0.001)

def test_limiting_case_easy():
    """
    This tests that the number of objects sampled by the `PowerLawRates` is
    equal to the number expected for a simple case with a rate given by
    3.0e-5 * (h/0.7)**3, when 3.0e-5 is passed as the argument.
    """
    rng = np.random.RandomState(1)
    pl = PowerLawRates(rng, sky_area=None, zlower=1.0e-8, zhigher=1.2,
                       num_bins=200, sky_fraction=1., zbin_edges=None,
                       alpha_rate=3.0e-5,
                       survey_duration=1.0,
                       cosmo=cosmo,
                       beta_rate=1.0)

    vol =  cosmo.comoving_volume(z=1.2).value
    np.testing.assert_allclose(vol * 3.0e-5 * (cosmo.h/0.7)**3, pl.z_sample_sizes.sum(), rtol=0.001) 
