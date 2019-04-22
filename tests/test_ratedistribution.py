from __future__ import print_function, division, absolute_import
import os
import pytest
import numpy as np

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
