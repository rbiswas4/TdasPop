from __future__ import print_function, division, absolute_import
import os
import pytest
import numpy as np

from tdaspop import double_gaussian

def test_running_dg():
    rng = np.random.RandomState(1)

    samps = double_gaussian(1., 1., 2., size=1000000, rng=rng)

    assert len(samps) == 1000000

def test_limiting_case():

    rng = np.random.RandomState(1)
    samps = double_gaussian(0., 1., 1., size=1000000, rng=rng)
    n_samps = np.random.normal(0., 1., size=1000000)
    bins = np.arange(-2., 2., 0.1)
    counts, bins = np.histogram(samps, bins)
    counts_n, bins_n = np.histogram(samps, bins)
    assert all(counts_n/counts > 0.8)
    assert all(counts_n/counts < 1.2)
    samps = double_gaussian(0., 1., 9., size=1000000, rng=rng)
    assert np.abs(samps[samps < 0.].size/samps.size - 0.1) < 0.05
    samps = double_gaussian(20., 1., 9., size=1000000, rng=rng)
    assert np.abs(samps[samps < 20.].size/samps.size - 0.1) < 0.05
