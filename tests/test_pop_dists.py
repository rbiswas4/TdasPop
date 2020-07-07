from __future__ import print_function, division, absolute_import
import os
import pytest
import numpy as np

from tdaspop import double_gaussian, double_gaussian_pdf, double_gaussian_logpdf

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

def test_double_gaussian_logpdf():

    # Hard coded Double Gaussian
    mode = 20.
    sigmam = 1.
    sigmap = 8.

    # Evauate the pdf at a set of poits
    # points -> x
    num_points = 2 * 1000000
    min_val = -1000. 
    max_val = 1000
    x = np.linspace(min_val, max_val, num_points) 
    pdf = double_gaussian_pdf(x, mode=mode, sigmam=sigmam, sigmap=sigmap)
    logpdf = double_gaussian_logpdf(x, mode=mode, sigmam=sigmam, sigmap=sigmap)
    np.testing.assert_allclose(pdf, np.exp(logpdf), rtol=5)

def test_double_gaussian_pdf_integral():

    # Hard coded Double Gaussian
    mode = 20.
    sigmam = 1.
    sigmap = 8.

    # Do integral to mode by MC
    num_points = 2 * 1000000
    min_val = -1000. 
    max_val = 1000

    x = np.linspace(min_val, mode, num_points) 
    intl = (mode - min_val) * np.sum(double_gaussian_pdf(x, mode=mode, sigmam=sigmam,
        sigmap=sigmap))/num_points 
    x = np.linspace(mode, max_val, num_points) 
    inth = (max_val - mode) * np.sum(double_gaussian_pdf(x, mode=mode, sigmam=sigmam,
        sigmap=sigmap))/num_points 

    np.testing.assert_almost_equal(inth + intl, 1, decimal=3)
    np.testing.assert_almost_equal(inth/intl, 8., decimal=3)
