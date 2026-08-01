from __future__ import print_function, division, absolute_import
import os
import pytest
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tdaspop import Sample1D, GMMSampler

def test_running_dg():
    rng = np.random.RandomState(1)

    xx = np.linspace(-5., 5., 100000)
    pd = norm.pdf(xx)
    s1d  = Sample1D(xx, pd)
    r = s1d.sample_pdf(np.random.RandomState(1), size=1000000)
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(r, bins=np.linspace(-5., 5., 100), cumulative=True,
                                    histtype='stepfilled', density=True)
    xv = 0.5 * (bins[0:-1] + bins[1:])
    assert max(counts - norm.cdf(xv)) < 0.02

def test_default_rng_reproducible():
    # Repeated calls without an explicit `rng` should be reproducible,
    # each drawing from a fresh `RandomState(0)` rather than sharing state
    # across calls.
    xx = np.linspace(-5., 5., 10000)
    pd = norm.pdf(xx)
    s1d = Sample1D(xx, pd)
    r1 = s1d.sample_pdf(size=100)
    r2 = s1d.sample_pdf(size=100)
    np.testing.assert_array_equal(r1, r2)


def test_gmm_sampler():
    weights = np.array([0.3, 0.7])
    means = np.array([[0., 0.], [10., 10.]])
    covariances = np.array([np.eye(2), np.eye(2) * 4.])

    gmm = GMMSampler(weights, means, covariances)
    X, component = gmm.sample(size=100000, rng=np.random.RandomState(1))

    assert X.shape == (100000, 2)
    assert component.shape == (100000,)

    # Component fractions should match the input weights
    frac0 = np.mean(component == 0)
    np.testing.assert_allclose(frac0, weights[0], atol=0.01)

    # Per component means and stds should match the input parameters
    np.testing.assert_allclose(X[component == 0].mean(axis=0), means[0], atol=0.05)
    np.testing.assert_allclose(X[component == 1].mean(axis=0), means[1], atol=0.1)
    np.testing.assert_allclose(X[component == 0].std(axis=0), 1., atol=0.05)
    np.testing.assert_allclose(X[component == 1].std(axis=0), 2., atol=0.05)

    # Reproducibility with the same rng seed
    X2, component2 = gmm.sample(size=100, rng=np.random.RandomState(42))
    X3, component3 = gmm.sample(size=100, rng=np.random.RandomState(42))
    np.testing.assert_array_equal(X2, X3)
    np.testing.assert_array_equal(component2, component3)
