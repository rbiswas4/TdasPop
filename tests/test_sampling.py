from __future__ import print_function, division, absolute_import
import os
import pytest
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tdaspop import Sample1D

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
