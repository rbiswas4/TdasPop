# Time Domain Astronomy Sources Populations

[![tests](https://github.com/rbiswas4/TdasPop/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rbiswas4/TdasPop/actions/workflows/tests.yml)[![PyPI version](https://badge.fury.io/py/tdaspop.svg)](https://badge.fury.io/py/tdaspop)

A base repository to provide common infrastructure in describing populations of Time Domain Astronomy Sources (astrophysical objects with luminosities varying over time scales to be detected as changing by LSST) of different classes, sampling those populations and validating the distributions.

## Installation
This project can be installed directly from github by cloning and running `setup.py`
```
python setup.py install
```
or it can be installed from the pypi server:
```
pip install tdaspop
```
## Description
While the implementations of these classes will be specific for the case of any variable object, any simulated variable with this minimal class structure will be simulated by varsims. In practice, we expect that for a particular variable, it will be best to inherit from the classes in `varPop`. 

A very simple example based on a a light curve being sinosoidal, devoid of astrophysics is shown [here](./examples/Demo_Population.ipynb). This demonstrates how to set up such a population without knowing the astrophysics of a particular example. It also demonstrates that this infrastructure does not require one to have a stochastic distribution in the sense that it samples each parameter. Therefore, one could use a finite set of template objects and parametrize the templates through a discrete index (as done here).

A couple of implementations of more realistic, astrophysical distributions are shown Supernovae Type Ia, modeled using the well known SALT model are setup in the [SNPop](https://github.com/rbiswas4/SNPop) repository. These population models `SimpleSALTPopulation` and `GMM_SALTPopulation`, coded up within the `snpop` package inherit from `varpop` populations and the code can be seen in a package [module](https://github.com/rbiswas4/SNPop/blob/master/snpop/saltpop.py), and their basic functionality is demonstrated as JuPyteR notebooks [here](https://github.com/rbiswas4/SNPop/blob/master/Examples/Demo_Gmm.ipynb) and [here](https://github.com/rbiswas4/SNPop/blob/master/Examples/Demo_SimpleSALTPopulation.ipynb). 

## Releasing

Versioning is managed with [bump-my-version](https://github.com/callowayproject/bump-my-version) (`pip install bump-my-version`, or already included in `install/pip-requirements.txt`). To cut a release, bump `tdaspop/version.py` and tag the commit with one command:
```
bump-my-version bump patch   # or minor / major
```
This updates `tdaspop/version.py`, commits the change, and creates a matching `vX.Y.Z` git tag.

## Code style

Code is formatted with [Black](https://github.com/psf/black), enforced via [pre-commit](https://pre-commit.com/) (`pip install pre-commit`, or already included in `install/pip-requirements.txt`). To enable it locally, run once per clone:
```
pre-commit install
```
This reformats changed files with Black on every commit.


