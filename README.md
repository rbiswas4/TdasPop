# VarPop
A base repository to provide common infrastructure in describing populations of variables (astrophysical objects with luminosities varying over time scales to be detected as changing by LSST) of different classes, sampling those populations and validating the distributions.

While the implementations of these classes will be specific for the case of any variable object, any simulated variable with this minimal class structure will be simulated by varsims. In practice, we expect that for a particular variable, it will be best to inherit from the classes in `varPop`. 

A very simple example based on a a light curve being sinosoidal, devoid of astrophysics is shown [here](./examples/Demo_Population.ipynb). This demonstrates how to set up such a population without knowing the astrophysics of a particular example. It also demonstrates that this infrastructure does not require one to have a stochastic distribution in the sense that it samples each parameter. Therefore, one could use a finite set of template objects and parametrize the templates through a discrete index (as done here).

A couple of implementations of more realistic, astrophysical distributions are shown Supernovae Type Ia, modeled using the well known SALT model are setup in the [SNPop](https://github.com/rbiswas4/SNPop) repository. These population models `SimpleSALTPopulation` and `GMM_SALTPopulation`, coded up within the `snpop` package inherit from `varpop` populations and the code can be seen in a package [module](https://github.com/rbiswas4/SNPop/blob/master/snpop/saltpop.py), and their basic functionality is demonstrated as JuPyteR notebooks [here](https://github.com/rbiswas4/SNPop/blob/master/Examples/Demo_Gmm.ipynb) and [here](https://github.com/rbiswas4/SNPop/blob/master/Examples/Demo_SimpleSALTPopulation.ipynb). 


