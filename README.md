# VarPop
A base repository to provide common infrastructure in describing populations of variables (astrophysical objects with luminosities varying over time scales to be detected as changing by LSST) of different classes, sampling those populations and validating the distributions.

While the implementations of these classes will be specific for the case of any variable object, any simulated variable with this minimal class structure will be simulated by varsims. In practice, we expect that for a particular variable, it will be best to inherit from the classes in `varPop`. A very simple example, devoid of astrophysics is shown [here](./examples/Demo_Population.ipynb)


