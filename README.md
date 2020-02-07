# smoggy

`smoggy` is a restricted N-body code written in python-3. Its primary purpose, at the time of writing, is to understand the impact of chameleon fifth forces on stellar streams arising from dark matter dominated dwarfs around the Milky Way.

This code was used to generate the results in Naik et al., (2020). More details about the science can be found there. Please direct any comments/questions to the author, Aneesh Naik, at an485@[Cambridge University]

## Prerequisites

This code was written and implemented with python 3.6, and requires the following external packages (the version numbers in parentheses indicate the versions employed at time of writing, not particular dependencies):

* `h5py` (2.9.0)
* `numpy` (1.16.2)
* `emcee` (2.2.1)

## Usage

Setting up and running a simulation all done via the 'SmogSimulation' object, while all post-analysis is done via the 'SmogOutput' object, which reads a saved simulation output file and loads up all of the data.

The following examples should cover all of the basic use cases. More examples can be found at **insert link here**, which contains all the runscripts and plotting scripts used for Naik et al., (2020).

**Example 1:** 1-body, standard gravity

While there are a great deal of arguments `SmogSimulation` will *accept*, the only ones it *needs* are ones which tell it where the satellite start, its initial velocity, its scale radius, and its mass. Then, after initialising, one only needs to use the self-explanatory `run` and `save` methods.

So setting up, running, and saving such a simulation takes the following code:
```python
import smoggy
import numpy as np
from smoggy.constants import kpc, M_sun


sim = smoggy.SmogSimulation(sat_x0=np.array([50*kpc, 0, 0]),
                            sat_v0=np.array([0, 200000, 0]),
                            sat_radius=0.1*kpc, sat_mass=5e+8*M_sun)
sim.run(t_max=1e+17, N_snapshots=500)
sim.save("example_1")
```
The simulation data can all then be found in `example_1.hdf5`.

As a default, the gravity is 'standard', i.e. fifth forces are switched off, and there are no tracer particles, i.e. the simulation is a simply 1-body problem, with a single satellite 'particle' moving in the Milky Way potential. The default Milky Way potential is a NFW halo (virial mass 10<sup>12</sup> M<sub>sun</sub> and concentration 12), plus a Miyamoto-Nagai disc and Hernquist bulge, both using parameter values from [Law and Majewski (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...714..229L/abstract). However, the Milky Way potential is *very* customisable through the various keyword arguments fed to `SmogSimulation`.

**Example 2:** Restricted N-body, standard gravity

To add tracer particles, set the `tracers` flag to True, and add specify the numbers of type 1 (dark matter) and type 2 (stars) particles via `N1` and `N2`. The only difference between the two particle types is whether they couple to the fifth force (stars don't), so they are indistinguishable in the standard gravity case.
```python
sim = smoggy.SmogSimulation(sat_x0=np.array([50*kpc, 0, 0]),
                            sat_v0=np.array([0, 200000, 0]),
                            sat_radius=0.1*kpc, sat_mass=5e+8*M_sun,
                            tracers=True, N1=1000, N2=1000)
```


**Example 3:** Restricted N-body, modified gravity

To switch on fifth forces, set the `modgrav` flag to True, specify the fifth force coupling strength via `beta`, and the Milky Way and satellite screening radii via `MW_r_screen` and `sat_r_screen` respectively.
```python
sim = smoggy.SmogSimulation(sat_x0=np.array([50*kpc, 0, 0]),
                            sat_v0=np.array([0, 200000, 0]),
                            sat_radius=0.1*kpc, sat_mass=5e+8*M_sun,
                            tracers=True, N1=1000, N2=1000,
                            modgrav=True, beta=0.1, 
                            MW_r_screen=10*kpc, sat_r_screen=0.5*kpc)
```

**Example 4:** Plotting a stream image

To load a saved simulation file and create a stream image:
```python
import smoggy
from smoggy.constants import kpc
import matplotlib.pyplot as plt


# load data
d = smoggy.SmogOutput("saved_simulation.hdf5")

# get particle positions
x0 = d.p0_positions[-1]/kpc
v0 = d.p0_velocities[-1]
x1 = d.p1_positions[-1]/kpc
v1 = d.p1_velocities[-1]
x2 = d.p2_positions[-1]/kpc
v2 = d.p2_velocities[-1]

# set up axes
ax = plt.subplot()
ax.set_aspect('equal')

# plot particles
ax.scatter(x1[:, 0], x1[:, 2], s=3, alpha=0.5, rasterized=True, c='red')
ax.scatter(x2[:, 0], x2[:, 2], s=3, alpha=0.5, rasterized=True, c='green')
plt.show()
```


## Authors

This code was written by **Aneesh Naik** ([website](https://www.ast.cam.ac.uk/~an485/)). The research was performed in collaboration with the co-authors of Naik et al. (2019):

* [N. Wyn Evans](https://people.ast.cam.ac.uk/~nwe/)
* [Ewald Puchwein](https://www.aip.de/Members/epuchwein)
* [Anne-Christine Davis](http://www.damtp.cam.ac.uk/user/acd/)
* [Hongsheng Zhao](http://www-star.st-and.ac.uk/~hz4/)


## License

Copyright (2020) Aneesh Naik.

`smoggy` is free software made available under the MIT license. For details see LICENSE.

If you make use of `smoggy` in your work, please cite our paper ([arXiv](), [ADS]()).


## Acknowledgments

Please see the acknowledgments in the paper for a list of the many people and institutions I and my co-authors indebted!
