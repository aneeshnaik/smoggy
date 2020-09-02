#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 14th April 2019
Author: A. P. Naik
Description: Various procedures to initialise tracer particles.
"""
from emcee import EnsembleSampler as Sampler
from .constants import G, pi
from .util import print_progress
import numpy as np
sqrt = np.sqrt


def hernquist_df_iso(theta, M, a):
    """
    Isotropic Hernquist distribution function; calculates log-probability of
    given phase space position theta.

    Functional form is Eq. (43) in Naik et al., (2020).

    Parameters
    ----------
    theta: array-like, shape (6,)
        Array containing phase space position (x, y, z, vx, vy, vz). UNITS:
        metres and metres/second for positions/velocities respectively.
    M: float
        Total mass of Hernquist blob. UNITS: kilograms.
    a: float
        Scale radius of Hernquist blob. UNITS: metres.

    Returns
    -------
    lnf: float
        Unnormalised ln-probability associated with phase space position.
    """

    v = np.linalg.norm(theta[3:])
    r = np.linalg.norm(theta[:3])

    E = 0.5*v**2 - G*M/(r+a)
    x = E/(G*M/a)

    if x >= 0.:
        return -1e+20
    elif x <= -1.:
        return -1e+20
    else:
        B = np.abs(x)
        prefac = np.sqrt(B)/(1-B)**2 * 1/(np.sqrt(2)*(2*pi)**3*(G*M*a)**(3/2))
        term1 = (1-2*B)*(8*B**2-8*B-3)
        term2 = 3*np.arcsin(np.sqrt(B))/np.sqrt(B*(1-B))
        lnf = np.log(prefac*(term1+term2))
    return lnf


def hernquist_df_aniso(theta, M, a):
    """
    Anisotropic Hernquist distribution function; calculates log-probability of
    given phase space position theta.

    Functional form is Eq. (44) in Naik et al., (2020)

    Parameters
    ----------
    theta: array-like, shape (6,)
        Array containing phase space position (x, y, z, vx, vy, vz). UNITS:
        metres and metres/second for positions/velocities respectively.
    M: float
        Total mass of Hernquist blob. UNITS: kilograms.
    a: float
        Scale radius of Hernquist blob. UNITS: metres.

    Returns
    -------
    lnf: float
        Unnormalised ln-probability associated with phase space position.
    """

    v = np.linalg.norm(theta[3:])
    r = np.linalg.norm(theta[:3])

    E = 0.5*v**2 - G*M/(r+a)
    x = E/(G*M/a)
    L = np.linalg.norm(np.cross(theta[:3], theta[3:]))

    if x >= 0.:
        return -1e+20
    elif x <= -1.:
        return -1e+20
    else:
        prefac = (3*a) / (4*pi**3)
        lnf = np.log(prefac * E**2 / (G**3*M**3*L))
    return lnf


def sample(N, M, a, df='isotropic'):
    """
    Sample N tracer particles from Hernquist distribution function df,
    parametrised by mass M and scale radius a.

    Sampler uses 50 MCMC walkers, each taking N iterations (after burn-in).
    These samples are then thinned by an interval of 50, giving N
    quasi-independent samples.

    Parameters
    ----------
    N: int
        Number of particles to sample. Note: this needs to be a multiple of 50.
    M: float
        Total mass of Hernquist blob. UNITS: kilograms.
    a: float
        Scale radius of Hernquist blob. UNITS: metres.
    df: {'isotropic' (default), 'anisoptropic'}, optional
        Whether to use isotropic or anisotropic distribution functions. The
        functions are Eqs. (43) and (44) respectively in Naik et al., (2020).

    Returns
    -------
    pos: (N, 3) array
        Positions of sampled particles, in Cartesian coordinates. UNITS:
        metres.
    vel: (N, 3) array
        Velocities of sampled particles, in Cartesian coordinates. UNITS:
        metres/second.
    """
    if df == 'isotropic':
        df_function = hernquist_df_iso
    elif df == 'anisotropic':
        df_function = hernquist_df_aniso
    else:
        raise KeyError("Distribution function not recognised")

    # set up sampler
    nwalkers, ndim = 50, 6
    n_burnin = 1000
    assert N % nwalkers == 0
    n_iter = N
    s = Sampler(nwalkers, ndim, df_function, args=[M, a])

    # set up initial walker positions
    v_sig = 0.5*np.sqrt(G*M/a)/np.sqrt(3)
    sig = np.array([0.3*a, 0.3*a, 0.3*a, v_sig, v_sig, v_sig])
    p0 = -sig + 2*sig*np.random.rand(nwalkers, ndim)

    # burn in
    print("Burning in...")
    s.run_mcmc(p0, n_burnin)
    #for i, result in enumerate(s.sample(p0, iterations=n_burnin)):
    #        print_progress(i, n_burnin, interval=n_burnin//50)

    # take final sample
    p0 = s.chain[:, -1, :]
    s.reset()
    print("Taking final sample...")
    s.run_mcmc(p0, n_iter, thin=50)
    #for i, result in enumerate(s.sample(p0, iterations=n_iter, thin=50)):
    #        print_progress(i, n_iter)
    pos = s.flatchain[:, :3]
    vel = s.flatchain[:, 3:]

    return pos, vel
