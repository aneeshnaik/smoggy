#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 14th April 2019
Author: A. P. Naik
Description: Various procedures to initialise tracer particles.
"""
from emcee import EnsembleSampler as Sampler
from constants import G, pi
from util import print_progress
import numpy as np
sqrt = np.sqrt


def hernquist_potential(r, M, a):
    phi = -G*M/(r+a)
    return phi


def hernquist_df(theta, M, a):

    v = np.linalg.norm(theta[3:])
    r = np.linalg.norm(theta[:3])

    E = 0.5*v**2 + hernquist_potential(r, M, a)
    x = E/(G*M/a)

    if x >= 0.:
        return -1e+20
    elif x <= -1.:
        return -1e+20
    else:
        B = np.abs(x)
        prefac = sqrt(B)/(1-B)**2 * 1/(sqrt(2)*(2*pi)**3*(G*M*a)**(3/2))
        term1 = (1-2*B)*(8*B**2-8*B-3)
        term2 = 3*np.arcsin(sqrt(B))/sqrt(B*(1-B))
        lnf = np.log(prefac*(term1+term2))
    return lnf


def sample_hernquist(N, M, a):

    # set up sampler
    nwalkers, ndim = 50, 6
    n_burnin = 1000
    assert N % nwalkers == 0
    n_iter = int(N/nwalkers)
    s = Sampler(nwalkers, ndim, hernquist_df, args=[M, a])

    # set up initial walker positions
    v_sig = 0.5*np.sqrt(G*M/a)/sqrt(3)
    sig = np.array([0.3*a, 0.3*a, 0.3*a, v_sig, v_sig, v_sig])
    p0 = -sig + 2*sig*np.random.rand(nwalkers, ndim)

    # burn in
    print("Burning in...")
    for i, result in enumerate(s.sample(p0, iterations=n_burnin)):
            print_progress(i, n_burnin, interval=n_burnin//50)

    # take final sample
    p0 = s.chain[:, -1, :]
    s.reset()
    print("Taking final sample...")
    for i, result in enumerate(s.sample(p0, iterations=n_iter)):
            print_progress(i, n_iter)
    pos = s.flatchain[:, :3]
    vel = s.flatchain[:, 3:]

    return pos, vel
