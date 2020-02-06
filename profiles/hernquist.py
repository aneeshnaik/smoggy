#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Various functions (potential, density, acceleration, enclosed
mass) for a Hernquist profile.
"""
from smoggy.constants import G, pi, kpc
import numpy as np


def potential(pos, M_hernquist, a):
    """
    Potential of a Hernquist profile.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    phi : numpy array, shape (N,) or float
        Potential at given positions. UNITS: m^2/s^2.
    """
    r = np.linalg.norm(pos, axis=-1)
    phi = -G*M_hernquist/(r+a)
    return phi


def density(pos, M_hernquist, a, softening=0.0001*kpc):
    """
    Density of a Hernquist profile.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate density. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.
    softening: float, optional
        Gravitational softening. Default is 1e-4 kpc. UNITS: metres.

    Returns
    -------
    rho : numpy array, shape (N,) or float
        Density at given positions. UNITS: kg/m^3.
    """
    r = np.linalg.norm(pos, axis=-1)
    x = r/a
    epsilon = softening/a
    denom = 2*pi*a**3*(x+epsilon)*(1+x)**3
    rho = M_hernquist/denom
    return rho


def acceleration(pos, M_hernquist, a):
    """
    Acceleration due to a Hernquist profile.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate acceleration. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    acc : numpy array, shape same as pos
        Acceleration at given positions. UNITS: m/s^2.
    """
    if pos.ndim == 1:
        r = np.linalg.norm(pos)
    else:
        r = np.linalg.norm(pos, axis=-1)[:, None]

    acc = -G*M_hernquist*(pos/r)/(r+a)**2
    return acc


def mass_enc(pos, M_hernquist, a):
    """
    Mass enclosed within given position in Hernquist profile.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate enclosed mass. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    M_enc : (N,) array or float, depending on shape of 'pos' parameter.
        Mass enclosed at given positions. UNITS: kilogram
    """
    r = np.linalg.norm(pos, axis=-1)
    x = r/a
    M_enc = M_hernquist*x**2/(1+x)**2
    return M_enc
