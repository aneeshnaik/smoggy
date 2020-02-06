#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Various functions (potential, density, acceleration, enclosed
mass) for a truncated Hernquist profile, i.e. a Hernquist profile truncated at
10 scale radii.
"""
import numpy as np
from smoggy.constants import pi, G


def density(pos, M, a):
    """
    Density of a truncated Hernquist profile, i.e. a Hernquist profile
    truncated at 10 scale radii.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate density. UNITS: metres.
    M : float
        Total mass of truncated profile (i.e. Hernquist mass within 10 scale
        radii). UNITS: kilograms.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    rho : numpy array, shape (N,) or float
        Density at given positions. UNITS: kg/m^3.
    """

    rho_0 = 121*M/(200*pi*a**3)
    x = np.sqrt(pos[..., 0]**2 + pos[..., 1]**2 + pos[..., 2]**2)/a

    if pos.ndim == 1:
        if x > 10:
            return 0
        else:
            rho = rho_0/(x*(1+x)**3)

    elif pos.ndim == 2:
        rho = np.zeros_like(x)
        mask = np.where(x <= 10)
        xmask = x[mask]
        rho[mask] = rho_0/(xmask*(1+xmask)**3)

    else:
        assert False

    return rho


def mass_enc(pos, M, a):
    """
    Mass enclosed within given position in truncated Hernquist profile, i.e.
    Hernquist profile truncated at 10 scale radii.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate enclosed mass. UNITS: metres.
    M : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    M_enc : (N,) array or float, depending on shape of 'pos' parameter.
        Mass enclosed at given positions. UNITS: kilogram
    """
    x = np.sqrt(pos[..., 0]**2 + pos[..., 1]**2 + pos[..., 2]**2)/a

    if pos.ndim == 1:
        if x > 10:
            M_enc = M
        else:
            M_enc = (121*M/100) * (x**2/(1+x)**2)

    elif pos.ndim == 2:
        M_enc = M*np.ones_like(x)
        mask = np.where(x <= 10)
        xmask = x[mask]
        M_enc[mask] = (121*M/100) * (xmask**2/(1+xmask)**2)

    else:
        assert False

    return M_enc


def potential(pos, M, a):
    """
    Potential of a truncated Hernquist profile, i.e. a Hernquist profile
    truncated at 10 scale radii.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    phi : numpy array, shape (N,) or float
        Potential at given positions. UNITS: m^2/s^2.
    """
    r = np.sqrt(pos[..., 0]**2 + pos[..., 1]**2 + pos[..., 2]**2)
    x = r/a

    if pos.ndim == 1:
        if x > 10:
            phi = -G*M/r
        else:
            phi = -G*M/(10*a) - (1.21*G*M/a)*(1/(1+x) - 1/11)

    elif pos.ndim == 2:
        phi = -G*M/r
        inds = np.where(x <= 10)
        phi[inds] = -G*M/(10*a) - (1.21*G*M/a)*(1/(1+x[inds]) - 1/11)

    else:
        assert False

    return phi


def acceleration(pos, M, a):
    """
    Acceleration due to a Hernquist profile, i.e. a Hernquist profile
    truncated at 10 scale radii.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate acceleration. UNITS: metres.
    M : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Scale radius. UNITS: metres.

    Returns
    -------
    acc : numpy array, shape same as pos
        Acceleration at given positions. UNITS: m/s^2.
    """
    r = np.sqrt(pos[..., 0]**2 + pos[..., 1]**2 + pos[..., 2]**2)
    x = r/a

    if pos.ndim == 1:
        poshat = pos/r
        if x > 10:
            acc = -G*M/r**2 * poshat
        else:
            acc = -(1.21*G*M/a**2)/(1+x)**2 * poshat

    elif pos.ndim == 2:
        r = r[:, None]
        x = x[:, None]
        poshat = pos/r
        acc = -(1.21*G*M/a**2)/(1+x)**2 * poshat
        inds = np.where(x > 10)[0]
        acc[inds] = -G*M/r[inds]**2 * poshat[inds]

    else:
        assert False

    return acc
