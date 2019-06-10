#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
from gravstream.constants import G, pi
import numpy as np


# default values for NFW halo, requiring v_circ = 220km/s at solar radius,
# assuming LM10 potentials for disc and bulge
#rho_0_NFW = 0.0025*M_sun/pc**3
#r_s_NFW = 27*kpc


def potential(pos, rho_0, r_s):
    """
    Potential of an NFW halo. Default scale radius is from Gerhard+Wegg, 2018.
    Density normalisation is such that circular velocity is roughly 220km/s at
    the Solar radius, when combined with Hernquist bulge and Miyamoto disc.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    rho_0 : float
        Overall density normalisation. Default is 2.5 x 10^-3 solar masses per
        cubic parsec. UNITS: kg/m^3.
    r_s : float
        Scale radius. Default is 27 kpc. UNITS: metres.

    Returns
    -------
    phi : numpy array, shape (N,) or float
        Potential at given positions. UNITS: m^2/s^2.
    """
    r = np.linalg.norm(pos, axis=-1)
    phi = - (4*pi*G*rho_0*r_s**3/r)*np.log(1+r/r_s)
    return phi


def density(pos, rho_0, r_s):
    """
    Density of an NFW halo. Default scale radius is from Gerhard+Wegg, 2018.
    Density normalisation is such that circular velocity is roughly 220km/s at
    the Solar radius, when combined with Hernquist bulge and Miyamoto disc.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    rho_0 : float
        Overall density normalisation. Default is 2.5 x 10^-3 solar masses per
        cubic parsec. UNITS: kg/m^3.
    r_s : float
        Scale radius. Default is 27 kpc. UNITS: metres.

    Returns
    -------
    rho : numpy array, shape (N,) or float
        Density at given positions. UNITS: kg/m^3.
    """
    r = np.linalg.norm(pos, axis=-1)
    x = r/r_s
    rho = rho_0/(x*(1+x)**2)
    return rho


def acceleration(pos, rho_0, r_s):
    """
    Acceleration due to an NFW halo. Default scale radius is from
    Gerhard+Wegg, 2018. Density normalisation is such that circular velocity
    is roughly 220km/s at the Solar radius, when combined with Hernquist bulge
    and Miyamoto disc.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    rho_0 : float
        Overall density normalisation. Default is 2.5 x 10^-3 solar masses per
        cubic parsec. UNITS: kg/m^3.
    r_s : float
        Scale radius. Default is 27 kpc. UNITS: metres.

    Returns
    -------
    acc : numpy array, shape same as pos
        Acceleration at given positions. UNITS: m/s^2.
    """
    if pos.ndim == 1:
        r = np.linalg.norm(pos)
    else:
        r = np.linalg.norm(pos, axis=-1)[:, None]

    prefac = 4*np.pi*G*rho_0*r_s**3
    term1 = np.log(1 + r/r_s)/r**2
    term2 = 1/(r*(r_s+r))
    acc = -prefac*(term1-term2)*(pos/r)
    return acc


def mass_enc(pos, rho_0, r_s):
    r = np.linalg.norm(pos, axis=-1)
    x = r/r_s
    M_enc = 4*pi*rho_0*r_s**3*(np.log(1+x)-x/(1+x))
    return M_enc
