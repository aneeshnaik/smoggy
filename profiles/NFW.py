#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Various functions (potential, density, acceleration, enclosed
mass) for an NFW Milky Way dark matter halo.
"""
from smoggy.constants import G, pi, pc, kpc
import numpy as np


def NFW_param_conversion(M_vir, c_vir, delta=200, h=0.7):
    """
    For given virial mass and concentration, calculate NFW scale density and
    scale radius.

    Note that the virial radius here means the radius containing a region of
    average density equal to delta times the cosmic critical density. The
    virial mass is then the mass enclosed within this radius, while the virial
    concentration is the ratio of the virial radius to the NFW scale radius.

    Note that this conversion is very mildly cosmology dependent,
    as the definition of the virial radius depends on the critical density,
    which in turn depends on h.

    Parameters
    ----------
    M_vir: float
        Virial mass of halo. UNITS: kilograms.
    c_vir: float
        Virial concentration of halo. Dimensionless.
    delta: float, optional
        Virial ratio --- See note above. Dimensionless.
    h : float, optional
        Dimensionless Hubble constant. Default is 0.7

    Returns
    -------
    rho_0: float
        NFW scale density. UNITS: kilograms/metre^3
    R_s: float
        NFW scale radius. UNITS: metres.
    """

    # calculate critical density
    H0 = h*100*1000/(1e+6*pc)
    rho_c = 3*H0**2/(8*pi*G)

    # calculate virial radius
    R_vir = (3*M_vir/(4*pi*delta*rho_c))**(1/3)

    # calculate scale radius
    R_s = R_vir/c_vir

    # calculate rho_0
    denom = 4*pi*R_s**3*(np.log(1+c_vir) - (c_vir/(1+c_vir)))
    rho_0 = M_vir/denom

    return rho_0, R_s


def potential(pos, M_vir, c_vir):
    """
    Potential of NFW halo at given position.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_vir: float
        Virial mass of halo. UNITS: kilograms.
    c_vir: float
        Virial concentration of halo. Dimensionless.

    Returns
    -------
    phi : (N,) array or float, depending on shape of 'pos' parameter.
        Potential at given positions. UNITS: metres^2/seconds^2.
    """

    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    r = np.linalg.norm(pos, axis=-1)
    phi = - (4*pi*G*rho_0*r_s**3/r)*np.log(1+r/r_s)
    return phi


def density(pos, M_vir, c_vir, softening=0.0001*kpc):
    """
    Density of NFW halo at given position.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_vir: float
        Virial mass of halo. UNITS: kilograms.
    c_vir: float
        Virial concentration of halo. Dimensionless.
    softening: float, optional
        Gravitational softening. Default is 1e-4 kpc. UNITS: metres.

    Returns
    -------
    phi : (N,) array or float, depending on shape of 'pos' parameter.
        Density at given positions. UNITS: kg/m^3.
    """
    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    r = np.linalg.norm(pos, axis=-1)
    x = r/r_s
    epsilon = softening/r_s
    rho = rho_0/((x+epsilon)*(1+x)**2)
    return rho


def acceleration(pos, M_vir, c_vir):
    """

    """
    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    if pos.ndim == 1:
        r = np.linalg.norm(pos)
    else:
        r = np.linalg.norm(pos, axis=-1)[:, None]

    prefac = 4*pi*G*rho_0*r_s**3
    term1 = np.log(1 + r/r_s)/r**2
    term2 = 1/(r*(r_s+r))
    acc = -prefac*(term1-term2)*(pos/r)
    return acc


def mass_enc(pos, M_vir, c_vir):
    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    r = np.linalg.norm(pos, axis=-1)
    x = r/r_s
    M_enc = 4*pi*rho_0*r_s**3*(np.log(1+x)-x/(1+x))
    return M_enc

