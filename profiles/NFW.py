#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
from smog.constants import G, pi, pc
import numpy as np
delta = 200
h = 0.7
H0 = h*100*1000/(1e+6*pc)
rho_c = 3*H0**2/(8*np.pi*G)


def NFW_param_conversion(M_vir, c_vir):

    # calculate virial radius
    R_vir = (3*M_vir/(4*np.pi*delta*rho_c))**(1/3)

    # calculate scale radius
    R_s = R_vir/c_vir

    # calculate rho_0
    denom = 4*np.pi*R_s**3*(np.log(1+c_vir) - (c_vir/(1+c_vir)))
    rho_0 = M_vir/denom

    return rho_0, R_s


# default values for NFW halo, requiring v_circ = 220km/s at solar radius,
# assuming LM10 potentials for disc and bulge

def potential(pos, M_vir, c_vir):
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

    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)
    
    r = np.linalg.norm(pos, axis=-1)
    phi = - (4*pi*G*rho_0*r_s**3/r)*np.log(1+r/r_s)
    return phi


def density(pos, M_vir, c_vir):
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
    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    r = np.linalg.norm(pos, axis=-1)
    x = r/r_s
    rho = rho_0/(x*(1+x)**2)
    return rho


def acceleration(pos, M_vir, c_vir):
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
    rho_0, r_s = NFW_param_conversion(M_vir, c_vir)

    if pos.ndim == 1:
        r = np.linalg.norm(pos)
    else:
        r = np.linalg.norm(pos, axis=-1)[:, None]

    prefac = 4*np.pi*G*rho_0*r_s**3
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
