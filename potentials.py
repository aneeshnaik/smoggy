#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Milky Way potential/density
"""
import numpy as np
from constants import G, M_sun, kpc, pi, pc
sqrt = np.sqrt

# default values for Miyamoto-Nagai disc, from LM10
M_disc = 1e+11*M_sun
a_disc = 6.5*kpc
b_disc = 0.26*kpc

# default values for Hernquist bulge, from LM10
M_hernquist = 3.4e+10*M_sun
a_hernquist = 0.7*kpc

# default values for NFW halo, requiring v_circ = 220km/s at solar radius
rho_0_NFW = 0.0025*M_sun/pc**3
r_s_NFW = 27*kpc


def miyamoto_potential(pos, M_disc=M_disc, a=a_disc, b=b_disc):
    """
    Potential of a Miyamoto-Nagai disc. Default parameter values are MW
    parameters from Law and Majewski 2010.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_disc : float
        Overall mass normalisation. Default is 10^11 solar masses. UNITS: kg.
    a : float
        Disc scale length. Default is 6.5 kpc. UNITS: metres.
    b : float
        Disc scale width. Default is 0.26 kpc. UNITS: metres.

    Returns
    -------
    phi : numpy array, shape (N,), or float
        Potential at given positions. UNITS: m^2/s^2.
    """

    if pos.ndim == 1:
        R = np.linalg.norm(pos[:2])
        z = pos[2]
    else:
        R = np.linalg.norm(pos[:, :2], axis=-1)
        z = pos[:, 2]

    denom = sqrt(R**2+(a+sqrt(z**2+b**2))**2)
    phi = -G*M_disc/denom
    return phi


def miyamoto_density(pos, M_disc=M_disc, a=a_disc, b=b_disc):
    """
    Density of a Miyamoto-Nagai disc. Default parameter values are MW
    parameters from Law and Majewski 2010.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres
    M_disc : float
        Overall mass normalisation. Default is 10^11 solar masses. UNITS: kg
    a : float
        Disc scale length. Default is 6.5 kpc. UNITS: metres.
    b : float
        Disc scale width. Default is 0.26 kpc. UNITS: metres.

    Returns
    -------
    rho : numpy array, shape (N,), or float
        Density at given positions. UNITS: kg/m^3.
    """
    if pos.ndim == 1:
        R = np.linalg.norm(pos[:2])
        z = pos[2]
    else:
        R = np.linalg.norm(pos[:, :2], axis=-1)
        z = pos[:, 2]

    z_plus = sqrt(z**2+b**2)
    prefac = b**2*M_disc/(4*pi)
    top = a*R**2 + (a+3*z_plus)*(a+z_plus)**2
    bottom = z_plus**3*(R**2+(a+z_plus)**2)**(5/2)
    rho = prefac*top/bottom
    return rho


def miyamoto_acceleration(pos, M_disc=M_disc, a=a_disc, b=b_disc):
    """
    Acceleration due to a Miyamoto-Nagai disc. Default parameter values are MW
    parameters from Law and Majewski 2010.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres
    M_disc : float
        Overall mass normalisation. Default is 10^11 solar masses. UNITS: kg
    a : float
        Disc scale length. Default is 6.5 kpc. UNITS: metres.
    b : float
        Disc scale width. Default is 0.26 kpc. UNITS: metres.

    Returns
    -------
    acc : numpy array, shape same as pos
        Acceleration at given positions. UNITS: m/s^2.
    """

    if pos.ndim == 1:
        R = np.linalg.norm(pos[:2])
        z = pos[2]
        z_plus = sqrt(z**2+b**2)
        denom = (R**2+(a+z_plus)**2)**(1.5)
        acc = -G*M_disc*pos/denom
        acc[2] *= (a+z_plus)/z_plus
    else:
        R = np.linalg.norm(pos[:, :2], axis=-1)
        z = pos[:, 2]
        z_plus = sqrt(z**2+b**2)
        denom = (R**2+(a+z_plus)**2)**(1.5)
        acc = -G*M_disc*pos/denom[:, None]
        acc[:, 2] *= (a+z_plus)/z_plus
    return acc


def hernquist_potential(pos, M_hernquist=M_hernquist, a=a_hernquist):
    """
    Potential of a Hernquist bulge. Default parameter values are MW
    parameters from Law and Majewski 2010.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. Default is 3.4 x 10^10 solar masses.
        UNITS: kg.
    a : float
        Scale radius. Default is 0.7 kpc. UNITS: metres.

    Returns
    -------
    phi : numpy array, shape (N,) or float
        Potential at given positions. UNITS: m^2/s^2.
    """
    r = np.linalg.norm(pos, axis=-1)
    phi = -G*M_hernquist/(r+a)
    return phi


def hernquist_density(pos, M_hernquist=M_hernquist, a=a_hernquist):
    """
    Density of a Hernquist bulge. Default parameter values are MW
    parameters from Law and Majewski 2010.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. Default is 3.4 x 10^10 solar masses.
        UNITS: kg.
    a : float
        Scale radius. Default is 0.7 kpc. UNITS: metres.

    Returns
    -------
    rho : numpy array, shape (N,) or float
        Density at given positions. UNITS: kg/m^3.
    """
    r = np.linalg.norm(pos, axis=-1)
    x = r/a
    denom = 2*pi*a**3*x*(1+x)**3
    rho = M_hernquist/denom
    return rho


def hernquist_acceleration(pos, M_hernquist=M_hernquist, a=a_hernquist):
    """
    Acceleration due to a Hernquist bulge. Default parameter values are MW
    parameters from Law and Majewski 2010.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_hernquist : float
        Overall mass normalisation. Default is 3.4 x 10^10 solar masses.
        UNITS: kg.
    a : float
        Scale radius. Default is 0.7 kpc. UNITS: metres.

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


def NFW_potential(pos, rho_0=rho_0_NFW, r_s=r_s_NFW):
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


def NFW_density(pos, rho_0=rho_0_NFW, r_s=r_s_NFW):
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


def NFW_acceleration(pos, rho_0=rho_0_NFW, r_s=r_s_NFW):
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


if __name__ == '__main__':

    pos_array = -30*kpc + 60*kpc*np.random.rand(1000, 3)
    pos_single = np.array([10*kpc, -10*kpc, 0.01*kpc])
    pos_array[250] = pos_single
