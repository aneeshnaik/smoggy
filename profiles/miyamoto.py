#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Various functions (potential, density, acceleration, enclosed
mass) for a Miyamoto-Nagai galactic disc.
"""
from smoggy.constants import G, pi
import numpy as np


def potential(pos, M_disc, a, b):
    """
    Potential of a Miyamoto-Nagai disc.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M_disc : float
        Overall mass normalisation. UNITS: kg.
    a : float
        Disc scale length. UNITS: metres.
    b : float
        Disc scale width. UNITS: metres.

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

    denom = np.sqrt(R**2+(a+np.sqrt(z**2+b**2))**2)
    phi = -G*M_disc/denom
    return phi


def density(pos, M_disc, a, b):
    """
    Density of a Miyamoto-Nagai disc.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres
    M_disc : float
        Overall mass normalisation. UNITS: kg
    a : float
        Disc scale length. UNITS: metres.
    b : float
        Disc scale width. UNITS: metres.

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

    z_plus = np.sqrt(z**2+b**2)
    prefac = b**2*M_disc/(4*pi)
    top = a*R**2 + (a+3*z_plus)*(a+z_plus)**2
    bottom = z_plus**3*(R**2+(a+z_plus)**2)**(5/2)
    rho = prefac*top/bottom
    return rho


def acceleration(pos, M_disc, a, b):
    """
    Acceleration due to a Miyamoto-Nagai disc.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres
    M_disc : float
        Overall mass normalisation. UNITS: kg
    a : float
        Disc scale length. UNITS: metres.
    b : float
        Disc scale width. UNITS: metres.

    Returns
    -------
    acc : numpy array, shape same as pos
        Acceleration at given positions. UNITS: m/s^2.
    """

    if pos.ndim == 1:
        R = np.linalg.norm(pos[:2])
        z = pos[2]
        z_plus = np.sqrt(z**2+b**2)
        denom = (R**2+(a+z_plus)**2)**(1.5)
        acc = -G*M_disc*pos/denom
        acc[2] *= (a+z_plus)/z_plus
    else:
        R = np.linalg.norm(pos[:, :2], axis=-1)
        z = pos[:, 2]
        z_plus = np.sqrt(z**2+b**2)
        denom = (R**2+(a+z_plus)**2)**(1.5)
        acc = -G*M_disc*pos/denom[:, None]
        acc[:, 2] *= (a+z_plus)/z_plus
    return acc


def M_enc_grid(N_r, N_th, r_max, M_disc, a, b):
    """
    Calculate enclosed mass of Miyamoto-Nagai disc in spherical shells, so that
    enclosed mass at arbitrary radius can be interpolated.

    Parameters
    ----------
    N_r : int
        Number of radial bins
    N_th : int
        Number of polar bins
    r_max : float
        Radius of outermost spherical sphell. UNITS: metres.
    M_disc : float
        Overall mass normalisation. UNITS: kg
    a : float
        Disc scale length. UNITS: metres.
    b : float
        Disc scale width. UNITS: metres.

    Returns
    -------
    r_edges : (N_r+1,) array
        Spherical radii. First radius is 0. UNITS: metres.
    M_enc : (N_r+1,) array
        Mass enclosed in r_edges. First is be 0. UNITS: kilograms.
    """

    # set up r grid
    r_edges = np.linspace(0, r_max, num=N_r+1)  # cell edges
    r_grid = 0.5*(r_edges[1:] + r_edges[:-1])  # cell centres
    dr = np.diff(r_edges)[0]

    # set up theta grid
    th_edges = np.linspace(0, pi/2, num=N_th+1)  # cell edges
    th_grid = 0.5*(th_edges[1:] + th_edges[:-1])  # cell centres
    dth = np.diff(th_edges)[0]

    # calculate density on grid
    x = np.outer(r_grid, np.sin(th_grid))
    z = np.outer(r_grid, np.cos(th_grid))
    pos = np.zeros((N_r*N_th, 3))
    pos[:, 0] = x.flatten()
    pos[:, 2] = z.flatten()
    rho = density(pos, M_disc, a, b).reshape((N_r, N_th))

    # integrate
    const = dr*dth*4*pi
    dM = np.sum(rho*np.sin(th_grid[None, :])*r_grid[:, None]**2*const, axis=1)
    M_enc = np.cumsum(dM)
    M_enc = np.append(0, M_enc)

    return r_edges, M_enc
