#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: June 2019
Author: A. P. Naik
Description: Various functions (potential, density, acceleration) for the
triaxial logarithmic profile (see Law and Majewski, 2010) for the Milky Way
dark matter halo.
"""
from smoggy.constants import pi, G
import numpy as np


def potential(pos, v_halo, q1, qz, phi, r_halo):
    """
    Potential of triaxial logarithmic halo (see Law and Majewski, 2010) at
    given position.

    Parameters
    ----------
    pos: (3,) or (N, 3) array
        Position, in Galactocentric Cartesian coordinates, at which to evaluate
        potential. UNITS: metres.
    v_halo: float
        Overall normalisation. UNITS: metres/second.
    q1: float
        Axial flattening along equatorial axis. Dimensionless.
    qz: float
        Axial flattening perpendicular to galactic disc. Dimensionless.
    phi: float
        Rotation of ellipsoid about Galactic Z-axis. phi=0 corresponds with
        the q1 flattening being coincident with the Galactic X-axis. UNITS:
        radians.
    r_halo: float
        Scale length of halo. UNITS: metres.

    Returns
    -------
    pot: float or (N,) array, depending on shape of 'pos' parameter
        Potential at 'pos'. UNITS: metres^2/second^2.
    """

    c = np.cos(phi)
    s = np.sin(phi)
    C1 = c**2/q1**2 + s**2
    C2 = s**2/q1**2 + c**2
    C3 = 2*c*s*(1/q1**2 - 1)

    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]

    f = C1*x**2 + C2*y**2 + C3*x*y + (z/qz)**2 + r_halo**2
    pot = v_halo**2*np.log(f)

    return pot


def density(pos, v_halo, q1, qz, phi, r_halo):
    """
    Density of triaxial logarithmic halo (see Law and Majewski, 2010) at
    given position.

    Parameters
    ----------
    pos: (3,) or (N, 3) array
        Position, in Galactocentric Cartesian coordinates, at which to evaluate
        potential. UNITS: metres.
    v_halo: float
        Overall normalisation. UNITS: metres/second.
    q1: float
        Axial flattening along equatorial axis. Dimensionless.
    qz: float
        Axial flattening perpendicular to galactic disc. Dimensionless.
    phi: float
        Rotation of ellipsoid about Galactic Z-axis. phi=0 corresponds with
        the q1 flattening being coincident with the Galactic X-axis. UNITS:
        radians.
    r_halo: float
        Scale length of halo. UNITS: metres.

    Returns
    -------
    rho: float or (N,) array, depending on shape of 'pos' parameter
        Density at 'pos'. UNITS: kilograms/metres^3
    """

    c = np.cos(phi)
    s = np.sin(phi)
    C1 = c**2/q1**2 + s**2
    C2 = s**2/q1**2 + c**2
    C3 = 2*c*s*(1/q1**2 - 1)

    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]

    f = C1*x**2 + C2*y**2 + C3*x*y + (z/qz)**2 + r_halo**2
    dfdx = 2*C1*x + C3*y
    dfdy = 2*C2*y + C3*x
    dfdz = 2*z/qz**2

    term3 = (dfdx**2 + dfdy**2 + dfdz**2)/f
    rho = (v_halo**2/(4*pi*G*f))*(2*C1 + 2*C2 + 2/qz - term3)

    return rho


def acceleration(pos, v_halo, q1, qz, phi, r_halo):
    """
    Acceleration due to triaxial logarithmic halo (see Law and Majewski, 2010)
    at given position.

    Parameters
    ----------
    pos: (3,) or (N, 3) array
        Position, in Galactocentric Cartesian coordinates, at which to evaluate
        potential. UNITS: metres.
    v_halo: float
        Overall normalisation. UNITS: metres/second.
    q1: float
        Axial flattening along equatorial axis. Dimensionless.
    qz: float
        Axial flattening perpendicular to galactic disc. Dimensionless.
    phi: float
        Rotation of ellipsoid about Galactic Z-axis. phi=0 corresponds with
        the q1 flattening being coincident with the Galactic X-axis. UNITS:
        radians.
    r_halo: float
        Scale length of halo. UNITS: metres.

    Returns
    -------
    acc: array, same shape as 'pos' parameter
        Acceleration at 'pos'. UNITS: metres/second^2
    """

    c = np.cos(phi)
    s = np.sin(phi)
    C1 = c**2/q1**2 + s**2
    C2 = s**2/q1**2 + c**2
    C3 = 2*c*s*(1/q1**2 - 1)

    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]

    f = C1*x**2 + C2*y**2 + C3*x*y + (z/qz)**2 + r_halo**2
    prefac = v_halo**2/f
    dphidx = prefac*(2*C1*x + C3*y)
    dphidy = prefac*(2*C2*y + C3*x)
    dphidz = prefac*(2*z/qz**2)

    return np.stack((-dphidx, -dphidy, -dphidz), axis=-1)
