#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
import numpy as np
from smoggy.constants import G


def potential(pos, M):
    """
    Potential of a point mass.

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M : float
        Mass of point mass. UNITS: kg.

    Returns
    -------
    phi : numpy array, shape (N,) or float
        Potential at given positions. UNITS: m^2/s^2.
    """
    r = np.linalg.norm(pos, axis=-1)
    phi = -G*M/r
    return phi


def acceleration(pos, M):
    """
    Acceleration due to a point mass

    Parameters
    ----------
    pos : numpy array, shape (N, 3) or (3,)
        Positions at which to calculate potential. UNITS: metres.
    M : float
        Mass of point mass. UNITS: kg.

    Returns
    -------
    acc : numpy array, shape same as pos
        Acceleration at given positions. UNITS: m/s^2.
    """

    if pos.ndim == 1:
        r = np.linalg.norm(pos)
    else:
        r = np.linalg.norm(pos, axis=-1)[:, None]

    acc = -G*M*(pos/r**3)
    return acc


def mass_enc(pos, M):

    r = np.linalg.norm(pos, axis=-1)

    if pos.ndim == 1:
        if r == 0:
            M_enc = 0
        else:
            M_enc = M
    else:
        M_enc = M*np.ones_like(r)
        M_enc[np.where(r == 0)] = 0
    return M_enc
