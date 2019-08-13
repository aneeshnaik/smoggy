#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
from smog.constants import pi, G
import numpy as np


def potential(pos, v_halo, q1, qz, phi, r_halo):

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


def mass_enc(pos, rho_0, r_s):
    assert False
    return
