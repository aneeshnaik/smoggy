#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
import numpy as np
from gravstream.constants import pi, G


def density(pos, M, a):

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
