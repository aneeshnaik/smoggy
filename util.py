#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Miscellaneous functions used in gravstream module.

Functions
---------
print_progress :
    Display a percentage progress bar within a for loop.
hernquist_acc
NFW_M_enc
NFW_acc
"""
import sys as sys
import numpy as np
from .constants import G, pi


def print_progress(i, n, interval=1):
    """
    Display a precentage progress bar within a for loop. Should be called at
    each iteration of loop (see example below).

    Parameters
    ----------
    i : int
        Current iteration number.
    n : int
        Length of for loop.
    interval : int
        How often to update progress bar, in number of iterations. Default is
        1, i.e. every iteration. For larger loops, it can make sense to have
        interval = n//50 or similar.

    Example
    -------
    >>> n = 1234
    >>> for i in range(n):
    ...     print_progress(i, n)
    ...     do_something()

    """

    if (i+1) % interval == 0:
        sys.stdout.write('\r')
        j = (i + 1) / n
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.flush()
        if i+1 == n:
            sys.stdout.write('\n')
            sys.stdout.flush()

    return


def hernquist_acc(pos, centre, M, a):
    r_vec = pos-centre
    r = np.linalg.norm(r_vec, axis=-1)[:, None]
    acc = -G*M*(r_vec/r)/(r+a)**(2)
    return acc


def NFW_M_enc(r, rho_0, r_s):
    K = 4*pi*rho_0*r_s**3
    M = K*(np.log(1+r/r_s) - r/(r+r_s))
    return M


def NFW_acc(pos, r, centre, rho_0, r_s):
    """
    pos is 3-vector with position at which to calculate acceleration. rho_0 and
    r_s are NFW parameters. Everything in SI units. Returns acceleration vector
    in SI units.
    """
    K = 4*np.pi*G*rho_0*r_s**3
    r_vec = pos-centre
    term1 = np.log(1 + r/r_s)/r**2
    term2 = 1/(r*(r_s+r))
    acc = -K*(term1-term2)*(r_vec/r)
    return acc
