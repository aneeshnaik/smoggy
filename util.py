#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: Miscellaneous functions used in smoggy.

Functions
---------
print_progress :
    Display a percentage progress bar within a for loop.
"""
import sys as sys


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
