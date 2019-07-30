#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 11th April 2019
Author: A. P. Naik
Description: Physical constants and unit conversions, for general use. All SI
units.
"""
import numpy as np


# mathematical constants
pi = 3.141592653589793


# physical constants
G = 6.67408e-11  # m^3 kg s^-2
c = 299792458.0  # m/s
h_bar = 1.0545718001391127e-34  # kg m^3 s^-2
M_pl = np.sqrt(h_bar*c/(8*pi*G))

# units
pc = 3.0857e+16  # metres
kpc = 3.0857e+19  # metres
Mpc = 3.0857e+22  # metres
M_sun = 1.9885e+30  # kg
year = 31536000.0  # seconds


# colours for visualisation
green1 = '#EDF8E9'
green2 = '#BAE4B3'
green3 = '#10C476'
green4 = '#31A354'
green5 = '#006D2C'
claret = '#A33159'
