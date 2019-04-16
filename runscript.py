#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 16th April 2019
Author: A. P. Naik
Description: Runscript
"""
from main import Simulation
from constants import kpc, M_sun
import pickle


x0 = [60*kpc, 0, 0]
v0 = [0, 1e+5, 0]
s = Simulation(MW_M_vir=1e+12*M_sun, MW_r_vir=200*kpc, MW_c_vir=10,
               sat_r=0.5*kpc, sat_M=5e+8*M_sun, sat_x0=x0, sat_v0=v0)
s.add_tracers(N_DM=1000, N_stars=1000, r_cutoff=3*kpc)
s.relax_tracers()
s.run(beta=0.2, r_screen=70*kpc)

f = open("/data/an485/gravstream_sims/sim_test.obj", 'wb')
pickle.dump(s, f)
f.close()
