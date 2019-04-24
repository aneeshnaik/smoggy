#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description:
"""
import numpy as np
from constants import kpc, M_sun, pi
from .tracers import sample_hernquist
from .potentials import MW_acceleration
from .util import print_progress


class Simulation:
    def __init__(self, sat_x0=[60*kpc, 0, 0], sat_v0=[0, 1e+5, 0]):

        self.sat_x0 = np.array(sat_x0, dtype=np.float64)
        self.sat_v0 = np.array(sat_v0, dtype=np.float64)
        self.sat_x = np.copy(self.sat_x0)
        self.sat_v = np.copy(self.sat_v0)

        self.tracers = False

        return

    def run(self, modgrav=None, t_max=1e+17, dt=1e+12, N_frames=1000):
        """

        Parameters
        ----------
        t_max : float
            Total relaxation time. UNITS: seconds
        dt : float
            Size of timesteps. Note: Timesteps larger than 1e+12 seconds might
            have convergence issues.
        N_frames : int
        """

        if modgrav not in ['None']:
            raise ValueError("Unrecognised value for 'modgrav'")
        self.modgrav = modgrav
        self.t_max = t_max
        self.dt = dt

        # N_iter is number of iterations, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        N_iter = int(t_max/dt)
        if N_iter % N_frames != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        frame_interval = int(N_iter/N_frames)

        # *_tr are the arrays that store the trajectories of the particles
        self.sat_tr = np.copy(self.sat_x0)

        # desynchronise velocities for leapfrog integration
        sat_acc = MW_acceleration(self.sat_x)
        sat_v_half = self.sat_v - 0.5*dt*sat_acc

        # main loop
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            # calculate accelerations
            sat_acc = MW_acceleration(self.sat_x)

            # timestep
            sat_v_half = sat_v_half + sat_acc*dt
            self.sat_x = self.sat_x + sat_v_half*dt

            # snapshot
            if (i+1) % frame_interval == 0:
                self.sat_tr = np.vstack((self.sat_tr, self.sat_x))

        # resynchronise velocities
        self.sat_v = sat_v_half - 0.5*dt*sat_acc
        return
