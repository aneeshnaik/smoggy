#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description:
"""
import numpy as np
from .constants import kpc, Mpc, M_sun, pi
from .tracers import sample_hernquist
from .potentials import MW_acceleration
from .potentials import hernquist_acceleration as h_acc
from .potentials import hernquist_density as h_rho
from .potentials import NFW_density as NFW_rho
from .potentials import miyamoto_density as disc_rho
from .util import print_progress
from .MG_solver import Grid2D


class Simulation:
    def __init__(self, sat_x0=[60*kpc, 0, 0], sat_v0=[0, 1e+5, 0],
                 sat_r=0.5*kpc, sat_M=5e+8*M_sun):

        self.sat_r = sat_r
        self.sat_M = sat_M
        self.sat_x0 = np.array(sat_x0, dtype=np.float64)
        self.sat_v0 = np.array(sat_v0, dtype=np.float64)
        self.sat_x = np.copy(self.sat_x0)
        self.sat_v = np.copy(self.sat_v0)

        self.tracers = False

        return

    def add_tracers(self, N_DM=1000, N_stars=1000, r_cutoff=3*kpc):
        """
        Create tracer particles, arranged in a Hernquist profile around the
        satellite particles. Sample positions and velocities using equilibrium
        distribution function from Binney and Tremaine. If r_cutoff is set,
        then this routine oversamples by a factor of 5, then only selects
        particles within the given cut_off radius.

        Parameters
        ----------
        N_DM : int
            Number of dark matter tracer particles.
        N_stars : int
            Number of stellar tracer particles
        r_cutoff : float
        """

        self.tracers = True

        # create tracer particles; initialise positions and velocities
        N_tracers = N_DM + N_stars
        if N_tracers % 50 != 0:
            raise ValueError("N_tracers should be multiple of 50")
        self.N_DM = N_DM
        self.N_stars = N_stars
        self.N_tracers = N_tracers
        if r_cutoff is None:
            x, v = sample_hernquist(N=N_tracers, M=self.sat_M, a=self.sat_r)
        else:
            x, v = sample_hernquist(N=5*N_tracers, M=self.sat_M, a=self.sat_r)
            r = np.linalg.norm(x, axis=-1)
            eligible = np.where(r < r_cutoff)[0]
            if eligible.size < N_tracers:
                raise ValueError("Not enough particles within cutoff radius!")
            inds = np.random.choice(eligible, N_tracers, replace=False)
            x = x[inds]
            v = v[inds]

        DM_inds = np.random.choice(np.arange(N_tracers), N_DM, replace=False)
        DM_mask = np.zeros(N_tracers, dtype=bool)
        DM_mask[DM_inds] = True

        DM_x = x[DM_mask]
        DM_v = v[DM_mask]
        stars_x = x[~DM_mask]
        stars_v = v[~DM_mask]

        self.DM_x0 = DM_x + self.sat_x0
        self.DM_v0 = DM_v + self.sat_v0
        self.DM_x = np.copy(self.DM_x0)
        self.DM_v = np.copy(self.DM_v0)
        self.stars_x0 = stars_x + self.sat_x0
        self.stars_v0 = stars_v + self.sat_v0
        self.stars_x = np.copy(self.stars_x0)
        self.stars_v = np.copy(self.stars_v0)
        return

    def relax_tracers(self, t_max=1e+17, dt=1e+12):
        """

        Parameters
        ----------
        t_max : float
            Total relaxation time. UNITS: seconds
        dt : float
            Size of timesteps. Note: Timesteps larger than 1e+12 seconds might
            have convergence issues.
        """

        assert self.tracers

        x0 = self.sat_x0
        M = self.sat_M
        a = self.sat_r
        N_iter = int(t_max/dt)

        self.DM_v = self.DM_v - self.sat_v0
        self.stars_v = self.stars_v - self.sat_v0

        DM_acc = h_acc(self.DM_x-x0, M, a)
        DM_v_half = self.DM_v - 0.5*dt*DM_acc
        stars_acc = h_acc(self.stars_x-x0, M, a)
        stars_v_half = self.stars_v - 0.5*dt*stars_acc

        print("Relaxing tracer particles...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            DM_acc = h_acc(self.DM_x-x0, M, a)
            stars_acc = h_acc(self.stars_x-x0, M, a)
            DM_v_half = DM_v_half + DM_acc*dt
            stars_v_half = stars_v_half + stars_acc*dt
            self.DM_x = self.DM_x + DM_v_half*dt
            self.stars_x = self.stars_x + stars_v_half*dt

        self.DM_v = DM_v_half - 0.5*dt*DM_acc
        self.stars_v = stars_v_half - 0.5*dt*stars_acc

        self.DM_v = self.DM_v + self.sat_v0
        self.stars_v = self.stars_v + self.sat_v0

        return

    def run(self, modgrav=False, fR0=None,
            t_max=1e+17, dt=1e+12, N_frames=1000):
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

        self.modgrav = modgrav
        if modgrav:
            assert fR0 is not None
        self.fR0 = fR0
        self.t_max = t_max
        self.dt = dt
        sat_M = self.sat_M
        sat_r = self.sat_r

        # N_iter is number of iterations, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        N_iter = int(t_max/dt)
        if N_iter % N_frames != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        frame_interval = int(N_iter/N_frames)

        # *_tr are the arrays that store the trajectories of the particles
        self.sat_tr = np.copy(self.sat_x0)
        if self.tracers:
            self.DM_tr = np.copy(self.DM_x0)
            self.stars_tr = np.copy(self.stars_x0)

        if modgrav:
            self.grid = Grid2D(ngrid=200, nth=101, rmax=10*Mpc)
            self.grid.set_cosmology(h=0.7, omega_m=0.3)

            self.grid.drho = np.zeros(self.grid.grid_shape)
            for i in range(self.grid.ngrid):
                for j in range(self.grid.nth):
                    x = self.grid.r[i]*np.sin(self.grid.th[j])
                    y = 0
                    z = self.grid.r[i]*np.cos(self.grid.th[j])
                    pos = np.array([x, y, z])
                    rho = h_rho(pos) + NFW_rho(pos) + disc_rho(pos)
                    self.grid.drho[i, j] = rho

            self.grid.iter_solve(niter=1000000, F0=fR0, verbose=True)

        # desynchronise velocities for leapfrog integration
        sat_acc = MW_acceleration(self.sat_x)
        if modgrav:
                sat_acc += self.grid.accel(self.sat_x)
        sat_v_half = self.sat_v - 0.5*dt*sat_acc
        if self.tracers:
            DM_acc = MW_acceleration(self.DM_x)
            DM_acc += (4/3)*h_acc(self.DM_x-self.sat_x, sat_M, sat_r)
            if modgrav:
                DM_acc += self.grid.accel(self.DM_x)
            stars_acc = MW_acceleration(self.stars_x)
            stars_acc += h_acc(self.stars_x-self.sat_x, sat_M, sat_r)
            DM_v_half = self.DM_v - 0.5*dt*DM_acc
            stars_v_half = self.stars_v - 0.5*dt*stars_acc

        # main loop
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            # calculate accelerations
            sat_acc = MW_acceleration(self.sat_x)
            if modgrav:
                sat_acc += self.grid.accel(self.sat_x)
            if self.tracers:
                DM_acc = MW_acceleration(self.DM_x)
                DM_acc += (4/3)*h_acc(self.DM_x-self.sat_x, sat_M, sat_r)
                if modgrav:
                    DM_acc += self.grid.accel(self.DM_x)
                stars_acc = MW_acceleration(self.stars_x)
                stars_acc += h_acc(self.stars_x-self.sat_x, sat_M, sat_r)

            # timestep
            sat_v_half = sat_v_half + sat_acc*dt
            self.sat_x = self.sat_x + sat_v_half*dt
            if self.tracers:
                DM_v_half = DM_v_half + DM_acc*dt
                stars_v_half = stars_v_half + stars_acc*dt
                self.DM_x = self.DM_x + DM_v_half*dt
                self.stars_x = self.stars_x + stars_v_half*dt

            # snapshot
            if (i+1) % frame_interval == 0:
                self.sat_tr = np.vstack((self.sat_tr, self.sat_x))
                if self.tracers:
                    self.DM_tr = np.dstack((self.DM_tr, self.DM_x))
                    self.stars_tr = np.dstack((self.stars_tr, self.stars_x))

        # resynchronise velocities
        self.sat_v = sat_v_half - 0.5*dt*sat_acc
        if self.tracers:
            self.DM_v = DM_v_half - 0.5*dt*DM_acc
            self.stars_v = stars_v_half - 0.5*dt*stars_acc
        return
