#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 7th April 2019
Author: A. P. Naik
Description: Stellar stream simulation
"""
import numpy as np
from .constants import M_sun, kpc, pi
from .util import print_progress, NFW_acc, hernquist_acc, NFW_M_enc
from .tracers import sample_hernquist


class Simulation:
    """
    Class setting up system of Milky Way (NFW), satellite (Hernquist), and
    tracer particles.

    Parameters
    ----------
    MW_M_vir : float
        Virial mass of the Milky Way, in kg.
    MW_r_vir : float
        Virial radius of the Milky Way, in metres.
    MW_c_vir : float
        Virial concentration (i.e. virial radius / NFW scale radius).
    sat_r : float
        Hernquist radius of satellite. UNITS: metres
    sat_M : float
        Total mass of satellite. UNITS: kg
    sat_x0 : 1D array, length 3.
        Initial position of satellite with respect to galactic centre.
        UNITS: metres
    sat_v0 : 1D array, length 3.
        Initial velocity of satellite with respect to galactic centre.
        UNITS: m/s

    Attributes
    ----------
    All of the above parameters are stored as class attributes, in addition to
    the following:

    tracers : bool
        Flag indicating whether there are tracer particles, experiencing the
        gravitational field of both the Milky Way and the satellite. This is
        initially set to False, but is switched on by the 'add_tracers' method.

    Methods
    -------
    add_tracers :
        Add tracer particles to the simulation.
    """
    def __init__(self, MW_M_vir=1e+12*M_sun, MW_r_vir=200*kpc, MW_c_vir=10,
                 sat_r=0.5*kpc, sat_M=5e+8*M_sun,
                 sat_x0=[60*kpc, 0, 0], sat_v0=[0, 1e+5, 0]):

        self.MW_M_vir = MW_M_vir
        self.MW_r_vir = MW_r_vir
        self.MW_c_vir = MW_c_vir
        self.sat_r = sat_r
        self.sat_M = sat_M
        self.sat_x0 = np.array(sat_x0, dtype=np.float64)
        self.sat_v0 = np.array(sat_v0, dtype=np.float64)
        self.sat_x = np.copy(self.sat_x0)
        self.sat_v = np.copy(self.sat_v0)

        # # convert MW parameters to canonical NFW parameters
        self.MW_r_NFW = self.MW_r_vir/self.MW_c_vir
        c_fac = np.log(1+self.MW_c_vir) - self.MW_c_vir/(1+self.MW_c_vir)
        self.MW_rho_NFW = self.MW_M_vir/(4*pi*self.MW_r_NFW**3*c_fac)

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

        DM_acc = hernquist_acc(self.DM_x, x0, M, a)
        DM_v_half = self.DM_v - 0.5*dt*DM_acc
        stars_acc = hernquist_acc(self.stars_x, x0, M, a)
        stars_v_half = self.stars_v - 0.5*dt*stars_acc

        print("Relaxing tracer particles...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            DM_acc = hernquist_acc(self.DM_x, x0, M, a)
            stars_acc = hernquist_acc(self.stars_x, x0, M, a)
            DM_v_half = DM_v_half + DM_acc*dt
            stars_v_half = stars_v_half + stars_acc*dt
            self.DM_x = self.DM_x + DM_v_half*dt
            self.stars_x = self.stars_x + stars_v_half*dt

        self.DM_v = DM_v_half - 0.5*dt*DM_acc
        self.stars_v = stars_v_half - 0.5*dt*stars_acc

        self.DM_v = self.DM_v + self.sat_v0
        self.stars_v = self.stars_v + self.sat_v0

        return

    def run(self, beta, r_screen, t_max=1e+17, dt=1e+12, N_frames=1000):
        """

        Parameters
        ----------
        beta : float
        r_screen : float
        t_max : float
            Total relaxation time. UNITS: seconds
        dt : float
            Size of timesteps. Note: Timesteps larger than 1e+12 seconds might
            have convergence issues.
        N_frames : int
        """

        self.beta = beta
        self.r_screen = r_screen
        self.t_max = t_max
        self.dt = dt
        rho_0 = self.MW_rho_NFW
        r_s = self.MW_r_NFW
        M = self.sat_M
        a = self.sat_r

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

        # mass enclosed within screening radius
        M_screen = NFW_M_enc(r_screen, self.MW_rho_NFW, self.MW_r_NFW)

        sat_r = np.linalg.norm(self.sat_x)
        DM_r = np.linalg.norm(self.DM_x, axis=-1)[:, None]
        stars_r = np.linalg.norm(self.stars_x, axis=-1)[:, None]

        # calculate enhancement of acceleration on satellite and DM
        # if satellite is outside r_screen, then enhance
        if sat_r > r_screen:
            sat_M_enc = NFW_M_enc(sat_r, self.MW_rho_NFW, self.MW_r_NFW)
            sat_MG = 1+2*beta**2*(1-M_screen/sat_M_enc)
        else:
            sat_MG = 1
        if self.tracers:
            # if DM is outside r_screen, then enhance
            DM_MG = np.ones_like(DM_r)
            DM_M_enc = NFW_M_enc(DM_r, self.MW_rho_NFW, self.MW_r_NFW)
            inds = np.where(DM_r > r_screen)[0]
            DM_MG[inds] = 1+2*beta**2*(1-M_screen/DM_M_enc[inds])
            # enhancement for DM acceleration from satellite
            beta_eff = np.zeros_like(DM_r)
            if sat_r > r_screen:
                beta_eff[inds] = beta
            fac = (1+2*beta_eff**2)

        # desynchronise velocities for leapfrog integration
        sat_acc = sat_MG*NFW_acc(self.sat_x, sat_r, np.zeros(3,), rho_0, r_s)
        sat_v_half = self.sat_v - 0.5*dt*sat_acc
        if self.tracers:
            DM_acc = DM_MG*NFW_acc(self.DM_x, DM_r, np.zeros(3,), rho_0, r_s)
            DM_acc += fac*hernquist_acc(self.DM_x, self.sat_x, M, a)
            DM_v_half = self.DM_v - 0.5*dt*DM_acc
            stars_acc = NFW_acc(self.stars_x, stars_r, np.zeros(3,), rho_0, r_s)
            stars_acc += hernquist_acc(self.stars_x, self.sat_x, M, a)
            stars_v_half = self.stars_v - 0.5*dt*stars_acc

        # main loop
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            sat_r = np.linalg.norm(self.sat_x)
            DM_r = np.linalg.norm(self.DM_x, axis=-1)[:, None]
            stars_r = np.linalg.norm(self.stars_x, axis=-1)[:, None]

            # calculate enhancement of acceleration on satellite and DM
            # if satellite is outside r_screen, then enhance
            if sat_r > r_screen:
                sat_M_enc = NFW_M_enc(sat_r, self.MW_rho_NFW, self.MW_r_NFW)
                sat_MG = 1+2*beta**2*(1-M_screen/sat_M_enc)
            else:
                sat_MG = 1
            if self.tracers:
                # if DM is outside r_screen, then enhance
                DM_MG = np.ones_like(DM_r)
                DM_M_enc = NFW_M_enc(DM_r, self.MW_rho_NFW, self.MW_r_NFW)
                inds = np.where(DM_r > r_screen)[0]
                DM_MG[inds] = 1+2*beta**2*(1-M_screen/DM_M_enc[inds])
                # enhancement for DM acceleration from satellite
                beta_eff = np.zeros_like(DM_r)
                if sat_r > r_screen:
                    beta_eff[inds] = beta
                fac = (1+2*beta_eff**2)

            # calculate accelerations
            sat_acc = sat_MG*NFW_acc(self.sat_x, sat_r, np.zeros(3,), rho_0, r_s)
            if self.tracers:
                DM_acc = DM_MG*NFW_acc(self.DM_x, DM_r, np.zeros(3,), rho_0, r_s)
                DM_acc += fac*hernquist_acc(self.DM_x, self.sat_x, M, a)
                stars_acc = NFW_acc(self.stars_x, stars_r, np.zeros(3,), rho_0, r_s)
                stars_acc += hernquist_acc(self.stars_x, self.sat_x, M, a)

            # timestep
            sat_v_half = sat_v_half + sat_acc*dt
            self.sat_x = self.sat_x + sat_v_half*dt
            if self.tracers:
                DM_v_half = DM_v_half + DM_acc*dt
                self.DM_x = self.DM_x + DM_v_half*dt
                stars_v_half = stars_v_half + stars_acc*dt
                self.stars_x = self.stars_x + stars_v_half*dt

            # snapshot
            if (i+1) % frame_interval == 0:
                self.sat_tr = np.vstack((self.sat_tr, self.sat_x))
                if self.tracers:
                    self.DM_tr = np.dstack((self.DM_tr, self.DM_x))
                    self.stars_tr = np.dstack((self.stars_tr, self.stars_x))

        # resynchronise velocities
        self.sat_v = sat_v_half - 0.5*dt*sat_acc
        self.DM_v = DM_v_half - 0.5*dt*DM_acc
        self.stars_v = stars_v_half - 0.5*dt*stars_acc
        return
