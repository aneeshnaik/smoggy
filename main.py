#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 7th April 2019
Author: A. P. Naik
Description: Stellar stream simulation
"""
import numpy as np
from constants import G, M_sun, kpc, pi
from util import print_progress
from tracers import sample_hernquist


def hernquist_acc(pos, centre, M, a):
    r_vec = pos-centre
    r = np.linalg.norm(r_vec, axis=-1)[:, None]
    acc = -G*M*(r_vec/r)/(r+a)**(2)
    return acc


def NFW_M_enc(r, rho_0, r_s):
    K = 4*pi*rho_0*r_s**3
    M = K*(np.log(1+r/r_s) - r/(r+r_s))
    return M


def NFW_acc(pos, centre, rho_0, r_s):
    """
    pos is 3-vector with position at which to calculate acceleration. rho_0 and
    r_s are NFW parameters. Everything in SI units. Returns acceleration vector
    in SI units.
    """
    K = 4*np.pi*G*rho_0*r_s**3
    r_vec = pos-centre
    if r_vec.ndim == 1:
        r = np.linalg.norm(r_vec, axis=-1)
    else:
        r = np.linalg.norm(r_vec, axis=-1)[:, None]
    term1 = np.log(1 + r/r_s)/r**2
    term2 = 1/(r*(r_s+r))
    acc = -K*(term1-term2)*(r_vec/r)
    return acc


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
    sat_pos_init : 1D array, length 3.
        Initial position of satellite with respect to galactic centre.
        UNITS: metres
    sat_pos_init : 1D array, length 3.
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
    def __init__(self, MW_M_vir, MW_r_vir, MW_c_vir,
                 sat_r, sat_M, sat_x0, sat_v0):

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
        self.MW_rho_NFW = self.MW_M_vir/(4*np.pi*self.MW_r_NFW**3*c_fac)

        self.tracers = False

        return

    def add_tracers(self, N_DM, N_stars, r_cutoff=None):
        """
        Create tracer particles, arranged in a Hernquist profile around the
        satellite particles. Sample positions and velocities using equilibrium
        distribution function from Binney and Tremaine. If r_cutoff is set,
        then this routine oversamples by a factor of 5, then only selects
        particles within the given cut_off radius.

        Parameters
        ----------
        N_tracers : int
            Number of tracer particles.
        """
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

    def relax_tracers(self, t_max=1e+17, dt=5e+11):
        """

        Parameters
        ----------
        t_max : float
            Total relaxation time. UNITS: seconds
        dt : float
            Size of timesteps. Note: Timesteps larger than 1e+12 seconds might
            have convergence issues.
        """
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

        self.beta = beta
        self.r_screen = r_screen
        self.t_max = t_max
        self.dt = dt

        rho_0 = self.MW_rho_NFW
        r_s = self.MW_r_NFW
        M = self.sat_M
        a = self.sat_r
        N_iter = int(t_max/dt)
        if N_iter % N_frames != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        frame_interval = int(N_iter/N_frames)

        self.sat_traj = np.copy(self.sat_x0)
        self.DM_traj = np.copy(self.DM_x0)
        self.stars_traj = np.copy(self.stars_x0)

        M_screen = NFW_M_enc(r_screen, self.MW_rho_NFW, self.MW_r_NFW)

        sat_r = np.linalg.norm(self.sat_x)
        if sat_r > r_screen:
            sat_M_enc = NFW_M_enc(sat_r, self.MW_rho_NFW, self.MW_r_NFW)
            sat_MG = 1+2*beta**2*(1-M_screen/sat_M_enc)
        else:
            sat_MG = 1
        DM_r = np.linalg.norm(self.DM_x, axis=-1)
        DM_MG = np.ones_like(DM_r)
        DM_M_enc = NFW_M_enc(DM_r, self.MW_rho_NFW, self.MW_r_NFW)
        inds = np.where(DM_r > r_screen)[0]
        DM_MG[inds] = 1+2*beta**2*(1-M_screen/DM_M_enc[inds])
        DM_MG = DM_MG[:, None]

        # desynchronise velocities for leapfrog integration
        sat_acc = sat_MG*NFW_acc(self.sat_x, np.zeros(3,), rho_0, r_s)
        sat_v_half = self.sat_v - 0.5*dt*sat_acc
        DM_acc = DM_MG*NFW_acc(self.DM_x, np.zeros(3,), rho_0, r_s)
        DM_acc += (1+2*beta**2)*hernquist_acc(self.DM_x, self.sat_x, M, a)
        DM_v_half = self.DM_v - 0.5*dt*DM_acc
        stars_acc = NFW_acc(self.stars_x, np.zeros(3,), rho_0, r_s)
        stars_acc += hernquist_acc(self.stars_x, self.sat_x, M, a)
        stars_v_half = self.stars_v - 0.5*dt*stars_acc

        # main loop
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            sat_r = np.linalg.norm(self.sat_x)
            if sat_r > r_screen:
                sat_M_enc = NFW_M_enc(sat_r, self.MW_rho_NFW, self.MW_r_NFW)
                sat_MG = 1+2*beta**2*(1-M_screen/sat_M_enc)
            else:
                sat_MG = 1
            DM_r = np.linalg.norm(self.DM_x, axis=-1)
            DM_MG = np.ones_like(DM_r)
            DM_M_enc = NFW_M_enc(DM_r, self.MW_rho_NFW, self.MW_r_NFW)
            inds = np.where(DM_r > r_screen)[0]
            DM_MG[inds] = 1+2*beta**2*(1-M_screen/DM_M_enc[inds])
            DM_MG = DM_MG[:, None]
            beta_eff = np.zeros_like(DM_r)
            if sat_r > r_screen:
                inds = np.where(DM_r > r_screen)[0]
                beta_eff[inds] = beta
            fac = (1+2*beta_eff[:, None]**2)

            # calculate accelerations
            sat_acc = sat_MG*NFW_acc(self.sat_x, np.zeros(3,), rho_0, r_s)
            DM_acc = DM_MG*NFW_acc(self.DM_x, np.zeros(3,), rho_0, r_s)
            DM_acc += fac*hernquist_acc(self.DM_x, self.sat_x, M, a)
            stars_acc = NFW_acc(self.stars_x, np.zeros(3,), rho_0, r_s)
            stars_acc += hernquist_acc(self.stars_x, self.sat_x, M, a)

            sat_v_half = sat_v_half + sat_acc*dt
            DM_v_half = DM_v_half + DM_acc*dt
            stars_v_half = stars_v_half + stars_acc*dt

            self.sat_x = self.sat_x + sat_v_half*dt
            self.DM_x = self.DM_x + DM_v_half*dt
            self.stars_x = self.stars_x + stars_v_half*dt

            if (i+1) % frame_interval == 0:
                self.sat_traj = np.vstack((self.sat_traj, self.sat_x))
                self.DM_traj = np.dstack((self.DM_traj, self.DM_x))
                self.stars_traj = np.dstack((self.stars_traj, self.stars_x))

        self.sat_v = sat_v_half - 0.5*dt*sat_acc
        self.DM_v = DM_v_half - 0.5*dt*DM_acc
        self.stars_v = stars_v_half - 0.5*dt*stars_acc
        return

