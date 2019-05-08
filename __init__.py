#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: __init__ file of gravstream package, containing main 'Simulation'
object.
"""
import numpy as _np
from .constants import kpc, Mpc, M_sun
from .sample_hernquist import sample_particles as _sample
from .potentials import hernquist_acceleration as _hacc
from .potentials import hernquist_density as _hrho
from .potentials import NFW_density as _NFWrho
from .potentials import miyamoto_density as _discrho
from .potentials import MW_acceleration as _MWacc
from .util import print_progress
from .MG_solver import Grid2D
from .animate import movie


class Simulation:
    """
    Object that sets up and runs simulations of satellite and tracer particles
    in Milky Way potential, with optional fifth force.

    Parameters
    ----------
    sat_x0 : 1D array-like, shape (3,)
    sat_v0 : 1D array-like, shape (3,)
    sat_r : float
    sat_M : float
    tracers : bool
    N_DM : int
    N_stars : int

    Attributes
    ----------


    Methods
    -------


    Examples
    --------
    """
    def __init__(self, sat_x0=[60*kpc, 0, 0], sat_v0=[0, 1e+5, 0],
                 sat_r=0.5*kpc, sat_M=5e+8*M_sun, tracers=True,
                 N_DM=5000, N_stars=5000):
        """
        Initialise an instance of 'Simulation' class. See class docstring for
        more information.
        """
        self.sat_r = sat_r
        self.sat_M = sat_M
        self.sat_x0 = _np.array(sat_x0, dtype=_np.float64)
        self.sat_v0 = _np.array(sat_v0, dtype=_np.float64)
        self.sat_x = _np.copy(self.sat_x0)
        self.sat_v = _np.copy(self.sat_v0)

        self.tracers = tracers
        if self.tracers:
            self.add_tracers(N_DM=N_DM, N_stars=N_stars)
            self.relax_tracers()

    def add_tracers(self, N_DM, N_stars):
        """
        Add tracer particles, sampled from a (truncated) Hernquist profile
        around the central satellite particle. Sample positions and velocities
        using equilibrium phase-space distribution function from Binney and
        Tremaine.

        NOTE: Total number of tracer particles N_DM+N_stars needs to be a
        multiple of 50 for sampling to work.

        Parameters
        ----------
        N_DM : int
            Number of dark matter tracer particles.
        N_stars : int
            Number of stellar tracer particles
        """

        if not self.tracers:
            self.tracers = True

        # total tracer number needs to be multiple of 50 for sampling to work
        N_tracers = N_DM + N_stars
        if N_tracers % 50 != 0:
            raise ValueError("N_tracers should be multiple of 50")
        self.N_DM = N_DM
        self.N_stars = N_stars
        self.N_tracers = N_tracers

        # sample positions and velocities
        x, v = _sample(N=N_tracers, M=self.sat_M, a=self.sat_r)

        # from sampled tracer particles, randomly choose which are dark matter
        DM_inds = _np.random.choice(_np.arange(N_tracers), N_DM, replace=False)
        DM_mask = _np.zeros(N_tracers, dtype=bool)
        DM_mask[DM_inds] = True
        DM_x = x[DM_mask]
        DM_v = v[DM_mask]
        stars_x = x[~DM_mask]
        stars_v = v[~DM_mask]

        # stored sampled positions and velocities
        self.DM_x0 = DM_x + self.sat_x0
        self.DM_v0 = DM_v + self.sat_v0
        self.DM_x = _np.copy(self.DM_x0)
        self.DM_v = _np.copy(self.DM_v0)
        self.stars_x0 = stars_x + self.sat_x0
        self.stars_v0 = stars_v + self.sat_v0
        self.stars_x = _np.copy(self.stars_x0)
        self.stars_v = _np.copy(self.stars_v0)

        return

    def relax_tracers(self, t_max=1e+16, dt=1e+11):
        """
        Evolve tracers in satellite potential for a while (t_max), in the
        absence of the external MW potential and any fifth forces.

        Orbit integrator uses a 'leapfrog' scheme.

        Parameters
        ----------
        t_max : float
            Total relaxation time. Default is 1e+17 seconds, around 3Gyr.
            UNITS: seconds
        dt : float
            Size of timesteps. Note: Timesteps larger than 1e+12 seconds might
            have convergence issues.
        """

        assert self.tracers

        x0 = self.sat_x0
        M = self.sat_M
        a = self.sat_r
        N_iter = int(t_max/dt)

        # subtract satellite bulk velocity from tracers
        self.DM_v = self.DM_v - self.sat_v0
        self.stars_v = self.stars_v - self.sat_v0

        # calculate initial accelerations, then desynchronise velocities
        # for leapfrog integration
        DM_acc = _hacc(self.DM_x-x0, M, a)
        DM_v_half = self.DM_v - 0.5*dt*DM_acc
        stars_acc = _hacc(self.stars_x-x0, M, a)
        stars_v_half = self.stars_v - 0.5*dt*stars_acc

        # main loop
        print("Relaxing tracer particles...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            # calculate accelerations
            DM_acc = _hacc(self.DM_x-x0, M, a)
            stars_acc = _hacc(self.stars_x-x0, M, a)

            # update velocities
            DM_v_half = DM_v_half + DM_acc*dt
            stars_v_half = stars_v_half + stars_acc*dt

            # update positions
            self.DM_x = self.DM_x + DM_v_half*dt
            self.stars_x = self.stars_x + stars_v_half*dt

        # resynchronise velocities
        self.DM_v = DM_v_half - 0.5*dt*DM_acc
        self.stars_v = stars_v_half - 0.5*dt*stars_acc

        # restore satellite bulk velocity
        self.DM_v = self.DM_v + self.sat_v0
        self.stars_v = self.stars_v + self.sat_v0

        return

    def run(self, modgrav=False, fR0=None, beta=_np.sqrt(1/6), sat_r_screen=0,
            t_max=1e+17, dt=1e+11, N_frames=1000):
        """
        Run simulation.

        Parameters
        ----------
        modgrav : bool
            Whether the dark matter particles feel an additional fifth force.
        fR0 : float
            Cosmic background value of the scalar field. Should be negative,
            e.g. -1e-6 for the so-called 'F6' theory.
        t_max : float
            Total run time. Default is 1e+17 seconds, around 3 Gyr.
            UNITS: seconds
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

        # N_iter is number of timesteps, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        N_iter = int(t_max/dt)
        if N_iter % N_frames != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        frame_interval = int(N_iter/N_frames)

        # *_tr are the arrays that store the saved particle trajectories
        self.sat_tr = _np.copy(self.sat_x)
        if self.tracers:
            self.DM_tr = _np.copy(self.DM_x)
            self.stars_tr = _np.copy(self.stars_x)

        # if adding 5th force, then calculate scalar field map
        if modgrav:
            # set up scalar field grid
            self.grid = Grid2D(ngrid=200, nth=101, rmax=10*Mpc)
            self.grid.set_cosmology(h=0.7, omega_m=0.3)

            # add MW density to grid
            self.grid.drho = _np.zeros(self.grid.grid_shape)
            for i in range(self.grid.ngrid):
                for j in range(self.grid.nth):
                    x = self.grid.r[i]*_np.sin(self.grid.th[j])
                    y = 0
                    z = self.grid.r[i]*_np.cos(self.grid.th[j])
                    pos = _np.array([x, y, z])
                    rho = _hrho(pos) + _NFWrho(pos) + _discrho(pos)
                    self.grid.drho[i, j] = rho

            # solve scalar field EOMs
            self.grid.iter_solve(niter=1000000, F0=fR0, verbose=True)
            beta_fac = (2*beta**2)/(1/3)

        # calculate initial accelerations, then desynchronise velocities
        # for leapfrog integration
        sat_acc = _MWacc(self.sat_x)
        if modgrav:
                sat_acc += beta_fac*self.grid.accel(self.sat_x)
        sat_v_half = self.sat_v - 0.5*dt*sat_acc
        if self.tracers:
            DM_acc = _MWacc(self.DM_x)
            DM_acc += _hacc(self.DM_x-self.sat_x, sat_M, sat_r)
            if modgrav:
                DM_acc += 2*beta**2*_hacc(self.DM_x-self.sat_x, sat_M, sat_r)
                DM_acc += beta_fac*self.grid.accel(self.DM_x)
            stars_acc = _MWacc(self.stars_x)
            stars_acc += _hacc(self.stars_x-self.sat_x, sat_M, sat_r)
            DM_v_half = self.DM_v - 0.5*dt*DM_acc
            stars_v_half = self.stars_v - 0.5*dt*stars_acc

        # main loop
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            # calculate accelerations
            sat_acc = _MWacc(self.sat_x)
            if modgrav:
                sat_acc += beta_fac*self.grid.accel(self.sat_x)
            if self.tracers:
                DM_acc = _MWacc(self.DM_x)
                DM_acc += _hacc(self.DM_x-self.sat_x, sat_M, sat_r)
                if modgrav:
                    DM_acc += 2*beta**2*_hacc(self.DM_x-self.sat_x, sat_M, sat_r)
                    DM_acc += beta_fac*self.grid.accel(self.DM_x)
                stars_acc = _MWacc(self.stars_x)
                stars_acc += _hacc(self.stars_x-self.sat_x, sat_M, sat_r)

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
                self.sat_tr = _np.vstack((self.sat_tr, self.sat_x))
                if self.tracers:
                    self.DM_tr = _np.dstack((self.DM_tr, self.DM_x))
                    self.stars_tr = _np.dstack((self.stars_tr, self.stars_x))

        # resynchronise velocities
        self.sat_v = sat_v_half - 0.5*dt*sat_acc
        if self.tracers:
            self.DM_v = DM_v_half - 0.5*dt*DM_acc
            self.stars_v = stars_v_half - 0.5*dt*stars_acc
        return


__all__ = ['Simulation', 'movie']
