#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description: __init__ file of gravstream package, containing main 'Simulation'
object.

TO-DO LIST
----------

- adaptive timesteps
- snapshots of position and velocity
- identify when particles are disrupted
- identify leading vs. trailing stream
- add support for no halo etc.
"""
import numpy as _np
from .constants import pc, kpc, M_sun, pi
from .sample_hernquist import sample_particles as _sample
from .util import print_progress


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
    def __init__(self, halo='NFW', halo_pars='default',
                 disc='miyamoto', disc_pars='default',
                 bulge='hernquist', bulge_pars='default',
                 sat_x0=[60*kpc, 0, 0], sat_v0=[0, 1e+5, 0],
                 sat_radius=0.5*kpc, sat_mass=5e+8*M_sun,
                 modgrav=False, beta=None, MW_r_screen=None, sat_r_screen=None,
                 tracers=True, N1=5000, N2=5000):
        """
        Initialise an instance of 'Simulation' class. See class docstring for
        more information.
        """

        self._setup_MW(halo=halo, halo_pars=halo_pars,
                       disc=disc, disc_pars=disc_pars,
                       bulge=bulge, bulge_pars=bulge_pars)

        self._setup_satellite(sat_x0=sat_x0, sat_v0=sat_v0,
                              sat_radius=sat_radius, sat_mass=sat_mass,
                              tracers=tracers, N1=N1, N2=N2)

        self._setup_modgrav(modgrav=modgrav, beta=beta,
                            MW_r_screen=MW_r_screen, sat_r_screen=sat_r_screen)

        return

    def _setup_MW(self, halo, halo_pars, disc, disc_pars, bulge, bulge_pars):

        # read halo type and get default parameters if necessary
        if halo == 'trilog':
            self.halo = halo
            from .potentials.trilog import acceleration as halo_acc
            from .potentials.trilog import mass_enc as halo_mass
            if halo_pars == 'default':
                # default parameters from LM10
                self.halo_pars = {'v_halo': 121900,
                                  'q1': 1.38,
                                  'qz': 1.36,
                                  'phi': 97*pi/180,
                                  'r_halo': 12*kpc}
            else:
                assert False, "Not supported yet"
        elif halo == 'NFW':
            self.halo = halo
            from .potentials.NFW import acceleration as halo_acc
            from .potentials.NFW import mass_enc as halo_mass
            if halo_pars == 'default':
                self.halo_pars = {'rho_0': 0.0025*M_sun/pc**3,
                                  'r_s': 27*kpc}
            else:
                assert False, "Not supported yet"
        elif halo is None:
            assert False, "Not supported yet"
        else:
            raise ValueError("Unrecognised halo type")

        # read disc type and get default parameters if necessary
        if disc == 'miyamoto':
            self.disc = disc
            from .potentials.miyamoto import acceleration as disc_acc
            from .potentials.miyamoto import M_enc_grid
            if disc_pars == 'default':
                # default parameters from LM10
                self.disc_pars = {'M_disc': 1e+11*M_sun,
                                  'a': 6.5*kpc,
                                  'b': 0.26*kpc}
            else:
                assert False, "Not supported yet"

            # don't have analytic disc mass; set up grid to integrate density
            # and then can interpolate disc mass
            N_r = 2000  # number of cells in radial dimension
            N_th = 1000  # number of cells in theta dimension
            r_max = 400*kpc
            r_edges, M_enc = M_enc_grid(N_r, N_th, r_max, **self.disc_pars)

            def disc_mass(pos, **kwargs):
                r = _np.linalg.norm(pos, axis=-1)
                mass = _np.interp(r, r_edges, M_enc)
                return mass

        elif disc is None:
            assert False, "Not supported yet"
        else:
            raise ValueError("Unrecognised disc type")

        # read bulge type and get default parameters if necessary
        if bulge == 'hernquist':
            self.bulge = bulge
            from .potentials.hernquist import acceleration as bulge_acc
            from .potentials.hernquist import mass_enc as bulge_mass
            if bulge_pars == 'default':
                # default values from LM10
                self.bulge_pars = {'M_hernquist': 3.4e+10*M_sun,
                                   'a': 0.7*kpc}
            else:
                assert False, "Not supported yet"
        elif bulge is None:
            assert False, "Not supported yet"
        else:
            raise ValueError("Unrecognised bulge type")

        # create function for acceleration due to milky way
        def MW_acc(pos):
            acc = (halo_acc(pos, **self.halo_pars) +
                   disc_acc(pos, **self.disc_pars) +
                   bulge_acc(pos, **self.bulge_pars))
            return acc
        self.MW_acc = MW_acc

        # create function for MW mass enclosed within given position
        def MW_M_enc(pos):
            M_enc = (halo_mass(pos, **self.halo_pars) +
                     disc_mass(pos, **self.disc_pars) +
                     bulge_mass(pos, **self.bulge_pars))
            return M_enc
        self.MW_M_enc = MW_M_enc

        return

    def _setup_satellite(self, sat_x0, sat_v0, sat_radius, sat_mass,
                         tracers, N1, N2):

        from .potentials.hernquist import acceleration as hacc
        from .potentials.hernquist import mass_enc as hmass

        # read satellite parameters
        self.sat_radius = sat_radius
        self.sat_mass = sat_mass
        self.sat_x0 = _np.array(sat_x0, dtype=_np.float64)
        self.sat_v0 = _np.array(sat_v0, dtype=_np.float64)
        self.p0_x = _np.copy(self.sat_x0)
        self.p0_v = _np.copy(self.sat_v0)

        # create function for acceleration due to satellite
        def sat_acc(pos):
            acc = hacc(pos-self.p0_x, self.sat_mass, self.sat_radius)
            return acc
        self.sat_acc = sat_acc

        # create function for satellite mass enclosed within given position
        def sat_M_enc(pos):
            M_enc = hmass(pos-self.p0_x, self.sat_mass, self.sat_radius)
            return M_enc
        self.sat_M_enc = sat_M_enc

        # add tracers if required
        self.tracers = tracers
        if self.tracers:
            self._add_tracers(N1=N1, N2=N2)
            self._relax_tracers()

        return

    def _setup_modgrav(self, modgrav, beta, MW_r_screen, sat_r_screen):

        self.modgrav = modgrav
        self.beta = beta
        self.MW_r_screen = MW_r_screen
        self.sat_r_screen = sat_r_screen

        if modgrav is False:

            def mg_acc_tracer(pos):
                acc = _np.zeros_like(pos)
                return acc

            def mg_acc_satellite(pos):
                acc = _np.zeros_like(pos)
                return acc

        else:

            # satellite mass within screening radius
            sat_M_screen = self.sat_M_enc(self.p0_x + _np.array([self.sat_r_screen, 0, 0]))

            # MW mass within screening radius
            MW_M_screen = self.MW_M_enc([self.MW_r_screen, 0, 0])

            # satellite mass fraction outside the screening radius;
            # multiplies coupling constant to scalar field
            sat_Q = 1 - sat_M_screen/self.sat_mass

            # fifth force on tracer particle
            def mg_acc_tracer(pos):

                r1 = _np.linalg.norm(pos, axis=-1)
                r2 = _np.linalg.norm(pos-self.p0_x, axis=-1)

                if pos.ndim == 1:
                    if r1 < MW_r_screen or r2 < sat_r_screen:
                        return _np.zeros_like(pos)

                    MW_mass_fac = 1 - MW_M_screen/self.MW_M_enc(pos)
                    sat_mass_fac = 1 - sat_M_screen/self.sat_M_enc(pos)
                    acc = 2*beta**2*MW_mass_fac*self.MW_acc(pos)
                    acc += 2*beta**2*sat_mass_fac*self.sat_acc(pos)
                    return acc

                inds = _np.where((r1 > MW_r_screen) & (r2 > sat_r_screen))
                acc = _np.zeros_like(pos)

                MW_mass_fac = 1 - MW_M_screen/self.MW_M_enc(pos[inds])
                sat_mass_fac = 1 - sat_M_screen/self.sat_M_enc(pos[inds])

                acc[inds] = 2*beta**2*MW_mass_fac[:, None]*self.MW_acc(pos[inds])
                acc[inds] += 2*beta**2*sat_mass_fac[:, None]*self.sat_acc(pos[inds])

                return acc

            # fifth force on satellite
            def mg_acc_satellite(pos):
                r = _np.linalg.norm(pos, axis=-1)
                if r < self.MW_r_screen:
                    acc = _np.zeros_like(pos)
                else:
                    MW_mass_fac = 1 - MW_M_screen/self.MW_M_enc(pos)
                    acc = 2*beta**2*sat_Q*MW_mass_fac*self.MW_acc(pos)
                return acc

        self.mg_acc_tracer = mg_acc_tracer
        self.mg_acc_satellite = mg_acc_satellite
        return

    def _add_tracers(self, N1, N2):
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
        N_tracers = N1 + N2
        if N_tracers % 50 != 0:
            raise ValueError("N_tracers should be multiple of 50")
        self.N1 = N1
        self.N2 = N2
        self.N_tracers = N_tracers

        # sample positions and velocities
        x, v = _sample(N=N_tracers, M=self.sat_mass, a=self.sat_radius)

        # from sampled tracer particles, randomly choose which are type 1
        inds = _np.random.choice(_np.arange(N_tracers), N1, replace=False)
        mask = _np.zeros(N_tracers, dtype=bool)
        mask[inds] = True
        p1_x = x[mask]
        p1_v = v[mask]
        p2_x = x[~mask]
        p2_v = v[~mask]

        # stored sampled positions and velocities
        self.p1_x0 = p1_x + self.sat_x0
        self.p1_v0 = p1_v + self.sat_v0
        self.p1_x = _np.copy(self.p1_x0)
        self.p1_v = _np.copy(self.p1_v0)
        self.p2_x0 = p2_x + self.sat_x0
        self.p2_v0 = p2_v + self.sat_v0
        self.p2_x = _np.copy(self.p2_x0)
        self.p2_v = _np.copy(self.p2_v0)

        return

    def _relax_tracers(self, t_max=1e+16, dt=5e+11):
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
        N_iter = int(t_max/dt)

        # subtract satellite bulk velocity from tracers
        self.p1_v = self.p1_v - self.sat_v0
        self.p2_v = self.p2_v - self.sat_v0

        # calculate initial accelerations, then desynchronise velocities
        # for leapfrog integration
        p1_acc = self.sat_acc(self.p1_x)
        p1_v_half = self.p1_v - 0.5*dt*p1_acc
        p2_acc = self.sat_acc(self.p2_x)
        p2_v_half = self.p2_v - 0.5*dt*p2_acc

        # main loop
        print("Relaxing tracer particles...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

            # calculate accelerations
            p1_acc = self.sat_acc(self.p1_x)
            p2_acc = self.sat_acc(self.p2_x)

            # update velocities
            p1_v_half = p1_v_half + p1_acc*dt
            p2_v_half = p2_v_half + p2_acc*dt

            # update positions
            self.p1_x = self.p1_x + p1_v_half*dt
            self.p2_x = self.p2_x + p2_v_half*dt

        # resynchronise velocities
        self.p1_v = p1_v_half - 0.5*dt*p1_acc
        self.p2_v = p2_v_half - 0.5*dt*p2_acc

        # restore satellite bulk velocity
        self.p1_v = self.p1_v + self.sat_v0
        self.p2_v = self.p2_v + self.sat_v0

        return

    def run(self, t_max=1e+17, dt=5e+11, N_frames=1000, mass_loss=False):
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

        self.t_max = t_max
        self.dt = dt

        # N_iter is number of timesteps, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        N_iter = int(t_max/dt)
        if N_iter % N_frames != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        frame_interval = int(N_iter/N_frames)

        # *_tr are the arrays that store the saved particle trajectories
        self.p0_trajectory = _np.hstack((self.p0_x, self.p0_v))[None, :]
        if self.tracers:
            self.p1_trajectory = _np.hstack((self.p1_x, self.p1_v))[None, :]
            self.p2_trajectory = _np.hstack((self.p2_x, self.p2_v))[None, :]

# =============================================================================
#         # if adding 5th force, then calculate scalar field map
#         if modgrav:
#             # set up scalar field grid
#             self.grid = Grid2D(ngrid=200, nth=101, rmax=10*Mpc)
#             self.grid.set_cosmology(h=0.7, omega_m=0.3)
#
#             # add MW density to grid
#             self.grid.drho = _np.zeros(self.grid.grid_shape)
#             for i in range(self.grid.ngrid):
#                 for j in range(self.grid.nth):
#                     x = self.grid.r[i]*_np.sin(self.grid.th[j])
#                     y = 0
#                     z = self.grid.r[i]*_np.cos(self.grid.th[j])
#                     pos = _np.array([x, y, z])
#                     rho = _hrho(pos) + _NFWrho(pos) + _discrho(pos)
#                     self.grid.drho[i, j] = rho
#
#             # solve scalar field EOMs
#             self.grid.iter_solve(niter=1000000, F0=fR0, verbose=True)
#             beta_fac = (2*beta**2)/(1/3)
# =============================================================================
# =============================================================================
# 
#         if mass_loss:
#             assert self.tracers
# 
#             # calculate number of particles within 10*sat_r initially
#             DM_r = _np.linalg.norm((self.DM_x - self.sat_x), axis=-1)
#             stars_r = _np.linalg.norm((self.stars_x - self.sat_x), axis=-1)
#             all_r = _np.hstack((DM_r, stars_r))
#             N_bound_init = _np.where(all_r < 10*sat_r)[0].size
# 
#             self.sat_M_evo = sat_M
# 
# =============================================================================
        # calculate initial accelerations, then desynchronise velocities
        # for leapfrog integration
        p0_acc = self.MW_acc(self.p0_x)
        p0_acc += self.mg_acc_satellite(self.p0_x)
        p0_v_half = self.p0_v - 0.5*dt*p0_acc
        if self.tracers:
            p1_acc = self.MW_acc(self.p1_x)
            p1_acc += self.sat_acc(self.p1_x)
            p1_acc += self.mg_acc_tracer(self.p1_x)
            p2_acc = self.MW_acc(self.p2_x)
            p2_acc += self.sat_acc(self.p2_x)
            p1_v_half = self.p1_v - 0.5*dt*p1_acc
            p2_v_half = self.p2_v - 0.5*dt*p2_acc

        # main loop
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

# =============================================================================
#             if mass_loss:
#                 assert self.tracers
#
#                 # calculate number of particles within 10*sat_r initially
#                 DM_r = _np.linalg.norm((self.DM_x - self.sat_x), axis=-1)
#                 stars_r = _np.linalg.norm((self.stars_x - self.sat_x), axis=-1)
#                 all_r = _np.hstack((DM_r, stars_r))
#                 N_bound = _np.where(all_r < 10*sat_r)[0].size
#
#                 # reduce mass correspondingly
#                 sat_M = (N_bound/N_bound_init)*self.sat_M
# =============================================================================

            # calculate accelerations
            p0_acc = self.MW_acc(self.p0_x)
            p0_acc += self.mg_acc_satellite(self.p0_x)
            if self.tracers:
                p1_acc = self.MW_acc(self.p1_x)
                p1_acc += self.sat_acc(self.p1_x)
                p1_acc += self.mg_acc_tracer(self.p1_x)
                p2_acc = self.MW_acc(self.p2_x)
                p2_acc += self.sat_acc(self.p2_x)

            # timestep
            p0_v_half = p0_v_half + p0_acc*dt
            self.p0_x = self.p0_x + p0_v_half*dt
            if self.tracers:
                p1_v_half = p1_v_half + p1_acc*dt
                p2_v_half = p2_v_half + p2_acc*dt
                self.p1_x = self.p1_x + p1_v_half*dt
                self.p2_x = self.p2_x + p2_v_half*dt

            # snapshot
            if (i+1) % frame_interval == 0:
                p0_v = p0_v_half - 0.5*p0_acc*dt
                snap = _np.hstack((self.p0_x, p0_v))[None, :]
                self.p0_trajectory = _np.vstack((self.p0_trajectory, snap))
                if self.tracers:
                    p1_v = p1_v_half - 0.5*p1_acc*dt
                    snap = _np.hstack((self.p1_x, p1_v))[None, :]
                    self.p1_trajectory = _np.vstack((self.p1_trajectory, snap))
                    p2_v = p2_v_half - 0.5*p2_acc*dt
                    snap = _np.hstack((self.p2_x, p2_v))[None, :]
                    self.p2_trajectory = _np.vstack((self.p2_trajectory, snap))
                    #if mass_loss:
                    #    self.sat_M_evo = _np.append(self.sat_M_evo, sat_M)

        # resynchronise velocities
        self.p0_v = p0_v_half - 0.5*dt*p0_acc
        if self.tracers:
            self.p1_v = p1_v_half - 0.5*dt*p1_acc
            self.p2_v = p2_v_half - 0.5*dt*p2_acc
        return
