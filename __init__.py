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
import numpy as np
import h5py
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
                # checking halo parameters are understood
                for k in halo_pars.keys():
                    assert k in ['rho_0', 'r_s']
                self.halo_pars = halo_pars
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
                r = np.linalg.norm(pos, axis=-1)
                mass = np.interp(r, r_edges, M_enc)
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
        from .potentials.hernquist import potential as hphi
        from .potentials.hernquist import mass_enc as hmass

        # read satellite parameters
        self.sat_radius = sat_radius
        self.sat_mass = sat_mass
        self.sat_x0 = np.array(sat_x0, dtype=np.float64)
        self.sat_v0 = np.array(sat_v0, dtype=np.float64)
        self.p0_x = np.copy(self.sat_x0)
        self.p0_v = np.copy(self.sat_v0)

        # create function for acceleration due to satellite
        def sat_acc(pos):
            acc = hacc(pos-self.p0_x, self.sat_mass, self.sat_radius)
            return acc
        self.sat_acc = sat_acc

        # create function for potential due to satellite
        def sat_phi(pos):
            phi = hphi(pos-self.p0_x, self.sat_mass, self.sat_radius)
            return phi
        self.sat_phi = sat_phi

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
                acc = np.zeros_like(pos)
                return acc

            def mg_acc_satellite(pos):
                acc = np.zeros_like(pos)
                return acc

        else:

            # satellite mass within screening radius
            sat_M_screen = self.sat_M_enc(self.p0_x + np.array([self.sat_r_screen, 0, 0]))

            # MW mass within screening radius
            MW_M_screen = self.MW_M_enc([self.MW_r_screen, 0, 0])

            # satellite mass fraction outside the screening radius;
            # multiplies coupling constant to scalar field
            sat_Q = 1 - sat_M_screen/self.sat_mass

            # fifth force on tracer particle
            def mg_acc_tracer(pos):

                r1 = np.linalg.norm(pos, axis=-1)
                r2 = np.linalg.norm(pos-self.p0_x, axis=-1)

                if pos.ndim == 1:
                    if r1 < MW_r_screen or r2 < sat_r_screen:
                        return np.zeros_like(pos)

                    MW_mass_fac = 1 - MW_M_screen/self.MW_M_enc(pos)
                    sat_mass_fac = 1 - sat_M_screen/self.sat_M_enc(pos)
                    acc = 2*beta**2*MW_mass_fac*self.MW_acc(pos)
                    acc += 2*beta**2*sat_mass_fac*self.sat_acc(pos)
                    return acc

                inds = np.where((r1 > MW_r_screen) & (r2 > sat_r_screen))
                acc = np.zeros_like(pos)

                MW_fac = 1 - MW_M_screen/self.MW_M_enc(pos[inds])
                sat_fac = 1 - sat_M_screen/self.sat_M_enc(pos[inds])

                acc[inds] = 2*beta**2*MW_fac[:, None]*self.MW_acc(pos[inds])
                acc[inds] += 2*beta**2*sat_fac[:, None]*self.sat_acc(pos[inds])

                return acc

            # fifth force on satellite
            def mg_acc_satellite(pos):
                r = np.linalg.norm(pos, axis=-1)
                if r < self.MW_r_screen:
                    acc = np.zeros_like(pos)
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

        # sample positions and velocities
        x, v = _sample(N=N_tracers, M=self.sat_mass, a=self.sat_radius)

        # from sampled tracer particles, randomly choose which are type 1
        inds = np.random.choice(np.arange(N_tracers), N1, replace=False)
        mask = np.zeros(N_tracers, dtype=bool)
        mask[inds] = True
        p1_x = x[mask]
        p1_v = v[mask]
        p2_x = x[~mask]
        p2_v = v[~mask]

        # stored sampled positions and velocities
        self.p1_x0 = p1_x + self.sat_x0
        self.p1_v0 = p1_v + self.sat_v0
        self.p1_x = np.copy(self.p1_x0)
        self.p1_v = np.copy(self.p1_v0)
        self.p2_x0 = p2_x + self.sat_x0
        self.p2_v0 = p2_v + self.sat_v0
        self.p2_x = np.copy(self.p2_x0)
        self.p2_v = np.copy(self.p2_v0)

        return

    def _relax_tracers(self, t_max=1e+16, dt=1e+11):
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

    def run(self, t_max=1e+17, dt=5e+11, N_snapshots=1000, mass_loss=False):
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

        self.times = np.array([0])
        self.t_max = t_max
        self.dt = dt

        # N_iter is number of timesteps, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        self.N_snapshots = N_snapshots
        N_iter = int(t_max/dt)
        if N_iter % N_snapshots != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        snap_interval = int(N_iter/N_snapshots)

        # create arrays in which snapshot pos and vel are stored
        p0_positions = np.copy(self.p0_x)
        p0_velocities = np.copy(self.p0_v)
        if self.tracers:
            p1_positions = np.copy(self.p1_x)[None, :]
            p2_positions = np.copy(self.p2_x)[None, :]
            p1_velocities = np.copy(self.p1_v)[None, :]
            p2_velocities = np.copy(self.p2_v)[None, :]

            self.p1_disrupted = np.zeros((self.N1), dtype=bool)
            self.p2_disrupted = np.zeros((self.N2), dtype=bool)

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
        t = 0
        print("Main loop...")
        for i in range(N_iter):

            print_progress(i, N_iter, interval=N_iter//50)

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
            t += self.dt
            p0_v_half = p0_v_half + p0_acc*dt
            self.p0_x = self.p0_x + p0_v_half*dt
            if self.tracers:
                p1_v_half = p1_v_half + p1_acc*dt
                p2_v_half = p2_v_half + p2_acc*dt
                self.p1_x = self.p1_x + p1_v_half*dt
                self.p2_x = self.p2_x + p2_v_half*dt

            # snapshot
            if (i+1) % snap_interval == 0:

                self.times = np.append(self.times, t)

                # resynchronised satellite velocity
                p0_v = p0_v_half - 0.5*p0_acc*dt
                p0_positions = np.vstack((p0_positions, self.p0_x))
                p0_velocities = np.vstack((p0_velocities, p0_v))

                if self.tracers:
                    # resynchronised tracer velocities
                    p1_v = p1_v_half - 0.5*p1_acc*dt
                    p2_v = p2_v_half - 0.5*p2_acc*dt

                    # store positions and velocities
                    p1_positions = np.vstack((p1_positions, self.p1_x[None, :]))
                    p1_velocities = np.vstack((p1_velocities, p1_v[None, :]))
                    p2_positions = np.vstack((p2_positions, self.p2_x[None, :]))
                    p2_velocities = np.vstack((p2_velocities, p2_v[None, :]))

                    # masks indicating which particles are disrupted
                    dv1 = p1_v - p0_v
                    dv2 = p2_v - p0_v
                    K1 = 0.5*(dv1[:, 0]**2 + dv1[:, 1]**2 + dv1[:, 2]**2)
                    K2 = 0.5*(dv2[:, 0]**2 + dv2[:, 1]**2 + dv2[:, 2]**2)
                    U1 = self.sat_phi(self.p1_x)
                    U2 = self.sat_phi(self.p2_x)
                    E1 = K1 + U1
                    E2 = K2 + U2
                    mask1 = np.zeros((self.N1), dtype=bool)
                    mask2 = np.zeros((self.N2), dtype=bool)
                    mask1[np.where(E1 > 0)] = True
                    mask2[np.where(E2 > 0)] = True
                    self.p1_disrupted = np.vstack((self.p1_disrupted, mask1))
                    self.p2_disrupted = np.vstack((self.p2_disrupted, mask2))

        self.p0_positions = p0_positions
        self.p0_velocities = p0_velocities
        if self.tracers:
            self.p1_positions = p1_positions
            self.p1_velocities = p1_velocities
            self.p2_positions = p2_positions
            self.p2_velocities = p2_velocities

        # resynchronise velocities
        self.p0_v = p0_v_half - 0.5*dt*p0_acc
        if self.tracers:
            self.p1_v = p1_v_half - 0.5*dt*p1_acc
            self.p2_v = p2_v_half - 0.5*dt*p2_acc

        # store arrays of particle disruption times
        if self.tracers:
            self.p1_disruption_time = -np.ones((self.N1), dtype=int)
            self.p2_disruption_time = -np.ones((self.N2), dtype=int)
            for i in range(self.N1):
                mask = self.p1_disrupted[:, i]
                if mask.any():
                    self.p1_disruption_time[i] = np.where(mask)[0][0]
            for i in range(self.N2):
                mask = self.p2_disrupted[:, i]
                if mask.any():
                    self.p2_disruption_time[i] = np.where(mask)[0][0]

        # store arrays of orbital phase
        zp = np.cross(self.sat_x0, self.sat_v0)
        zp = zp/np.linalg.norm(zp)
        xp = self.sat_x0/np.linalg.norm(self.sat_x0)
        yp = np.cross(zp, xp)
        p0_xp = np.dot(self.p0_positions, xp)
        p0_yp = np.dot(self.p0_positions, yp)
        self.p0_phi = np.arctan2(p0_yp, p0_xp)
        dphi = np.append(0, np.diff(self.p0_phi))
        changes = np.where(dphi < -pi)[0]
        for i in range(changes.size):
            self.p0_phi[changes[i]:] += 2*pi

        if self.tracers:
            p1_xp = np.dot(self.p1_positions[..., :3], xp)
            p1_yp = np.dot(self.p1_positions[..., :3], yp)
            p2_xp = np.dot(self.p2_positions[..., :3], xp)
            p2_yp = np.dot(self.p2_positions[..., :3], yp)

            self.p1_phi = np.arctan2(p1_yp, p1_xp)
            self.p2_phi = np.arctan2(p2_yp, p2_xp)
            dphi = np.vstack((np.zeros((1, self.N1)), np.diff(self.p1_phi, axis=0)))
            for j in range(self.N1):
                changes = np.where(dphi[:, j] < -pi)[0]
                for i in range(changes.size):
                    self.p1_phi[changes[i]:, j] += 2*pi
            dphi = np.vstack((np.zeros((1, self.N2)), np.diff(self.p2_phi, axis=0)))
            for j in range(self.N2):
                changes = np.where(dphi[:, j] < -pi)[0]
                for i in range(changes.size):
                    self.p2_phi[changes[i]:, j] += 2*pi

            # difference in orbital phase and leading stream mask
            p1_dphi = self.p1_phi - self.p0_phi[:, None]
            p2_dphi = self.p2_phi - self.p0_phi[:, None]
            self.p1_leading = np.zeros_like(p1_dphi, dtype=bool)
            self.p1_leading[np.where(p1_dphi > 0)] = True
            self.p2_leading = np.zeros_like(p2_dphi, dtype=bool)
            self.p2_leading[np.where(p2_dphi > 0)] = True

        return

    def save(self, filename):

        # check correct file ending
        if filename[-5:] == '.hdf5':
            f = h5py.File(filename, 'w')
        else:
            f = h5py.File(filename+".hdf5", 'w')

        # set up header group
        header = f.create_group("Header")
        header.attrs['HaloType'] = self.halo
        header.attrs['DiscType'] = self.disc
        header.attrs['BulgeType'] = self.bulge
        header.attrs['SatelliteMass'] = self.sat_mass
        header.attrs['SatelliteRadius'] = self.sat_radius
        header.attrs['ModGrav'] = self.modgrav
        if self.modgrav:
            header.attrs['Beta'] = self.beta
            header.attrs['MW_RScreen'] = self.MW_r_screen
            header.attrs['Satellite_RScreen'] = self.sat_r_screen
        header.attrs['Tracers'] = self.tracers
        if self.tracers:
            header.attrs['NPart_Type1'] = self.N1
            header.attrs['NPart_Type2'] = self.N2
        header.attrs['NSnapshots'] = self.N_snapshots
        header.attrs['TimeMax'] = self.t_max
        header.attrs['TimestepSize'] = self.dt

        # add subgroups to header containing parameters for halo, disc, bulge
        halo = f.create_group("Header/Halo")
        halo.attrs['HaloType'] = self.halo
        for k, v in self.halo_pars.items():
            halo.attrs[k] = v
        disc = f.create_group("Header/Disc")
        disc.attrs['DiscType'] = self.disc
        for k, v in self.disc_pars.items():
            disc.attrs[k] = v
        bulge = f.create_group("Header/Bulge")
        bulge.attrs['BulgeType'] = self.bulge
        for k, v in self.bulge_pars.items():
            bulge.attrs[k] = v

        # array of times
        f.create_dataset('SnapshotTimes', data=self.times)

        # satellite group and data
        sat_grp = f.create_group("Satellite")
        sat_grp.create_dataset("Position", data=self.p0_positions)
        sat_grp.create_dataset("Velocity", data=self.p0_velocities)
        sat_grp.create_dataset("OrbitalPhase", data=self.p0_phi)

        # groups and data for tracer particles
        if self.tracers:
            p1_grp = f.create_group("PartType1")
            p2_grp = f.create_group("PartType2")

            p1_grp.create_dataset("Position", data=self.p1_positions)
            p1_grp.create_dataset("Velocity", data=self.p1_velocities)
            p1_grp.create_dataset("OrbitalPhase", data=self.p1_phi)
            p1_grp.create_dataset("Disrupted", data=self.p1_disrupted)
            p1_grp.create_dataset("DisruptionTime", data=self.p1_disruption_time)
            p1_grp.create_dataset("LeadingStream", data=self.p1_leading)

            p2_grp.create_dataset("Position", data=self.p2_positions)
            p2_grp.create_dataset("Velocity", data=self.p2_velocities)
            p2_grp.create_dataset("OrbitalPhase", data=self.p2_phi)
            p2_grp.create_dataset("Disrupted", data=self.p2_disrupted)
            p2_grp.create_dataset("DisruptionTime", data=self.p2_disruption_time)
            p2_grp.create_dataset("LeadingStream", data=self.p2_leading)

        f.close()
        return
