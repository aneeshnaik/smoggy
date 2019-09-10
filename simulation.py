#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
import numpy as np
import h5py
from .constants import kpc, M_sun, pi
from .util import print_progress


class SatelliteSimulation:
    """
    Object that sets up and runs simulations of satellite and tracer particles
    in Milky Way potential, with optional fifth force.

    Parameters
    ----------
    sat_x0 : 1D array-like, shape (3,)
    sat_v0 : 1D array-like, shape (3,)
    sat_radius : float
    sat_mass : float
    halo : str
    halo_pars : dict
    disc : str
    disc_pars : dict
    bulge : str
    bulge_pars : dict
    modgrav : bool
    beta : float
    MW_r_screen : float
    sat_r_screen : float
    tracers : bool
    N1 : int
    N2 : int

    Attributes
    ----------


    Methods
    -------


    Examples
    --------
    """
    def __init__(self, sat_x0, sat_v0, sat_radius, sat_mass,
                 halo='NFW', halo_pars='default',
                 disc='miyamoto', disc_pars='default',
                 bulge='hernquist', bulge_pars='default',
                 modgrav=False, beta=None, MW_r_screen=None, sat_r_screen=None,
                 tracers=True, N1=5000, N2=5000):
        """
        Initialise an instance of class. See class docstring for more info.
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
        """
        Set up Milky Way profile. Main purpose of this function is to create
        two class methods: self.MW_acc and self.MW_M_enc, i.e. functions to
        calculate acceleration due to Milky Way at a given point and MW mass
        enclosed by a given radius.
        """
        # read halo type and get default parameters if necessary
        if halo == 'trilog':  # triaxial logarithmic halo
            self.halo = halo
            from .profiles.trilog import acceleration as halo_acc
            from .profiles.trilog import mass_enc as halo_mass
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
            from .profiles.NFW import acceleration as halo_acc
            from .profiles.NFW import mass_enc as halo_mass
            if halo_pars == 'default':
                self.halo_pars = {'M_vir': 1e+12*M_sun,
                                  'c_vir': 12}
            else:
                # checking halo parameters are understood
                for k in halo_pars.keys():
                    assert k in ['M_vir', 'c_vir']
                self.halo_pars = halo_pars
        elif halo is None:
            assert False, "Not supported yet"
        else:
            raise ValueError("Unrecognised halo type")

        # read disc type and get default parameters if necessary
        if disc == 'miyamoto':
            self.disc = disc
            from .profiles.miyamoto import acceleration as disc_acc
            from .profiles.miyamoto import M_enc_grid
            if disc_pars == 'default':
                # default parameters from LM10
                self.disc_pars = {'M_disc': 1e+11*M_sun,
                                  'a': 6.5*kpc,
                                  'b': 0.26*kpc}
            else:
                # checking disc parameters are understood
                for k in disc_pars.keys():
                    assert k in ['M_disc', 'a', 'b']
                self.disc_pars = disc_pars

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
            from .profiles.hernquist import acceleration as bulge_acc
            from .profiles.hernquist import mass_enc as bulge_mass
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
        """
        Set up satellite, i.e. create class methods for acceleration,
        potential, and enclosed mass of satellite. Also, if tracer particles
        are required, the function to initialise them is called here.
        """
        from .profiles.hernquist_truncated import acceleration as hacc
        from .profiles.hernquist_truncated import potential as hphi
        from .profiles.hernquist_truncated import mass_enc as hmass

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

            # MW mass within screening radius
            MW_M_screen = self.MW_M_enc([self.MW_r_screen, 0, 0])

            screen_pos = self.sat_x0 + np.array([self.sat_r_screen, 0, 0])
            sat_M_screen = self.sat_M_enc(screen_pos)

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

                else:

                    inds = np.where((r1 > MW_r_screen) & (r2 > sat_r_screen))
                    acc = np.zeros_like(pos)

                    Q1 = 1 - MW_M_screen/self.MW_M_enc(pos[inds])
                    Q2 = 1 - sat_M_screen/self.sat_M_enc(pos[inds])

                    acc[inds] = 2*beta**2*Q1[:, None]*self.MW_acc(pos[inds])
                    acc[inds] += 2*beta**2*Q2[:, None]*self.sat_acc(pos[inds])

                    return acc

            # 'scalar charge' of satellite, i.e. mass fraction outside
            # screening radius.
            if self.sat_r_screen > 10*self.sat_radius:
                sat_Q = 0
            else:
                xs = self.sat_r_screen/self.sat_radius
                sat_Q = 1 - (121*xs**2)/(100*(1+xs)**2)

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

        from .tracers import sample

        # total tracer number needs to be multiple of 50 for sampling to work
        N_tracers = N1 + N2
        if N_tracers % 50 != 0:
            raise ValueError("Total number of tracer particles should be"
                             "multiple of 50 for sampling to work")
        self.N1 = N1
        self.N2 = N2
        M = self.sat_mass
        a = self.sat_radius

        # sample 2N initial positions and velocities form B+T hernquist DF.
        # N.B. 121M/100 is mass of full (untruncated) Hernquist distribution
        x0, v0 = sample(2*N_tracers, M=121*M/100, a=a)

        # displace to satellite position (add bulk velocity later)
        x0 += self.sat_x0

        # integrate these tracers under satellite potential for a Gyr or so
        pos, vel = self._relax_tracers(x0, v0, t_max=1e+17)

        # downsample N of these, excluding those which are ever more than 10a
        # from satellite centre
        r = np.linalg.norm(pos-self.sat_x0, axis=-1)
        allowed = [(r[:, i] < 10*a).all() for i in range(2*N_tracers)]
        inds1 = np.where(np.array(allowed))[0]
        assert inds1.size > N_tracers

        inds2 = np.random.choice(inds1, N_tracers, replace=False)
        x1 = pos[-1, inds2]
        v1 = vel[-1, inds2]

        # from sampled tracer particles, randomly choose which are type 1
        inds = np.random.choice(np.arange(N_tracers), N1, replace=False)
        mask = np.zeros(N_tracers, dtype=bool)
        mask[inds] = True
        p1_x = x1[mask]
        p1_v = v1[mask]
        p2_x = x1[~mask]
        p2_v = v1[~mask]

        # stored sampled positions and velocities
        self.p1_x0 = p1_x
        self.p1_v0 = p1_v + self.sat_v0
        self.p1_x = np.copy(self.p1_x0)
        self.p1_v = np.copy(self.p1_v0)
        self.p2_x0 = p2_x
        self.p2_v0 = p2_v + self.sat_v0
        self.p2_x = np.copy(self.p2_x0)
        self.p2_v = np.copy(self.p2_v0)

        return

    def _relax_tracers(self, x0, v0, t_max):

        N_part = x0.shape[0]
        N_snapshots = 500
        positions = np.zeros((N_snapshots+1, N_part, 3))
        velocities = np.zeros((N_snapshots+1, N_part, 3))
        positions[0] = x0
        velocities[0] = v0

        dt = t_max / N_snapshots
        res = 1
        tol = 0.02
        while res > tol:

            x = np.copy(x0)
            v = np.copy(v0)

            N_iter = int(t_max/dt)

            assert N_iter % N_snapshots == 0
            snap_interval = int(N_iter/N_snapshots)
            snapcount = 1

            acc = self.sat_acc(x)
            v_half = v - 0.5*dt*acc

            # main loop
            print("Relaxing tracer particles... trying dt={0:.2e}".format(dt))
            for i in range(N_iter):

                print_progress(i, N_iter, interval=N_iter//50)

                # calculate accelerations
                acc = self.sat_acc(x)

                # update velocities
                v_half = v_half + acc*dt

                # update positions
                x = x + v_half*dt

                # snapshot
                if (i+1) % snap_interval == 0:
                    # resync velocities
                    v = v_half - 0.5*acc*dt

                    # store x and v
                    positions[snapcount] = x
                    velocities[snapcount] = v
                    snapcount += 1

            vi = velocities[0]
            vf = velocities[-1]
            xi = positions[0]
            xf = positions[-1]
            Ki = 0.5*(vi[:, 0]**2 + vi[:, 1]**2 + vi[:, 2]**2)
            Kf = 0.5*(vf[:, 0]**2 + vf[:, 1]**2 + vf[:, 2]**2)
            Ui = self.sat_phi(xi)
            Uf = self.sat_phi(xf)
            Ei = Ki + Ui
            Ef = Kf + Uf

            res = np.max(np.abs((Ef-Ei)/Ei))
            if res > tol:
                dt /= 2

        self.dt = dt
        return positions, velocities

    def run(self, t_max=1e+17, N_snapshots=500):
        """
        Run simulation.

        Parameters
        ----------
        t_max : float
            Total run time. Default is 1e+17 seconds, around 3 Gyr.
            UNITS: seconds
        N_snapshots : int
            Number of snapshots to store, EXCLUDING initial snapshot.
            Default is 500 (so 501 snapshots are saved overall).
        """

        self.times = np.array([0])
        self.t_max = t_max

        if self.tracers:
            dt = self.dt
        else:
            dt = 1e+12
            self.dt = dt

        # N_iter is number of timesteps, need to be integer multiple of
        # N_frames, so that a snapshot can be made at a regular interval
        self.N_snapshots = N_snapshots
        N_iter = int(t_max/dt)
        if N_iter % N_snapshots != 0:
            raise ValueError("Need N_iter to be multiple of N_frames")
        snap_interval = int(N_iter/N_snapshots)

        # create arrays in which outputs are stored
        snapcount = 0
        p0_positions = np.zeros((N_snapshots+1, 3))
        p0_positions[0] = self.p0_x
        p0_velocities = np.zeros((N_snapshots+1, 3))
        p0_velocities[0] = self.p0_v
        if self.tracers:
            p1_positions = np.zeros((N_snapshots+1, self.N1, 3))
            p1_positions[0] = self.p1_x
            p1_velocities = np.zeros((N_snapshots+1, self.N1, 3))
            p1_velocities[0] = self.p1_v
            p2_positions = np.zeros((N_snapshots+1, self.N2, 3))
            p2_positions[0] = self.p2_x
            p2_velocities = np.zeros((N_snapshots+1, self.N2, 3))
            p2_velocities[0] = self.p2_v
            p1_disrupted = np.zeros((N_snapshots+1, self.N1), dtype=bool)
            p2_disrupted = np.zeros((N_snapshots+1, self.N2), dtype=bool)

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

                snapcount += 1
                self.times = np.append(self.times, t)

                # resynchronise satellite velocity
                self.p0_v = p0_v_half - 0.5*p0_acc*dt

                # store satellite position and velocity
                p0_positions[snapcount] = self.p0_x
                p0_velocities[snapcount] = self.p0_v

                if self.tracers:
                    # resynchronised tracer velocities
                    self.p1_v = p1_v_half - 0.5*p1_acc*dt
                    self.p2_v = p2_v_half - 0.5*p2_acc*dt

                    # store positions and velocities
                    p1_positions[snapcount] = self.p1_x
                    p1_velocities[snapcount] = self.p1_v
                    p2_positions[snapcount] = self.p2_x
                    p2_velocities[snapcount] = self.p2_v

                    # masks indicating which particles are disrupted
                    dv1 = self.p1_v - self.p0_v
                    dv2 = self.p2_v - self.p0_v
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
                    p1_disrupted[snapcount] = mask1
                    p2_disrupted[snapcount] = mask2

        self.p0_positions = p0_positions
        self.p0_velocities = p0_velocities
        if self.tracers:
            self.p1_positions = p1_positions
            self.p1_velocities = p1_velocities
            self.p1_disrupted = p1_disrupted
            self.p2_positions = p2_positions
            self.p2_velocities = p2_velocities
            self.p2_disrupted = p2_disrupted

        # resynchronise velocities
        self.p0_v = p0_v_half - 0.5*dt*p0_acc
        if self.tracers:
            self.p1_v = p1_v_half - 0.5*dt*p1_acc
            self.p2_v = p2_v_half - 0.5*dt*p2_acc

        # calculate and store arrays of particle disruption times
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
        grp0 = f.create_group("Satellite")
        grp0.create_dataset("Position", data=self.p0_positions)
        grp0.create_dataset("Velocity", data=self.p0_velocities)

        # groups and data for tracer particles
        if self.tracers:
            grp1 = f.create_group("PartType1")
            grp2 = f.create_group("PartType2")

            grp1.create_dataset("Position", data=self.p1_positions)
            grp1.create_dataset("Velocity", data=self.p1_velocities)
            grp1.create_dataset("Disrupted", data=self.p1_disrupted)
            grp1.create_dataset("DisruptionTime", data=self.p1_disruption_time)

            grp2.create_dataset("Position", data=self.p2_positions)
            grp2.create_dataset("Velocity", data=self.p2_velocities)
            grp2.create_dataset("Disrupted", data=self.p2_disrupted)
            grp2.create_dataset("DisruptionTime", data=self.p2_disruption_time)

        f.close()
        return
