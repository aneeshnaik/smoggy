#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: July 2019
Author: A. P. Naik
Description: 'SmogOutput' class is main object of smoggy.analysis submodule.
Has convenient functions for plotting/analysis etc.
"""
from .constants import pi
import h5py
import numpy as np


class SmogOutput:
    """
    Object that loads and reads smoggy simulation output data.

    For usage examples, see smoggy README and template runscripts.

    Parameters
    ----------
    filename : str
        Name of saved smoggy simulation file. If '.hdf5' is not at the end then
        it is automatically appended.

    Attributes
    ----------
    ndiscs : int
        Number of disc components in galaxy.
    dpars : list of dicts, length ndiscs
        See smoggy.Simulation documentation for more info.
    nspheroids : int
        Number of spheroidal components in galaxy. Default is 2.
    spars : list of dicts, length nspheroids
        See smoggy.Simulation documentation for more info.
    sat_mass : float
        Total mass of (truncated Hernquist) satellite. UNITS: kilograms.
    sat_radius : float
        Scale radius of (truncated Hernquist) satellite. UNITS: metres.
    modgrav : bool, optional
        Whether the fifth force is switched on. Default: False.
    beta : float
        If 'modgrav' is True, then 'beta' is the (dimensionless) coupling
        strength of the fifth force. The fifth force (in the absence of
        screening) is 2*beta**2 * gravity.
    MW_r_screen : float
        If 'modgrav' is True, Milky Way screening radius. UNITS: metres.
    sat_r_screen : float
        If 'modgrav' is True, satellite screening radius. UNITS: metres.
    tracers : bool
        Whether tracer particles (i.e. stream particles) are present. If False,
        simulation is simply single satellite particle orbiting the Milky Way.
    N1 : int
        If 'tracers' is true, number of type 1 (i.e. dark matter) tracer
        particles. Type 1 particles couple to the fifth force.
    N2 : int
        If 'tracers' is true, number of type 2 (i.e. star) tracer
        particles. Type 2 particles do not couple to the fifth force.
    N_snapshots : int
        Number of snapshots to store, EXCLUDING initial snapshot, e.g. if
        N_snapshots is 500 then 501 snapshots are saved overall.
    t_max : float
        Total run time. UNITS: seconds
    dt : float
        Timestep size. UNITS: seconds.
    times : (N_snapshots+1,) array of floats
        Times of each snapshot. First is 0. UNITS: seconds.
    p0_positions : (N_snapshots+1, 3) array of floats
        Position of satellite at each time, in Galactocentric Cartesian
        coordinates. UNITS: metres.
    p0_velocities : (N_snapshots+1, 3) array of floats
        Velocity of satellite at each time, in Galactocentric Cartesian
        coordinates. UNITS: metres/second.
    p1_positions : (N_snapshots+1, N1, 3) array of floats
        Positions of type 1 (dark matter) particles at each time, in
        Galactocentric Cartesian coordinates. Type 1 particles couple to the
        fifth force. UNITS: metres.
    p1_velocities : (N_snapshots+1, N1, 3) array of floats
        Velocities of type 1 (dark matter) particles at each time, in
        Galactocentric Cartesian coordinates. Type 1 particles couple to the
        fifth force. UNITS: metres/second.
    p1_disrupted : (N_snapshots+1, N1) array of bools
        At each time, a flag to indicate whether each type 1 (dark matter)
        particle has become unbound from the satellite.
    p1_disruption_time : (N1,) array of ints
        For each type 1 particle, an integer index corresponding to the time at
        which it became unbound from the satellite. For example, if
        self.p1_disruption_time[345] is 31, then particle 345 become unbound at
        self.times[31].
    p1_longitudes : (N_snapshots+1, N1) array of floats
        At each time, the longitudes of all type 1 (dark matter) particles in
        the instantaneous orbital plane of the satellite. A positive (negative)
        value indicates that the particle is leading (trailing) the satellite.
        Note that multiple wraps are accounted for, e.g. a leading stream with
        two complete wraps will stretch from 0 to 4pi, while a trailing stream
        with 3 wraps will stretch from 0 to -6pi. UNITS: radians.
    p2_positions : (N_snapshots+1, N2, 3) array
        Positions of type 2 (star) particles at each time, in
        Galactocentric Cartesian coordinates. UNITS: metres.
    p2_velocities : (N_snapshots+1, N2, 3) array
        Velocities of type 2 (star) particles at each time, in Galactocentric
        Cartesian coordinates. UNITS: metres/second.
    p2_disrupted : (N_snapshots+1, N2) array of bools
        See p1_disrupted above.
    p2_disruption_time : (N2,) array of ints
        See p1_disruption_time above.
    p2_longitudes : (N_snapshots+1, N2) array of floats
        See p1_longitudes above.
    """

    def __init__(self, filename):

        # check correct file ending
        if filename[-5:] == '.hdf5':
            f = h5py.File(filename, 'r')
        else:
            f = h5py.File(filename + ".hdf5", 'r')

        # read header
        header = f["Header"]
        self.ndiscs = header.attrs['NDiscs']
        self.nspheroids = header.attrs['NSpheroids']
        self.sat_mass = header.attrs['SatelliteMass']
        self.sat_radius = header.attrs['SatelliteRadius']
        self.modgrav = header.attrs['ModGrav']
        if self.modgrav:
            self.beta = header.attrs['Beta']
            self.MW_r_screen = header.attrs['MW_RScreen']
            self.sat_r_screen = header.attrs['Satellite_RScreen']
        else:
            self.beta = None
            self.MW_r_screen = None
            self.sat_r_screen = None
        self.tracers = header.attrs['Tracers']
        if self.tracers:
            self.N1 = header.attrs['NPart_Type1']
            self.N2 = header.attrs['NPart_Type2']
        else:
            self.N1 = None
            self.N2 = None
        self.N_snapshots = header.attrs['NSnapshots']
        self.t_max = header.attrs['TimeMax']
        self.dt = header.attrs['TimestepSize']

        # disc and spheroid parameters
        self.dpars = []
        self.spars = []
        for i in range(self.ndiscs):
            self.dpars.append(dict(f['Header/Disc_' + str(i)].attrs))
        for i in range(self.nspheroids):
            self.spars.append(dict(f['Header/Spheroid_' + str(i)].attrs))

        # array of times
        self.times = np.array(f['SnapshotTimes'])

        # satellite data
        self.p0_positions = np.array(f['Satellite/Position'])
        self.p0_velocities = np.array(f['Satellite/Velocity'])

        # tracer particle data
        if self.tracers:
            self.p1_positions = np.array(f['PartType1/Position'])
            self.p1_velocities = np.array(f['PartType1/Velocity'])
            self.p1_disrupted = np.array(f['PartType1/Disrupted'])
            self.p1_disruption_time = np.array(f['PartType1/DisruptionTime'])

            self.p2_positions = np.array(f['PartType2/Position'])
            self.p2_velocities = np.array(f['PartType2/Velocity'])
            self.p2_disrupted = np.array(f['PartType2/Disrupted'])
            self.p2_disruption_time = np.array(f['PartType2/DisruptionTime'])

            # calculate orbital plane longitudes
            self.p1_longitudes, self.p2_longitudes = self._calc_longitudes()

        else:
            self.p1_positions = None
            self.p1_velocities = None
            self.p1_disrupted = None
            self.p1_disruption_time = None
            self.p1_longitudes = None

            self.p2_positions = None
            self.p2_velocities = None
            self.p2_disrupted = None
            self.p2_disruption_time = None
            self.p2_longitudes = None

        f.close()

        return

    def _calc_longitudes(self):
        """
        Calculate longitudes of tracer particles.

        Note: calculation performed in instantaneous orbital plane of
        satellite.
        """
        assert self.tracers

        # zp is z unit vector at all times, shape 501 x 3
        zp = np.cross(self.p0_positions, self.p0_velocities)
        zp = zp / np.linalg.norm(zp, axis=-1)[:, None]

        # xp and yp are x and y unit vectors
        xp = self.p0_positions
        xp = xp / np.linalg.norm(xp, axis=-1)[:, None]
        yp = np.cross(zp, xp)

        # project particle positions into orbital x-y plane
        p1_xp = np.sum(self.p1_positions * xp[:, None, :], axis=-1)
        p1_yp = np.sum(self.p1_positions * yp[:, None, :], axis=-1)
        p2_xp = np.sum(self.p2_positions * xp[:, None, :], axis=-1)
        p2_yp = np.sum(self.p2_positions * yp[:, None, :], axis=-1)

        # get longitudes
        p1_phi = np.arctan2(p1_yp, p1_xp)
        p2_phi = np.arctan2(p2_yp, p2_xp)

        # add/subtract multiples of 2pi for particles on higher wraps.
        dp = np.vstack((np.zeros((1, self.N1)), np.diff(p1_phi, axis=0)))
        for j in range(self.N1):
            changes = np.where(np.abs(dp[:, j]) > 1.1 * pi)[0]
            for i in range(changes.size):
                p1_phi[changes[i]:, j] -= 2 * pi * np.sign(dp[changes[i], j])
        dp = np.vstack((np.zeros((1, self.N2)), np.diff(p2_phi, axis=0)))
        for j in range(self.N2):
            changes = np.where(np.abs(dp[:, j]) > 1.1 * pi)[0]
            for i in range(changes.size):
                p2_phi[changes[i]:, j] -= 2 * pi * np.sign(dp[changes[i], j])

        return p1_phi, p2_phi
