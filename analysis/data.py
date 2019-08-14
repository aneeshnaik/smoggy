#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: July 2019
Author: A. P. Naik
Description: Class that reads in saved simulation outputs. Has convenient
functions for plotting/analysis etc.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from smoggy.constants import kpc, year, pi


class SimData:
    def __init__(self, filename):

        # check correct file ending
        if filename[-5:] == '.hdf5':
            f = h5py.File(filename, 'r')
        else:
            f = h5py.File(filename+".hdf5", 'r')

        # read header
        header = f["Header"]
        self.halo = header.attrs['HaloType']
        self.disc = header.attrs['DiscType']
        self.bulge = header.attrs['BulgeType']
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

        # add subgroups to header containing parameters for halo, disc, bulge
        self.halo_pars = dict(f['Header/Halo'].attrs)
        self.disc_pars = dict(f['Header/Disc'].attrs)
        self.bulge_pars = dict(f['Header/Bulge'].attrs)

        # array of times
        self.times = np.array(f['SnapshotTimes'])

        # satellite data
        self.p0_positions = np.array(f['Satellite/Position'])
        self.p0_velocities = np.array(f['Satellite/Velocity'])
        self.p0_phi = np.array(f['Satellite/OrbitalPhase'])

        # tracer particle data
        if self.tracers:
            self.p1_positions = np.array(f['PartType1/Position'])
            self.p1_velocities = np.array(f['PartType1/Velocity'])
            self.p1_phi = np.array(f['PartType1/OrbitalPhase'])
            self.p1_disrupted = np.array(f['PartType1/Disrupted'])
            self.p1_disruption_time = np.array(f['PartType1/DisruptionTime'])
            self.p1_leading = np.array(f['PartType1/LeadingStream'])

            self.p2_positions = np.array(f['PartType2/Position'])
            self.p2_velocities = np.array(f['PartType2/Velocity'])
            self.p2_phi = np.array(f['PartType2/OrbitalPhase'])
            self.p2_disrupted = np.array(f['PartType2/Disrupted'])
            self.p2_disruption_time = np.array(f['PartType2/DisruptionTime'])
            self.p2_leading = np.array(f['PartType2/LeadingStream'])
        else:
            self.p1_positions = None
            self.p1_velocities = None
            self.p1_phi = None
            self.p1_disrupted = None
            self.p1_disruption_time = None
            self.p1_leading = None

            self.p2_positions = None
            self.p2_velocities = None
            self.p2_phi = None
            self.p2_disrupted = None
            self.p2_disruption_time = None
            self.p2_leading = None

        f.close()

        return

    def plot_stream_image(self, ax=None, c1='r', c2='g'):

        assert self.tracers

        # create axis if one isn't provided
        if ax is None:
            ax = plt.subplot()

        x0 = self.p0_positions[-1]/kpc
        x1 = self.p1_positions[-1]/kpc
        x2 = self.p2_positions[-1]/kpc

        # plot particles
        ax.scatter(x1[:, 0], x1[:, 2], s=3, alpha=0.5, rasterized=True, c=c1)
        ax.scatter(x2[:, 0], x2[:, 2], s=3, alpha=0.5, rasterized=True, c=c2)

        # plot satellite and MW centres
        ax.scatter(x0[0], x0[2], marker='x', c='k')
        ax.scatter([0], [0], marker='x', c='k')

        return

    def plot_trajectories(self, ax1=None, ax2=None, c1='r', c2='g'):

        assert self.tracers
        from matplotlib.collections import LineCollection as LineColl

        long_1 = self.p1_phi - self.p0_phi[:, None]
        long_2 = self.p2_phi - self.p0_phi[:, None]

        x = np.linspace(-1e+17/(1e+9*year), 0, self.N_snapshots+1)
        y1 = long_1.T * 180/pi
        y2 = long_2.T * 180/pi

        segs1 = np.zeros((self.N1, self.N_snapshots+1, 2))
        segs2 = np.zeros((self.N2, self.N_snapshots+1, 2))
        segs1[:, :, 1] = y1
        segs1[:, :, 0] = x
        segs2[:, :, 1] = y2
        segs2[:, :, 0] = x

        if ax1 is None and ax2 is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ymin = min(y1.min(), y2.min())
            ymax = max(y1.max(), y2.max())

            ax1.set_ylim(ymin, ymax)
            ax2.set_ylim(ymin, ymax)

            ax1.set_xlim(-self.times[-1]/(1e+9*year), 0)
            ax2.set_xlim(-self.times[-1]/(1e+9*year), 0)

        coll1 = LineColl(segs1, color=c1, lw=0.5, alpha=0.05, rasterized=True)
        coll2 = LineColl(segs2, color=c2, lw=0.5, alpha=0.05, rasterized=True)
        ax1.add_collection(coll1)
        ax2.add_collection(coll2)

        return

    def movie(self, fig=None, length=10, orientation='y'):

        x0 = self.p0_positions/kpc
        x1 = self.p1_positions/kpc
        x2 = self.p2_positions/kpc

        interval = length*1000/self.N_snapshots

        if orientation == 'z':
            inds = (0, 1)
        elif orientation == 'y':
            inds = (0, 2)
        else:
            assert False

        # create axis if one isn't provided
        if fig is None:
            fig = plt.figure()
            ax = fig.subplots()
        else:
            ax = fig.axes[0]

        # show initial positions in background
        ax.plot(x0[:, inds[0]], x0[:, inds[1]], ls='dashed', c='grey')
        ax.scatter(x1[0, :, inds[0]], x1[0, :, inds[1]], s=1, c='grey')
        ax.scatter(x2[0, :, inds[0]], x2[0, :, inds[1]], s=1, c='grey')

        # set up initial artists for animation
        l, = ax.plot([], [], c='grey')
        artists = [l]
        s1 = ax.scatter(x1[0, :, inds[0]], x1[0, :, inds[1]], s=1, c='green')
        s2 = ax.scatter(x2[0, :, inds[0]], x2[0, :, inds[1]], s=1, c='blue')
        artists.append(s1)
        artists.append(s2)

        # central MW blob
        ax.scatter([0], [0], c='k')

        fargs = (x0, x1, x2, artists, inds)
        a = FuncAnimation(fig, frame_update, frames=self.N_snapshots+1,
                          blit=True, fargs=fargs, interval=interval)

        return a


def frame_update(num, x0, x1, x2, artists, inds):

    line = artists[0]
    line.set_data(x0[:num, inds[0]], x0[:num, inds[1]])

    s1 = artists[1]
    s2 = artists[2]
    s1.set_offsets(x1[num, :, inds].T)
    s2.set_offsets(x2[num, :, inds].T)

    return artists
