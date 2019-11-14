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
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14


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
        if self.disc != 'None':
            self.disc_pars = dict(f['Header/Disc'].attrs)
        else:
            self.disc_pars = {}
        if self.bulge != 'None':
            self.bulge_pars = dict(f['Header/Bulge'].attrs)
        else:
            self.bulge_pars = {}

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

        else:
            self.p1_positions = None
            self.p1_velocities = None
            self.p1_disrupted = None
            self.p1_disruption_time = None

            self.p2_positions = None
            self.p2_velocities = None
            self.p2_disrupted = None
            self.p2_disruption_time = None

        f.close()

        return

    def plot_stream_image(self, ax=None, c1='r', c2='g', frame=-1):

        assert self.tracers

        # create axis if one isn't provided
        if ax is None:
            ax = plt.subplot()

        x0 = self.p0_positions[frame]/kpc
        x1 = self.p1_positions[frame]/kpc
        x2 = self.p2_positions[frame]/kpc

        # plot particles
        ax.scatter(x1[:, 0], x1[:, 2], s=3, alpha=0.5, rasterized=True, c=c1)
        ax.scatter(x2[:, 0], x2[:, 2], s=3, alpha=0.5, rasterized=True, c=c2)

        # plot satellite and MW centres
        ax.scatter(x0[0], x0[2], marker='x', c='k')
        ax.scatter([0], [0], marker='o', c='k')

        return

    def calculate_longitudes(self):

        # zp is z vector at all times, shape 501 x 3
        zp = np.cross(self.p0_positions, self.p0_velocities)
        zp = zp/np.linalg.norm(zp, axis=-1)[:, None]

        xp = self.p0_positions
        xp = xp/np.linalg.norm(xp, axis=-1)[:, None]
        yp = np.cross(zp, xp)

        p1_xp = np.sum(self.p1_positions*xp[:, None, :], axis=-1)
        p1_yp = np.sum(self.p1_positions*yp[:, None, :], axis=-1)
        p2_xp = np.sum(self.p2_positions*xp[:, None, :], axis=-1)
        p2_yp = np.sum(self.p2_positions*yp[:, None, :], axis=-1)

        p1_phi = np.arctan2(p1_yp, p1_xp)
        p2_phi = np.arctan2(p2_yp, p2_xp)

        dp = np.vstack((np.zeros((1, self.N1)), np.diff(p1_phi, axis=0)))
        for j in range(self.N1):
            changes = np.where(np.abs(dp[:, j]) > 1.1*pi)[0]
            for i in range(changes.size):
                p1_phi[changes[i]:, j] -= 2*pi*np.sign(dp[changes[i], j])

        dp = np.vstack((np.zeros((1, self.N2)), np.diff(p2_phi, axis=0)))
        for j in range(self.N2):
            changes = np.where(np.abs(dp[:, j]) > 1.1*pi)[0]
            for i in range(changes.size):
                p2_phi[changes[i]:, j] -= 2*pi*np.sign(dp[changes[i], j])

        return p1_phi, p2_phi

    def plot_trajectories(self, ax1=None, ax2=None, c1='r', c2='g'):

        assert self.tracers
        from matplotlib.collections import LineCollection as LineColl

        long_1, long_2 = self.calculate_longitudes()

        x = np.linspace(-self.times[-1]/(1e+9*year), 0, self.N_snapshots+1)
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

    def movie(self, length=20, orientation='y',
              c1='#31A354', c2='#A33159'):
        """
        Make movie of smoggy simulation.

        Parameters
        ----------
        d : smoggy.analysis.SimData instance
            Simulation data
        coords : {'galactocentric', 'lagrangian', 'both'}
            Type of movie. Default is galactocentric.
        orientation : {'y', 'z', 'x'}
            Orientation of movie
        """


        # calculate delay between frames in millisecs
        interval = length*1000/self.N_snapshots

        # get relevant coordinates in kpc
        if orientation == 'z':
            inds = (0, 1)
            axlabels = ['$x\ [\mathrm{kpc}]$', '$y\ [\mathrm{kpc}]$']
        elif orientation == 'y':
            inds = (0, 2)
            axlabels = ['$x\ [\mathrm{kpc}]$', '$z\ [\mathrm{kpc}]$']
        else:
            inds = (1, 2)
            axlabels = ['$y\ [\mathrm{kpc}]$', '$z\ [\mathrm{kpc}]$']
        times = self.times/(1e+9*year)
        x0 = self.p0_positions[:, inds]/kpc
        x1 = self.p1_positions[:, :, inds]/kpc
        x2 = self.p2_positions[:, :, inds]/kpc
        dx1 = (x1 - x0[:, None, :])
        dx2 = (x2 - x0[:, None, :])

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2 = fig.add_axes([0.7, 0.7, 0.18, 0.18])

        ax1lim = np.max(np.linalg.norm(x0, axis=-1))*1.5
        ax2lim = 12*self.sat_radius/kpc
        ax1.set_xlim(-ax1lim, ax1lim)
        ax1.set_ylim(-ax1lim, ax1lim)
        ax2.set_xlim(-ax2lim, ax2lim)
        ax2.set_ylim(-ax2lim, ax2lim)
        ax1.set_xlabel(axlabels[0])
        ax1.set_ylabel(axlabels[1])

        # set up initial artists for animation
        l, = ax1.plot([], [], c='lightgrey')
        s1 = ax1.scatter(x1[0, :, 0], x1[0, :, 1], s=1, c=c1, alpha=0.6, label='Dark Matter', zorder=1)
        s2 = ax1.scatter(x2[0, :, 0], x2[0, :, 1], s=1, c=c2, alpha=0.6, label='Stars', zorder=1)
        t = ax1.text(0.5, 0.975, "{0:.2f} Gyr".format(times[0]), transform=ax1.transAxes, ha='center', va='top')

        # central MW blob
        ax1.scatter([0], [0], c='k')

        if self.MW_r_screen is not None:
            circ = plt.Circle((0, 0), radius=self.MW_r_screen/kpc, fill=False, ls=':', ec='k')
            ax1.add_artist(circ)

        ax1.legend(markerscale=10, frameon=False, loc='upper left')

        s3 = ax2.scatter(dx1[0, :, 0], dx1[0, :, 1], s=1, c=c1, alpha=0.6)
        s4 = ax2.scatter(dx2[0, :, 0], dx2[0, :, 1], s=1, c=c2, alpha=0.6)

        artists = [l, s1, s2, s3, s4, t]

        fargs = (x0, x1, x2, dx1, dx2, times, artists)
        a = FuncAnimation(fig, frame_update, frames=self.N_snapshots+1,
                          blit=False, fargs=fargs, interval=interval)

        return a


def frame_update(num, x0, x1, x2, dx1, dx2, times, artists):

    line = artists[0]
    line.set_data(x0[:num, 0], x0[:num, 1])

    s1 = artists[1]
    s2 = artists[2]
    s1.set_offsets(x1[num, :])
    s2.set_offsets(x2[num, :])
    s1.set_zorder(1)
    s2.set_zorder(1)
    s3 = artists[3]
    s4 = artists[4]
    s3.set_offsets(dx1[num, :])
    s4.set_offsets(dx2[num, :])

    t = artists[5]
    t.set_text("{0:.2f} Gyr".format(times[num]))

    return artists
