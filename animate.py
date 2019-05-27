#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 8th April 2019
Author: A. P. Naik
Description: Animate trajectories
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from .constants import kpc
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')

green1 = '#EDF8E9'
green2 = '#BAE4B3'
green3 = '#10C476'
green4 = '#31A354'
green5 = '#006D2C'
comp = '#A33159'


def movie_update(num, simulation, artists, inds):

    sat = simulation.sat_tr/kpc
    line = artists[0]
    line.set_data(sat[:num, inds[0]], sat[:num, inds[1]])

    if simulation.tracers:
        tracer1 = simulation.DM_tr/kpc
        tracer2 = simulation.stars_tr/kpc
        s1 = artists[1]
        s2 = artists[2]
        s1.set_offsets(tracer1[:, inds, num])
        s2.set_offsets(tracer2[:, inds, num])

    return artists


def movie(s, orientation='z', length=10):

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    sat_t = s.sat_tr/kpc
    if s.tracers:
        DM_t = s.DM_tr/kpc
        stars_t = s.stars_tr/kpc

    N_frames = sat_t.shape[0]
    interval = length*1000/(N_frames-1)

    if orientation == 'z':
        inds = (0, 1)
    elif orientation == 'y':
        inds = (0, 2)
    # show initial positions in background
    ax1.plot(sat_t[:, inds[0]], sat_t[:, inds[1]], ls='dashed', c='white')
    if s.tracers:
        ax1.scatter(DM_t[:, inds[0], 0], DM_t[:, inds[1], 0], s=1, c='white')
        ax1.scatter(stars_t[:, inds[0], 0], stars_t[:, inds[1], 0], s=1, c='white')

    # set up initial artists for animation
    l, = ax1.plot([], [], c='white')
    artists = [l]
    if s.tracers:
        s1 = ax1.scatter(DM_t[:, inds[0], 0], DM_t[:, inds[1], 0], s=1, c=green3, label='DM')
        s2 = ax1.scatter(stars_t[:, inds[0], 0], stars_t[:, inds[1], 0], s=1, c=green1)
        artists.append(s1)
        artists.append(s2)

    # central MW blob
    ax1.scatter([0], [0], c=comp)

    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    if orientation == 'z':
        ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
        ax1.set_ylabel(r'$y\ [\mathrm{kpc}]$')
    elif orientation == 'y':
        ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
        ax1.set_ylabel(r'$z\ [\mathrm{kpc}]$')

    handles = [Line2D([0], [0], marker='.', lw=0, label="Dark matter",
                      mfc=green3, mec=green3, ms=10),
               Line2D([0], [0], marker='.', lw=0, label="Stars",
                      mfc=green1, mec=green1, ms=10)]
    ax1.legend(frameon=False, handles=handles)

    a = FuncAnimation(fig, movie_update, frames=N_frames, blit=True,
                      fargs=(s, artists, inds), interval=interval)

    return a
