#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: June 2019
Author: A. P. Naik
Description: Create a movie of a gravstream simulation
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
import numpy as np
from gravstream.constants import kpc


def frame_update(num, x0, x1, x2, artists, inds):

    line = artists[0]
    line.set_data(x0[:num, inds[0]], x0[:num, inds[1]])

    s1 = artists[1]
    s2 = artists[2]
    s1.set_offsets(x1[num, :, inds].T)
    s2.set_offsets(x2[num, :, inds].T)

    return artists


def movie(filename, length=10, orientation='y', axlim=100):

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    f = h5py.File(filename)
    x0 = np.array(f['Satellite/Position'])/kpc
    x1 = np.array(f['PartType1/Position'])/kpc
    x2 = np.array(f['PartType2/Position'])/kpc
    f.close()

    N_frames = x0.shape[0]
    interval = length*1000/(N_frames-1)

    if orientation == 'z':
        inds = (0, 1)
    elif orientation == 'y':
        inds = (0, 2)
    else:
        assert False

    # show initial positions in background
    ax1.plot(x0[:, inds[0]], x0[:, inds[1]], ls='dashed', c='grey')
    ax1.scatter(x1[0, :, inds[0]], x1[0, :, inds[1]], s=1, c='grey')
    ax1.scatter(x2[0, :, inds[0]], x2[0, :, inds[1]], s=1, c='grey')

    # set up initial artists for animation
    l, = ax1.plot([], [], c='grey')
    artists = [l]
    s1 = ax1.scatter(x1[0, :, inds[0]], x1[0, :, inds[1]], s=1, c='green')
    s2 = ax1.scatter(x2[0, :, inds[0]], x2[0, :, inds[1]], s=1, c='blue')
    artists.append(s1)
    artists.append(s2)

    # central MW blob
    ax1.scatter([0], [0], c='k')

    ax1.set_xlim(-axlim, axlim)
    ax1.set_ylim(-axlim, axlim)
    if orientation == 'z':
        ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
        ax1.set_ylabel(r'$y\ [\mathrm{kpc}]$')
    elif orientation == 'y':
        ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
        ax1.set_ylabel(r'$z\ [\mathrm{kpc}]$')

    a = FuncAnimation(fig, frame_update, frames=N_frames, blit=True,
                      fargs=(x0, x1, x2, artists, inds), interval=interval)

    return a