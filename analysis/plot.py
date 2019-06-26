#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from gravstream.constants import kpc


# IMAGE OF FINAL STREAM

def plot_stream(i):

    # load file
    f = h5py.File("/data/an485/gravstream_sims/pal5/"+str(i)+".hdf5")
    x0 = np.array(f['Satellite/Position'])[-1]/kpc
    x1 = np.array(f['PartType1/Position'])[-1]/kpc
    x2 = np.array(f['PartType2/Position'])[-1]/kpc

    # set up figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.8])

    # plot particles
    ax.scatter(x1[:, 0], x1[:, 2], s=2, alpha=0.3)
    ax.scatter(x2[:, 0], x2[:, 2], s=2, alpha=0.3)

    # plot satellite and MW centres
    ax.scatter(x0[0], x0[2], marker='x', c='k')
    ax.scatter([0], [0], marker='x', c='k')

    return fig


def movie_update(num, x0, x1, x2, artists, inds):

    line = artists[0]
    line.set_data(x0[:num, inds[0]], x0[:num, inds[1]])

    s1 = artists[1]
    s2 = artists[2]
    s1.set_offsets(x1[num, :, inds].T)
    s2.set_offsets(x2[num, :, inds].T)

    return artists


fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
length = 30
orientation = 'y'

f = h5py.File("/data/an485/gravstream_sims/sgr/0.hdf5")
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

ax1.set_xlim(-100, 100)
ax1.set_ylim(-100, 100)
if orientation == 'z':
    ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
    ax1.set_ylabel(r'$y\ [\mathrm{kpc}]$')
elif orientation == 'y':
    ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
    ax1.set_ylabel(r'$z\ [\mathrm{kpc}]$')


a = FuncAnimation(fig, movie_update, frames=N_frames, blit=True,
                  fargs=(x0, x1, x2, artists, inds), interval=interval)
