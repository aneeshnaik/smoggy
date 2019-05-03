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


def movie_update(num, sat, tracer1, tracer2, line, s1, s2):
    line.set_data(sat[:num, 0], sat[:num, 1])
    s1.set_offsets(tracer1[:, :2, num])
    s2.set_offsets(tracer2[:, :2, num])
    return (line, s1, s2)


def movie(s, length=10):

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    sat_t = s.sat_tr/kpc
    DM_t = s.DM_tr/kpc
    stars_t = s.stars_tr/kpc

    N_frames = sat_t.shape[0]
    interval = length*1000/(N_frames-1)

    # show initial positions in background
    ax1.plot(sat_t[:, 0], sat_t[:, 1], ls='dashed', c='white')
    ax1.scatter(DM_t[:, 0, 0], DM_t[:, 1, 0], s=1, c='white')
    ax1.scatter(stars_t[:, 0, 0], stars_t[:, 1, 0], s=1, c='white')

    # set up initial artists for animation
    l, = ax1.plot([], [], c='white')
    s1 = ax1.scatter(DM_t[:, 0, 0], DM_t[:, 1, 0], s=1, c=green3, label='DM')
    s2 = ax1.scatter(stars_t[:, 0, 0], stars_t[:, 1, 0], s=1, c=green1)

    # central MW blob
    ax1.scatter([0], [0], c=comp)

    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.set_xlabel(r'$x\ [\mathrm{kpc}]$')
    ax1.set_ylabel(r'$y\ [\mathrm{kpc}]$')

    handles = [Line2D([0], [0], marker='.', lw=0, label="Dark matter",
                      mfc=green3, mec=green3, ms=10),
               Line2D([0], [0], marker='.', lw=0, label="Stars",
                      mfc=green1, mec=green1, ms=10)]
    ax1.legend(frameon=False, handles=handles)

    a = FuncAnimation(fig, movie_update, frames=N_frames, blit=True,
                      fargs=(sat_t, DM_t, stars_t, l, s1, s2),
                      interval=interval)

    return a
