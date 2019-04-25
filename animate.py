#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 8th April 2019
Author: A. P. Naik
Description: Animate trajectories
"""
import pickle
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


def movie(filename, length=10):

    f = open(filename, 'rb')
    s = pickle.load(f)
    f.close()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    sat_t = s.sat_tr/kpc
    DM_t = s.DM_tr/kpc
    stars_t = s.stars_tr/kpc

    N_frames = sat_t.shape[0]
    interval = length*1000/(N_frames-1)

    st = r'$\beta={0:.1f},\ r={1:.1f}$'.format(s.beta, s.r_screen/kpc)
    ax.text(0.05, 0.95, st, transform=ax.transAxes, va='center', ha='left')

    # show initial positions in background
    ax.plot(sat_t[:, 0], sat_t[:, 1], ls='dashed', c='white')
    ax.scatter(DM_t[:, 0, 0], DM_t[:, 1, 0], s=1, c='white')
    ax.scatter(stars_t[:, 0, 0], stars_t[:, 1, 0], s=1, c='white')

    # set up initial artists for animation
    l, = ax.plot([], [], c='white')
    s1 = ax.scatter(DM_t[:, 0, 0], DM_t[:, 1, 0], s=1, c=green3, label='DM')
    s2 = ax.scatter(stars_t[:, 0, 0], stars_t[:, 1, 0], s=1, c=green1)

    # central MW blob
    circ = plt.Circle((0, 0), s.r_screen/kpc, color=comp, fill=False)
    ax.add_artist(circ)
    ax.scatter([0], [0], c=comp)

    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.xlabel(r'$x\ [\mathrm{kpc}]$')
    plt.ylabel(r'$y\ [\mathrm{kpc}]$')

    handles = [Line2D([0], [0], marker='.', lw=0, label="Dark matter",
                      mfc=green3, mec=green3, ms=10),
               Line2D([0], [0], marker='.', lw=0, label="Stars",
                      mfc=green1, mec=green1, ms=10),
               Line2D([0], [0], lw=2, label="Screening Radius", color=comp)]
    plt.legend(frameon=False, handles=handles)

    a = FuncAnimation(fig, movie_update, frames=N_frames, blit=True,
                      fargs=(sat_t, DM_t, stars_t, l, s1, s2),
                      interval=interval)

    return a
