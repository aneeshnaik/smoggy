#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created:
Author: A. P. Naik
Description:
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from gravstream.constants import kpc


def stream_image(filename):

    # load file
    f = h5py.File(filename, 'r')
    x0 = np.array(f['Satellite/Position'])[-1]/kpc
    x1 = np.array(f['PartType1/Position'])[-1]/kpc
    x2 = np.array(f['PartType2/Position'])[-1]/kpc
    f.close()

    # set up figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.8])

    # plot particles
    ax.scatter(x1[:, 0], x1[:, 2], s=3, alpha=0.5)
    ax.scatter(x2[:, 0], x2[:, 2], s=3, alpha=0.5)

    # plot satellite and MW centres
    ax.scatter(x0[0], x0[2], marker='x', c='k')
    ax.scatter([0], [0], marker='x', c='k')

    return fig
