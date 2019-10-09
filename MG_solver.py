#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: April 2019
Author: A. P. Naik
Description:
"""
from .constants import Mpc, kpc, pi, G, c
import numpy as np
import time
from scipy.interpolate import RectBivariateSpline as RBS


class Grid2D:
    def __init__(self, ngrid=175, nth=101, rmin=0.05*kpc, rmax=10*Mpc):

        self.ngrid = ngrid
        self.rmax = rmax
        self.rmin = rmin

        # radial direction: log spaced grid from xmin=ln(rmin) to xmax=ln(rmax)
        # xin and xout are the inner and outer edges of each grid cell
        xmax = np.log(rmax)
        xmin = np.log(rmin)
        self.dx = (xmax-xmin)/ngrid
        self.x = xmin + (np.arange(self.ngrid, dtype=np.float64)+0.5)*self.dx
        self.xout = xmin + (np.arange(self.ngrid, dtype=np.float64)+1)*self.dx
        self.xin = xmin + np.arange(self.ngrid, dtype=np.float64)*self.dx
        self.r = np.exp(self.x)
        self.rout = np.exp(self.xout)
        self.rin = np.exp(self.xin)

        # theta direction: evenly spaced from 0 to pi; number of points is nth
        self.nth = nth
        self.disc_ind = self.nth//2
        self.dth = np.pi/self.nth
        self.th = (np.arange(self.nth, dtype=np.float64)+0.5)*self.dth
        self.thin = np.arange(self.nth, dtype=np.float64)*self.dth
        self.thout = (np.arange(self.nth, dtype=np.float64)+1)*self.dth
        self.grid_shape = (self.ngrid, self.nth)

        # set up grid structure; any object, e.g. rgrid, is a ngrid x nth array
        self.rgrid, self.thgrid = np.meshgrid(self.r, self.th, indexing='ij')
        self.ringrid, self.thingrid = np.meshgrid(self.rin, self.thin,
                                                  indexing='ij')
        self.routgrid, self.thoutgrid = np.meshgrid(self.rout, self.thout,
                                                    indexing='ij')
        self.sthgrid = np.sin(self.thgrid)
        self.sthingrid = np.sin(self.thingrid)
        self.sthoutgrid = np.sin(self.thoutgrid)

        # coefficient of each term in the discretised Laplacian
        self.__coeff1 = self.routgrid/(self.dx**2*self.rgrid**3)
        self.__coeff2 = self.ringrid/(self.dx**2*self.rgrid**3)
        self.__coeff3 = self.sthoutgrid/((self.dth*self.rgrid)**2*self.sthgrid)
        self.__coeff4 = self.sthingrid/((self.dth*self.rgrid)**2*self.sthgrid)

        # dLdu_const gives the final constant term in the Newton Raphson
        # expression for dL/du
        d1 = (self.rgrid**3*self.dx**2)
        d2 = (self.rgrid**2*self.sthgrid*self.dth**2)
        self.__dLdu_const = ((self.ringrid+self.routgrid)/d1 +
                             (self.sthingrid+self.sthoutgrid)/d2)

        # dvol is cell volume, full vol is volume of whole sphere
        self.dvol = 2*np.pi*self.rgrid**3*np.sin(self.thgrid)*self.dx*self.dth
        self.fullvol = 4.0/3.0*np.pi*(self.rmax**3 - self.rmin**3)

        # user change to True if setting a guess
        self.GuessFlag = False

        return

    def set_cosmology(self, h, omega_m, redshift=0):
        """
        Assign a background cosmology to the grid: Hubble constant, redshift,
        omega_m are input, then rho_crit and rho_mean are calculated.

        Parameters
        ----------
        h : float
            Dimensionless Hubble parameter, e.g. 0.7
        omega_m : float
            Cosmic matter fraction, e.g. 0.3
        redshift : float
            Redshift at time of solution. Default: 0.
        """

        self.h = h
        self.omega_m = omega_m
        self.omega_l = 1 - omega_m
        self.redshift = redshift

        # calculate dimensional Hubble constant
        self.H0 = self.h * 100.0 * 1000.0 / Mpc

        # rhocrit is evaluated today, rhomean at given redshift
        self.rhocrit = 3.0 * self.H0**2 / (8.0 * pi * G)
        self.rhomean = (1+self.redshift)**3*self.omega_m * self.rhocrit

        return

    def laplacian(self, expu):
        """
        Calculate discretised laplacian of e^u. The coefficients coeff1 etc.
        are initialised in __init__ above. The boundary conditions are
        implemented here: first derivative of u vanishes at inner and outer
        boundaries.

        Parameters
        ----------
        expu : numpy.ndarray, shape grid_shape
            Exponential of u=ln(fR/fa)

        Returns
        -------
        D2expu : numpy.ndarray, shape grid_shape
            Discretised Laplacian of e^u
        """

        # d(e^u)/dx. BCs: vanishes at both boundaries
        deudx = np.zeros((self.ngrid+1, self.nth))
        deudx[1:-1, :] = (expu[1:, :] - expu[:-1, :])

        # d(e^u)/dtheta. BCs: vanishes at both boundaries
        deudth = np.zeros((self.ngrid, self.nth+1))
        deudth[:, 1:-1] = (expu[:, 1:] - expu[:, :-1])

        # discretised laplacian
        D2expu = deudx[1:, :]*self.__coeff1 - deudx[:-1, :]*self.__coeff2
        D2expu += deudth[:, 1:]*self.__coeff3 - deudth[:, :-1]*self.__coeff4
        return D2expu

    def newton(self, expu, D2expu):
        """
        Calculate the change in u this iteration, i.e. du = - L / (dL/du)

        Parameters
        ----------
        expu : numpy.ndarray, shape grid_shape
            Exponential of u=ln(fR/fa)
        D2expu : numpy.ndarray, shape grid_shape
            Discretised Laplacian of e^u, as calculated in 'laplacian' method

        Returns
        -------
        du : numpy.ndarray, shape grid_shape
            Required change to u this iteration.
        """

        # Newton-Raphson step, as in MG-GADGET paper
        oneoverrteu = 1/np.sqrt(expu)
        L = D2expu + self.__const1*(1.0-oneoverrteu) - self.__const2*self.drho
        dLdu = 0.5*self.__const1*oneoverrteu - expu*self.__dLdu_const
        du = - L / dLdu

        return du

    def iter_solve(self, niter, F0, verbose=False, tol=1e-7):
        """
        Iteratively solve the scalar field equation of motion on grid, using
        a Newton-Gauss-Seidel relaxation technique, as in MG-GADGET. Perform
        iterations until the change in u=ln(fR/fa) is everywhere below the
        threshold specified by the parameter 'tol'.

        Parameters
        ----------
        niter : int
            Max number of NGS iterations, beyond which the computation stops
            with an error code.
        F0 : float
            Cosmic background value of the scalar field. Should be negative,
            e.g. -1e-6 for F6.
        verbose : bool
            Whether to print progress update periodically. Default: False.
        tol : float
            Tolerance level for iterative changes in u. Once du is everywhere
            below this threshold, the computation stops. Default: 1e-7.
        """

        # relevant constants
        msq = self.omega_m*self.H0*self.H0
        self.Ra = 3*msq*((1+self.redshift)**3 + 4*self.omega_l/self.omega_m)
        self.R0 = 3*msq*(1 + 4*self.omega_l/self.omega_m)
        self.F0 = F0
        self.Fa = self.F0*(self.R0/self.Ra)**2
        self.__const1 = self.Ra / (3.0 * c**2 * self.Fa)
        self.__const2 = -8.0*np.pi*G / (3.0 * c**2 * self.Fa)

        # u = ln(fR/Fa)
        if self.GuessFlag:
            u = self.u_guess
        else:
            u = np.zeros((self.ngrid, self.nth))

        # main loop
        t0 = time.time()
        for i in np.arange(niter):

            # calculate change du
            expu = np.exp(u)
            D2expu = self.laplacian(expu)
            du = self.newton(expu, D2expu)

            # check if du is too high or sufficiently low
            if((abs(du) > 1).any()):
                ind = np.where(du > 1.0)
                du[ind] = 1.0
                ind = np.where(du < -1.0)
                du[ind] = -1.0
            elif abs(du).max() < tol:
                break

            # NGS update
            u += du

            if(verbose and i % 1000 == 0):
                print("iteration", i, ", max(|du|)=", abs(du).max())
        t1 = time.time()
        if i == niter-1:
            raise Exception("Solver took too long!")

        # output results
        self.u = u
        self.fR = self.Fa*np.exp(u)
        self.time_taken = t1-t0
        self.iters_taken = i

        df = (self.fR[:, 2:] - self.fR[:, :-2])
        start = (self.fR[:, 1] - self.fR[:, 0])[:, None]
        end = (self.fR[:, -1] - self.fR[:, -2])[:, None]
        self.dfdth = np.hstack((start, df, end))/(2*self.dth)

        df = (self.fR[2:, :] - self.fR[:-2, :])
        start = (self.fR[1] - self.fR[0])[None, :]
        end = (self.fR[-1] - self.fR[-2])[None, :]
        self.dfdr = np.vstack((start, df, end))/(2*self.dx*self.rgrid)

        self.dfdr_spline = RBS(self.x, self.th, self.dfdr)
        self.dfdth_spline = RBS(self.x, self.th, self.dfdth)

        if verbose:
            print("Took ", self.time_taken, " seconds")

        return

    def accel(self, pos):

        if pos.ndim == 1:
            x = pos[0]
            y = pos[1]
            z = pos[2]
            R = np.sqrt(x**2+y**2)
        else:
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]
            R = np.linalg.norm(pos[:, :2], axis=-1)

        r = np.sqrt(R**2+z**2)
        theta = np.arctan2(R, z)
        dfdr = self.dfdr_spline.ev(np.log(r), theta)
        dfdth = self.dfdth_spline.ev(np.log(r), theta)

        dfdx = dfdr*x/r + dfdth*x*z/(R*r**2)
        dfdy = dfdr*y/r + dfdth*y*z/(R*r**2)
        dfdz = dfdr*z/r - dfdth*R/r**2

        if pos.ndim == 1:
            return 0.5*c**2*np.array([dfdx, dfdy, dfdz])
        else:
            return 0.5*c**2*np.array([dfdx, dfdy, dfdz]).T
