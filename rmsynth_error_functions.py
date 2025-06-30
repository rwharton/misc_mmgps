#!/usr/bin/env python
import numpy as np
from astropy import constants as const
from astropy.io import fits
import astropy.io.ascii as asc
from astropy.wcs import WCS
import pylab as pl
import sys
#sys.path.append('/home/abasu/bin/python_funcs')
#import faraday_models as mods
import os
import rmUtils_update as rmUtils
import gc
import argparse
import commands as com
from argparse import RawTextHelpFormatter
from joblib import Parallel, delayed
import multiprocessing
import time
from psutil import virtual_memory
import subprocess
import scipy
from scipy.optimize import minpack
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import scipy.constants as constants
from scipy import stats
from lmfit import minimize, Parameters, Parameter, report_errors, report_fit
import lmfit
import scipy.interpolate as interp
import re
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


"""
Written by: Aritra Basu [TLS, Tautenburg] on 28 March 2025

On 03 April 2025, a new feature to directly handle spectra has also been added.

*** The last stable version that works on a source catalog is 'makeRMsynth_catalog.py-v3' ***


On: 19 May 2025

This is a subset of functions used to compute the error on different polarization 
parameters from Faraday depth spectra.

These functions are collated for Robert Wharton and Anahat Cheema.
"""



def rmsynth(qarray, uarray, nuarray, outpath=None, name=None, weights=None, rmpar=None, ptype=None, fdmin=None, fdmax=None, nslice=20):
    """
    Performs RMsynthesis using pyrmsynth and estimates the errors

    qarray [numpy.array]: array of Q
    uarray [numpy.array]: array of U
    nuarray [numpy.array]: array of frequencies

    outpath: path to rmsynthesis products including temp files
    weights: path to the weight file [MUST BE THE SAME NAME SPECIFIED IN THE rmsynth PAR FILE]
    rmpar: path to the rmsynth parameter file
    ptype: polfrac or polInt [used only as the naming convention maintained on gigantix2.tls-tautenburg.de]

    fdmin, fdmax [float]: plotting range

    nslice [int]: number of FD slices to be used for error estimation
    """
    

    """
    pwd = os.getcwd()
    temploc = os.path.join(outpath,name)
    if not os.path.exists(temploc):
        os.makedirs(temploc)

    qarray[np.where(np.isnan(qarray))] = 0.0
    uarray[np.where(np.isnan(uarray))] = 0.0

    rmUtils.create_pixel_cube_general(nuarray/1e9, qarray, uarray, temploc, 'None')
    s, o = com.getstatusoutput('mkdir %s/stokes_q/' %temploc)
    s, o = com.getstatusoutput('mkdir %s/stokes_u/' %temploc)

    os.system('mv %s/stokesQ_for_testing_weird_name*.fits %s/stokes_q/' %(temploc, temploc))
    os.system('mv %s/stokesU_for_testing_weird_name*.fits %s/stokes_u/' %(temploc, temploc))
    os.system('cp %s %s' %(weights, temploc))

    os.chdir(temploc)
    #print "\n\n+++ RM-synthesis par file +++"
    #os.system('cat %s' %rmpar)
    #print "\n+++  par file +++\n"

    os.system('rmsynthesis_header.py %s -s' %rmpar) # modified pyrmsynth execution
    os.chdir(pwd)


    s, pFDcubeFile =  com.getstatusoutput('ls %s/rmSynth_%s_clean_p.fits' %(temploc, ptype)) # RM-CLEANed pol. intensity FD cube
    s, pccFDcubeFile =  com.getstatusoutput('ls %s/rmSynth_%s_cc_p.fits' %(temploc, ptype)) # RM-CLEANed pol. intensity FD cube
    s, qFDcubeFile =  com.getstatusoutput('ls %s/rmSynth_%s_clean_q.fits' %(temploc, ptype)) # RM-CLEANed stokes q
    s, uFDcubeFile =  com.getstatusoutput('ls %s/rmSynth_%s_clean_u.fits' %(temploc, ptype)) # RM-CLEANed stokes u
    """
    rmsf = fits.getheader(pFDcubeFile)['TFFWHM']

    phi_array, polI, cc = spectra_cc(pFDcubeFile, pccFDcubeFile, 2, 2, 100, 'k-', 'r-', '', '', False, xmin=fdmin, xmax=fdmax, plot_cc=True)
    try:
        rm_peak, polI_peak = phi_max_fit(phi_array, polI)
    except:
        rm_peak, polI_peak = 0, 0

    nsteps = int(rmsf/np.diff(phi_array)[0]) + 1 # number of FD pixels in RMSF
    print ("\n +++ RMSF has %i dphi units of %f rad/m/m+++" %(nsteps, np.diff(phi_array)[0]))
    
    idxFD1 = [5*nsteps + i*int(1.2*nsteps) for i in range(int(nslice/2.))]  
              # wing on the left-side of the peak
    idxFD2 = [len(phi_array) - 5*nsteps - i*int(1.2*nsteps) for i in range(int(nslice/2.))]
              # wing on the right-side of the peak
                 # 5*nsteps avoids edges in the FD spectrum by 5 RMSF widths
                 # 1.2*nsteps ensures adjacent slices are separated slightly 
                 # by more than the RMSF

    try:
        resq = fits.getdata('%s/rmSynth_%s_residual_q.fits' %(temploc, ptype))
        resu = fits.getdata('%s/rmSynth_%s_residual_u.fits' %(temploc, ptype))
    except:
        resq = fits.getdata(qFDcubeFile)
        resu = fits.getdata(uFDcubeFile)


    idxFD = idxFD1 + idxFD2  # NOTE: Be careful idxFD1 & idxFD2 are lists.
    noiseq = np.array([resq[idx] for idx in idxFD])  # arbitrary Q slices in FD
    noiseu = np.array([resu[idx] for idx in idxFD])

    qcube = fits.getdata(qFDcubeFile)[:,2,2]
    ucube = fits.getdata(uFDcubeFile)[:,2,2]

    qinterp = interp.interp1d(phi_array, qcube, fill_value="extrapolate")
    uinterp = interp.interp1d(phi_array, ucube, fill_value="extrapolate")

    q_peak = qinterp(rm_peak)
    u_peak = uinterp(rm_peak)

    rmsq = noiseq.std(axis=0)  # Stokes Q & U rms 
    rmsu = noiseu.std(axis=0)
    rmsp = np.sqrt(rmsq**2. + rmsu**2.)

    rmerror = rmsf/(2.*polI_peak/rmsp) # From Iacobelli
    print (q_peak, q_peak.shape, type(q_peak))

    print (rm_peak, np.nanmean(rmerror), polI_peak, np.nanmean(rmsp), float(q_peak), np.nanmean(rmsq), float(u_peak), np.nanmean(rmsu), np.sqrt(float(q_peak)**2 + float(u_peak)**2)) # taking mean --> all pixels have same value
    return rm_peak, np.nanmean(rmerror), polI_peak, np.nanmean(rmsp), q_peak, np.nanmean(rmsq), u_peak, np.nanmean(rmsu)




def spectra_cc(clean_p, cc_p, x, y, figno, scolor, ccolor, label, pwd, hold, **kwargs):
    """
       LAST WORKING VERSION IS COMMENTED OUT BELOW!
       HERE THE LABEL SIZES HAVE BEEN CHANGED FOR THE GALAXIES PAPER!!!


       Plots the RMcleaned spectra and the Clean components (CC)

       clean_p : RMcleaned Faraday depth cube [from 'rmsynthesis.py']
       cc_p : The clean component cube [from 'rmsynthesis.py']
              'None' if no CC present or to be plotted

       x, y : The pixel for which the spectra is to be plotted [AIPS/ds9 convention]
       figno : Matplotlib figure number [INTEGER] to be plotted in.
    
       scolor : Line color for the spectra [e.g: 'k-', 'b--']
       ccolor : Color of the CC [e.g: 'r-', 'g--']
       label : label for the plot legend

       pwd : working directory of code execution

       hold : Boolean [for over plotting]
    """

    
    print('\n\n\n RMCUBE LOCATION')
    print(pwd+'/'+clean_p)
    rmcube = fits.getdata(pwd+'/'+clean_p)
    hdr = fits.getheader(pwd+'/'+clean_p)

    if cc_p == 'None':
        print '\n*** No CC file provided ***\n'
        rmcc = np.zeros(rmcube.shape)
    else:
        rmcc = fits.getdata(pwd+'/'+cc_p)

    unit = hdr['CUNIT3']
    ncube = hdr['NAXIS3']
    phi_ref = hdr['CRVAL3']  # Reference pixel RM value
    ref_pix = hdr['CRPIX3']  # Reference pixel
    dphi = hdr['CDELT3']

    phi_start = phi_ref - dphi * (ref_pix - 1)  # Starts from 1 (unlike 0 in python)
    phi_end = phi_start + ncube * dphi

    #phi_array = np.arange(phi_start, phi_end+dphi, dphi)
    phi_array = np.arange(phi_start, phi_end, dphi)


    print "\n--------------------------------------------------------"
    print "All phi values in unit: %s" %unit
    print "Reference phi: %4.3f" %phi_ref
    print "at pixel: %i" %ref_pix
    print "N phi: %i" %ncube
    print "dphi: %4.3f" %dphi
    print "phi start: %4.3f" %phi_start
    print "phi_end: %4.3f" %phi_end
    print "--------------------------------------------------------\n"

    rm = rmcube[:,y-1,x-1]
    cc = rmcc[:,y-1,x-1]
    
    goodcc = np.where(cc!=0.0)
    print 'There are %i CC [At pixel %i, %i]' %(len(goodcc[0]), x, y)
    
    return phi_array, rm, cc



def phi_max_fit(phi_array, rm):  # bad coding (rm --> poli)
    """
      Fits the peak of the FD spectra with an inverted parabola

      phi_array : Array of the Faraday depth
      rm : This is the value of the cleaned pol. I 
               ['rm' variable used is bad naming]

       RETURNS: peak_FD, peak_polI
    """

    idx = (np.abs(rm - rm.max())).argmin()

    phi_peak_array = phi_array[idx-2:idx+3]
    
    peak_polI_array = rm[idx-2:idx+3]
    a,b,c, erra, errb, errc = fit_parabola(phi_peak_array, peak_polI_array)
    
    print '\nPEAK FITTED RM: %3.3f rad/m/m' %b
    print 'PEAK POLARIZED EMISSION: %f' %c

    return b, c



