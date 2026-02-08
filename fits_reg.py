import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

from regions import Regions
from regions import PixCoord 
from regions import CircleSkyRegion, CircleAnnulusSkyRegion

import scipy.optimize as opt


matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


## Define power law function centred at <reffreq> Hz
def pl(x, alpha, k):
   return k * (x/3100)**alpha

def fit_spec(freqs, spec, e_spec):
    popt, pcov = opt.curve_fit(pl, freqs, spec, None, e_spec, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def get_internal_pixels(pix_reg, xlims=(None, None), 
                        ylims=(None, None)):
    """
    Get the pixel indices within the pixel region 
    pix_reg.  Optionally set x,y max
    """
    bb = pix_reg.bounding_box

    xmin, xmax = xlims
    ymin, ymax = ylims

    if xmin is None:
        xmin = bb.ixmin
    if ymin is None:
        ymin = bb.iymin

    if xmax is None:
        xmax = bb.ixmax
    if ymax is None:
        ymax = bb.iymax

    ixmin = max(xmin, bb.ixmin)
    ixmax = min(bb.ixmax, xmax)

    iymin = max(ymin, bb.iymin)
    iymax = min(bb.iymax, ymax)

    mm = np.mgrid[iymin:iymax, ixmin:ixmax]
    y, x = mm
    x = x.flatten()
    y = y.flatten()

    pp = PixCoord(x, y) 
    zz = np.where(pix_reg.contains(pp))[0]

    xin = x[zz]
    yin = y[zz]

    return xin, yin
     

def get_internal_pixels_for_skyreg(sky_reg, wcs):
    """
    Given a SkyRegion and WCS, find the pixels 
    that are in the image and fall within the region
    """
    pix_reg = sky_reg.to_pixel(wcs)
    bb = pix_reg.bounding_box

    data_ymax, data_xmax = wcs.celestial.array_shape
    
    ixmin = max(0, bb.ixmin)
    ixmax = min(bb.ixmax, data_xmax)

    iymin = max(0, bb.iymin)
    iymax = min(bb.iymax, data_ymax)
    
    xlims = (ixmin, ixmax)
    ylims = (iymin, iymax)

    xin, yin = get_internal_pixels(pix_reg, xlims=xlims, ylims=ylims)

    return xin, yin


def get_pixel_area(header):
    """
    Get beam and pixel info from fits header 
    and return the pixel area in beams.  This 
    will be used to convert Jy/beam to Jy
    """
    bmaj = header['BMAJ']
    bmin = header['BMIN']

    pix_1 = np.abs( header['CDELT1'] )
    pix_2 = np.abs( header['CDELT2'] )

    pix_A = pix_1 * pix_2
    beam_A = np.pi * bmaj * bmin / (4 * np.log(2))

    pix_beams = pix_A / beam_A

    return pix_beams 


def get_flux_from_region(sky_reg, dat, wcs, pix_beams):
    """
    Get flux density contained within sky_reg for for 
    fits data dat using wcs 
    """
    xin, yin = get_internal_pixels_for_skyreg(sky_reg, wcs)
    flux = np.sum(dat[yin, xin]) * pix_beams
    return flux


def get_flux_from_region_bg_sub(sky_reg, dat, wcs, pix_beams):
    """
    Get flux density contained within sky_reg for for 
    fits data dat using wcs.  Subtract background from 
    surrouding annulus
    """
    # Make annulus region and get stats
    bg_reg = get_bg_annulus(sky_reg)
    bg_flux, pix_med, pix_std = get_stats_from_region(bg_reg, dat, wcs, pix_beams)    

    # Now get data from target region
    xin, yin = get_internal_pixels_for_skyreg(sky_reg, wcs)
    flux = np.sum(dat[yin, xin] - pix_med) * pix_beams

    # Est flux err (???)
    flux_err = pix_std * np.sqrt( len(xin) * pix_beams )
    return flux, flux_err, pix_std


def get_stats_from_region(sky_reg, dat, wcs, pix_beams):
    """
    Get flux density contained within sky_reg for for 
    fits data dat using wcs 
    """
    xin, yin = get_internal_pixels_for_skyreg(sky_reg, wcs)
    flux = np.sum(dat[yin, xin]) * pix_beams
    pix_median = np.median(dat[yin, xin])
    pix_std    = np.std(dat[yin, xin])
    return flux, pix_median, pix_std


def get_bg_annulus(sky_reg):
    """
    Get a circular annulus around circular region

    Take Router = sqrt(2) * R

    which gives same area in annulus as circle
    """
    center = sky_reg.center
    inner_radius = sky_reg.radius
    outer_radius = sky_reg.radius * np.sqrt(2)
    cir_annulus = CircleAnnulusSkyRegion(center=center, 
                                    inner_radius=inner_radius, 
                                    outer_radius=outer_radius)
    return cir_annulus


def get_cube_spec_from_region(sky_reg, cube_dat, wcs, 
                              pix_beams):
    """
    Get flux of region in each channel of a data 
    cube of shape (Nchan, y, x)
    """
    spec = np.zeros(len(cube_dat))
    for ii, dd in enumerate(cube_dat):
        spec[ii] = get_flux_from_region(sky_reg, dd, wcs, 
                                        pix_beams)

    return spec
     

def get_cube_spec_from_region_bg_sub(sky_reg, cube_dat, wcs, 
                                     pix_beams):
    """
    Get flux of region in each channel of a data 
    cube of shape (Nchan, y, x)

    Account for background by 
    """
    spec  = np.zeros(len(cube_dat))
    espec = np.zeros(len(cube_dat))
    bg_std = np.zeros(len(cube_dat))
    for ii, dd in enumerate(cube_dat):
        flux_ii, eflux_ii, std_ii = \
                      get_flux_from_region_bg_sub(sky_reg, dd, wcs, pix_beams)
        spec[ii] = flux_ii
        espec[ii] = eflux_ii
        bg_std[ii] = std_ii

    return spec, espec, bg_std


def make_plot(freqs, spec, espec=None, fit=False, outfile=None, title=None):
    """
    Make a spectrum plot
    """
    if outfile is not None:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if espec is None:
        ax.plot(freqs, spec, ls='', marker='o', c='k')
    else:
        ax.errorbar(freqs, y=spec, yerr=espec, ls='', marker='o', c='k')

    if fit:
        if espec is None:
            espec = 0.1 * spec

        popt, perr = fit_spec(freqs, spec, espec)
        mm = popt[1] * (freqs/3100)**popt[0]
        ax.plot(freqs, mm, c='r', lw=2)

        alpha_fit = popt[0]
        aerr_fit  = perr[0]

        red_chi_sqr = np.sum( ((spec - mm) / espec)**2 / len(spec) )

        afit_str = f"$\\alpha_{{\\rm fit}} = {alpha_fit:+.1f} \\pm {aerr_fit:.1f}$"
        rchi_str = f"$\\chi^2_{{\\rm red}} = {red_chi_sqr:.2f}$"


    ax.set_xlabel("Frequency (MHz)", fontsize=14)
    ax.set_ylabel("Flux Density (Jy)", fontsize=14)

    tstr = ""
    if title is not None:
        tstr = title

    if fit:
        if len(tstr):
            tstr += ", "
        tstr += f"{afit_str} ({rchi_str})"

    if len(tstr):
        ax.set_title(tstr, fontsize=14)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    return


def make_many_plots(regfile, freqfile, fitsfile, fit=True, outbase=None):
    """
    """
    # Read in fits data
    hdulist = fits.open(fitsfile)
    hdu = hdulist[0]
    dat = np.squeeze(hdu.data)

    # Get pix beams + WCS
    pix_beams = get_pixel_area(hdu.header)
    wcs = WCS(hdu.header).celestial

    # Read in regions
    reg_list = Regions.read(regfile, format='ds9') 

    # Read in freqs
    freqs = np.loadtxt(freqfile)

    for ii, rr in enumerate(reg_list):
        spec, espec, bg_std = \
              get_cube_spec_from_region_bg_sub(rr, dat, wcs, pix_beams)  

        if outbase is not None:
            outfile = f"{outbase}_{ii:03d}.png"
        else:
            outfile = f"src_{ii:03d}.png"

        tstr = f"Src {ii:03d}" 
    
        make_plot(freqs, spec, espec=espec, fit=fit, outfile=outfile, title=tstr)

    return
