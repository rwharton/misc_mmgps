import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from astropy.visualization import make_lupton_rgb

from astropy.table import Table

from astropy.coordinates import SkyCoord
import astropy.units as u

import bdsf

import os


def fit_one(fitsfile, cc, outfits, fix_point=True):
    """
    From the image in `fitsfile`, try to fit a single 
    source around coordinate given by SkyCoord object 
    `cc`.  Write output to a fits table in `outfits`.

    fix_point = True -- do not fit shape, use synth beam
    """
    ra_dec = [(cc.ra.deg, cc.dec.deg)]
    img = bdsf.process_image(fitsfile, advanced_opts=True, 
                             src_ra_dec=ra_dec, 
                             fix_to_beam=fix_point)

    img.write_catalog(outfile=outfits, 
                     format='fits', catalog_type='srl')
    
    return


def radec_str_to_cc(ra_str, dec_str):
    """
    Make skycoord  
    """
    return SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))


def image_fits(infile, top_FITS, outdir):
    """
    Read in a list of pulsar with names, coords, and fits files 
    and do pybdsf fit.  Write output fits (in... FITS) to 
    out_fits
    """
    with open(infile) as fin:
        for line in fin:
            if line[0] == "#":
                continue
            cols = line.split()
            psr_name = cols[0].strip()
            ra_str   = cols[1].strip()
            dec_str  = cols[2].strip()
            fits_fp  = cols[3].strip()

            fitsfile = f"{top_FITS}/{fits_fp}"

            outfile = f"{outdir}/{psr_name}_cat.fits"

            cc = radec_str_to_cc(ra_str, dec_str)

            fit_one(fitsfile, cc, outfile)

    return


def fits_to_txt(infile, outfile, top_fits):
    """
    Write out text file of fits

    infile is just used to read in source names and order
    
    fits catalogs are in top_fits
    """
    psr_names = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] == "#":
                continue
            
            psr_name = line.split()[0]
            psr_names.append( psr_name.strip() )

    with open(outfile, 'w') as fout:
        hdr_str1 = f"#{' Pulsar':<14} {'RA':^15} {'e_RA':^8} {'e_RA':^8} " +\
                   f"{'DEC':^15} {'e_DEC':^8} {'SNR':^8}"
        hdr_str2 = f"#{'':<14} {'(hms)':^15} {'(sec)':^8} {'(arcsec)':^8} " +\
                   f"{'(dms)':^15} {'(arcsec)':^8} {'':^8}"

        fout.write(hdr_str1 + "\n")
        fout.write(hdr_str2 + "\n")

        for psr in psr_names:
            fits_file = f"{top_fits}/{psr}_cat.fits"

            if not os.path.exists(fits_file):
                print(f"File not found: {fits_file}")
                continue

            T = Table.read(fits_file)
            
            cc = SkyCoord(T['RA'][0], T['DEC'][0], unit=u.deg)

            ra_str, dec_str = cc.to_string('hmsdms', sep=':', precision=3).split()

            e_ra_arcsec = T['E_RA'].to('arcsec').value[0]
            e_dec_arcsec = T['E_DEC'].to('arcsec').value[0]

            e_ra_s = e_ra_arcsec / (15 * np.cos(cc.dec.rad) )

            snr = T['Total_flux'][0] / T['Resid_Isl_rms'][0]

            out_str = f"{psr:<15} {ra_str:^15} {e_ra_s:^8.3f} {e_ra_arcsec:^8.3f} " +\
                      f"{dec_str:^15} {e_dec_arcsec:^8.2f} {snr:^8.1f}"

            fout.write(out_str + "\n")

    return
        

def fits_to_reg(infile, outfile, top_fits, show='beam', nsig=3):
    """
    Write out region file of fits

    infile is just used to read in source names and order
    
    fits catalogs are in top_fits
    """
    psr_names = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] == "#":
                continue
            
            psr_name = line.split()[0]
            psr_names.append( psr_name.strip() )

    with open(outfile, 'w') as fout:
        hdr_str1 = f"#{' Pulsar':<14} {'RA':^15} {'e_RA':^8} {'e_RA':^8} " +\
                   f"{'DEC':^15} {'e_DEC':^8} {'SNR':^8}"
        hdr_str2 = f"#{'':<14} {'(hms)':^15} {'(sec)':^8} {'(arcsec)':^8} " +\
                   f"{'(dms)':^15} {'(arcsec)':^8} {'':^8}"

        hdr = "global color=white dashlist=8 3 width=1 " +\
              "font=\"helvetica 10 normal roman\" select=1 highlite=1 " +\
              "dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1" +\
              "\n" +\
              "fk5\n"

        fout.write(hdr)

        for psr in psr_names:
            fits_file = f"{top_fits}/{psr}_cat.fits"

            if not os.path.exists(fits_file):
                print(f"File not found: {fits_file}")
                continue

            T = Table.read(fits_file)
            
            cc = SkyCoord(T['RA'][0], T['DEC'][0], unit=u.deg)

            ra_str, dec_str = cc.to_string('hmsdms', sep=':', precision=3).split()

            if show == "beam":
                bm_maj = T['Maj'].to('arcsec').value[0]
                bm_min = T['Min'].to('arcsec').value[0]
                bm_pa  = T['PA'].to('deg').value[0]
                out_str = f"ellipse({ra_str}, {dec_str}, "+\
                          f"{bm_min/2:.3f}\", {bm_maj/2:.3f}\", {bm_pa:.1f}) #text={{{psr}}}"

            elif show == "pos":
                e_ra_arcsec = T['E_RA'].to('arcsec').value[0] * nsig
                e_dec_arcsec = T['E_DEC'].to('arcsec').value[0] * nsig
                out_str = f"ellipse({ra_str}, {dec_str}, "+\
                          f"{e_ra_arcsec:.3f}\", {e_dec_arcsec:.3f}\", 0) #text={{{psr}}}"

            else:
                out_str = f"point {ra_str} {dec_str} #text={{{psr}}}"

            
            fout.write(out_str + "\n")
                       
    return
        
