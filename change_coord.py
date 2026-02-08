from astropy.io import fits 
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def convert_to_gal(infile, outfile):
    """
    Convert RA/DEC fitsfile infile to 
    GLON/GLAT fitsfile outfile 
    """
    hdulist = fits.open(infile)
    hdu = hdulist[0]
    
    # Get RA and DEC and convert to
    # GL and Gb
    ra  = hdu.header['CRVAL1']  
    dec = hdu.header['CRVAL2']  
    cc  = SkyCoord(ra, dec, unit=u.deg)
       
    gl  = cc.galactic.l.deg
    gb  = cc.galactic.b.deg

    ohdr = hdu.header[:]
    ohdr['CTYPE1'] = 'GLON-TAN'
    ohdr['CTYPE2'] = 'GLAT-TAN' 
    ohdr['CRVAL1'] = gl
    ohdr['CRVAL2'] = gb

    new_image, footprint = reproject_interp(hdu, ohdr)

    fits.writeto(outfile, new_image, ohdr)

    return 

"""
hdulist1 = fits.open('Lband.fits')
hdu1 = hdulist1[0]
w1 = WCS(hdu1.header)

hdulist2 = fits.open('small.fits')
hdu2 = hdulist2[0]
w2 = WCS(hdu2.header)
"""
