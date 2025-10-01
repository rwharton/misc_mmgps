from astropy.io import fits 
from astropy.wcs import WCS
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np


def remove_pol_freq(infile, outfile):
    """
    Remove the FREQ and STOKES channels
    so we just have the image data

    We are assuming that these only have 
    one entry (ie, just one pol and one chan)
    otherwise, you cant just remove them.
    """
    hdulist = fits.open(infile)
    hdu = hdulist[0]

    hshape = hdu.shape
    N = len(hshape)

    for ii in range(len(hshape)):
        anum = N - ii
        print(f"axis {anum} has size {hshape[ii]}")
        if hshape[ii] == 1:
            print(f"Removing axis {anum}")
            CTYPE = f"CTYPE{anum}" 
            CRPIX = f"CRPIX{anum}" 
            CRVAL = f"CRVAL{anum}" 
            CDELT = f"CDELT{anum}" 
            CUNIT = f"CUNIT{anum}"

            hname = hdu.header[CTYPE]
            print(f"Name = {hname}")

            hdu.header.remove(CTYPE)
            hdu.header.remove(CRPIX)
            hdu.header.remove(CRVAL)
            hdu.header.remove(CDELT)
            hdu.header.remove(CUNIT)

    hdu.data = np.squeeze( hdu.data )

    hdu.writeto(outfile)

    return 

