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

    for ii in range(N):
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

            for ii in range(1, N+1):
                for jj in range(1, N+1):
                    if (ii == anum) or (jj == anum):
                        PC = f"PC{ii}_{jj}"
                        if hdu.header.get(PC) is not None:
                            hdu.header.remove(PC)

    hdu.data = np.squeeze( hdu.data )

    hdu.writeto(outfile)

    return 


def remove_pol_freq2(infile, outfile):
    """
    Remove the FREQ and STOKES channels
    so we just have the image data

    We are assuming that these only have 
    one entry (ie, just one pol and one chan)
    otherwise, you cant just remove them.
    """
    hdulist = fits.open(infile)
    hdu = hdulist[0]
    
    w_in = WCS(hdu.header)
    w_out = w_in.celestial

    hdu.data = np.squeeze( hdu.data )
    out_header = w_out.to_header()

    hdu.header = out_header

    hdu.writeto(outfile)

    return 

