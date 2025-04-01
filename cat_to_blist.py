import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

def make_beam_list(coords):
    """
    Take list of astropy SkyCoord objects and 
    convert them to a string format that will 
    be used by the beamforming pipeline
    """
    blist = []
    for cc in coords:
        ra_str  = cc.ra.to_string(u.hour, sep=':', pad=True)
        dec_str = cc.dec.to_string(u.deg, sep='.', pad=True,
                                   alwayssign=True)
        blist.append( ["J2000", ra_str, dec_str] )
    blist = np.array(blist, dtype=str)
    return blist


def cat_to_blist(fits_file, userows=[]):
    """
    Get coords of sources in fits_file (optionally selecting 
    a list of rows) and return a string format list of coords 
    to be used for beamlist 
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    cc_list = []

    u_ra  = tab['RA'].unit
    u_dec = tab['DEC'].unit
    
    for trow in tab:
        ra  = trow['RA']
        dec = trow['DEC']
    
        cc = SkyCoord(ra, dec, unit=(u_ra, u_dec))
        cc_list.append(cc)

    blist = make_beam_list(cc_list)

    return blist


def get_flux_sorted_rows(fits_file, smin=0):
    """
    Get flux sorted rows of fits file to give to blist

    Optionally give a cutoff
    """
    tab = Table.read(fits_file)
    
    flux = tab['Total_flux']

    xx = np.argsort( flux )[::-1]
    
    yy = np.where( flux[xx] >= smin )[0]

    return xx[yy] 
     

def flux_sorted_blist(fits_file, smin=0):
    """
    get blist in descending flux order down to smin
    """ 
    xx = get_flux_sorted_rows(fits_file, smin=smin)
    blist =  cat_to_blist(fits_file, userows=xx)

    return blist
