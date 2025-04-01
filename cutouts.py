import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle, Rectangle
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord


def one_beam_to_coord(ra_str, dec_str):
    """
    Convert one ra dec string pair from 
    the beam list to an astropy SkyCoord

    These have format:

    ra_str  = 'hh:mm:ss.ss'
    dec_str = 'dd.mm.ss.ss'

    which is annoying, i know 
    """
    # Careful
    dec_fix = dec_str.replace('.', ':', 2)

    cc = SkyCoord(ra_str, dec_fix, unit=(u.hourangle, u.deg), 
                  frame='fk5')

    return cc


def beamlist_to_coords(bfile):
    """
    Read in a beam list file (npy array)
    and convert to list of SkyCoord objects
    """
    beams = np.load(bfile)
    cc_list = []
    for bb in beams:
        cc = one_beam_to_coord(bb[1], bb[2])
        cc_list.append(cc)

    return cc_list


def get_cutout(cc, data, wcs, size):
    """
    Get the subarray of data centered on the 
    SkyCoord cc using the WCS wcs

    size is the size in pixels
    """
    # Get row and column index for coordinate
    r0, c0 = wcs.world_to_array_index(cc)

    cdat = data[ r0 - size//2 : r0 + size//2, 
                 c0 - size//2 : c0 + size//2 ]

    return cdat, (r0, c0)


def make_one_cutout(cdat, dx_deg, dy_deg, vmin=None, vmax=None,
                    title=None, outfile=None, idx_off=None, xx=None, 
                    radius=None):
    """
    Make a nice cutout image 
    """ 
    if outfile is not None:
        plt.ioff()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cx, cy = cdat.shape
    l_arc = -1 * (cx//2) * dx_deg * 3600
    r_arc = +1 * (cx//2) * dx_deg * 3600
    
    b_arc = -1 * (cy//2) * dy_deg * 3600
    t_arc = +1 * (cy//2) * dy_deg * 3600

    ext = [l_arc, r_arc, b_arc, t_arc] 

    im = ax.imshow(cdat, aspect='equal', origin='lower', 
                   vmin=vmin, vmax=vmax, interpolation='nearest',
                   extent=ext)

    if (idx_off is not None) and (len(idx_off)):
        rxy = (ext[0], ext[2])
        rw = -1 * np.abs(ext[1] - ext[0])
        rh = np.abs(ext[3] - ext[2])
        for ii, idx in enumerate(idx_off):
            y = idx[0] * dy_deg * 3600
            x = idx[1] * dx_deg * 3600
            if (x == 0) and (y == 0):
                ls = '-'
            else:
                ls = '--'
            p = Circle((x, y), radius, fc='none', ec='w', lw=2, ls=ls)
            ax.add_artist(p)
            ax.text(x, y + 1.25 * radius, f"{xx[ii]:03d}", 
                    color='w', ha='center', va='center', 
                    fontsize=12)
            
    ax.set_xlabel("Offset (arcsec)", fontsize=14)
    ax.set_ylabel("Offset (arcsec)", fontsize=14)
    
    if title is not None:
        ax.set_title(title, fontsize=16)

    cbar = plt.colorbar(im)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches=None)
        plt.close()
        plt.ion()

    else:
        plt.show()

    return


def get_beam_centers(cc_list, wcs):
    """
    Get beam centers in array indices
    """
    idx_arr = np.zeros( shape=(len(cc_list), 2) ) 
    for ii, cc in enumerate(cc_list):
        # Get row and column index for coordinate
        r0, c0 = wcs.world_to_array_index(cc)
        idx_arr[ii] = [r0, c0]
    return idx_arr


def find_beams_in_field( idx_arr, idx0, size, rpix):
    """
    Look for beams that fall within the field of view 
    centered on idx0 with side length size

    assume the beam has radiux rpix

    return array of center locs and ids
    """
    rs = idx_arr[:, 0]
    cs = idx_arr[:, 1]

    cond1 = np.abs( rs - idx0[0] ) < (size // 2) + rpix
    cond2 = np.abs( cs - idx0[1] ) < (size // 2) + rpix

    xx = np.where( cond1 & cond2 )[0]

    idx_offset = idx_arr[xx] - idx0

    return idx_offset, xx


def make_cutouts(cc_list, fits_file, size_arcsec=120, 
                 radius_arcsec=20, vmin=None, vmax=None):
    """
    Make cutouts from coord list cc_list
    """
    hdulist = fits.open(fits_file)
    hdu = hdulist[0]

    cdelt1 = hdu.header['CDELT1']
    cdelt2 = hdu.header['CDELT2']

    wcs = WCS(hdu.header)
    
    # Assuming we also have freq and stokes axis
    dat = hdu.data[0, 0, :, :]
    subwcs = wcs[0, 0, :, :]

    idx_arr = get_beam_centers(cc_list, subwcs)

    size = int( size_arcsec / (np.abs(cdelt1) * 3600) )
    rpix = int( radius_arcsec / (np.abs(cdelt1) * 3600) )

    for ii, cc in enumerate(cc_list):
        print(ii)
        bname = f"beam{ii:03d}"
        outname = f"{bname}_cutout.png"
        cdat, idx0 = get_cutout(cc, dat, subwcs, size)
        idx_offset, xx = find_beams_in_field( idx_arr, idx0, 
                                              size, rpix)
        make_one_cutout(cdat * 1e3, cdelt1, cdelt2, 
                        idx_off=idx_offset, xx=xx, 
                        radius=radius_arcsec, 
                        vmin=vmin, vmax=vmax,
                        title=bname, outfile=outname)

    hdulist.close()
    return
