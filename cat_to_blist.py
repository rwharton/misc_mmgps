import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from collections import Counter


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


def size_filter(fits_file, max_dc_maj=1.0, userows=[]):
    """
    Read in fits file (optionally using rows specified 
    by use_rows) and select only those remaining rows where
    the source has a deconvolved major axis size less than 
    max_dc_maj arcseconds
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    dc_maj = tab['DC_Maj'] * 3600
    
    xx = np.where( dc_maj <= max_dc_maj )[0]

    if len(userows):
        xx_out = userows[xx]
    else:
        xx_out = xx

    return xx_out


def isl_filter(fits_file, max_comp=1, min_flux=0, userows=[]):
    """
    Set a maximum number of components a given 
    'island' can have.  These are clusters of 
    sources that are close to one another. While 
    real component certainly exist (e.g. lobes 
    or resolved structure), these can often be 
    artifacts near bright sources

    If min_flux > 0, then we only consider islands 
    with the brightest component brigher than min_flux.
    Other islands are not filtered.  This can be useful 
    if you mainly want to focus on artifacts. 
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    isl_ids = tab['Isl_id']
    flux = tab['Total_flux']

    isl_count = Counter(isl_ids)
  
    idx_bad = []
    for isl_id, isl_num in isl_count.items():
        if isl_num <= max_comp:
            continue

        # get row nums of sources in that isl
        xx = np.where(isl_ids == isl_id)[0]

        # get fluxes, skip if max flux is less than 
        # the min flux 
        flux_isl = flux[xx]
        if np.max(flux[xx]) <= min_flux:
            continue

        # sort by descending flux
        yy = np.argsort(flux_isl)[::-1]
        
        # Get the indices beyond max_comp that 
        # we want to remove
        xx_bad = xx[ yy[max_comp:] ]

        idx_bad += xx_bad.tolist()

    idx_good = np.setdiff1d( np.arange(len(tab)), np.array(idx_bad) )

    if len(userows):
        xx_out = userows[idx_good]
    else:
        xx_out = idx_good

    return xx_out


def filter_nearby(fits_file, min_sep=0.0, min_flux=0, 
                  userows=[]):
    """
    sort by flux and filter out sources that are closer 
    than min_sep from a brighter source
    
    can optionally can only do this down to min_flux
    """
    full_tab = Table.read(fits_file)

    if len(userows):
        tab = full_tab[userows]
    else:
        tab = full_tab

    flux = tab['Total_flux']
    
    u_ra  = tab['RA'].unit
    u_dec = tab['DEC'].unit
    
    ras  = tab['RA']
    decs = tab['DEC']

    coords = SkyCoord(ras, decs, unit=(u_ra, u_dec))
     
    idx_bad = []
    xx = np.argsort(flux)[::-1]

    flux_srt = flux[xx]
    coord_srt = coords[xx]

    for ii, fsrt in enumerate(flux_srt):
        if fsrt <= min_flux:
            break
    
        if ii == len(flux_srt) - 1:
            break

        if ii in idx_bad:
            continue
        
        cc_ii = coord_srt[ii]
        cc_rest = coord_srt[ii+1:]

        seps = cc_ii.separation(cc_rest).arcsec 

        yy = np.where(seps <=  min_sep)[0]
        
        if len(yy):
            cur_bad = (yy+ii+1).tolist()
            idx_bad += cur_bad

    idx_bad_arr = np.unique( np.array( idx_bad ) )
    idx_good = np.setdiff1d( np.arange(len(xx)), idx_bad_arr )
    
    if len(userows):
        xx_out = userows[xx[idx_good]]
    else:
        xx_out = xx[idx_good]

    return xx_out

    
def get_flux_sorted_rows(fits_file, smin=0, point=True):
    """
    Get flux sorted rows of fits file to give to blist

    Optionally give a cutoff
    """
    tab = Table.read(fits_file)
    
    flux = tab['Total_flux']

    xx = np.argsort( flux )[::-1]
    
    yy = np.where( flux[xx] >= smin )[0]

    idx = xx[yy]

    if 'Resolved' in tab.colnames:
        zz = np.where(tab['Resolved'][idx] == False)[0]
        idx = idx[zz]

    return idx
     

def flux_sorted_blist(fits_file, smin=0, point=True, regfile=None):
    """
    get blist in descending flux order down to smin
    
    if res == True, only select point sources (unresolved)
    """ 
    xx = get_flux_sorted_rows(fits_file, smin=smin, point=True)
    blist =  cat_to_blist(fits_file, userows=xx)

    if regfile is not None:
        table_to_reg(fits_file, regfile, use_rows=xx, 
                     radius=20, color='white', shape='circle')

    return blist


def get_reg_str(ra, dec, radius, shape, cc_str=False, label=None):
    # Are ra,dec already strings?
    if cc_str == False:
        loc_str = f"{ra:.6f}, {dec:.6f}"
    else:
        loc_str = f"{ra}, {dec}"

    if shape == "circle":
        size_str = f"{radius}\""
    elif shape == "box":
        size_str = f"{2 * radius}\", {2 * radius}\""
    elif shape == "point":
        pass
    else:
        print("Shape must be circle or box")
        return ""

    if shape == "point":
        out_str = f"{shape}({loc_str}) #point=x"
    else:
        out_str = f"{shape}({loc_str}, {size_str})"

    if label is not None:
        lstr = f"text={{{label}}}"
        if shape == "point":
            out_str += f" {lstr}\n"
        else:
            out_str += f" #{lstr}\n"

    else:
        out_str += "\n"

    return out_str


def table_to_reg(fits_table, reg_file, use_rows=None, col_add=None,
                 radius=20, color='white', shape='circle'):
    """
    Read in a fits_table and make a ds9 region file
    from its contents.  Optionally only use rows
    given in use_rows (def: None = use all)

    make regions circles with radius in arcsec

    col_add = string to append to column name (only needed)
    when using matched columns
    """
    slist = ["circle", "box", "point"]
    if shape not in slist:
        print("Shape must be one of: " + ",".join(slist))
        return

    hdr = "" +\
          "# Region file format: DS9 version 4.0\n" +\
          f"global color={color} font=\"helvetica 10 normal\" select=1 "+\
          "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n"+\
          "fk5\n"

    full_tab = Table.read(fits_table)
    if use_rows is not None:
        tab = full_tab[use_rows]
    else:
        tab = full_tab

    if col_add == None:
        ra_col = "RA"
        dec_col = "DEC"

    else:
        ra_col = f"RA_{col_add}"
        dec_col = f"DEC_{col_add}"

    with open(reg_file, 'w') as fout:
        fout.write(hdr)
        for ii, row in enumerate(tab):
            ra = row[ ra_col ]
            dec = row[ dec_col ]
            out = get_reg_str(ra, dec, radius, shape)
            fout.write(out)

    return


def blist_to_reg(blist, reg_file, radius=20, 
                 color='white', shape='circle'):
    slist = ["circle", "box", "point"]
    if shape not in slist:
        print("Shape must be one of: " + ",".join(slist))
        return

    hdr = "" +\
          "# Region file format: DS9 version 4.0\n" +\
          f"global color={color} font=\"helvetica 10 normal\" select=1 "+\
          "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n"+\
          "fk5\n"

    with open(reg_file, 'w') as fout:
        fout.write(hdr)
        for ii, col in enumerate(blist):
            ra_str = col[1]
            dec_str = col[2]
            dec_str = ':'.join(dec_str.split('.', 2))
            out = get_reg_str(ra_str, dec_str, radius, shape, 
                              cc_str=True, label=f"{ii}")
            fout.write(out)

    return


def get_spec(tab, idx, nchan=8):
    """
    go to row idx of table tab and extract 
    the spectrum 

    May not have values for every frequency
    """
    freqs = []
    fluxes = []
    
    for ii in range(nchan):
        freq_key = f"Freq_ch{ii:d}"
        flux_key = f"Aperture_flux_ch{ii:d}"
        
        freq = tab[idx].get(freq_key, -1)
        flux = tab[idx].get(flux_key, -1)

        print(freq, flux)

        if (freq == -1) or (freq == np.ma.is_masked(freq)):
            continue
        
        if (flux == -1) or np.ma.is_masked( flux ):
            continue

        freqs.append(freq)
        fluxes.append(flux)

    freqs = np.array(freqs)
    fluxes = np.array(fluxes)

    alpha = tab[idx]['Alpha']
    e_alpha = tab[idx]['E_Alpha']
    print(f"spindex = {alpha:.2f} ({e_alpha:.2f})") 

    return freqs, fluxes
        
        
    
    
