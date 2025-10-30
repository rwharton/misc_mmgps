import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
from astropy.io import fits
from astropy.table import Table
from glob import glob
import bdsf
import os
import sys
import shutil


def get_pol_files(indir, basename, pols="IQUV", suffix=None):
    """
    Get list of paths to the polarization FITS files.

    Will glob on indir/basename*[pols].fits 

    Assumes polarization is indicated by the last letter 
    before the fits extension, so: "[whatever][IQUV][suffix].fits"

    Returns list of file paths for each pol letter in "pols", 
    in that order.  So "IV" gives [Ifile, Vfile] and "VUQI" 
    gives [Vfile, Ufile, Qfile, Ifile]
    
    Also returns pol_list 
    """
    pol_list = list(pols)
    pol_files = []  

    for pp in pol_list:
        if suffix is not None:
            mstr = f"{indir}/{basename}*{pp}{suffix}.fits"
        else:
            mstr = f"{indir}/{basename}*{pp}.fits"
        mpaths = glob(mstr)
        if len(mpaths) == 0:
            print(f"No files found matching {mstr}")
            pol_files.append(None)
        elif len(mpaths) > 1:
            print(f"More than one file found matching {mstr}!")
            print(mpaths)
            print("Using first one...")
            pol_files.append(mpaths[0])
        else:
            pol_files.append(mpaths[0])

    if None in pol_files:
        print("At least one pol missing...")
        sys.exit(0)

    return pol_files, pol_list


def check_and_copy(src, dst):
    if not os.path.exists( dst ):
        shutil.copyfile(src, dst)
    else:
        print(f"File exists: {dst}")
    return 


def setup_workdir(workdir, indir, inbase, suffix=None, outbase=None):
    """
    Make directory where we will do our processing, 
    """
    # Get files and polarizations
    # Need all pols in IQUV order for combined fits file
    # to make sense
    pol_files, pol_list = get_pol_files(indir, inbase, pols="IQUV", suffix=suffix)

    # Make work directory if required 
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # If outbase not given, use inbase for naming 
    if outbase is None:
        outbase = inbase

    # Copy over Stokes I fits file to work directory 
    # and rename to "IQUV"
    opath = f"{workdir}/{outbase}-IQUV.fits"
    check_and_copy(pol_files[0], opath)

    # Open up output file to modify
    h = fits.open(opath, mode='update')

    # Now get data from all pols
    dd = h[0].data
    for ii in range(1, 4):
        h_ii = fits.open(pol_files[ii]) 
        dd = np.vstack( (dd, h_ii[0].data) )
        h_ii.close()

    # Now replace the full file data with the 
    # 4 pols of data
    h[0].data = dd
    
    # close to make changes
    h.close()   
    
    return opath


def srcfind_pol(fitsfile, cat_fmt='fits', cat_type='srl', regfile=True):
    """
    Run the pybdsf source finding on polarization
    images
    """
    img = bdsf.process_image(fitsfile, polarisation_do=True, 
                             pi_fit=False, rms_box=(200, 50))
    img.write_catalog(format=cat_fmt, catalog_type=cat_type)
    if regfile:
        img.write_catalog(format='ds9', catalog_type=cat_type)

    img.export_image(img_type='rms')
   
    return 


def get_reg_str(ra, dec, radius, shape):
    loc_str = f"{ra:.6f}, {dec:.6f}"
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
        out_str = f"{shape}({loc_str}) #point=x\n"
    else:
        out_str = f"{shape}({loc_str}, {size_str})\n"

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


def get_pol_regions(fits_table, basename, radius=20, shape='circle'):
    """
    make a plot of the polarized sources
    """ 
    tab = Table.read(fits_table)
    
    I = tab['Total_flux'] * 1e3
    I_err = tab['E_Total_flux'] * 1e3

    LPF = tab['Linear_Pol_frac']
    LPF_ehi = tab['Ehigh_Linear_Pol_frac']
    LPF_elo = tab['Elow_Linear_Pol_frac']

    CPF = tab['Circ_Pol_Frac']
    CPF_ehi = tab['Ehigh_Circ_Pol_Frac']
    CPF_elo = tab['Elow_Circ_Pol_Frac']

    xx = np.where(CPF > 0)[0]
    yy = np.where(LPF > 0)[0] 

    print(f"Total sources = {len(tab)}")
    print(f"Circ Pol = {len(xx)}")
    print(f"Lin Pol = {len(yy)}")

    # Write circ in white
    out_circ = f"{basename}_circ.reg"
    table_to_reg(fits_table, out_circ, use_rows=xx, color='white', 
                 radius=radius, shape=shape)
    
    # Write lin in yellow
    out_lin = f"{basename}_lin.reg"
    table_to_reg(fits_table, out_lin, use_rows=yy, color='yellow', 
                 radius=radius*1.5, shape=shape)

    return


def print_tab_info(fits_table):
    """
    make a plot of the polarized sources
    """ 
    tab = Table.read(fits_table)
    
    I = tab['Total_flux'] * 1e3
    I_err = tab['E_Total_flux'] * 1e3

    LPF = tab['Linear_Pol_frac']
    LPF_ehi = tab['Ehigh_Linear_Pol_frac']
    LPF_elo = tab['Elow_Linear_Pol_frac']

    CPF = tab['Circ_Pol_Frac']
    CPF_ehi = tab['Ehigh_Circ_Pol_Frac']
    CPF_elo = tab['Elow_Circ_Pol_Frac']

    xx = np.where(CPF > 0)[0]
    yy = np.where(LPF > 0)[0] 

    print(f"Total sources = {len(tab)}")
    print(f"Circ Pol = {len(xx)}")
    print(f"Lin Pol = {len(yy)}")

    return


def make_polfrac_plot(fits_table, rms_ujy=14, title=None):
    """
    make a plot of the polarized sources
    """ 
    tab = Table.read(fits_table)
    
    I = tab['Total_flux'] * 1e3
    I_err = tab['E_Total_flux'] * 1e3

    LPF = tab['Linear_Pol_frac']
    LPF_ehi = tab['Ehigh_Linear_Pol_frac']
    LPF_elo = tab['Elow_Linear_Pol_frac']

    CPF = tab['Circ_Pol_Frac']
    CPF_ehi = tab['Ehigh_Circ_Pol_Frac']
    CPF_elo = tab['Elow_Circ_Pol_Frac']

    xx = np.where(CPF > 0)[0]
    yy = np.where(LPF > 0)[0] 

    print(f"Total sources = {len(tab)}")
    print(f"Circ Pol = {len(xx)}")
    print(f"Lin Pol = {len(yy)}")


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.errorbar(I[xx], CPF[xx], xerr=I_err[xx], 
                yerr=(CPF_elo[xx], CPF_ehi[xx]), 
                marker='o', ms=12, ls='', 
                label="Circ Pol Frac")

    ax.errorbar(I[yy], LPF[yy], xerr=I_err[yy], 
                yerr=(LPF_elo[yy], LPF_ehi[yy]), 
                marker='s', ms=8, ls='', 
                label="Lin Pol Frac")

    sI = np.logspace(-4, 4, 100)
    fp_lim = 5 * rms_ujy * 1e-3 / sI
    fp_lim[ fp_lim > 1 ] = 1

    #ax.set_xlim(0.1, 1000)
    #ax.set_ylim(2e-3, 1.5)

    ax.plot(sI, fp_lim, c='k', ls='--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:g}'.format(x)))

    ax.set_xlabel("Stokes I Flux Density (mJy)", fontsize=12)
    ax.set_ylabel("Polarization Fraction", fontsize=12)

    if title is not None:
        ax.set_title(title, fontsize=14)

    
    plt.legend()
    plt.show()

    return


def all_stokes_frac_fov(fits_table, Imin=0):
    """
    Make a plot showing fractional QUV 
    on the sky 
    """
    full_tab = Table.read(fits_table)

    # Imin cutoff
    xx = np.where( full_tab['Total_flux'] > Imin )[0]
    tab = full_tab[xx]

    I = tab['Total_flux']
    Q = tab['Total_Q']
    U = tab['Total_U']
    V = tab['Total_V']

    Ie = tab['E_Total_flux']
    Qe = tab['E_Total_Q']
    Ue = tab['E_Total_U']
    Ve = tab['E_Total_V']

    fQ = Q / I
    fU = U / I 
    fV = V / I

    qq = np.where( np.abs(Q) > 3 * Qe )[0]
    uu = np.where( np.abs(U) > 3 * Ue )[0]
    vv = np.where( np.abs(V) > 3 * Ve )[0]
    

    ra = tab['RA']
    dec = tab['DEC']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.scatter(ra[vv], dec[vv], c=fV[vv], s=np.sqrt(I[vv] / 1e-5), 
                     vmin=-0.1, vmax=0.1, cmap='coolwarm')
    cbar = fig.colorbar(cax)
    

    plt.show()
    return 
    

def stokes_frac_fov(fits_table, Imin=0, pol='Q', 
                    cc0=(0, 0), size=-1, rlist=[], 
                    add_nulls=True):
    """
    Make a plot showing fractional polarized flux 
    on the sky (pol = Q, U, or V
    """
    full_tab = Table.read(fits_table)

    pol = pol.upper()
    if pol not in "QUV":
        print(f"pol {pol} not valid.  Must be one of Q, U, V")
        return

    # Imin cutoff
    xx = np.where( full_tab['Total_flux'] > Imin )[0]
    tab = full_tab[xx]
    
    pcol  = f"Total_{pol}"
    epcol = f"E_Total_{pol}"


    I = tab['Total_flux']
    Ie = tab['E_Total_flux']
    
    P  = tab[pcol]
    Pe = tab[epcol]

    fP = P / I

    yy = np.where( np.abs(P) > 3 * Pe )[0]

    ra = tab['RA']
    dec = tab['DEC']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    dra = (ra - cc0[0]) * np.abs( np.cos( dec * np.pi/180.) )
    ddec = dec - cc0[1]

    cax = ax.scatter(dra[yy], ddec[yy], c=fP[yy], s=np.sqrt(I[yy] / 1e-5), 
                     vmin=-0.1, vmax=0.1, cmap='coolwarm')
    cbar = fig.colorbar(cax)
    ax.set_title(f"Fractional Stokes {pol}", fontsize=14) 
   
    if len(rlist): 
        for rr in rlist:
            p = plt.Circle((0,0), radius=rr, fc='none', ec='k', 
                           ls='-', zorder=-1)
            ax.add_artist(p)
            if add_nulls:
                pn = plt.Circle((0,0), radius=rr * 2.5, fc='none', ec='k', 
                            ls='--', zorder=-1)
                ax.add_artist(pn)
                

    if size > 0:
        ax.set_xlim(0.5 * size, -0.5 * size)
        ax.set_ylim(-0.5 * size, 0.5 * size)

    ax.grid(alpha=0.3, ls='--')
    ax.set_aspect('equal')
    

    plt.show()
    return 


# L1825 pars
#"""
ra0 = -174.317916666667
dec0 = -64.2009722222222
size = 3.33
fc = 1.278984375
bw = 0.758404296
freqs = fc + np.array([-0.5, 0.5]) * bw
rlist = 1.0 * (1.4/freqs) / 2.
#"""

# S0748  / S0759
"""
ra0  = -174.069958333333
dec0 = -64.0645555555556
ra1  = -173.413625
dec1 = -64.0940833333333
size = 1.33
fc = 2.4011
bw = 0.775
freqs = fc + np.array([-0.5, 0.5]) * bw
rlist = 0.8 * (2.0/freqs) * bw / 2.
"""


# J1306 pars
"""
ra0 = -163.414458333333
dec0 = -60.7306388888889
size = 3.33
fc = 1.278984375
bw = 0.758404296
freqs = fc + np.array([-0.5, 0.5]) * bw
rlist = 1.0 * (1.4/freqs) / 2.
"""


