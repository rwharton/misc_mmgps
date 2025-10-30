import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from astropy.visualization import make_lupton_rgb

from astropy.coordinates import SkyCoord
import astropy.units as u

def cscale(x, bias, contrast):
    """
    From https://github.com/glue-viz/ds9norm

    Apply bias and contrast scaling. Overwrite input

    Parameters
    ----------
    x : array
      Values between 0 and 1
    bias : float
    contrast : float

    Returns
    -------
    The input x, scaled inplace
    """
    x = np.subtract(x, bias, out=x)
    x = np.multiply(x, contrast, out=x)
    x = np.add(x, 0.5, out=x)
    x = np.clip(x, 0, 1, out=x)
    return x

def norm(x, vmin, vmax):
    """
    From https://github.com/glue-viz/ds9norm
    
    Linearly scale data between [vmin, vmax] to [0, 1]. Clip outliers
    """
    result = (x - 1.0 * vmin)
    result = np.divide(result, vmax - vmin, out=result)
    result = np.clip(result, 0, 1, out=result)
    return result


def linear_warp(x, vmin, vmax, bias, contrast):
    return cscale(norm(x, vmin, vmax), bias, contrast)


def log_warp(x, vmin, vmax, bias, contrast, exp=1000.0):
    black = x < vmin
    x = norm(x, vmin, vmax)
    x = np.multiply(exp, x, out=x)
    # sidestep numpy bug that masks log(1)
    # when out is provided
    x = np.add(x, 1.001, out=x)
    x = np.log(x, out=x)
    x = np.divide(x, np.log(exp + 1.0), out=x)
    x = cscale(x, bias, contrast)
    return x



def get_rgb(fitslist, vr_list, bias_list, contrast_list, 
            stretch='linear'):
    """
    Make rbg image cube with ds9 scalings
    """
    # get data
    dd_list = []
    for ii, fitsfile in enumerate(fitslist):
        hdulist = fits.open(fitsfile)
        dat = hdulist[0].data[:]
        vrange   = vr_list[ii]
        bias     = bias_list[ii]
        contrast = contrast_list[ii]
        if vrange is None:
            vrange = ( np.nanmin(dat), np.nanmax(dat) )

        if bias is None:
            bias = 0

        if contrast is None:
            contrast = 1

        if stretch == 'log':
            dd = log_warp(dat, vrange[0], vrange[1], 
                          bias, contrast)
        
        else: 
            dd = linear_warp(dat, vrange[0], vrange[1], 
                             bias, contrast)

        dd_list.append(dd)

    if len(dd_list) == 2:
        dd_list.append( np.zeros( dd_list[0].shape ) )

    rr = np.vstack( (dd_list[0].ravel(), 
                     dd_list[1].ravel(),
                     dd_list[2].ravel()) )

    new_shape = (dd_list[0].shape[0], dd_list[0].shape[1], 3)
    rgb = np.reshape(rr.T, new_shape)

    return rgb


def read_psr_data(infile):
    """
    Read in pulsar data from text file
    """
    names = []
    ra_str = []
    dec_str = []       
    pers = []
    dms = []
    rms = []
    his = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", " ", "\n"]:
                continue
            cols = line.split()
            names.append(cols[0])
            ra_str.append(cols[1])
            dec_str.append(cols[2])
            pers.append( float(cols[3]) )

            if cols[4] == "***":
                dms.append( 0 )
            else:
                dms.append( float(cols[4]) )

            if cols[5] == "***":
                rms.append( 0 )
            else:
                rms.append( float(cols[5]) )

            his.append( int(cols[8]) )

    cc_list = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
    
    names = np.array( names )
    pers = np.array( pers )
    dms = np.array( dms )
    rms = np.array( rms )
    his = np.array( his )

    return names, cc_list, pers, dms, rms, his
    

def make_plot(fitsfile, vrange=(-0.001, 0.01)):
    """
    make plot
    """
    hdu = fits.open(fitsfile)[0]
    wcs = WCS(hdu.header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(hdu.data, vmin=-0.001, vmax=0.01, origin='lower')
    ax.set_xlabel("Glon", fontsize=14)
    ax.set_ylabel("Glat", fontsize=14)

    gc_distance = 8.2 * u.kpc
    scalebar_length = 10 * u.pc
    scalebar_angle = (scalebar_length / gc_distance).to(
                      u.deg, equivalencies=u.dimensionless_angles()
                     )

    # Add a scale bar
    add_scalebar(ax, scalebar_angle, label="10 pc", color="k") 
    plt.show()
    return


def transform_lim( xlim, ylim, wcs ):
    """
    convert limits (in degrees) to pixels
    """
    ny, nx = wcs.array_shape
    
    cc_ll = SkyCoord( xlim[0], ylim[0], frame='galactic', 
                      unit=u.deg)
    cc_ur = SkyCoord( xlim[1], ylim[1], frame='galactic', 
                      unit=u.deg)

    print(cc_ll)
    print(cc_ur)
    
    pix_ll = wcs.world_to_array_index(cc_ll)
    pix_ur = wcs.world_to_array_index(cc_ur)

    pix_xlim = ( float(pix_ll[1]), float(pix_ur[1]) )
    pix_ylim = ( float(pix_ll[0]), float(pix_ur[0]) )

    return pix_xlim, pix_ylim


def make_plot_psrs(fitslist, psr_file, vr_list = [], bias_list = [], 
                   contrast_list=[], lon_lim=(1.28, 358.45), 
                   lat_lim=(-1.05,1.05), cval='rm', clog=False, 
                   crange=(None, None), bounds=None, 
                   nocolor='k', dark=False):
    """
    make plot
    """
    if dark:
        plt.style.use('dark_background')

    # get pulsar data
    names, cc_list, pers, dms, rms, highlights = read_psr_data(psr_file)
    
    # get ras, decs in fk5 deg
    glons  = cc_list.galactic.l.deg
    glats  = cc_list.galactic.b.deg

    # Setup figure and axis
    hdu = fits.open(fitslist[0])[0]
    wcs = WCS(hdu.header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    fig.set_figheight(8)
    fig.set_figwidth(12)

    # Get image data
    if len(fitslist) == 1:
        cmap = 'bone'
        dat = hdu.data
        vrange = vr_list[0]
        if vrange is None:
            vrange = (np.nanmin(dat), np.nanmax(dat))
        ax.imshow(dat, vmin=vrange[0], vmax=vrange[1], 
                  origin='lower', cmap=cmap, 
                  interpolation='nearest')

    elif (len(fitslist)==2) or (len(fitslist==3)):
        dat = get_rgb(fitslist, vr_list, bias_list, contrast_list, stretch='log')
        ax.imshow(dat, origin='lower', interpolation='nearest')

    else: 
        print("can only do one, two, or three images")
        return

    lcol = 'k'
    if dark:
        lcol = 'w'
        fig.set_facecolor('k')
        fig.set_edgecolor('k')
        #ax.set_frame_on(False)
        for key, spine in ax.spines.items():
            spine.set_color('w')
        
        
    ax.set_xlabel('$\\ell$', fontsize=18, color=lcol)
    ax.set_ylabel('$b$', fontsize=18, color=lcol)
    ax.tick_params(axis='both', labelcolor=lcol, color=lcol, 
                   labelsize=14, direction="in", length=10)

    ax.coords[0].set_major_formatter('d.dd')
    ax.coords[1].set_major_formatter('d.dd')

    ax.set_facecolor('k')

    mcol = None

    hnums = np.unique(highlights)  
    hnums = hnums[hnums > 0]
    hshapes = ['8', 's', 'o', 'd', '^', 'h']
   
    vals = dms
    if cval == 'rm':
        vals = rms
        lstr = '${\\rm RM}~({\\rm rad~m}^{-2})$'
        cmap = 'coolwarm'
    elif cval == 'per':
        vals = pers
        lstr = '${\\rm P}~({\\rm s})$'
        cmap = 'plasma_r'
    else:
        lstr = '${\\rm DM}~({\\rm pc~cm}^{-3})$'
        cmap = 'plasma_r'
        
    xx = np.where( np.abs(vals) > 0 )[0]
    yy = np.setdiff1d( np.arange(len(glons)), xx )

    val_min = np.min(vals[xx])
    val_max = np.max(vals[xx])
        
    if crange[0] is None:
        cmin = val_min
    else:
        cmin = crange[0]

    if crange[-1] is None:
        cmax = val_max
    else:
        cmax = crange[-1]

    if bounds is None:
        norm = colors.Normalize(vmin=cmin, vmax=cmax)
    else:
        cmin = bounds[0]
        cmax = bounds[-1]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256) 

    if (val_min < cmin) & (val_max > cmax):
        extend = 'both'
    elif (val_min < cmin) & (val_max <= cmax):
        extend = 'min'
    elif (val_min >= cmin) & (val_max > cmax):
        extend = 'max'
    else:
        extend = 'neither'
 
    ccm = matplotlib.colormaps.get_cmap(cmap)
    cax = ax.scatter(glons[xx], glats[xx], 
                     transform=ax.get_transform('galactic'), 
                     s=70, c=vals[xx], cmap=ccm, norm=norm)

    cbar = plt.colorbar(cax, shrink=0.8, extend=extend)
    cbar.ax.set_ylabel(lstr)
        
    if len(yy):
        ax.scatter(glons[yy], glats[yy], 
                   transform=ax.get_transform('galactic'), s=70,
                   marker='x', c=nocolor)

    for hh in hnums:
        zz = np.where( highlights == hh )[0]
        
        vvb = np.where(vals[zz] == 0)[0]
        vvg = np.setdiff1d( np.arange(len(zz)), vvb )
        
        ax.scatter(glons[zz[vvg]], glats[zz[vvg]], 
                   transform=ax.get_transform('galactic'), 
                   s=200, marker=hshapes[hh % 6], lw=2,  
                   facecolor='None', edgecolor=ccm(norm(vals[zz[vvg]])))
        
        ax.scatter(glons[zz[vvb]], glats[zz[vvb]], 
                   transform=ax.get_transform('galactic'), 
                   s=200, marker=hshapes[hh % 6], lw=2,  
                   facecolor='None', edgecolor=nocolor)
    
    xlim, ylim = transform_lim( lon_lim, lat_lim, wcs )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Make and add Scalebar
    gc_distance = 8.2 * u.kpc
    scalebar_length = 100 * u.pc
    scalebar_angle = (scalebar_length / gc_distance).to(
                      u.deg, equivalencies=u.dimensionless_angles()
                     )
    #add_scalebar(ax, scalebar_angle, label="100 pc", color=nocolor, 
    #             corner='bottom', width=5)
    
    plt.show()

    if dark:
        plt.style.use('default')
    return


psr_file = '/Users/wharton/work/MMGPS/GC/gc_psr_data.txt'

ir_bias_list = [0.5, 0.5]
ir_contrast_list = [2, 2]
ir_vr_list = [None, None]
ir_fitslist = ['/Users/wharton/work/evla_sgr/infrared/mips_mosaic_match.fits', 
              '/Users/wharton/work/evla_sgr/infrared/glimpse/GLM_00000+0000_mosaic_I4.fits']

L_bias_list = [None]
L_contrast_list = [None]
L_vr_list = [(-1e-3, 3e-3)]
radio_fitslist = ['/Users/wharton/work/MMGPS/GC/LBAND/Lband_gal.fits']

H_bias_list = [None]
H_contrast_list = [None]
H_vr_list = [None]
shassa_fitslist = ['/Users/wharton/work/evla_sgr/HALPHA/shassa_gc.fits']
