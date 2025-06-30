import matplotlib.pyplot as plt
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization.wcsaxes import add_beam, add_scalebar

from astropy.coordinates import SkyCoord
import astropy.units as u

def read_psr_data(infile):
    """
    Read in pulsar data from text file
    """
    names = []
    ra_str = []
    dec_str = []       
    dms = []
    rms = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", " ", "\n"]:
                continue
            cols = line.split()
            names.append(cols[0])
            ra_str.append(cols[1])
            dec_str.append(cols[2])
            if cols[3] == "***":
                dms.append( 0 )
            else:
                dms.append( float(cols[3]) )

            if cols[4] == "***":
                rms.append( 0 )
            else:
                rms.append( float(cols[4]) )

    cc_list = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
    
    names = np.array( names )
    dms = np.array( dms )
    rms = np.array( rms )

    return names, cc_list, dms, rms
    

def make_plot(fitsfile, vrange=(-0.001, 0.01)):
    """
    make plot
    """
    hdu = fits.open(fitsfile)[0]
    wcs = WCS(hdu.header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    ax.imshow(hdu.data, vmin=-0.001, vmax=0.01, origin='lower')
    ax.set_xlabel("RA", fontsize=14)
    ax.set_ylabel("Dec", fontsize=14)

    gc_distance = 8.2 * u.kpc
    scalebar_length = 10 * u.pc
    scalebar_angle = (scalebar_length / gc_distance).to(
                      u.deg, equivalencies=u.dimensionless_angles()
                     )

    # Add a scale bar
    add_scalebar(ax, scalebar_angle, label="10 pc", color="k") 
    plt.show()
    return
    

def make_plot_psrs(fitsfile, psr_file, vrange=(-0.001, 0.01), 
                   cval='rm'):
    """
    make plot
    """
    # get pulsar data
    names, cc_list, dms, rms = read_psr_data(psr_file)

    # get ras, decs in fk5 deg
    ras  = cc_list.fk5.ra.deg
    decs = cc_list.fk5.dec.deg

    # Get image
    hdu = fits.open(fitsfile)[0]
    wcs = WCS(hdu.header)
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    fig.set_figheight(8)
    fig.set_figwidth(8)

    ax.imshow(hdu.data, vmin=-0.001, vmax=0.005, origin='lower', 
              cmap='bone')
    ax.set_xlabel("RA", fontsize=14)
    ax.set_ylabel("Dec", fontsize=14)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_facecolor('k')

    # Add pulsars

    if cval == 'rm':
        xx = np.where( np.abs(rms) > 0 )[0]

        cax = ax.scatter(ras[xx], decs[xx], 
                         transform=ax.get_transform('fk5'), s=70,
                         c=rms[xx], cmap='coolwarm', 
                         vmin=-1.5e4, vmax=1.5e4)
        cbar = plt.colorbar(cax, shrink=0.8, extend='both')
        cbar.ax.set_ylabel('${\\rm RM}~({\\rm rad~m}^{-2})$')

        yy = np.setdiff1d( np.arange(len(ras)), xx )
        if len(yy):
            ax.scatter(ras[yy], decs[yy], 
                       transform=ax.get_transform('fk5'), s=70,
                       marker='x', c='w')

    elif cval == 'dm':
        xx = np.where( np.abs(dms) > 0 )[0]

        cax = ax.scatter(ras[xx], decs[xx], 
                         transform=ax.get_transform('fk5'), s=70,
                         c=dms[xx], cmap='plasma_r', 
                         )#vmin=500, vmax=1800)
        cbar = plt.colorbar(cax, shrink=0.8, extend='neither')
        cbar.ax.set_ylabel('${\\rm DM}~({\\rm pc~cm}^{-3})$')

        yy = np.setdiff1d( np.arange(len(ras)), xx )
        if len(yy):
            ax.scatter(ras[yy], decs[yy], 
                       transform=ax.get_transform('fk5'), s=70,
                       marker='x', c='w')

    """ 
    for ii, nn in enumerate(names): 
        ax.text(ras[ii] - 0.01, decs[ii], nn, 
                transform=ax.get_transform('fk5'), 
                color='w', va='center', ha='left')
    """

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    dsize = hdu.data.shape
    xfracs = (0.2, 0.75) # S_small (0.1, 0.8)  L_small (0.35, 0.6) 
    yfracs = (0.25, 0.9) # S_small (0.15, 0.9) L_small (0.45, 0.7) 
    ax.set_xlim( dsize[1] * xfracs[0], xfracs[1] * dsize[1] )
    ax.set_ylim( dsize[0] * yfracs[0], yfracs[1] * dsize[0] )

    # Make and add Scalebar
    gc_distance = 8.2 * u.kpc
    scalebar_length = 20 * u.pc
    scalebar_angle = (scalebar_length / gc_distance).to(
                      u.deg, equivalencies=u.dimensionless_angles()
                     )
    add_scalebar(ax, scalebar_angle, label="20 pc", color="w", 
                 corner='bottom left')
    
    plt.show()
    return
