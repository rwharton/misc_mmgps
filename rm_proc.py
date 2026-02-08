import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import subprocess as sub
import os
import glob
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import json

def blist_to_ccs(blist_file):
    """
    convert beam file to ccs
    """
    ras = []
    decs = []
    blist = np.load(blist_file)
    for ii, col in enumerate(blist):
        ra_str = col[1]
        dec_str = col[2]
        dec_str = ':'.join(dec_str.split('.', 2))
        ras.append(ra_str)
        decs.append(dec_str)

    ccs = SkyCoord(ras, decs, unit=(u.hourangle, u.deg))

    return ccs


def running_median(dd, nwin=30):
    """
    Calculate running median with window size nwin bins 
    """
    mdd = np.copy(dd)
    for ii in range( 0 + nwin//2, len(dd) - nwin//2 ):
        mdd[ii] = np.median( dd[ii - nwin//2 : ii + nwin//2 ] )

    return mdd


def avg_chan(dat, edat, nchan=4):
    """
    average data by nchan chans
    """
    Np = len(dat) // nchan
    dd = np.reshape( dat[: Np * nchan], (-1, nchan) )
    dd_avg = np.mean(dd, axis=1)
    
    edd = np.reshape( edat[: Np * nchan], (-1, nchan) )
    edd_avg = np.sqrt(np.sum( edd**2, axis=1 )) / nchan #/ 10
    
    return dd_avg, edd_avg


def avg_chan_darr(freqs, darr, nchan=4):
    """
    Run channel averaging on darr
    """    
    Np = len(freqs) // nchan
    favg = np.mean( np.reshape(freqs, (-1, nchan)), axis=1 )

    darr_avg = np.zeros( (len(favg), darr.shape[1]), dtype='float')
    for ii in range(4):
        dd_ii, edd_ii = avg_chan(darr[:, 2 * ii], darr[:, 2 *ii+1], nchan=nchan)
        darr_avg[:, 2 * ii] = dd_ii
        darr_avg[:, 2 * ii+1] = edd_ii

    return favg, darr_avg  


def get_bdat(bfile):
    """
    Load in bdat file and return IQUV
    """
    bdat = np.load(bfile)

    I = np.mean(bdat[0, :, :], axis=0)
    Q = np.mean(bdat[1, :, :], axis=0)
    U = np.mean(bdat[2, :, :], axis=0)
    V = np.mean(bdat[3, :, :], axis=0)
    
    #Q -= np.mean(Q)
    #U -= np.mean(U)

    Nt = bdat.shape[1]
    Nf = bdat.shape[2]

    # Uncertainty... this is probably not right
    dI = np.std(bdat[0, :, :], axis=0) / np.sqrt(Nt)
    dQ = np.std(bdat[1, :, :], axis=0) / np.sqrt(Nt)
    dU = np.std(bdat[2, :, :], axis=0) / np.sqrt(Nt)
    dV = np.std(bdat[3, :, :], axis=0) / np.sqrt(Nt)

    dlist = [ I, dI, Q, dQ, U, dU, V, dV ]

    darr = np.vstack( dlist ).T

    return darr


def get_idat(ifile):
    """
    Get imaging data spectrum from Aritra's dat file
    """
    idat = np.loadtxt(ifile, dtype='float')

    if len(idat) == 0:
        freqs = []
        darr = []

    else:
        freqs = idat[:, 0] * 1e9
        darr  = idat[:, 1:9]

    return [freqs, darr]


def darr_to_txt(outfile, darr, freqs):
    """
    Write spectra to text file format that 
    can be passed to RMSynth1D

    'RMsynth1D expects the input to be the 
     form of an ASCII text file, with 7 
     columns: frequency (in Hz), Stokes I, 
     Stokes Q, Stokes U, error in I, error in Q, 
     error in U. Each row is a single channel. 
     No header is expected.'
    
    Can also provide 5 columns and skip I

    Can also mask with NaNs
    """
    with open(outfile, 'w') as fout:
        for ii, freq in enumerate(freqs):
            xx = [0, 2, 4, 1, 3, 5]
            I, Q, U, dI, dQ, dU = darr[ii, xx]
            
            ostr = f"{freq:12.1f} " +\
                   f"{I:10.6f} {Q:10.6f} {U:10.6f} " +\
                   f"{dI:10.6f} {dQ:10.6f} {dU:10.6f} \n"

            fout.write(ostr)

    return


def run_rmsynth1d(infile, phimax=None):
    """
    Run rmsynth1d on the input text file infile 
    (assumed to be in proper format).  Optionally 
    include a phimax, otherwise the default will 
    be used (which depends on particular frequency 
    channel arrangement)
    """
    phi_str = ""
    if phimax is not None:
        phi_str = f"-l {phimax:.2f}"
    
    cmd = f"rmsynth1d -i -S {phi_str} {infile}"
    print(cmd)
    
    try:
        ret = sub.run(cmd, shell=True, check=True)
    except sub.CalledProcessError:
        print(f"cmd failed: {cmd}")
    except:
        print("Something else failed somehow")

    return ret.returncode 


def run_rmclean1d(infile, niter, threshold):
    """
    Run rmclean1d on the data set given 
    by the input text file.  Assumes that 
    rmsynth1d has already been run.  

    niter = number of clean interations

    threshold = cleaning threshold in Jy (if positive)
                or in sigma (if negative) 
    """
    cmd = f"rmclean1d -S -n {niter} -c {threshold} {infile}"
    print(cmd)
    
    try:
        ret = sub.run(cmd, shell=True, check=True)
    except sub.CalledProcessError:
        print(f"cmd failed: {cmd}")
    except:
        print("Something else failed somehow")

    return ret.returncode 


def rm_clean_many(npy_files, freq_file, mask_file=None, outdir='.', 
                  niter=200, threshold=-2, phimax=None):
    """
    Given a list of npy data files, run rm clean on 
    all and output to individual directories
    """
    for npy_file in npy_files:
        fname = npy_file.split('/')[-1]
        basenm = fname.split('.npy')[0]
        
        # directory where clean data will go
        bdir = f"{outdir}/{basenm}"
        if not os.path.exists(bdir):
            os.mkdir(bdir)

        # read in npy data and write to text 
        # file for rm synthesis and clean
        freqs = np.load(freq_file)
        darr = get_bdat(npy_file)   
        if mask_file is not None:
            mask = np.load(mask_file)
            xx = np.where( mask )[0]
        else:
            xx = np.arange(len(freqs))
        dat_file = f"{bdir}/{basenm}.txt"
        darr_to_txt(dat_file, darr[xx], freqs[xx])

        # run rm synthesis
        ret1 = run_rmsynth1d(dat_file, phimax=phimax)
        
        # run rm clean
        ret2 = run_rmclean1d(dat_file, niter, threshold)
    
    return


def read_FDF(datfile):
    """
    Read in faraday depth spectra

    Data in form (phi_rad_m2, real spec, imag spec)
    """
    dat = np.loadtxt(datfile)
    phis = dat[:, 0]
    r_spec = dat[:, 1]
    i_spec = dat[:, 2]
    return phis, r_spec, i_spec


def make_clean_plot(datfile, rm=None, show_ri=False, xlim=None,
                    title=None, outfile=None):
    """
    Make a nice plot of clean FDF
    
    mark known rm at `rm` if not none
    """
    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)

    phis, r_spec, i_spec = read_FDF(datfile)

    # Put in uJy
    r_spec *= 1e6
    i_spec *= 1e6

    m_spec = np.sqrt(r_spec**2 + i_spec**2)
    
    if show_ri:
        ax.plot(phis, r_spec, lw=1, alpha=0.5, label="Real")
        ax.plot(phis, i_spec, lw=1, alpha=0.5, label="Imag")
    
    ax.plot(phis, m_spec, lw=1.5, c='k', label="Mag")

    if rm is not None:
        ax.axvline(x=rm, ls='--', c='r', alpha=0.5, zorder=0)

    ax.set_xlabel("$\\phi~({\\rm rad~m}^{-2})$", fontsize=16)
    ax.set_ylabel("${\\rm Flux~Density}~(\\mu{\\rm Jy~beam}^{-1})$", 
                  fontsize=14)

    max_val = np.max(m_spec)
    ax.set_ylim(-0.1 * max_val, 1.2 * max_val)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    plt.tick_params(axis='both', which='major', 
                    direction='in', length=7, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)
    
    plt.tick_params(axis='both', which='minor', 
                    direction='in', length=4, 
                    top=True, bottom=True, 
                    left=True, right=True, 
                    labelsize=12)

    plt.grid(alpha=0.2)

    if title is not None:
        plt.title(title, fontsize=16)

    if outfile is not None:
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    return


def make_psr_rm_plots(bnums, bdir, nbeams, 
                      psr_names, psr_nums, psr_rms):
    """
    Make plots for pulsars
    """
    for bnum in bnums:
        bpsr = (bnum // nbeams) * nbeams
        print(bpsr)
        xx = np.where( psr_nums == bpsr)[0]
        print(xx)
        if len(xx) == 0:
            continue
        pname = psr_names[xx[0]]
        prm = psr_rms[xx[0]]
    
        #bname = f"beam{bnum:03d}_full_split"
        bname = f"beam{bnum:03d}_full_split"
        datfile = f"{bdir}/{bname}/{bname}_FDFclean.dat"
        print(datfile)
   
        outfile = f"{pname}_beam{bnum:03d}.png"
        xlim = (-7e4, 7e4)
        make_clean_plot(datfile, rm=prm, show_ri=False, xlim=xlim,
                        title=pname, outfile=outfile)
        
    return


def make_rm_plots(bnums, bdir, xlim=(-1e4, 1e4)):
    """
    Make rm plots
    """
    for bnum in bnums:
        bname = f"beam{bnum:03d}_full"
        datfile = f"{bdir}/{bname}/{bname}_FDFclean.dat"
        print(datfile)
   
        outfile = f"beam{bnum:03d}_rm_clean.png"
        make_clean_plot(datfile, rm=None, show_ri=False, xlim=xlim,
                        title=None, outfile=outfile)
        
    return


def json_to_cat(bnum, json_file):
    """
    read in json file results for clean, and convert 
    to a string to be print in catalog
    """
    with open(json_file, "r") as fin:
        dd = json.load(fin)

    rm = dd['phiPeakPIfit_rm2']
    rm_err = dd['dPhiPeakPIfit_rm2']

    # Convert to mJy
    PI = dd['ampPeakPIfit'] * 1e3
    PI_err = dd['dAmpPeakPIfit'] * 1e3

    PI_snr = dd['snrPIfit']

    PA0 = dd['polAngle0Fit_deg']
    PA0_err = dd['dPolAngle0Fit_deg']

    ostr = f"{bnum:03d}   {rm:10.2f}  {rm_err:10.2f}  " +\
           f"{PI:8.3f}  {PI_err:8.3f}  {PI_snr:7.1f}  "+\
           f"{PA0:8.2f}  {PA0_err:8.2f} \n"

    return ostr
    

def make_catalog(bdir, outfile):
    """
    Parse all the RMclean json files for every beam
    and write values of interest to the catalog
    """
    blist = glob.glob(f"{bdir}/beam*")
    blist.sort()

    bnames = [bb.split('/')[-1] for bb in blist]
    bnums  = [int( ff.lstrip('beam').split('_')[0] ) for ff in bnames]
    
    # get cat lines
    olines = []
    for ii, bdir in enumerate(blist):
        jfile = f"{bdir}/{bnames[ii]}_RMclean.json"
        ostr = json_to_cat(bnums[ii], jfile)
        olines.append(ostr)

    # Write catalog 
    with open(outfile, "w") as fout:
        hdr1 = f"#{'Beam':^5} {'phi':^10}  {'phi_err':^10}  "+\
               f"{'PI':^8}  {'PI_err':^8}  {'SNR':^7}  "+\
               f"{'PA0':^8}  {'PA0_err':^8} \n"
        hdr2 = f"#{'':^5} {'(rad/m^2)':^10}  {'(rad/m^2)':^10}  "+\
               f"{'(mJy)':^8}  {'(mJy)':^8}  {'':^7}  "+\
               f"{'(deg)':^8}  {'(deg)':^8} \n"
        hdr3 = "#" + "="*len(hdr1) + "\n"
        fout.write(hdr1)
        fout.write(hdr2)
        fout.write(hdr3)
        for oline in olines:
            fout.write(oline)
    return


def plot_rms(infile, rm_txt, cc0, rm0=0, vmin=None, vmax=None, 
             cattype='blist', snr_min=-1, frame='fk5', offset=True):
    """
    Make a plot using coords from cat_fits and 
    rm values from rm_txt centered on cc0
    """
    if cattype == 'blist':
        cc = blist_to_ccs(infile)
    else:
        tab = Table.read(infile)
        cc = SkyCoord(tab["RA"], tab["DEC"])

    rdat = np.loadtxt(rm_txt)
    bnums = rdat[:, 0].astype('int')
    rms = rdat[:, 1]
    pflux = rdat[:, 3]
    snrs = rdat[:, 5]
    xx = np.where( snrs > snr_min )[0]

    if offset:
        if frame == 'fk5':
            dra = (cc.fk5.ra - cc0.fk5.ra) * np.cos(cc0.fk5.dec)
            ddec = (cc.fk5.dec - cc0.fk5.dec)

            dx = dra.arcmin
            dy = ddec.arcmin

        else:
            dlon = (cc.galactic.l - cc0.galactic.l) * np.cos(cc0.galactic.b)
            dlat = (cc.galactic.b - cc0.galactic.b)
        
            dx = dlon.arcmin
            dy = dlat.arcmin

    else:
        if frame == 'fk5':
            dx = cc.fk5.ra
            dy = cc.fk5.dec

        else:
            dx = cc.galactic.l.deg
            dy = cc.galactic.b.deg

            #yy = np.where( dx > 180 )[0]
            #print(yy)
            #if len(yy):
            #    dx[yy] = dx[yy] - 360 
            #    print(dx)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    #ax.plot(dx, dy, marker='o', ls='')

    print(np.mean(rms))
    cax = ax.scatter(dx[xx], dy[xx], marker='o', c=rms[xx]-rm0, cmap='coolwarm', 
                     s=3 * (snrs[xx]), edgecolor='k', lw=0.5,
                     vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cax, shrink=0.95, extend='both')
    cbar.ax.set_ylabel('${\\rm RM}~({\\rm rad~m}^{-2})$', fontsize=12)

    xlim = ax.get_xlim()
    ax.set_xlim( xlim[1], xlim[0])
   
    if frame == 'fk5': 
        if offset:
            ax.set_xlabel("RA Offset (arcmin)", fontsize=14)
            ax.set_ylabel("DEC Offset (arcmin)", fontsize=14)
        else:
            ax.set_xlabel("RA (deg)", fontsize=14)
            ax.set_ylabel("DEC (deg)", fontsize=14)
    else:
        if offset:
            ax.set_xlabel("Gal Lon Offset (arcmin)", fontsize=14)
            ax.set_ylabel("Gal Lat Offset (arcmin)", fontsize=14)
        else:
            ax.set_xlabel("Gal Lon (deg)", fontsize=14)
            ax.set_ylabel("Gal Lat (deg)", fontsize=14)

    #ax.plot(0, 0, marker='x', c='r')

    plt.show()
    return
    


def plot_frac_pol(infile, rm_txt, cc0, vmin=None, vmax=None, 
                  snr_min=-1, frame='fk5'):
    """
    Make a plot using coords from cat_fits and 
    rm values from rm_txt centered on cc0
    """
    tab = Table.read(infile)
    cc = SkyCoord(tab["RA"], tab["DEC"])
    fluxes = tab['Total_flux'] * 1e3

    rdat = np.loadtxt(rm_txt)
    bnums = rdat[:, 0].astype('int')
    rms = rdat[:, 1]
    pflux = rdat[:, 3]
    snrs = rdat[:, 5]
    pfrac = pflux / fluxes
    xx = np.where( snrs > snr_min )[0]

    if frame == 'fk5':
        dx = cc.fk5.ra.deg
        dy = cc.fk5.dec.deg

    else:
        dx = cc.galactic.l.deg
        dy = cc.galactic.b.deg

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    #ax.plot(dx, dy, marker='o', ls='')

    cax = ax.scatter(dx[xx], dy[xx], marker='o', c=pfrac[xx], cmap='inferno', 
                     s=3 * (snrs[xx]), edgecolor='k', lw=0.5,  
                     norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(cax, shrink=0.95)
    cbar.ax.set_ylabel('Lin Pol Frac', fontsize=12)

    xlim = ax.get_xlim()
    ax.set_xlim( xlim[1], xlim[0])
   
    if frame == 'fk5': 
       ax.set_xlabel("RA (deg)", fontsize=14)
       ax.set_ylabel("DEC (deg)", fontsize=14)
    else:
       ax.set_xlabel("Gal Lon (deg)", fontsize=14)
       ax.set_ylabel("Gal Lat (deg)", fontsize=14)

    #ax.plot(0, 0, marker='x', c='r')
    plt.show()
    return
    




def plot_rm_diffs(cat_fits, rm_txt, rm_txt2, cc0, vmin=None, vmax=None):
    """
    Make a plot using coords from cat_fits and 
    rm values from rm_txt centered on cc0
    """
    tab = Table.read(cat_fits)
    cc = SkyCoord(tab["RA"], tab["DEC"])
    rdat = np.loadtxt(rm_txt)
    bnums = rdat[:, 0].astype('int')
    rms = rdat[:, 1]
    pflux = rdat[:, 3]
    
    rdat2 = np.loadtxt(rm_txt2)
    rms2 = rdat2[:, 1]
    pflux2 = rdat2[:, 3]

    dra = (cc.ra - cc0.ra) * np.cos(cc0.dec)
    ddec = (cc.dec - cc0.dec)

    dx = dra.arcmin
    dy = ddec.arcmin

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_aspect('equal')
    #ax.plot(dx, dy, marker='o', ls='')

    print(np.mean(rms))
    cax = ax.scatter(dx, dy, marker='o', 
                     c=rms-rms2, cmap='coolwarm', vmin=vmin, vmax=vmax, 
                     s=1 * (pflux2/pflux))
    cbar = plt.colorbar(cax, shrink=0.95, extend='both')

    xlim = ax.get_xlim()
    ax.set_xlim( xlim[1], xlim[0])

    #ax.plot(0, 0, marker='x', c='r')

    plt.show()
    return
    

    

    

     

# S1102
#cc0 = (206.6635833333, -63.06052777778) 

#L3044
#cc0 = (237.809250, -56.031166)

#S1224
#cc0 = (212.872125, -62.034)

#S3021
#cc0 = SkyCoord(-94.4631, -29.4174, unit=u.deg)

# S3007
cc0 = SkyCoord(-93.816, -29.9337, unit=u.deg)

