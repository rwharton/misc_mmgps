import numpy as np
import matplotlib.pyplot as plt 
from astropy.coordinates import SkyCoord
import astropy.units as u
import subprocess as sub
import os
import glob
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import json
from astropy.table import Table

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


def darr_to_txt(outfile, darr, freqs, model=None):
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

            if model is not None:
                S0, alpha, freq0 = model
                I = S0 * (freq/freq0)**alpha 
            
            ostr = f"{freq:12.1f} " +\
                   f"{I:10.6f} {Q:10.6f} {U:10.6f} " +\
                   f"{dI:10.6f} {dQ:10.6f} {dU:10.6f} \n"

            fout.write(ostr)

    return


def run_rmsynth1d(infile, phimax=None, use_I=False):
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
    
    if use_I:
        cmd = f"rmsynth1d -S {phi_str} {infile}"
    else:
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


def get_model(table, row):
    """
    Get model from fits table 

    return (S0, alpha)
    """
    S0 = table['Total_flux'][row]
    alpha = table['Alpha'][row]
    return S0, alpha


def rm_clean_many(npy_files, freq_file, mask_file=None, outdir='.', 
                  cat_fits=None, row_file=None, 
                  niter=200, threshold=-2, phimax=None):
    """
    Given a list of npy data files, run rm clean on 
    all and output to individual directories

    """
    use_model = False

    if cat_fits is not None:
        tab =  Table.read(cat_fits)
        if row_file is not None:
            rr = np.load(row_file)
            use_model = True

    for npy_file in npy_files:
        fname = npy_file.split('/')[-1]
        basenm = fname.split('.npy')[0]

        # get bnum
        bnum = int(basenm.split('_')[0].split('beam')[-1])
        
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

        # Get the model (if desired)
        if use_model:
            S0, alpha = get_model(tab, rr[bnum])
            freq0 = np.mean(freqs)
            model = (S0, alpha, freq0)
        else:
            model = None

        print(model)
         
        darr_to_txt(dat_file, darr[xx], freqs[xx], model=model)

        # run rm synthesis
        ret1 = run_rmsynth1d(dat_file, phimax=phimax, use_I=True)
        
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
        try:
            ostr = json_to_cat(bnums[ii], jfile)
            olines.append(ostr)
        except:
            print(f"{bnames[ii]} rm json file not found!")

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

    

     

# S1102
#cc0 = (206.6635833333, -63.06052777778) 

#L3044
#cc0 = (237.809250, -56.031166)

#S1224
#cc0 = (212.872125, -62.034)


