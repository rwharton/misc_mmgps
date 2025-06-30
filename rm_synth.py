import numpy as np
import matplotlib.pyplot as plt 
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from astropy.coordinates import SkyCoord
import astropy.units as u


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


def swap_QU(darr, xx):
    """
    Swap Q and U for channels xx
    """
    Qxx = darr[xx, 2]
    eQxx = darr[xx, 3]
    
    Uxx = darr[xx, 4]
    eUxx = darr[xx, 5]

    darr2 = np.copy(darr)
    
    darr2[xx, 2] = -1 * Uxx
    darr2[xx, 3] = eUxx
    darr2[xx, 4] = -1 * Qxx
    darr2[xx, 5] = eQxx

    return darr2


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


def rm_synth_beam(darr, freqs, outbase=None, noStokesI=False, 
                  phiMax_radm2=None):
    """
    Inputs are our beam data and frequencies, 
    which we get in the appropriate format for 
    `run_rmsynth` and then run... `run_rmsynth`
    """
    I, dI, Q, dQ, U, dU, V, dV = darr.T

    dlist = [ freqs, I, Q, U, dI, dQ, dU ]

    if outbase is None:
        saveFigures=False
    else:
        saveFigures=True

    md, ad = run_rmsynth(dlist, fitRMSF=True, showPlots=False, prefixOut=outbase, 
                         saveFigures=saveFigures, noStokesI=noStokesI, 
                         phiMax_radm2=phiMax_radm2)

    return md, ad


def write_output(outfile, phi_pks, dphi_pks, PI_pks, dPI_pks):
    with open(outfile, 'w') as fout:
        hdr = f"# {'Beam' : <10} {'RM':^12} {'RM_err':^12} {'PI':^14} {'PI_err':^14}"
        hdr2 = f"# {'' : <10} {'(rad/m^2)':^12} {'(rad/m^2)':^12} {'(mJy)':^14} {'(mJy)':^14}"
        fout.write(hdr + "\n" + hdr2 + "\n")

        for ii in range(len(phi_pks)):
            src_str = f"beam{ii:03d}"
            ostr = ""
            ostr += f"{src_str: <10} " 
            ostr += f"{phi_pks[ii] : >12.2f} {dphi_pks[ii] : >12.2f} "
            ostr += f"{PI_pks[ii]*1e3 : >14.5f} {dPI_pks[ii]*1e3 : >14.5f}"

            fout.write(ostr + "\n")

    return 


def get_base(infile, dat_type='beam'):
    bfname = infile.split('/')[-1]
    if dat_type == 'beam':
        bbase = bfname.split('.npy')[0]
    else:
        bbase = bfname.split('.dat')[0]
    return bbase


def rm_synth_many_beams(blist, freq_file, outfile=None, 
                        noStokesI=False, phiMax_radm2=None, 
                        use_freqs=[], chan_avg=1, dat_type='beam'):
    """
    blist = list of beam dat files
    freq_file = frequency channel file (in Hz)
    use_freqs = option list of freq indices to use
    dat_type = 'image' or 'beam'
    """
    plt.ioff()

    phi_pks  = []
    dphi_pks = []
    
    PI_pks  = []
    dPI_pks = []
    
    for bbfile in blist:
        if dat_type == 'beam':
            freqs = np.load(freq_file)
            darr = get_bdat(bbfile)
        else:
            freqs, darr = get_idat(bbfile)
            print(len(freqs))
            if len(freqs) == 0:
                print("here")
                phi_pks.append( np.nan  )
                dphi_pks.append( np.nan )
                PI_pks.append( np.nan )
                dPI_pks.append( np.nan )
                continue

        if chan_avg > 1:
            freqs, darr = avg_chan_darr(freqs, darr, nchan=chan_avg)

        if len(use_freqs):
            darr = darr[use_freqs, :]
            ff = freqs[use_freqs]
        else:
            ff = freqs[:]

        # get basename
        bbase = get_base(bbfile, dat_type=dat_type)

        # run rm synth
        md, ad = rm_synth_beam(darr, ff, bbase, noStokesI=noStokesI,  
                               phiMax_radm2=phiMax_radm2)

        phi_pks.append( md['phiPeakPIfit_rm2'] )
        dphi_pks.append( md['dPhiPeakPIfit_rm2'] )

        PI_pks.append( md['ampPeakPIfit'] )
        dPI_pks.append( md['dAmpPeakPIfit'] )
        plt.close('all')


    phi_pks = np.array(phi_pks)
    dphi_pks = np.array(dphi_pks)

    PI_pks = np.array( PI_pks )
    dPI_pks = np.array( dPI_pks )
    
    plt.ion()

 
    if outfile is not None:
         write_output(outfile, phi_pks, dphi_pks, PI_pks, dPI_pks)

    return phi_pks, dphi_pks, PI_pks, dPI_pks


def read_and_convert_locs(bloc_file):
    """
    read in the beam location file and convert 
    to ra_deg, dec_deg
    """
    blocs = np.load(bloc_file)
    cc_list = []
    for bb in blocs:
        ra_str = bb[1]
        dec_str = bb[2].replace('.', ':', 2)
        cc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg)) 
        cc_list.append(cc)

    ra_deg = np.array([ cc.ra.deg for cc in cc_list ])
    dec_deg = np.array([ cc.dec.deg for cc in cc_list ])
    return ra_deg, dec_deg


def make_plot(rm_file, bloc_file, cc0=(0, 0), size=0, 
              vmin=None, vmax=None):
    """
    Read in the rm_file text file to get RMs and PIs
    
    Read in the bloc_file numpy array to get coords

    cc0 is (ra_deg, dec_deg) of cetner of image

    size is size of image in degrees
    """
    # Read rm file
    dat = np.loadtxt(rm_file, usecols=(1,2,3,4))
    rms, drms, pis, dpis = dat.T 

    # read in coordinates in ra dec deg
    ra, dec = read_and_convert_locs(bloc_file)

    # get offsets
    dra = (ra - cc0[0]) * np.abs( np.cos( dec * np.pi/180.) )
    ddec = dec - cc0[1]

    # start fig
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # color axis is RM, size is PI snr
    pi_snr = pis / dpis 
    cax = ax.scatter(dra, ddec, c=rms, s=np.sqrt(pi_snr*10),
                     cmap='coolwarm', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax)

    if size > 0:
        ax.set_xlim(0.5 * size, -0.5 * size)
        ax.set_ylim(-0.5 * size, 0.5 * size)

    ax.grid(alpha=0.3, ls='--')
    ax.set_aspect('equal')

    plt.show()
    return


def write_cat_output(outfile, ras, decs, phi_pks, dphi_pks, PI_pks, dPI_pks):
    with open(outfile, 'w') as fout:
        hdr = f"# {'Beam' : <10} {'ra':^12} {'dec':^12} {'RM':^12} {'RM_err':^12} {'PI':^12} {'PI_err':^12}"
        hdr2 = f"# {'' : <10} {'(deg)':^12} {'(deg)':^12} {'(rad/m^2)':^12} {'(rad/m^2)':^12} {'(mJy)':^12} {'(mJy)':^12}"
        fout.write(hdr + "\n" + hdr2 + "\n")

        for ii in range(len(phi_pks)):
            src_str = f"beam{ii:03d}"
            ostr = ""
            ostr += f"{src_str: <10} " 
            ostr += f"{ras[ii] : >12.6f} {decs[ii] : >12.6f} "
            ostr += f"{phi_pks[ii] : >12.2f} {dphi_pks[ii] : >12.2f} "
            ostr += f"{PI_pks[ii] : >12.3f} {dPI_pks[ii] : >12.3f}"

            fout.write(ostr + "\n")

    return 


def write_pos_rm_cat(rm_file, bloc_file, outfile):
    """
    Make a catalog with ra dec rm pi
    """
    # Read rm file
    dat = np.loadtxt(rm_file, usecols=(1,2,3,4))
    rms, drms, pis, dpis = dat.T 

    # read in coordinates in ra dec deg
    ra, dec = read_and_convert_locs(bloc_file)

    write_cat_output(outfile, ra, dec, rms, drms, pis, dpis)

    return


def write_cat_output2(outfile, ras, decs, phi_pks, dphi_pks, PI_pks, dPI_pks, resolved):
    with open(outfile, 'w') as fout:
        hdr = f"# {'Beam' : <10} {'ra':^12} {'dec':^12} {'RM':^12} {'RM_err':^12} {'PI':^12} {'PI_err':^12} {'Resolved?':^10}"
        hdr2 = f"# {'' : <10} {'(deg)':^12} {'(deg)':^12} {'(rad/m^2)':^12} {'(rad/m^2)':^12} {'(mJy)':^12} {'(mJy)':^12} {'':^10}"
        fout.write(hdr + "\n" + hdr2 + "\n")

        for ii in range(len(phi_pks)):
            src_str = f"beam{ii:03d}"
            ostr = ""
            ostr += f"{src_str: <10} " 
            ostr += f"{ras[ii] : >12.6f} {decs[ii] : >12.6f} "
            ostr += f"{phi_pks[ii] : >12.2f} {dphi_pks[ii] : >12.2f} "
            ostr += f"{PI_pks[ii] : >12.3f} {dPI_pks[ii] : >12.3f} "
            ostr += f"{resolved[ii] : ^10d}"

            fout.write(ostr + "\n")

    return 


def write_pos_rm_cat2(rm_file, bloc_file, outfile, res_file):
    """
    Make a catalog with ra dec rm pi
    """
    # Read rm file
    dat = np.loadtxt(rm_file, usecols=(1,2,3,4))
    rms, drms, pis, dpis = dat.T 

    # read in coordinates in ra dec deg
    ra, dec = read_and_convert_locs(bloc_file)

    # read in resolved flag
    res = np.load(res_file)

    write_cat_output2(outfile, ra, dec, rms, drms, pis, dpis, res)

    return


# S1102
#cc0 = (206.6635833333, -63.06052777778) 

#L3044
#cc0 = (237.809250, -56.031166)

#S1224
cc0 = (212.872125, -62.034)

