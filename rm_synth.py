import numpy as np
import matplotlib.pyplot as plt 
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from astropy.coordinates import SkyCoord
import astropy.units as u


def rm_synth_beam(bdat, freqs):
    """
    Inputs are our beam data and frequencies, 
    which we get in the appropriate format for 
    `run_rmsynth` and then run... `run_rmsynth`
    """
    I = np.mean(bdat[0, :, :], axis=0)
    Q = np.mean(bdat[1, :, :], axis=0)
    U = np.mean(bdat[2, :, :], axis=0)

    I = np.ones(shape=I.shape)

    Nt = bdat.shape[1]
    Nf = bdat.shape[2]

    # Uncertainty... this is probably not right
    dI = np.std(bdat[0, :, :], axis=0) / np.sqrt(Nt)
    dQ = np.std(bdat[1, :, :], axis=0) / np.sqrt(Nt)
    dU = np.std(bdat[2, :, :], axis=0) / np.sqrt(Nt)

    #dI = dQ = dU = np.ones(len(I)) * 14e-6 * np.sqrt(Nf)

    dlist = [ freqs, I, Q, U, dI, dQ, dU ]

    md, ad = run_rmsynth(dlist, fitRMSF=True, showPlots=False, saveFigures='test')

    return md, ad


def write_output(outfile, phi_pks, dphi_pks, PI_pks, dPI_pks):
    with open(outfile, 'w') as fout:
        hdr = f"# {'Beam' : <10} {'RM':^12} {'RM_err':^12} {'PI':^12} {'PI_err':^12}"
        hdr2 = f"# {'' : <10} {'(rad/m^2)':^12} {'(rad/m^2)':^12} {'(mJy)':^12} {'(mJy)':^12}"
        fout.write(hdr + "\n" + hdr2 + "\n")

        for ii in range(len(phi_pks)):
            src_str = f"beam{ii:03d}"
            ostr = ""
            ostr += f"{src_str: <10} " 
            ostr += f"{phi_pks[ii] : >12.2f} {dphi_pks[ii] : >12.2f} "
            ostr += f"{PI_pks[ii]*1e3 : >12.3f} {dPI_pks[ii]*1e3 : >12.3f}"

            fout.write(ostr + "\n")

    return 


def rm_synth_many_beams(blist, freq_file, outfile=None, use_freqs=[]):
    """
    blist = list of beam dat files
    freq_file = frequency channel file (in Hz)
    use_freqs = option list of freq indices to use
    """
    plt.ioff()
    freqs = np.load(freq_file)

    phi_pks  = []
    dphi_pks = []
    
    PI_pks  = []
    dPI_pks = []
    
    for bbfile in blist:
        bdat = np.load(bbfile)
        if len(use_freqs):
            bdat = bdat[:, :, use_freqs]
            ff = freqs[use_freqs]

        # run rm synth
        md, ad = rm_synth_beam(bdat, ff)

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


# S1102
#cc0 = (206.6635833333, -63.06052777778) 

#L3044
cc0 = (237.809250, -56.031166)

