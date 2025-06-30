import numpy as np
import matplotlib.pyplot as plt
import os

def avg_chan(dat, edat, nchan=4):
    """
    average data by nchan chans
    """
    Np = len(dat) // nchan
    dd = np.reshape( dat[: Np * nchan], (-1, nchan) )
    dd_avg = np.mean(dd, axis=1)

    edd = np.reshape( edat[: Np * nchan], (-1, nchan) )
    edd_avg = np.sqrt(np.sum( edd**2, axis=1 )) / nchan / 10

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

    #I = np.ones(shape=I.shape)

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
        freqs = idat[:, 0]
        darr  = idat[:, 1:9]

    return [freqs, darr]


def get_yrange(dat, frac=0.95, pfac=1.2):
    """
    Try to make ylim sensible
    """
    sdat = np.sort(dat)
    N = len(sdat)

    slo = sdat[int(N * (1-frac))]
    shi = sdat[int(N * frac)]

    ylo = slo - pfac * (shi - slo)
    yhi = shi + pfac * (shi - slo)

    return (ylo, yhi)


def get_yrange2(dat1, dat2, frac=0.95, pfac=1.1):
    """
    Do the samee but for 2 data sets
    """
    lo1, hi1 = get_yrange(dat1, frac=frac, pfac=pfac)
    lo2, hi2 = get_yrange(dat2, frac=frac, pfac=pfac)

    ylo = min(lo1, lo2)
    yhi = max(hi1, hi2)

    return (ylo, yhi)
    

def compare_plot(ifile, bfile, bfreqs, title=None, outfile=None, bchan_avg=1):
    """
    Plot a comparison of the beam results vs 
    image results
    """
    bdat = get_bdat(bfile)
    if bchan_avg > 1:
        bfreqs, bdat = avg_chan_darr(bfreqs, bdat, nchan=bchan_avg)
  
    ifreqs, idat = get_idat(ifile)

    if len(ifreqs) == 0:
        return

    # get to mJy
    bdat *= 1e3/2
    idat *= 1e3

    fig = plt.figure(figsize=(6, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axI = fig.add_subplot(311)
    axQ = fig.add_subplot(312, sharex=axI)
    axU = fig.add_subplot(313, sharex=axI)

    I_i, eI_i, Q_i, eQ_i, U_i, eU_i, V_i, eV_i = idat.T
    I_b, eI_b, Q_b, eQ_b, U_b, eU_b, V_b, eV_b = bdat.T

    aval = 0.6
    img_c = 'k'

    axI.plot(bfreqs, I_b, c=colors[0], alpha=aval, lw=2)
    axI.plot(ifreqs, I_i, c=img_c, lw=2)
    yI = get_yrange2(I_i, I_b)
    
    axQ.plot(bfreqs, Q_b, c=colors[1], alpha=aval, lw=2)
    axQ.plot(ifreqs, Q_i, c=img_c, lw=2)
    yQ = get_yrange2(Q_i, Q_b)
    
    axU.plot(bfreqs, U_b, c=colors[2], alpha=aval, lw=2)
    axU.plot(ifreqs, U_i, c=img_c, lw=2)
    yU = get_yrange2(U_i, U_b)
    
    axI.set_ylabel("$S_I \\, \\rm{ (mJy)}$", fontsize=14)
    axQ.set_ylabel("$S_Q \\, \\rm{ (mJy)}$", fontsize=14)
    axU.set_ylabel("$S_U \\, \\rm{ (mJy)}$", fontsize=14)

    tp_kwargs = {'which' : 'major', 'direction': 'in', 'labelbottom' : False,
                 'top': True, 'bottom': True, 'left' : True, 'right' : True,
                 'length' : 5}

    tp_kwargs_bot = tp_kwargs.copy()
    tp_kwargs_bot['labelbottom'] = True

    axI.tick_params(**tp_kwargs)
    axQ.tick_params(**tp_kwargs)
    axU.tick_params(**tp_kwargs_bot)

    g_kwargs = {'alpha' : 0.3 }
    axI.grid(**g_kwargs)
    axQ.grid(**g_kwargs)
    axU.grid(**g_kwargs)

    axI.set_ylim(yI)
    axQ.set_ylim(yQ)
    axU.set_ylim(yU)

    plt.subplots_adjust(hspace=0.0)

    axU.set_xlabel("Frequency (GHz)", fontsize=14)

    if title is not None:
        axI.set_title(title, fontsize=14)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    return


def many_compare_plots(idir, bdir, freq_file, bnums, bchan_avg=1):
    """
    Make many compare plots (as advertized!)

    idir is directory containing image dat files with 
    names of the form

        beamXXXX_qu_spectra.dat

    bdir is the directory containing beam data files 
    of the form 

        beamXXX_full.npy

    bfreqs is the file containing the freqs in Hz
    for the beam (image has its own freq)

    bnums is list of beam numbers
    """     
    # Read in frequencies and convert to GHz
    bfreqs = np.load(freq_file) / 1e9

    for bb in bnums:
        bstr = f"beam{bb:03d}"
        ifile = f"{idir}/beam{bb:03d}_qu_spectra.dat"
        bfile = f"{bdir}/{bstr}_full.npy"
        ofile = f"{bstr}_compare.png"

        if not os.path.exists(ifile):
            continue
  
        compare_plot(ifile, bfile, bfreqs, title=bstr, outfile=ofile, bchan_avg=bchan_avg)     

    return 

    
