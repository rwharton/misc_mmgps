import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from astropy.table import Table


matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

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


def get_spec(tab, idx, nchan=8, spindex=False):
    """
    go to row idx of table tab and extract
    the spectrum

    May not have values for every frequency
    """
    freqs = []
    fluxes = []
    e_fluxes = []

    for ii in range(nchan):
        freq_key = f"Freq_ch{ii:d}"
        flux_key = f"Aperture_flux_ch{ii:d}"
        e_flux_key = f"Aperture_rms_ch{ii:d}"

        freq = tab[idx].get(freq_key, -1)
        flux = tab[idx].get(flux_key, -1)
        e_flux = tab[idx].get(e_flux_key, -1)

        #print(freq, flux)

        if (freq == -1) or (freq == np.ma.is_masked(freq)):
            continue

        if (flux == -1) or np.ma.is_masked( flux ):
            continue

        freqs.append(freq)
        fluxes.append(flux)
        e_fluxes.append(e_flux)

    freqs = np.array(freqs)
    fluxes = np.array(fluxes)
    e_fluxes = np.array(e_fluxes)

    alpha = tab[idx]['Alpha']
    e_alpha = tab[idx]['E_Alpha']
    #print(f"spindex = {alpha:.2f} ({e_alpha:.2f})")
    
    if spindex:
        return freqs, fluxes, e_fluxes, alpha, e_alpha

    else:
        return freqs, fluxes, e_fluxes




def spec_plot(dat_file, freqsHz, cat_spec=None, avg_to_nchan=-1, 
              outfile=None, title=None, use_freqs=None):
    """
    Plot spec

    dat_file = npy file with data
    freq_file = npy file with freqs
    """
    dat = np.load(dat_file)
    freqs = freqsHz/ 1e9

    fmid = np.mean(freqsHz)

    if use_freqs is not None:
        dat = dat[:, :, use_freqs]
        freqs = freqs[use_freqs]
    
    dat *= 1000  # mJy

    I = np.mean(dat[0], axis=0)

    if outfile is not None:
        plt.ioff()

    fig = plt.figure(figsize=(8,4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax = fig.add_subplot(111)
    ax.plot(freqs, I, c='k', alpha=0.3, zorder=-1)

    if cat_spec is not None:
        cfreqs, cI, e_cI, alpha, e_alpha = cat_spec
        cI_mid = np.mean(cI)

        #ax.errorbar(cfreqs/1e9, cI*1e3, yerr=e_cI*1e3, marker='o', c='r', ls='')
        ax.plot(cfreqs/1e9, cI*1e3, marker='o', c=colors[0], ls='', ms=10)
        ax.plot(cfreqs/1e9, cI_mid * 1e3* (cfreqs / fmid)**alpha, c=colors[0], ls='-')

    if avg_to_nchan > 0:
        navg = int(len(freqs) / avg_to_nchan)
        freqs_avg, _ = avg_chan(freqs, np.zeros(len(freqs)), nchan=navg)
        I_avg, _ = avg_chan(I, np.zeros(len(I)), nchan=navg)
        ax.scatter(freqs_avg, I_avg, marker='s', 
                   ec=colors[1], fc='none', lw=3, s=200)


    frac = 0.95
    pfac = 1.2
    
    I_ylim = get_yrange(I, frac=frac, pfac=pfac)
    #ax.set_ylim(I_ylim)
    
    ax.set_ylabel("$S_I \\, \\rm{ (mJy)}$", fontsize=14)
    
    tp_kwargs = {'which' : 'major', 'direction': 'in', 'labelbottom' : False, 
                 'top': True, 'bottom': True, 'left' : True, 'right' : True, 
                 'length' : 5} 

    tp_kwargs_bot = tp_kwargs.copy()
    tp_kwargs_bot['labelbottom'] = True

    ax.tick_params(**tp_kwargs_bot)

    g_kwargs = {'alpha' : 0.3 }
    ax.grid(**g_kwargs)

    ax.set_xlabel("Frequency (GHz)", fontsize=14)

    if title is not None:
        ax.set_title(title, fontsize=14)

    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    return 


def many_spec_plots(bdir, freq_file, cat_fits, row_file, avg_to_nchan=-1):
    """
    Make spec plots for all beams
    """
    freqsHz = np.load(freq_file)
    rows = np.load(row_file)
    tab = Table.read(cat_fits)
    
    for ii, rr in enumerate(rows):
        cat_spec = get_spec(tab, rr)
        bfile = f"{bdir}/beam{ii:03d}_full_pbcorr.npy"
        cspec = get_spec(tab, rr, spindex=True)

        outfile = f"beam{ii:03d}_spec.png"

        spec_plot(bfile, freqsHz, cat_spec=cspec, 
                  avg_to_nchan=avg_to_nchan, outfile=outfile, 
                  title=f"beam{ii:03d}")
    
    return
