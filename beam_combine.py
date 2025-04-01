import numpy as np
import glob

def combine_spw(indir, bnum):
    dd_list = []
    for ii in range(16):
        infile = "%s/spw%02d_beam%05d_step000.npy" %(indir, ii, bnum)
        dd_ii = np.load(infile)
        dd_list.append(dd_ii)

    dd0 = np.hstack( [ddi[:, :, 0] for ddi in dd_list ] )
    dd1 = np.hstack( [ddi[:, :, 1] for ddi in dd_list ] )
    dd2 = np.hstack( [ddi[:, :, 2] for ddi in dd_list ] )
    dd3 = np.hstack( [ddi[:, :, 3] for ddi in dd_list ] )

    dd_out = np.array([dd0, dd1, dd2, dd3])

    outfile = "%s/beam%03d_full.npy" %(indir, bnum)
    np.save(outfile, dd_out)

    return 

def multibeam_combine(topdir, Nbeams):
    """
    assume b00000 etc 
    """
    for bnum in range(Nbeams):
        indir = f"{topdir}/beam{bnum:05d}"
        combine_spw(indir, bnum)

    return


def combine_freqs(basename):
    """
    combine freqs
    """
    freq_files = glob.glob('%s*spw*.npy' %basename)
    freq_files.sort()

    freqs = []
    
    for ff in freq_files:
        freq_ii = np.load(ff)
        freqs.append(freq_ii)

    freqs = np.hstack( freqs ) 
    freqs.sort()

    np.save("%s_full.npy" %basename, freqs)

    return 
