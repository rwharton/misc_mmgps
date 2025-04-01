import numpy as np
from astropy.io import fits
from glob import glob
import shutil
import os


def check_and_copy(src, dst):
    if not os.path.exists( dst ):
        shutil.copyfile(src, dst)
    else:
        print(f"File exists: {dst}")
    return


def get_chan_files(indir, pol='V'):
    """
    get sorted list of files 
    """
    infiles = glob(f"{indir}/*{pol}-image.fits")
    infiles.sort()
    return infiles


def combine(infiles, outfile):
    """
    combine infiles to outfile
    """
    # copy first file as outfile
    check_and_copy(infiles[0], outfile)

    # Open up output file to modify
    h = fits.open(outfile, mode='update')

    # Now get data from all channels
    dd = h[0].data
    for ii in range(1, len(infiles)):
        h_ii = fits.open(infiles[ii])
        dd = np.concatenate( (dd, h_ii[0].data), axis=1 )
        h_ii.close()

    # Replace orig data
    h[0].data = dd

    # close
    h.close()

    return

    
    
