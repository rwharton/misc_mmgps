import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

from katbeam import JimBeam

def one_beam_to_coord(ra_str, dec_str):
    """
    Convert one ra dec string pair from
    the beam list to an astropy SkyCoord

    These have format:

    ra_str  = 'hh:mm:ss.ss'
    dec_str = 'dd.mm.ss.ss'

    which is annoying, i know
    """
    # Careful
    dec_fix = dec_str.replace('.', ':', 2)

    cc = SkyCoord(ra_str, dec_fix, unit=(u.hourangle, u.deg),
                  frame='fk5')

    return cc


def beamlist_to_coords(bfile):
    """
    Read in a beam list file (npy array)
    and convert to list of SkyCoord objects
    """
    beams = np.load(bfile)
    cc_list = []
    for bb in beams:
        cc = one_beam_to_coord(bb[1], bb[2])
        cc_list.append(cc)

    return cc_list


def pb_corr_one(infile, beam, offset, freqs):
    """
    Using the katbeam `beam`, read in the data from `infile`, 
    make primary beam corrections for frequencies `freqs` 
    at angular offset `offset`.

    Will add 'pbcorr' to the infile name and write to file

    Since we are just using one offset value, we will take 
    it to be 
        
        (x, y) = (offset/sqrt(2), offset/sqrt(2))

    to split the difference between any asymmetries in 
    x and y
    """
    outfile = infile.rsplit('.npy', 1)[0] + '_pbcorr.npy'
    
    # put freqs in mhz
    freqsMHz = freqs/1e6
    pb = beam.I(offset/np.sqrt(2), offset/np.sqrt(2), freqsMHz)
    
    # get data
    dat = np.load(infile)
    pb_dat = dat / pb

    # write to file
    np.save(outfile, pb_dat)

    return


def pb_corr_many(bdir, cat_fits, row_file, freq_file):
    """
    Primary correct many dat files

    cat_fits is the FITS file catalog (needed for offset)

    row_file is a numpy file of table row numbers ordered 
    according to our beams 

    freq file is list of frequencies
    """
    freqs = np.load(freq_file)
    rows = np.load(row_file)

    # Set up beam
    beam = JimBeam('MKAT-AA-S-JIM-2020')

    tab = Table.read(cat_fits)

    for ii, rr in enumerate(rows):
        infile = f"{bdir}/beam{ii:03d}_full.npy"
        offset = tab[rr]['Sep_PC']

        pb_corr_one(infile, beam, offset, freqs)

    return 

