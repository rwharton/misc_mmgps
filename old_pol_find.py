import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, hstack, join
from glob import glob
import os
import sys
import shutil
import subprocess


##############################
## STUFF TO HELP MERGE CATS ##
## (lifted from Jonah's     ##
##  `combine_catalogs.py`   ##
##############################

from astropy.utils.metadata import enable_merge_strategies
from astropy.utils.metadata import MergeStrategy

class MergeNumbersAsList(MergeStrategy):
    types = ((int, float, str),  # left side types
             (int, float, str))  # right side types

    @classmethod
    def merge(cls, left, right):
        if left == right:
            return left
        else:
            return None

##############################


def get_pol_files(indir, basename, pols="IQUV"):
    """
    Get list of paths to the polarization FITS files.

    Will glob on indir/basename*[pols].fits 

    Assumes polarization is indicated by the last letter 
    before the fits extension, so: "[whatever][IQUV].fits"

    Returns list of file paths for each pol letter in "pols", 
    in that order.  So "IV" gives [Ifile, Vfile] and "VUQI" 
    gives [Vfile, Ufile, Qfile, Ifile]
    
    Also returns pol_list 
    """
    pol_list = list(pols)
    pol_files = []  

    for pp in pol_list:
        mstr = f"{indir}/{basename}*{pp}.fits"
        mpaths = glob(mstr)
        if len(mpaths) == 0:
            print(f"No files found matching {mstr}")
            pol_files.append(None)
        elif len(mpaths) > 1:
            print(f"More than one file found matching {mstr}!")
            print(mpaths)
            print("Using first one...")
            pol_files.append(mpaths[0])
        else:
            pol_files.append(mpaths[0])

    if None in pol_files:
        print("At least one pol missing...")
        sys.exit(0)

    return pol_files, pol_list


def check_and_copy(src, dst):
    if not os.path.exists( dst ):
        shutil.copyfile(src, dst)
    else:
        print(f"File exists: {dst}")
    return 


def check_and_update_fits(fitsfile, pol, sgn):
    """
    Update 'OBJECT' header parameter to append 
    "+/- [pol]".  If minus, multiply data by -1.

    As a check to see if we have already multiplied 
    by -1 (and hopefully to avoid future trouble), 
    we will first check to see if the object has been 
    appended (from, say, MSGPS_S_0759 to MSGPS_S_0759_Ipos)

    It's not foolproof, but...
    """
    hdulist = fits.open(fitsfile, mode='update')
    obj_name = hdulist[0].header['OBJECT']

    if obj_name is None:
        obj_name = "SRC"

    if sgn > 0:
        pstr = f"{pol}pos"
    else:
        pstr = f"{pol}neg"

    if obj_name[-4:] == pstr:
        print(f"file {fitsfile} already modified... skipping!")
        print("..... be careful...")
        
    else:
        hdulist[0].header['OBJECT'] = f"{obj_name}_{pstr}"
        
        if sgn < 0:
            hdulist[0].data = -1.0 * hdulist[0].data

    hdulist.close()
    return


def setup_workdir(workdir, indir, inbase, pols="IQUV"):
    """
    Make directory where we will do our processing, 
    copy over FITS files, make plus / minus files for 
    Q, U, and V
    """
    # Get files and polarizations
    pol_files, pol_list = get_pol_files(indir, inbase, pols=pols)

    # Make work directory if required 
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # Copy over FITS files to work directory... 
    # For pols Q, U, V make two copies: a "-plus" and "-minus" 
    # For stokes I just make a "-plus" (for consistency)
    for ii, fpol in enumerate(pol_files):
        pol_ii = pol_list[ii]
        fname = os.path.basename(fpol) 
        obase = fname.split('.fits')[0]
        opath_pos = f"{workdir}/{obase}-plus.fits"
        opath_neg = f"{workdir}/{obase}-minus.fits"
        # Copy to "positive"
        check_and_copy(fpol, opath_pos)
        check_and_update_fits(opath_pos, pol_ii, +1)

        # If in QUV, also copy to "negative"
        if pol_list[ii] in ["Q", "U", "V"]:
            check_and_copy(fpol, opath_neg)
            check_and_update_fits(opath_neg, pol_ii, -1)

    return
          

def sourcefind_all(workdir, src_dir):
    """
    Run PyBDSF sourcefinding on all FITS files in 
    workdir using Jonah's 'sourcefinding.py' script 
    in the 'image-processing' package.  

    "src_dir" points to the top of the 'image-processing' 
    directory 
    """ 
    fits_list = glob(f"{workdir}/*fits")
    N = len(fits_list)
    print(f"Found {N} fits files in {workdir}")

    for ii, fits_ii in enumerate(fits_list):
        print(f"\n\n...processing {ii+1}/{N}\n")
        cmd = f"python {src_dir}/sourcefinding.py catalog " +\
              f"{fits_ii} -o fits:srl ds9"

        subprocess.run(cmd, shell=True)

    return


def negate_cols(fits_minus):
    """
    Read in the fits file "fits_minus" and negate the 
    columns relating to flux density.

    return an astropy Table
    """
    # names of the columns we need to negate
    # (we will skip parameters that are 
    # uncertainties or rms values)
    # Not sure we care about the Island stuff...
    flux_cols = ['Total_flux', 'Peak_flux', 'Isl_Total_flux', 
                 'Isl_mean', 'Resid_Isl_mean' ] 
   
    # Read table from file 
    mtab = Table.read(fits_minus)

    # Loop over columns and *= -1
    for col in flux_cols:
        if col in mtab.colnames:
            mtab[col] *= -1.0
        else:
            print(f"Column {col} not found in {fits_minus}")

    return mtab


def combine_plus_minus(fits_plus, fits_minus, fits_out):
    """
    In order to get PyBDSF to find negative sources, we 
    made `negative images` where we just multiplied the 
    data table by -1.  This means that the source info 
    reports a positive flux density. We now want to go 
    back and `fix` the source table so that the flux 
    densities are actually reported with a minus sign. 

    Here we will take as input the +pol and -pol FITS
    files and output a combined catalog.  We will 
    multiply the relevant flux columns by -1 in the 
    -pol FITS file.

    So we will copy the `plus` file to the output 
    file and then append the `minus`
    """
    # copy plus fits file to output
    check_and_copy(fits_plus, fits_out)

    # Get positive table
    ptab = Table.read(fits_out)

    # Get minus-sign-corrected minus fits table
    mtab = negate_cols(fits_minus)

    # Combine tables
    with enable_merge_strategies(MergeNumbersAsList):
        otab = vstack( (ptab, mtab) )
    
    # Write to output file, and overwrite it
    otab.write(fits_out, format='fits', overwrite=True)
    
    # If we want compatibility with other image-processing 
    # scripts, we need to make sure we have the OBJECT 
    # header value assigned.
    hp = fits.open(fits_plus)
    obj_name = hp[1].header['OBJECT']
    hp.close()

    # Since we are taking the positive FITS, 
    # the object name should end in "[pol]pos"
    # so lets get rid of that
    if obj_name[-3:] == "pos":
        obj_name = obj_name[:-3]

    hout = fits.open(fits_out, mode='update')
    hout[1].header['OBJECT'] = obj_name
    hout.close()

    return






