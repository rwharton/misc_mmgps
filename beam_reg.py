from astropy.io import fits
import numpy as np
import argparse 
import sys

def get_fits_info(fitsfile):
    """
    Get ra, dec, freq, bw
    """
    hdulist = fits.open(fitsfile)
    # unfortunately these guys arent always in 
    # the same order, so we need to loop and check
    ra_deg = dec_deg = freq = None
    for ii in range(1, 5):
        ctype = hdulist[0].header[f"CTYPE{ii}"]
        if "RA" in ctype:
            ra_deg = hdulist[0].header[f"CRVAL{ii}"]
        elif "DEC" in ctype:
            dec_deg = hdulist[0].header[f"CRVAL{ii}"]
        elif "FREQ" in ctype:
            freq = hdulist[0].header[f"CRVAL{ii}"]  / 1e9
            bw = hdulist[0].header[f"CDELT{ii}"] / 1e9
        else:
            pass
    hdulist.close()

    print(f"{ra_deg=}, {dec_deg=}, {freq=}")

    if None in [ra_deg, dec_deg, freq]:
        print("Not all values found in header!")
        sys.exit(0)

    return ra_deg, dec_deg, freq, bw


def make_reg_file(regfile, ra_deg, dec_deg, radius_list):
    """
    Make region file
    """
    hdr = "" +\
          "# Region file format: DS9 version 4.0\n" +\
          "global color=white font=\"helvetica 10 normal\" select=1 "+\
          "highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n"+\
          "fk5\n"

    with open(regfile, 'w') as fout:
        fout.write(hdr)
        for rr in radius_list:
            out = f"circle({ra_deg:.6f}, {dec_deg:.6f}, {rr:.2f}\')\n"
            fout.write(out)

    return


def parse_input():
    """
    Use argparse to parse input
    """
    prog_desc = "For a primary beam of size hpbw0 arcmin at " +\
                "freq0 GHz, read in a FITS file and make a circle " +\
                "for that HPBW centered at the image center and " +\
                "adjusted as (freq/freq0)^-1 for the obs freq" 
    parser = argparse.ArgumentParser(description=prog_desc)

    parser.add_argument('fitsfile', help='FITS image file')
    parser.add_argument('outbase', help='output name of reg file')
    parser.add_argument('freq0', help='Reference frequency (GHz)', 
                         type=float)
    parser.add_argument('hpbw0', help='HPBW (in arcmin) at freq0', 
                         type=float)

    args = parser.parse_args()
    fitsfile = args.fitsfile
    outbase = args.outbase
    freq0 = args.freq0
    hpbw0 = args.hpbw0
    return fitsfile, outbase, freq0, hpbw0


if __name__ == "__main__":
    fitsfile, outbase, freq0, hpbw0 = parse_input()
    ra_deg, dec_deg, freq, bw = get_fits_info(fitsfile) 
    freqs = freq + np.array([ -0.5, +0.5 ]) * bw 
    radii = 0.5 * hpbw0 * (freqs/freq0)**-1.0
    make_reg_file(f"{outbase}.reg", ra_deg, dec_deg, radii)
