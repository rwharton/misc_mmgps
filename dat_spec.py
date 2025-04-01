import numpy as np
import glob

def dat_to_spec(dat_dir, outdir):
    """
    Take beam data and average in time to 
    get IQUV spectra
    """
    bfiles = glob.glob(f"{dat_dir}/beam*_full.npy")
    bfiles.sort()

    for bfile in bfiles:
        dat = np.load(bfile)
        spec = np.mean(dat, axis=1)
        
        dat_fn = bfile.split('/')[-1]
        out_fn = dat_fn.split('.npy')[0] + "_spec.npy"
        
        outfile = f"{outdir}/{out_fn}"
        np.save(outfile, spec)

    return
