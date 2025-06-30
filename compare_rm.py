import numpy as np
import matplotlib.pyplot as plt


def read_rm_compare(infile):
    """
    read in the rm compare file
    """
    bnames = []
    vals1 = []
    vals2 = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", " ", "\n"]:
                continue
            cols = line.split()
            if "nan" in cols[10]:
                continue
            
            bname = cols[0].strip() 
            vv1 = [ float(cols[3]), float(cols[4]), 
                    float(cols[5]), float(cols[6]) ]
            
            vv2 = [ float(cols[10]), float(cols[11]), 
                    float(cols[12]), float(cols[13]) ]

            bnames.append(bname)
            vals1.append(vv1)
            vals2.append(vv2)

    bnames = np.array(bnames)
    vals1  = np.array(vals1)
    vals2  = np.array(vals2)
    
    return bnames, vals1, vals2


def read_rm_one_orig(infile, use_bnames=None):
    """ 
    Read in one rm file (beam, rm, rm_err, PI, PIerr)
    optionally provide a list of bnames and will only 
    read in those beams
    """ 
    bnames = []
    vals = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", " ", "\n"]:
                continue
            cols = line.split()

            bname = cols[0].strip()
            vv = [ float(cols[1]), float(cols[2]), 
                   float(cols[3]), float(cols[4]) ]

            if use_bnames is not None:
                if bname not in use_bnames:
                    continue

            bnames.append(bname)
            vals.append(vv)

    bnames = np.array(bnames)
    vals = np.array(vals)
       
    return bnames, vals


def read_rm_one(infile, use_bnames=None):
    """ 
    Read in one rm file (beam, rm, rm_err, PI, PIerr, res)
    optionally provide a list of bnames and will only 
    read in those beams
    """ 
    bnames = []
    vals = []
    res = []
    with open(infile, 'r') as fin:
        for line in fin:
            if line[0] in ["#", " ", "\n"]:
                continue
            cols = line.split()

            bname = cols[0].strip()
            vv = [ float(cols[3]), float(cols[4]), 
                   float(cols[5]), float(cols[6]) ]

            res.append( float(cols[7]) )

            if use_bnames is not None:
                if bname not in use_bnames:
                    continue

            bnames.append(bname)
            vals.append(vv)

    bnames = np.array(bnames)
    vals = np.array(vals)
    res = np.array(res)
       
    return bnames, vals, res


def make_compare_plot(bnames, vref, v, val='rm'):
    """
    comparing image based rm/pi of vref to 
    orig beamform v and cut vcut

    val = 'rm' or 'pi'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if val == 'rm':
        ii = 0
    else:
        ii = 2

    snr = vref[:, 2] / vref[:, 3]

    #ax.plot(vref[:, ii], v[:, ii], ls='', marker='o', ms=10)
    #ax.errorbar(x=vref[:, ii], y=v[:, ii],
    #            xerr=vref[:, ii+1], yerr=v[:, ii+1], 
    #            marker='o', ms=10, ls='')
    
    cax = ax.scatter(x=vref[:, ii], y=v[:, ii],
                     marker='o', s=50, 
                     c=snr, cmap='inferno_r', 
                     vmin=5, vmax=15, ec='k', 
                     lw=0.5)

    plt.colorbar(cax)

    min1 = np.min(vref[:, ii])
    min2 = np.min(v[:, ii])

    max1 = np.max(vref[:, ii])
    max2 = np.max(vref[:, ii])

    min_val = min(min1, min2)
    max_val = max(max1, max2)

    x = np.linspace(min_val, max_val, 100)
    ax.plot(x, x, ls='--', c='k', alpha=0.8, zorder=-1)
    
    if val == 'pi':
        ax.plot(x, 2 * x, ls='--', c='r')

    if val == 'rm':
        lab_str = "${\\rm RM}~({\\rm rad~m}^{-2})$"
    else:
        lab_str = "${\\rm PI}~({\\rm mJy})$"

    xlab = "${\\rm Image}$" + " " + lab_str
    ylab = "${\\rm DFT}$" + " " + lab_str
    ax.set_xlabel(xlab, fontsize=14) 
    ax.set_ylabel(ylab, fontsize=14) 

    plt.show()
    return

def make_compare_frac_plot(bnames, vref, v, val='rm'):
    """
    comparing image based rm/pi of vref to 
    orig beamform v and cut vcut

    val = 'rm' or 'pi'
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if val == 'rm':
        ii = 0
    else:
        ii = 2

    snr = vref[:, 2] / vref[:, 3]

    #ax.plot(vref[:, ii], v[:, ii], ls='', marker='o', ms=10)
    fdiff = np.abs(v[:, ii] - vref[:, ii])
    cax = ax.scatter(x=vref[:, ii], y=fdiff,
                     marker='o', s=50, 
                     c=snr, cmap='inferno_r', 
                     vmin=5, vmax=15, ec='k', 
                     lw=0.5)
    
    xerr = vref[:, ii+1]
    yerr = np.sqrt( xerr**2 + v[:, ii+1]**2 ) 
    ax.errorbar(x=vref[:, ii], y=fdiff,
                xerr=xerr, yerr=yerr, 
                marker='', ls='', c='k', zorder=-1)
   

    plt.colorbar(cax)
    ax.axhline(y=0, ls='--', c='k', alpha=0.8, zorder=-1)
    
    if val == 'rm':
        lab_str = "${\\rm RM}~({\\rm rad~m}^{-2})$"
    else:
        lab_str = "${\\rm PI}~({\\rm mJy})$"

    xlab = "${\\rm Image}$" + " " + lab_str
    ylab = "${\\rm DFT}$" + " " + lab_str
    ax.set_xlabel(xlab, fontsize=14) 
    ax.set_ylabel(ylab, fontsize=14) 

    plt.show()
    return


def make_compare_plot_diff_name(bnames, vref, v, vcut, val='rm'):
    """
    comparing image based rm/pi of vref to 
    orig beamform v and cut vcut
    """
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)

    if val == 'rm':
        col = 0
    else:
        col = 2

    e1 = np.sqrt( v[:, col+1]**2 + vref[:, col+1]**2 )
    e1 = 1
    fd1 = np.abs( (v[:, col] - vref[:,col]) / e1 )

    e2 = np.sqrt( vcut[:, col+1]**2 + vref[:, col+1]**2 )
    e2 =1
    fd2 = np.abs( (vcut[:, col] - vref[:,col]) / e2)

    idx = np.arange(len(bnames))
    
    ax.plot(idx, fd1, ls='', marker='o', ms=10, label='natural')
    ax.plot(idx, fd2, ls='', marker='s', mfc='none', 
            mew=2, ms=12, label='weighted')

    for ii in range(len(idx)):
        x = np.array([idx[ii], idx[ii]])
        y = np.array([fd1[ii], fd2[ii]])
        if fd2[ii] < fd1[ii]:
            c = 'k'
        else:
            c = 'r'
        ax.plot(x, y, ls='-', c=c, alpha=0.3, lw=2)

    ax.set_yscale('log')

    #x = np.linspace(-2000, 2000, 100)
    #ax.plot(x, x, ls='--', c='k')

    bnums = np.array([ int( bb.lstrip('beam') ) for bb in bnames ])
    ax.set_xticks(np.arange(len(bnums)), bnums)

    ax.grid(alpha=0.2)

    ax.set_xlabel("Beam Number", fontsize=14) 
    if val == "rm":
        ax.set_ylabel("|DFT-Image RM|", fontsize=14) 
    else:
        ax.set_ylabel("|DFT-Image PI| / Err", fontsize=14) 


    plt.legend()

    plt.show()
    return


def make_compare_plot_name(bnames, vref, v, vcut, val='rm'):
    """
    comparing image based rm/pi of vref to 
    orig beamform v and cut vcut
    """
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)

    if val == 'rm':
        col = 0
    else:
        col = 2

    rm0 = vref[:, col]
    rm1 = v[:, col] 
    rm2 = vcut[:, col] 

    idx = np.arange(len(bnames))
    
    ax.plot(idx, rm1, ls='', marker='o', ms=10, label='natural')
    ax.plot(idx, rm2, ls='', marker='s', mfc='none', 
            mew=2, ms=12, label='weighted')
    ax.plot(idx, rm0, ls='', marker='x', ms=10, label='img', c='k')

    #x = np.linspace(-2000, 2000, 100)
    #ax.plot(x, x, ls='--', c='k')

    bnums = np.array([ int( bb.lstrip('beam') ) for bb in bnames ])
    ax.set_xticks(np.arange(len(bnums)), bnums)

    ax.grid(alpha=0.2)

    ax.set_xlabel("Beam Number", fontsize=14) 
    ax.set_ylabel("RM (rad/m^2)", fontsize=14) 

    plt.legend()

    plt.show()
    return



def make_compare_plot_frac_name(bnames, vref, v, vcut, val='rm'):
    """
    comparing image based rm/pi of vref to 
    orig beamform v and cut vcut
    """
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)

    if val == 'rm':
        col = 0
    else:
        col = 2

    fd1 = np.abs( (v[:, col] - vref[:,col]) / vref[:, col] )
    fd2 = np.abs( (vcut[:, col] - vref[:,col]) / vref[:, col] )

    idx = np.arange(len(bnames))
    
    ax.plot(idx, fd1, ls='', marker='o', ms=10, label='natural')
    ax.plot(idx, fd2, ls='', marker='s', mfc='none', 
            mew=2, ms=12, label='weighted')

    for ii in range(len(idx)):
        x = np.array([idx[ii], idx[ii]])
        y = np.array([fd1[ii], fd2[ii]])
        if fd2[ii] < fd1[ii]:
            c = 'k'
        else:
            c = 'r'
        ax.plot(x, y, ls='-', c=c, alpha=0.3, lw=2)

    ax.set_yscale('log')

    #x = np.linspace(-2000, 2000, 100)
    #ax.plot(x, x, ls='--', c='k')

    bnums = np.array([ int( bb.lstrip('beam') ) for bb in bnames ])
    ax.set_xticks(np.arange(len(bnums)), bnums)

    ax.grid(alpha=0.2)

    ax.set_xlabel("Beam Number", fontsize=14) 
    if val == "rm":
        ax.set_ylabel("DFT-Image RM Frac Diff", fontsize=14) 
    else:
        ax.set_ylabel("DFT-Image PI Frac Diff", fontsize=14) 


    plt.legend()

    plt.show()
    return

