import numpy as np
from astropy import units as u
from mw_plot import MWFaceOn, MWSkyMap
import matplotlib.pyplot as plt

def dsun_xy(d, lon, R0=8.125):
    """
    Get x,y from d and lon
    """
    x = R0 + d * np.cos((lon-180) * np.pi / 180)
    y = 0  + d * np.sin((lon-180) * np.pi / 180)
    return x, y

def err_angle(ax, d, derr, lon, R0=8.125, ms=6):
    x0, y0 = dsun_xy(d, lon, R0=R0) 
    
    ebar = d + np.linspace(-derr, derr, 100)
    xe, ye = dsun_xy(ebar, lon, R0=R0)

    ax.plot(x0, y0, marker='o', c='r', ms=ms, mec='k')
    ax.plot(xe, ye, ls='-', c='r', lw=1) 

    plt.show()

    return
    

mw1 = MWFaceOn(
    radius=20 * u.kpc,
    unit=u.kpc,
    angle=90, 
    coord="galactocentric",
    annotation=True,
    grayscale=False,
)

#mw1.title = "Bird's Eyes View"
#mw1.scatter(8 * u.kpc, 0 * u.kpc, c="r", s=2)
fig, ax = plt.subplots(figsize=(10,8))
mw1.transform(ax)
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 20)


hii_1 = [291.614, 7.30, 0.44]
hii_2 = [291.467, 6.79, 0.53]
hii_3 = [291.214, 6.56, 0.46]


ms_hii = 8
err_angle(ax, hii_1[1], hii_1[2], hii_1[0], ms=ms_hii)
err_angle(ax, hii_2[1], hii_2[2], hii_2[0], ms=ms_hii)
err_angle(ax, hii_3[1], hii_3[2], hii_3[0], ms=ms_hii)


psr_lon = 291.443
ymw16_dist = 5.505 
ne2001_dist = 15.885

x_ymw, y_ymw = dsun_xy(ymw16_dist, psr_lon)
x_ne, y_ne = dsun_xy(ne2001_dist, psr_lon)

ax.plot(x_ymw, y_ymw, c='yellow', marker='*', ms=14)
ax.plot(x_ne, y_ne, c='orange', marker='*', ms=14)

print(x_ymw, y_ymw)
print(x_ne, y_ne)


"""
mw1 = MWSkyMap(
    center=(-70, 0) * u.deg, 
    radius=(3,3) * u.deg,
    background='AllWISE color  Red (W4) , Green (W2) , Blue (W1) from raw Atlas Images', 
    #background="Mellinger color optical survey",
)

fig, ax = plt.subplots(figsize=(5, 5))
mw1.transform(ax)
"""

