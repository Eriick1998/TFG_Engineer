#!python3
# author: Moises Garin (moises.garin@uvic.cat)
# created: 3rd of April, 2021

import cv2 as cv
import numpy as np
from scipy.interpolate import splrep, splev


def centroid(contour):
    """
    Return the centroid (center of mass) coordinates of the
    contour.

    Parameters:
    contour: nd array.
             An opencv contour: 1D array of 2D points.
    """
    # Calculate using moments.
    # https://en.wikipedia.org/wiki/Image_moment#Moment_invariants

    m = cv.moments(contour)
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return x, y


def smooth_convex_spl(c, s=5000, N=None):
    """
    smooth a closed convex contour using a spline.
    
    Parameters:
    contour: nd array.
             An opencv contour: 1D array of 2D points.
    s: smooth coeficient.
       if s=1, no smooth.
       s>1 for smoothing.
    N: Number of points of the resulting profile.
       If None, return as much as allowed by the original
       profile.
    """

    xc, yc = centroid(c)

    # Express cotour as a parametric curve
    # function of the angle with respect the centroid.
    # x = f(p), y = f(p), where p is the angle. 
    x = c[:, 0, 0]
    y = c[:, 0, 1]
    p = np.arctan2(y - yc, x - xc)

    # Make sure theta is always increasing
    i = np.argsort(p)
    p = p[i]
    x = x[i]
    y = y[i]

    # Make sure that there are no repeating points
    # in theta.
    i = np.nonzero(np.diff(p) > 0)
    p = p[i]
    x = x[i]
    y = y[i]

    # Find the smoothing spline
    # t = np.linspace(p[0],p[-1],50)
    # t = t[1:-1]
    t = None
    splx = splrep(p, x, k=2, s=s, per=True, t=t)
    sply = splrep(p, y, k=2, s=s, per=True, t=t)

    # Recalculate the contour.
    if N is not None:
        p = np.linspace(p[0], p[-1], N)
    cout = np.empty((p.size, 1, 2), dtype=c.dtype)
    cout[:, 0, 0] = splev(p, splx)
    cout[:, 0, 1] = splev(p, sply)

    return cout


def smooth_contour_spl(c, s=5000, N=None):
    """
    smooth a closed contour using a spline.
    
    Parameters:
    contour: nd array.
             An opencv contour: 1D array of 2D points.
    s: smooth coeficient.
       if s=1, no smooth.
       s>1 for smoothing.
    N: Number of points of the resulting profile.
       If None, return as much as allowed by the original
       profile.
    """

    xc, yc = centroid(c)

    # Express cotour as a parametric curve
    # function of the arc length.
    # x = f(p), y = f(p), where p is the angle. 
    x = c[:, 0, 0]
    y = c[:, 0, 1]
    p = np.empty_like(x)
    p[0] = 0
    p[1:] = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    # Find the smoothing spline
    splx = splrep(p, x, s=s, per=True)
    sply = splrep(p, y, s=s, per=True)

    # Recalculate the contour.
    if N is not None:
        p = np.linspace(p[0], p[-1], N)
    cout = np.empty((p.size, 1, 2), dtype=c.dtype)
    cout[:, 0, 0] = splev(p, splx)
    cout[:, 0, 1] = splev(p, sply)

    return cout
