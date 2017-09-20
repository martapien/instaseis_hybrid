#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Library loading helper.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import ctypes as C
import glob
import inspect
import math
import os

import numpy as np


LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "lib")

cache = []


def load_lib():
    if cache:
        return cache[0]
    else:
        # Enable a couple of different library naming schemes.
        possible_files = glob.glob(os.path.join(LIB_DIR, "instaseis*.so"))
        if not possible_files:  # pragma: no cover
            raise ValueError("Could not find suitable instaseis shared "
                             "library.")
        filename = possible_files[0]
        lib = C.CDLL(filename)
        cache.append(lib)
        return lib


def get_band_code(dt):
    """
    Figure out the channel band code. Done as in SPECFEM.
    """
    if dt <= 0.001:
        band_code = "F"
    elif dt <= 0.004:
        band_code = "C"
    elif dt <= 0.0125:
        band_code = "H"
    elif dt <= 0.1:
        band_code = "B"
    elif dt < 1:
        band_code = "M"
    else:
        band_code = "L"
    return band_code


def elliptic_to_geocentric_latitude(lat, axis_a=6378137.0,
                                    axis_b=6356752.314245):
    """
    Convert a latitude defined on an ellipsoid to a geocentric one.

    :param lat: The latitude to convert.
    :param axis_a: The length of the major axis of the planet. Defaults to
        the value of the WGS84 ellipsoid.
    :param axis_b: The length of the minor axis of the planet. Defaults to
        the value of the WGS84 ellipsoid.

    >>> elliptic_to_geocentric_latitude(0.0)
    0.0
    >>> elliptic_to_geocentric_latitude(90.0)
    90.0
    >>> elliptic_to_geocentric_latitude(-90.0)
    -90.0
    >>> elliptic_to_geocentric_latitude(45.0)
    44.80757678401642
    >>> elliptic_to_geocentric_latitude(-45.0)
    -44.80757678401642
    """
    _f = (axis_a - axis_b) / axis_a
    E_2 = 2 * _f - _f ** 2

    # Singularities close to the pole and the equator. Just return the value
    # in that case.
    if abs(lat) < 1E-6 or abs(lat - 90) < 1E-6 or \
            abs(lat + 90.0) < 1E-6:
        return lat

    return math.degrees(math.atan((1 - E_2) * math.tan(math.radians(lat))))


def geocentric_to_elliptic_latitude(lat, axis_a=6378137.0,
                                    axis_b=6356752.314245):
    """
    Convert a geocentric latitude to one defined on an ellipsoid.

    :param lat: The latitude to convert.
    :param axis_a: The length of the major axis of the planet. Defaults to
        the value of the WGS84 ellipsoid.
    :param axis_b: The length of the minor axis of the planet. Defaults to
        the value of the WGS84 ellipsoid.

    >>> geocentric_to_elliptic_latitude(0.0)
    0.0
    >>> geocentric_to_elliptic_latitude(90.0)
    90.0
    >>> geocentric_to_elliptic_latitude(-90.0)
    -90.0
    >>> geocentric_to_elliptic_latitude(45.0)
    45.19242321598358
    >>> geocentric_to_elliptic_latitude(-45.0)
    -45.19242321598358
    """
    _f = (axis_a - axis_b) / axis_a
    E_2 = 2 * _f - _f ** 2

    # Singularities close to the pole and the equator. Just return the value
    # in that case.
    if abs(lat) < 1E-6 or abs(lat - 90) < 1E-6 or \
            abs(lat + 90.0) < 1E-6:
        return lat

    return math.degrees(math.atan(math.tan(math.radians(lat)) / (1 - E_2)))


def sizeof_fmt(num):
    """
    Handy formatting for human readable filesizes.

    From http://stackoverflow.com/a/1094933/1657047
    """
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, "TB")


def io_chunker(arr):
    """
    Assumes arr is an array of indices. Will return indices thus that
    adjacent items can be read in one go. Much faster for some cases!
    """
    idx = []
    for _i in range(len(arr)):
        if _i == 0:
            idx.append(arr[_i])
            continue
        diff = arr[_i] - arr[_i - 1]
        if diff == 1:
            if isinstance(idx[-1], list):
                idx[-1][-1] += 1
            else:
                idx[-1] = [idx[-1], idx[-1] + 2]
        else:
            idx.append(arr[_i])
    return idx


def rfftfreq(n, d=1.0):  # pragma: no cover
    """
    Polyfill for numpy's rfftfreq() for numpy versions that don't have it.
    """
    if hasattr(np.fft, "rfftfreq"):
        return np.fft.rfftfreq(n=n, d=d)

    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


def c_ijkl_ani(lbd, mu, xi_ani, phi_ani, eta_ani, theta_fa, phi_fa,
                i, j, k, l):
    """
    Outputs the ijkl element of the elasticity tensor (for a hexagonal solid) 
    from the elastic parameters lambda, mu, xi, phi and eta, and the angles 
    theta and phi that define the rotation axis.
    
    See Nolet (2008) equation 16.2
    """
    deltaf = np.zeros([3, 3])
    deltaf[0, 0] = 1.
    deltaf[1, 1] = 1
    deltaf[2, 2] = 1

    s = np.zeros(3)  # for transverse anisotropy
    s[0] = math.cos(phi_fa) * math.sin(theta_fa)  # 0.0
    s[1] = math.sin(phi_fa) * math.sin(theta_fa)  # 0.0
    s[2] = math.cos(theta_fa)  # 1.0

    c_ijkl_ani = 0.0

    # isotropic part:
    c_ijkl_ani += lbd * deltaf[i, j] * deltaf[k, l]

    c_ijkl_ani += mu * (deltaf[i, k] * deltaf[j, l]
                        + deltaf[i, l] * deltaf[j, k])

    # anisotropic part in xi, phi, eta
    c_ijkl_ani += ((eta_ani - 1.0) * lbd + 2.0 * eta_ani * mu *
                   (1.0 - 1.0 / xi_ani)) * (deltaf[i, j] * s[k] * s[l]
                                            + deltaf[k, l] * s[i] * s[j])

    c_ijkl_ani += mu * (1.0 / xi_ani - 1.0) *\
                  (deltaf[i, k] * s[j] * s[l]
                   + deltaf[i, l] * s[j] * s[k]
                   + deltaf[j, k] * s[i] * s[l]
                   + deltaf[j, l] * s[i] * s[k])

    c_ijkl_ani += ((1.0 - 2.0 * eta_ani + phi_ani) * (lbd + 2.0 * mu)
                   + (4. * eta_ani - 4.) * mu / xi_ani)\
                   * (s[i] * s[j] * s[k] * s[l])

    return c_ijkl_ani
