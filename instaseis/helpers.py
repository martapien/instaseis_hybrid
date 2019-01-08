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
from scipy.fftpack import rfft, irfft
from obspy.signal.filter import lowpass_cheby_2, lowpass
import h5py

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
    e_2 = 2 * _f - _f ** 2

    # Singularities close to the pole and the equator. Just return the value
    # in that case.
    if abs(lat) < 1E-6 or abs(lat - 90) < 1E-6 or \
            abs(lat + 90.0) < 1E-6:
        return lat

    return math.degrees(math.atan((1 - e_2) * math.tan(math.radians(lat))))


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
    e_2 = 2 * _f - _f ** 2

    # Singularities close to the pole and the equator. Just return the value
    # in that case.
    if abs(lat) < 1E-6 or abs(lat - 90) < 1E-6 or \
            abs(lat + 90.0) < 1E-6:
        return lat

    return math.degrees(math.atan(math.tan(math.radians(lat)) / (1 - e_2)))


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
    N = n // 2 + 1  # NOQA
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

    return np.float64(c_ijkl_ani)


def resample(data, old_sampling_rate, new_sampling_rate,
             old_npts, strict_length=True, no_filter=True,
             filter_type='lowpass_butterworth', maxorder=12, corners=4,
             zerophase=True):
    """
    Resample data using Fourier method. Spectra are linearly
    interpolated if required.

    :type data: array
    :param data: Data to resample.
    :type old_sampling_rate: float
    :param old_sampling_rate: The original sampling rate of the signal.
    :type new_sampling_rate: float
    :param new_sampling_rate: The sampling rate of the resampled signal.
    :type old_npts: float
    :param old_npts: Number of points of the original signal.
    :type no_filter: bool, optional
    :param no_filter: Deactivates automatic filtering if set to ``True``.
        Defaults to ``True``.
    :type filter_type: str, optional
    :param filter_type: Lowpass filter to  use if automatic filtering is set 
        to true. Possible options: ``lowpass_cheby_2`` or 
        ``lowpass_butterworth``. Defaults to  ``lowpass_butterworth``.
    :type strict_length: bool, optional
    :param strict_length: Leave traces unchanged for which end time of
        trace would change. Defaults to ``False``.

    :return: Resampled data. 
    
    Uses :func:`scipy.signal.resample`. Because a Fourier method is used,
    the signal is assumed to be periodic.

    """

    factor = float(old_sampling_rate) / float(new_sampling_rate)

    # check if end time changes and this is not explicitly allowed
    if strict_length:
        mod_check = int(len(data) % factor)
        if mod_check != 0.0:
            data = np.concatenate((data, np.zeros(mod_check)))
            mod_check2 = len(data) % factor
            if mod_check2 != 0.0:
                raise ValueError("something went wrong with padding")

    # do automatic lowpass filtering
    if not no_filter:
        # be sure filter still behaves good
        # ToDo does not seem to introduce problems (except the time shift)
        # with the specfem-Instaseis test, check with Lion what the 16 is about?
        #if factor > 16:
        #    msg = "Automatic filter design is unstable for resampling " + \
        #          "factors (current sampling rate/new sampling rate) " + \
        #          "above 16. Manual resampling is necessary."
        #    raise ArithmeticError(msg)
        freq = old_sampling_rate * 0.5 / float(factor)
        data = filter_data(data=data, filter_type=filter_type, freq=freq,
                           df=old_sampling_rate, maxorder=maxorder,
                           corners=corners, zerophase=zerophase)

    # resample in the frequency domain. Make sure the byteorder is native.
    x = rfft(data.newbyteorder("="))
    # Cast the value to be inserted to the same dtype as the array to avoid
    # issues with numpy rule 'safe'.
    x = np.insert(x, 1, x.dtype.type(0))
    if old_npts % 2 == 0:
        x = np.append(x, [0])
    x_r = x[::2]
    x_i = x[1::2]

    # interpolate
    num = int(old_npts / factor)
    df = 1.0 / old_npts * old_sampling_rate
    d_large_f = 1.0 / num * new_sampling_rate
    f = df * np.arange(0, old_npts // 2 + 1, dtype=np.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)
    large_y = np.zeros((2 * n_large_f))
    large_y[::2] = np.interp(large_f, f, x_r)
    large_y[1::2] = np.interp(large_f, f, x_i)

    large_y = np.delete(large_y, 1)
    if num % 2 == 0:
        large_y = np.delete(large_y, -1)
    data = irfft(large_y) * (float(num) / float(old_npts))

    return data


def filter_data(data, filter_type, freq, df, maxorder=12, corners=4,
                zerophase=True):

    """
    Filter data removing data over certain frequency ``freq``.
    
    :param data: Data to filter.
    :param filter_type: Type of filter to use.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param maxorder: Maximal order of the designed cheby2 filter
    :param corners: Filter corners / order for the butterworth lowpass filter.
    :param zerophase: If True, apply filter once forwards and once backwards. 
        This results in twice the number of corners but zero phase shift in 
        the resulting filtered trace.
        
    :return: Filtered data. 
    
    Uses :func:`scipy.signal.filter.lowpass_cheby_2` or  
    :func:`scipy.signal.filter.lowpass`.
    
    """

    if filter_type == "lowpass_cheby_2":
        data = lowpass_cheby_2(data=data, freq=freq, df=df,
                               maxorder=maxorder)

    elif filter_type == "lowpass_butterworth":
        data = lowpass(data=data, freq=freq, df=df, corners=corners,
                       zerophase=zerophase)

    else:
        raise ValueError("Unknown filter type.")

    return data


def resample_test(test_path, new_sampling_rate):
    # slow as reads and dumps pt by pt but just a test!

    file_in = h5py.File(test_path, 'r')

    data = file_in['local/velocity']

    npts_space = file_in['local/velocity'].shape[0]
    npts_time = file_in['local/velocity'].shape[1]
    dt = file_in['local'].attrs['dt']
    if type(dt) is np.ndarray:
        dt = dt[0]

    old_sampling_rate = 1. / dt
    file_out = h5py.File("fields_resampled", 'w')

    grp_out = file_out.create_group('local')

    dset_out = grp_out.create_dataset("velocity", file_in['local/velocity'],
                                      dtype=np.float32)

    for i in np.arange(npts_space):

        data_old = data[i, :, :]

        data_new = np.zeros_like(data_old)

        data_new[:, 0] = resample(
            data_old[:, 0], old_sampling_rate, new_sampling_rate, npts_time,
            strict_length=True, no_filter=False,
            filter_type='lowpass_butterworth', zerophase=True)
        data_new[:, 1] = resample(
            data_old[:, 1], old_sampling_rate, new_sampling_rate, npts_time,
            strict_length=True, no_filter=False,
            filter_type='lowpass_butterworth', zerophase=True)
        data_new[:, 2] = resample(
            data_old[:, 2], old_sampling_rate, new_sampling_rate, npts_time,
            strict_length=True, no_filter=False,
            filter_type='lowpass_butterworth', zerophase=True)

        dset_out[i, :, :] = data_new

    file_in.close()
    file_out.close()
    