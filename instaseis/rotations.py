#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions dealing with rotations.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from math import atan2, acos


def rotate_frame_rd(x, y, z, phi, theta):
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    # first rotation (longitude)
    xp_cp = x * np.cos(phi) + y * np.sin(phi)
    yp_cp = -x * np.sin(phi) + y * np.cos(phi)
    zp_cp = z

    # second rotation (colat)
    xp = xp_cp * np.cos(theta) - zp_cp * np.sin(theta)
    yp = yp_cp
    zp = xp_cp * np.sin(theta) + zp_cp * np.cos(theta)

    srd = np.sqrt(xp ** 2 + yp ** 2)
    zrd = zp
    phi_cp = np.arctan2(yp, xp)
    if phi_cp < 0.0:
        phird = 2.0 * np.pi + phi_cp
    else:
        phird = phi_cp
    return srd, phird, zrd


def rotate_symm_tensor_voigt_xyz_earth_to_xyz_src(mt, phi, theta):
    """
    rotates a tensor from a cartesian system xyz with z axis aligned with the
    north pole to a cartesian system x,y,z where z is aligned with the source /
    receiver

    input symmetric tensor in voigt notation:
    A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    rotation matrix from TNM 2007 eq 14
    R = {{ct*cp, -sp, st*cp}, {ct*sp , cp, st*sp}, {-st, 0, ct}}

    compute and ouput in voigt notation:
    Rt.A.R
    """
    A = np.array([[mt[0], mt[5], mt[4]],
                  [mt[5], mt[1], mt[3]],
                  [mt[4], mt[3], mt[2]]])

    ct = np.cos(theta)
    cp = np.cos(phi)
    st = np.sin(theta)
    sp = np.sin(phi)

    R = np.array([[ct * cp, -sp, st * cp],
                  [ct * sp, cp, st * sp],
                  [-st, 0, ct]])

    # This double matrix product involves number that might differ by 20
    # orders of magnitudes which makes it numerically tricky. Thus we employ
    # quad precision numbers to make it a bit more stable and reproducable.
    R = np.require(R, dtype=np.float128)
    A = np.require(A, dtype=np.float128)

    B = np.dot(np.dot(R.T, A), R)

    # Convert back to single precision.
    return np.require(
        np.array([B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]]),
        dtype=np.float64)


def rotate_symm_tensor_voigt_xyz_src_to_xyz_earth(mt, phi, theta):
    """
    rotates a tensor from a cartesian system xyz with z axis aligned with the
    source / receiver to a cartesian system x,y,z where z is aligned with the
    north pole

    input symmetric tensor in voigt notation:
    A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    rotation matrix from TNM 2007 eq 14
    R = {{ct*cp, -sp, st*cp}, {ct*sp , cp, st*sp}, {-st, 0, ct}}

    compute and ouput in voigt notation:
    R.A.Rt
    """
    A = np.array([[mt[0], mt[5], mt[4]],
                  [mt[5], mt[1], mt[3]],
                  [mt[4], mt[3], mt[2]]])

    ct = np.cos(theta)
    cp = np.cos(phi)
    st = np.sin(theta)
    sp = np.sin(phi)

    R = np.array([[ct * cp, -sp, st * cp],
                  [ct * sp, cp, st * sp],
                  [-st, 0, ct]])

    B = np.dot(np.dot(R, A), R.T)
    return np.array([B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]])


def rotate_symm_tensor_voigt_xyz_to_src(mt, phi):
    """
    rotates a tensor from a cartesian system x,y,z where z is aligned with the
    source and x with phi = 0 to the AxiSEM s, phi, z system aligned with the
    source on the s = 0 axis

    input symmetric tensor in voigt notation:
    A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    rotation matrix
    R = {{Cos[phi], Sin[phi], 0}, {-Sin[phi] , Cos[phi], 0}, {0, 0, 1}};

    compute and ouput in voigt notation:
    R.A.Rt
    """
    A = np.array([[mt[0], mt[5], mt[4]],
                  [mt[5], mt[1], mt[3]],
                  [mt[4], mt[3], mt[2]]])

    cp = np.cos(phi)
    sp = np.sin(phi)

    R = np.array([[cp, sp, 0.], [-sp, cp, 0], [0, 0, 1.]])

    B = np.dot(np.dot(R, A), R.T)
    return np.array([B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]])


def rotate_symm_tensor_voigt_src_to_xyz(mt, phi):
    """
    rotates a tensor from the AxiSEM s, phi, z system aligned with the
    source on the s = 0 axis to a cartesian system x,y,z where z is aligned
    with the source and x with phi = 0

    input symmetric tensor in voigt notation:
    A = {{a1, a6, a5}, {a6, a2, a4}, {a5, a4, a3}};
    rotation matrix
    R = {{Cos[phi], -Sin[phi], 0}, {Sin[phi] , Cos[phi], 0}, {0, 0, 1}};

    compute and ouput in voigt notation:
    R.A.Rt
    """
    A = np.array([[mt[0], mt[5], mt[4]],
                  [mt[5], mt[1], mt[3]],
                  [mt[4], mt[3], mt[2]]])

    cp = np.cos(phi)
    sp = np.sin(phi)

    R = np.array([[cp, -sp, 0.], [sp, cp, 0], [0, 0, 1.]])

    B = np.dot(np.dot(R, A), R.T)
    return np.array([B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]])


def rotate_vector_xyz_earth_to_xyz_src(vec, phi, theta):
    sp = np.sin(phi)
    cp = np.cos(phi)

    st = np.sin(theta)
    ct = np.cos(theta)

    return np.array([cp * ct * vec[0] + ct * sp * vec[1] - st * vec[2],
                     -(sp * vec[0]) + cp * vec[1],
                     cp * st * vec[0] + sp * st * vec[1] + ct * vec[2]])


def rotate_vector_xyz_src_to_xyz_earth(vec, phi, theta):
    sp = np.sin(phi)
    cp = np.cos(phi)

    st = np.sin(theta)
    ct = np.cos(theta)

    return np.array([cp * ct * vec[0] - sp * vec[1] + cp * st * vec[2],
                     ct * sp * vec[0] + cp * vec[1] + sp * st * vec[2],
                     -(st * vec[0]) + ct * vec[2]])


def rotate_vector_xyz_to_src(vec, phi):
    sp = np.sin(phi)
    cp = np.cos(phi)

    return np.array([cp * vec[0] + sp * vec[1],
                     - sp * vec[0] + cp * vec[1],
                     vec[2]])


def rotate_vector_src_to_xyz(vec, phi):
    sp = np.sin(phi)
    cp = np.cos(phi)

    return np.array([cp * vec[0] - sp * vec[1],
                     sp * vec[0] + cp * vec[1],
                     vec[2]])


def rotate_vector_src_to_NEZ(vec, phi, srclon, srccolat, reclon, reccolat):
    rotmat = np.eye(3)
    rotmat = rotate_vector_src_to_xyz(rotmat, phi)
    rotmat = rotate_vector_xyz_src_to_xyz_earth(rotmat, srclon, srccolat)
    rotmat = rotate_vector_xyz_earth_to_xyz_src(rotmat, reclon, reccolat)
    rotmat[0, :] *= -1  # N = - theta

    return np.dot(rotmat, vec)


def rotate_vector_src_to_tpr(vec, phi, srclon, srccolat, reclon, reccolat):
    # rotate from database spz with z aligned with the source to tpr
    # note that at the surface tpr = SEZ
    rotmat = np.eye(3)
    rotmat = rotate_vector_src_to_xyz(rotmat, phi)
    rotmat = rotate_vector_xyz_src_to_xyz_earth(rotmat, srclon, srccolat)
    rotmat = rotate_vector_xyz_earth_to_xyz_src(rotmat, reclon, reccolat)

    return np.dot(rotmat, vec)


def rotate_vector_xyz_src_to_xyz_rec(vec, srclon, srccolat, reclon, reccolat):
    rotmat = np.eye(3)
    rotmat = rotate_vector_xyz_src_to_xyz_earth(rotmat, srclon, srccolat)
    rotmat = rotate_vector_xyz_earth_to_xyz_src(rotmat, reclon, reccolat)

    return np.dot(rotmat, vec)


def coord_transform_lat_lon_depth_to_xyz(latitude, longitude, depth_in_m,
                                         planet_radius=6371e3):
    """
    Tansform coordinates from latitude, longitude, depth to global cartesian
    coordinates with z aligned with northpole
    """
    longitude_rad = np.radians(longitude)
    latitude_rad = np.radians(latitude)

    xyz = np.empty(3)
    xyz[0] = (planet_radius - depth_in_m) \
        * np.cos(latitude_rad) * np.cos(longitude_rad)
    xyz[1] = (planet_radius - depth_in_m) \
        * np.cos(latitude_rad) * np.sin(longitude_rad)
    xyz[2] = (planet_radius - depth_in_m) * np.sin(latitude_rad)

    return xyz


def coord_transform_xyz_to_lat_lon_depth(x, y, z, planet_radius=6371e3):
    """
    Tansform coordinates from global cartesian coordinates with z aligned with
    northpole to latitude, longitude, depth
    """

    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    longitude = np.rad2deg(phi)
    latitude = 90. - np.rad2deg(theta)
    depth_in_m = planet_radius - r

    return latitude, longitude, depth_in_m


def hybrid_coord_transform_local_cartesian_to_tpr(v, rot_mat):
    # rot_mat: xyz_local_to_xyz_global
    # local cartesian -> global xyz
    xyz = np.dot(v, rot_mat.T)
    # global xyz -> tpr
    spherical = np.zeros(xyz.shape)
    for i in np.arange(xyz.shape[0]):
        r = (xyz[i, 0] ** 2 + xyz[i, 1] ** 2 + xyz[i, 2] ** 2) ** 0.5
        p = atan2(xyz[i, 1], xyz[i, 0])  # longitude
        t = acos(xyz[i, 2] / r)       # colatitude
        spherical[i, :] = np.array([np.rad2deg(t), np.rad2deg(p), r])
        # tpr, r in metres
    return spherical

"""
def hybrid_coord_transform_tpr_to_local_cartesian(v, rot_mat):
    # rot_mat : xyz_global_to_xyz_local
    # tpr -> global xyz
    xyz = np.zeros(v.shape)
    for i in np.arange(v.shape[0]):
        v0 = np.deg2rad(v[i, 0])
        v1 = np.deg2rad(v[i, 1])
        v2 = v[i, 2]

        xyz[i, 0] = v2 * np.sin(v0) * np.cos(v1)
        xyz[i, 1] = v2 * np.sin(v0) * np.sin(v1)
        xyz[i, 2] = v2 * np.cos(v0)
    # global xyz -> local cartesian
    xyz_loc = np.dot(xyz, rot_mat.T)

    return xyz_loc
"""

def hybrid_vector_local_cartesian_to_tpr(v, rot_mat, phi, theta):

    # rot_mat: xyz_local_to_xyz_global
    # local cartesian -> global xyz
    xyz = np.dot(v, rot_mat.T)
    # global xyz -> tpr
    rot_mat_global_to_tpr = xyz_global_to_tpr(phi, theta)
    spherical = np.dot(xyz, rot_mat_global_to_tpr.T)

    return spherical


def hybrid_vector_tpr_to_local_cartesian(v, rot_mat, phi, theta):
    # rot_mat : xyz_global_to_xyz_local
    rot_mat_tpr_to_global = tpr_to_xyz_global(phi, theta)
    xyz = np.dot(v, rot_mat_tpr_to_global.T)
    xyz_loc = np.dot(xyz, rot_mat.T)
    return xyz_loc


def hybrid_vector_src_to_local_cartesian(v, rot_mat, phi, srclon, srccolat):
    rotmat = np.eye(3)
    rotmat = rotate_vector_src_to_xyz(rotmat, phi)
    rotmat = rotate_vector_xyz_src_to_xyz_earth(rotmat, srclon, srccolat)
    rotmat = np.dot(rot_mat, rotmat)
    return np.dot(rotmat, v)


def hybrid_tensor_tpr_to_local_cartesian(t, rot_mat, phi, theta):
    A = np.array([[t[0], t[5], t[4]],
                  [t[5], t[1], t[3]],
                  [t[4], t[3], t[2]]])
    # tpr -> global xyz
    rot_mat_tpr_to_global = tpr_to_xyz_global(phi, theta)
    mat_tpr_to_loc = np.dot(rot_mat, rot_mat_tpr_to_global)
    B = np.dot(np.dot(mat_tpr_to_loc, A), mat_tpr_to_loc.T)

    return np.require(np.array(
            [B[0, 0], B[1, 1], B[2, 2], B[1, 2], B[0, 2], B[0, 1]]),
            dtype=np.float64) # xyz_loc


def tpr_to_xyz_global(phi, theta):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    ct = np.cos(theta)
    cp = np.cos(phi)
    st = np.sin(theta)
    sp = np.sin(phi)

    rot_mat = np.array([[ct * cp, -sp, st * cp],
                        [ct * sp, cp, st * sp],
                        [-st, 0, ct]])

    return rot_mat


def xyz_global_to_tpr(phi, theta):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    ct = np.cos(theta)
    cp = np.cos(phi)
    st = np.sin(theta)
    sp = np.sin(phi)

    rot_mat = np.array([[ct * cp, -sp, st * cp],
                        [ct * sp, cp, st * sp],
                        [-st, 0, ct]])

    return rot_mat.T
