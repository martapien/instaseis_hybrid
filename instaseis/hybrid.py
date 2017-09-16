#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid classes of Instaseis.

:copyright:
    Marta Pienkowska-Cote (marta.pienkowska@earth.ox.ac.uk), 2017
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import h5py
from math import cos, sin

from . import rotations
from .source import Receiver


def hybrid_generate_output(outputfile, inputfile, source, database, dt,
                           filter_freqs=None,
                           dumpfields=("velocity", "strain"),
                           chunking="points", compression=5):
    # ToDo add , fileformat="hdf5" and let "netcdf" be an option??
    # :param fileformat: Format of the output file. Possible formats "hdf5" or
    #     "netcdf". Defaults to "hdf5".
    # type fileformat: str, optional
    """
    A method to generate the hdf5 file with the background field for the local
    hybrid simulation. Dumps displacements, velocities, strains or 
    tractions.

    :param outputfile: A path to the output .hdf5 file that is going to be 
        created, e.g. "/home/user/hybrid_output.hdf5"
    :type outputfile: str
    :param inputfile: A path to a .txt or .hdf5 file with spherical/local
        coordinates of points on the boundary of the local domain of the hybrid
        method. Txt file: three columns (radius, latitude, longitude). Hdf5
        file: group spherical with dataset coordinates ([npoints,3], 
        where the second dimension is rtp and with attribute 
        points-number defining the total number of boundary points.
    :type inputfile: str
    :param source: The source of the hybrid simulation.
    :type source: :class: '~instaseis.source.Source' object
    :param database: A forward Instaseis database to extract fields on the
        boundary of the local hybrid domain.
    :type database: :class: '~instaseis.InstaseisDB' object
    :param dt:
    :type dt: 
    :param filter_freqs: [0] - lowpass; [1] - highpass
    :type filter_freqs: 
    :param dumpfields: Which fields to dump. Possible options: "velocity", 
    "strain", "traction", "displacement". Defaults to ("velocity, strain").
    :type dumpfields: tuple of str, optional
    :param chunking: Flag to define the hdf5 chunking scheme. Possible 
        options are "points"(the fast read is a single time step for all 
        points on the boundary) and "times" (the fast read is an entire 
        time series for a single point on the boundary). Defaults to "points".
    :type chunking: str
    """

    if database.info.is_reciprocal:
        raise ValueError('Extraction of background wavefield requires a '
                         'forward Instaseis database.')

    f_in = h5py.File(inputfile, "r")
    f_out = h5py.File(outputfile, "w")

    if "spherical" in f_in:
        receivers = _make_receivers_from_spherical(inputfile)
    elif "local" in f_in:
        receivers = _make_receivers_from_local(inputfile)
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver. The latter needs to include a'
                                  'rotation matrix to spherical')

    # Check the bounds of the receivers vs the database
    _database_bounds_checks(receivers, database)

    if "traction" in dumpfields:
        if "spherical" in f_in:
            normals = f_in['spherical/normals'][:, :]  # review in tpr
        elif "local" in f_in:
            normals = f_in['local/normals'][:, :]
            rotmat = f_in['local'].attrs['rotation-matrix']
            normals = np.dot(normals, rotmat)
        mu_all = f_in['elastic_params/mu']
        lbd_all = f_in['elastic_params/lambda']
        xi_all = f_in['elastic_params/xi']
        phi_all = f_in['elastic_params/phi']
        eta_all = f_in['elastic_params/eta']

    npoints = len(receivers)
    ntimesteps = _get_ntimesteps(database, source, receivers[0], dt,
                                 filter_freqs)

    if chunking == "points":
        chunks = (npoints, 1, 3)
    elif chunking == "times":
        chunks = (1, ntimesteps, 3)
    else:
        raise NotImplementedError("Unknown chunking flag.")

    grp = f_out.create_group("spherical")
    grp.attrs['points-number'] = npoints

    if "velocity" in dumpfields:
        dset_vel = grp.create_dataset("velocity", (npoints, ntimesteps, 3),
                                      chunks=chunks,
                                      compression="gzip",
                                      compression_opts=compression)
        dset_vel.attrs['dt'] = dt
    if "displacement" in dumpfields:
        dset_disp = grp.create_dataset("displacement",
                                       (npoints, ntimesteps, 3),
                                       chunks=chunks,
                                       compression="gzip",
                                       compression_opts=compression)
        dset_disp.attrs['dt'] = dt
    if "strain" in dumpfields:
        dset_strn = grp.create_dataset("strain", (npoints, ntimesteps, 6),
                                       chunks=chunks,
                                       compression="gzip",
                                       compression_opts=compression)
        dset_strn.attrs['dt'] = dt
    if "traction" in dumpfields:
        dset_trac = grp.create_dataset("traction", (npoints, ntimesteps, 3),
                                       chunks=chunks,
                                       compression="gzip",
                                       compression_opts=compression)
        dset_trac.attrs['dt'] = dt

    for i in np.arange(npoints):
        data = database.get_data_hybrid(source, receivers[i], dt, dumpfields,
                                        filter_freqs=filter_freqs)

        if "velocity" in dumpfields:
            dset_vel[i, :, :] = data["velocity"]
        if "displacement" in dumpfields:
            dset_disp[i, :, :] = data["displacement"]
        if "strain" in dumpfields:
            dset_strn[i, :, :] = data["strain"]
        if "traction" in dumpfields:
            strain = data["strain"]
            e_tt = np.array(strain[:, 0])
            e_pp = np.array(strain[:, 1])
            e_rr = np.array(strain[:, 2])
            e_rp = np.array(strain[:, 3])
            e_rt = np.array(strain[:, 4])
            e_tp = np.array(strain[:, 5])

            n = normals[i, :]
            traction = np.zeros(ntimesteps, 3)

            mu = mu_all[i]
            lbd = lbd_all[i]
            xi = xi_all[i]
            phi = phi_all[i]
            eta = eta_all[i]
            fa_ani_thetal = 0.0
            fa_ani_phil = 0.0

            c_11 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 0, 0, 0, 0)
            c_12 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 0, 0, 1, 1)
            c_13 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 0, 0, 2, 2)
            c_15 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 0, 0, 2, 0)
            c_22 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 1, 1, 1, 1)
            c_23 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 1, 1, 2, 2)
            c_25 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 1, 1, 2, 0)
            c_33 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 2, 2, 2, 2)
            c_35 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 2, 2, 2, 0)
            c_44 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 1, 2, 1, 2)
            c_46 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 1, 2, 0, 1)
            c_55 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 2, 0, 2, 0)
            c_66 = _c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                                    fa_ani_phil, 0, 1, 0, 1)

            traction[:, 0] = n[0] * (c_11 * e_tt + 2.0 * c_15 * e_rt +
                                     c_12 * e_pp + c_13 * e_rr) + \
                             n[1] * 2.0 * (c_66 * e_tp + c_46 * e_rp) + \
                             n[2] * (c_15 * e_tt + c_25 * e_pp + c_35 * e_rr
                                     + 2.0 * c_55 * e_rt)

            traction[:, 1] = n[0] * 2.0 * (c_66 * e_tp + c_46 * e_rp) + \
                             n[1] * (c_12 * e_tt + 2.0 * c_25 * e_rt +
                                     c_22 * e_pp + c_23 * e_rr) + \
                             n[2] * 2.0 * (c_46 * e_tp + c_44 * e_rp)

            traction[:, 2] = n[0] * (c_15 * e_tt + 2.0 * c_55 * e_rt +
                                     c_25 * e_pp + c_35 * e_rr) + \
                             n[1] * 2.0 * (c_46 * e_tp + c_44 * e_rp) + \
                             n[2] * (c_13 * e_tt + 2.0 * c_35 * e_rt +
                                     c_23 * e_pp + c_33 * e_rr)

            dset_trac[i, :, :] = traction

    f_out.close()


def _make_receivers_from_spherical(inputfile):
    """
    Method to handle hybrid boundary input (in spherical coordinates) defined 
    by the mesh of a local solver.
    :param inputfile: path to a .txt or .hdf5 file with spherical/local
        coordinates of points on the boundary of the local domain of the hybrid
        method. See more in hybrid_generate_output.
    :type inputfile: str
    """
    receivers = []
    if inputfile.endswith('.txt'):
        f = open(inputfile, "r")
        for line in f:
            rad, lat, lon = line.split()
            lat = float(lat)
            lon = float(lon)
            depth = (6371.0 - float(rad)) * 1000
            receivers.append(Receiver(
                latitude=lat,
                longitude=lon,
                depth_in_m=depth))
        f.close()

    elif inputfile.endswith('.hdf5'):
        f = h5py.File(inputfile, 'r')
        if "spherical/coordinates" not in f:
            raise ValueError('spherical/coordinates not found in file')
        coords = f['spherical/coordinates'][:, :]  # review tpr
        items = f['spherical'].attrs['points-number']

        for i in np.arange(items):
            lat = 90.0 - coords[i, 0]
            lon = coords[i, 1]
            dep = (6371000.0 - coords[i, 2])
            receivers.append(Receiver(
                latitude=lat,
                longitude=lon,
                depth_in_m=dep))
        f.close()
    else:
        raise NotImplementedError('Provide input as .txt or .hdf5 file')

    return receivers


def _make_receivers_from_local(inputfile):
    """
    Method to handle hybrid boundary input (in local coordinates) defined 
    by the mesh of a local solver.
    :param inputfile: path to a .hdf5 file with coordinates of points on the 
        boundary of the local domain of the hybrid method. See more in 
        hybrid_generate_output.
    :type inputfile: str
    """
    receivers = []
    if not inputfile.endswith('.hdf5'):
        raise NotImplementedError('In local coordinates please '
                                  'provide a hdf5 file.')
    f = h5py.File(inputfile, 'r')
    if "local/coordinates" not in f:
        raise ValueError('local/coordinates not found in file')

    coords = f['local/coordinates'][:, :]
    # review does this store attribute or just point to it? TO TEST
    rotmat = f['local'].attrs['rotation-matrix']
    items = f['local'].attrs['points-number']

    # review rotate local into global spherical tpr
    coords = np.dot(coords, rotmat)

    for i in np.arange(items):
        lat = coords[i, 0]
        lon = coords[i, 1]
        dep = (6371000.0 - coords[i, 2])
        receivers.append(Receiver(
            latitude=lat,
            longitude=lon,
            depth_in_m=dep))
    f.close()

    return receivers


def _database_bounds_checks(receivers, database):

    db_min_depth = database.info.planet_radius - database.info.max_radius
    db_max_depth = database.info.planet_radius - database.info.min_radius
    db_min_colat = database.info.min_d
    db_max_colat = database.info.max_d
    db_min_lat = 90 - db_max_colat
    db_max_lat = 90 - db_min_colat
    min_depth = min(_i.depth_in_m for _i in receivers)
    max_depth = max(_i.depth_in_m for _i in receivers)
    min_lat = min(_i.latitude for _i in receivers)
    max_lat = max(_i.latitude for _i in receivers)

    if not (db_min_depth <= min_depth <= db_max_depth):
        raise ValueError("The shallowest receiver to construct a hybrid src"
                         " is %.1f km deep. The database only has a depth "
                         "range from %.1f km to %.1f km." % (
                             min_depth / 1000.0, db_min_depth / 1000.0,
                             db_max_depth / 1000.0))

    if not (db_min_depth <= max_depth <= db_max_depth):
        raise ValueError("The deepest receiver to construct a hybrid src"
                         " is %.1f km deep. The database only has a depth "
                         "range from %.1f km to %.1f km." % (
                             max_depth / 1000.0, db_min_depth / 1000.0,
                             db_max_depth / 1000.0))

    if not (db_min_lat <= min_lat <= db_max_lat):
        raise ValueError("Smallest receiver latitude is %.1f deg. The database "
                         "only has a latitude range from %.1f deg to %.1f "
                         "deg." % (min_lat, db_min_lat, db_max_lat))

    if not (db_min_lat <= max_lat <= db_max_lat):
        raise ValueError("Largest receiver latitude is %.1f deg. The database "
                         "only has a latitude range from %.1f deg to %.1f "
                         "deg." % (max_lat, db_min_lat, db_max_lat))


def _get_ntimesteps(database, source, receiver, dt, filter_freqs):
    data = database.get_data_hybrid(source, receiver, dt, dumpfields=(
        "displacement"), filter_freqs=filter_freqs)
    disp = data["displacement"][:, 0]

    return len(disp)


# review rethink structure as this function is defined in source.py (HybridS.)
def _c_ijkl_ani(lbd, mu, xi_ani, phi_ani, eta_ani, theta_fa, phi_fa,
                i, j, k, l):

    deltaf = np.zeros([3,3])
    deltaf[0, 0] = 1.
    deltaf[1, 1] = 1
    deltaf[2, 2] = 1

    s = np.zeros(3)  # for transverse anisotropy
    s[0] = cos(phi_fa) * sin(theta_fa)  # 0.0
    s[1] = sin(phi_fa) * sin(theta_fa)  # 0.0
    s[2] = cos(theta_fa)  # 1.0

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


class HybridReceiversBoundaryInternalTest(object):
    """Instaseis Internal Test
     a class to generate a network of receivers"""
    def __init__(self, latitude, longitude, depth_in_m, savepath, radius=45000,
                 recursion_level=3):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.depth_in_m = float(depth_in_m)
        self.radius = float(radius)
        self.network = self._network_on_sphere(latitude, longitude,
                                               depth_in_m, radius,
                                               recursion_level,
                                               savepath)

    def _network_on_sphere(self, latitude, longitude,
                           depth_in_m, radius, recursion_level,
                           savepath):
        """get list of receivers on a triangulated sphere"""
        x, y, z = \
            rotations.coord_transform_lat_lon_depth_to_xyz(latitude,
                                                           longitude,
                                                           depth_in_m)

        octahedron_vertices = np.array([
            [1, 0, 0],  # 0
            [-1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, -1, 0],  # 3
            [0, 0, 1],  # 4
            [0, 0, -1]  # 5
        ])

        octahedron_triangles = np.array([
            [0, 4, 2],
            [2, 4, 1],
            [1, 4, 3],
            [3, 4, 0],
            [0, 2, 5],
            [2, 1, 5],
            [1, 3, 5],
            [3, 0, 5]])

        vertices, triangles, centroids = self.make_sphere(
            octahedron_vertices, octahedron_triangles, radius,
            recursion_level)

        # shift the sphere
        vertices = np.hstack((vertices[:, 0] + x, vertices[:, 1] + y,
                              vertices[:, 2] + z)).reshape((len(vertices),
                                                            -1), order='F')
        centroids = np.hstack((centroids[:, 0] + x, centroids[:, 1] + y,
                              centroids[:, 2] + z)).reshape((len(centroids),
                                                            -1), order='F')

        receivers = []
        tpr = np.zeros([centroids.shape[0], 3])
        counter = 0
        for triangle in centroids:
            xx, yy, zz = triangle[:]
            lat, lon, depth = \
                rotations.coord_transform_xyz_to_lat_lon_depth(xx, yy, zz)
            receivers.append(Receiver(
                latitude=lat,
                longitude=lon,
                depth_in_m=depth))
            # save hdf5:
            tpr[counter, 0] = 90.0 - lat
            tpr[counter, 1] = lon
            tpr[counter, 2] = 6371000.0 - depth
            counter += 1

        normals, areas = self.sphere_surface_vars(vertices, triangles)

        # save hdf5:
        for i in np.arange(len(receivers)):
            normals[i, :] = rotations.rotate_vector_xyz_earth_to_xyz_src(
                normals[i, :], receivers[i].longitude_rad,
                receivers[i].colatitude_rad)
        normals *= -1.0
        f = h5py.File(savepath, 'w')
        grp = f.create_group("spherical")
        dset = grp.create_dataset("coordinates", data=tpr,
                                  compression="gzip", compression_opts=4)
        dset = grp.create_dataset("normals", data=normals,
                                  compression="gzip", compression_opts=4)
        dset = grp.create_dataset("weights", data=areas,
                                  compression="gzip", compression_opts=4)
        grp.attrs['points-number'] = centroids.shape[0]
        f.close()

        receivers = [receivers, normals, areas, centroids]

        return receivers

    def make_sphere(self, vertices, triangles, radius, recursion_level):
        vertex_array, index_array = vertices, triangles
        for i in range(recursion_level - 1):
            vertex_array, index_array = self.triangulate(vertex_array,
                                                         index_array)
        # multiply unit sphere by radius
        vertex_array *= radius

        centroid = np.zeros((len(index_array), 3))
        for i in range(len(index_array)):
            triangle = np.array(
                [vertex_array[index_array[i, 0]],
                 vertex_array[index_array[i, 1]],
                 vertex_array[index_array[i, 2]]])
            centroid[i, :] = np.dot(np.transpose(triangle),
                                  1 / 3. * np.ones(3))

        return vertex_array, index_array, centroid

    def sphere_surface_vars(self, vertex, triangle):
        n = len(triangle)
        normals = np.zeros((n, 3))
        areas = np.zeros(n)
        for i in range(n):
            y = vertex[triangle[i]]
            vect = np.array([y[1] - y[0], y[2] - y[1], y[0] - y[2]])
            normals[i, :] = np.cross(vect[0], vect[2])
            norm_normals = np.linalg.norm(normals[i, :])
            areas[i] = norm_normals / 2.0
            normals[i, :] /= norm_normals
        return normals, areas

    def triangulate(self, vertices, triangles):
        """
        Subdivide each triangle in the old approximation.
        Each input triangle (vertices [0,1,2]) is subdivided into four new
        triangles:
                1
               /\
              /  \
            b/____\ c
            /\    /\
           /  \  /  \
          /____\/____\
         0      a     2
        """
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        a = 0.5 * (v0 + v2)
        b = 0.5 * (v0 + v1)
        c = 0.5 * (v1 + v2)
        self.normalize(a)
        self.normalize(b)
        self.normalize(c)

        # Stack triangles together. See that vertices are duplicated.
        vertices = np.hstack(
            (v0, b, a, b, v1, c, a, b, c, a, c, v2)).reshape((-1, 3))

        return vertices, np.arange(len(vertices)).reshape((-1, 3))

    def normalize(self, arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr
