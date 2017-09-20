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
import netCDF4

from . import rotations
from .source import Receiver
from .helpers import c_ijkl_ani


def hybrid_generate_output(outputfile, inputfile, source, database, dt=None,
                           remove_source_shift=True,
                           reconvolve_stf=False,
                           filter_freqs=None,
                           dumpfields=("velocity", "strain"),
                           chunking="points", compression=4,
                           fileformat="hdf5"):
    """
    A method to generate the hdf5/netcdf file with the input (background, 
    injected) field for a local hybrid simulation. Dumps displacements, 
    velocities, strains or tractions.

    :param outputfile: A path to the output hdf5/netcdf file that is 
        created, e.g. "/home/user/hybrid_output.hdf5". The output file includes
        group 'spherical' with datasets corrseponding to dumpfields, 
        e.g. (by default):
        ['spherical/velocity'] and ['spherical/strain']
    :type outputfile: string
    :param inputfile: A path to a text or hdf5/netcdf file with spherical/local
        coordinates of points on the boundary of the local domain of the hybrid
        method. Txt file: three columns (radius, latitude, longitude). 
        Hdf5/netcdf file: group spherical with dataset coordinates 
        ([npoints, 3], where the second dimension is tpr, and with attribute 
        points_number defining the total number of gll boundary points. If 
        coordinates are in the local frame of reference, dataset spherical 
        requires an attribute rotation_matrix for right-multiplication to 
        rotate to tpr.
    :type inputfile: string
    :param source: The source of the hybrid simulation.
    :type source: :class: '~instaseis.source.Source' object
    :param database: A forward Instaseis database to extract fields on the
        boundary of the local hybrid domain.
    :type database: :class: '~instaseis.InstaseisDB' object
    :param dt: Desired sampling rate of the dumped fields. Resampling is 
        done using a Lanczos kernel. If None, defaults to the dt of the 
        daatbase. 
    :type dt: float, optional
    :param remove_source_shift: Cut all samples before the peak of the
            source time function. This has the effect that the first sample
            is the origin time of the source. Defaults to True.
    :type remove_source_shift: bool, optional
    :param reconvolve_stf: Deconvolve the source time function used in
            the AxiSEM run and convolve with the STF attached to the source.
            For this to be stable, the new STF needs to bandlimited. 
            Defaults to False.
    :type reconvolve_stf: bool, optional
    :param filter_freqs: A tuple (freq_min, freq_max) to bandpass filter 
            AxiSEM data. Defaults to None. 
    :type filter_freqs: tuple, optional
     :param dumpfields: Which fields to dump. Must be a tuple
            containing any combination of ``"displacement"``, ``"velocity"``, 
            ``"strain"``, and ``"traction"``. Defaults to ``"velocity"`` and 
            ``"traction"``.
    :type dumpfields: tuple of string, optional
    :param chunking: Flag to define the hdf5 chunking scheme. Possible 
        options are "points" (the fast read is a single time step for all 
        points on the boundary) and "times" (the fast read is an entire 
        time series for a single point on the boundary). Defaults to "points".
    :type chunking: string, optional
    :param compression: Compression level of gzip for hdf5/netcdf. 
        May be an integer from 0 to 9, default is 4.
    :type compression: integer, optional
    :param fileformat: Format of the output file. Possible formats "hdf5" or
        "netcdf". Defaults to "hdf5".
    :type fileformat: string, optional
    """

    if database.info.is_reciprocal:
        raise ValueError('Extraction of background wavefield requires a '
                         'forward Instaseis database.')

    f_in = h5py.File(inputfile, "r")

    if fileformat == "hdf5":
        if not outputfile.endswith('.hdf5'):
            raise ValueError("An hdf5 output file format required.")
        f_out = h5py.File(outputfile, "w")
    elif fileformat == "netcdf":
        if not outputfile.endswith('.nc'):
            raise ValueError("An nc output file format required.")
        f_out = netCDF4.Dataset(outputfile, "w", format="NETCDF4")
    else:
        raise NotImplementedError("Only hdf5 and netcdf outputs allowed.")

    if "spherical" in f_in:
        receivers = _make_receivers_from_spherical(inputfile)
    elif "local" in f_in:
        receivers = _make_receivers_from_local(inputfile)
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver. The latter needs to include a'
                                  'rotation matrix to spherical (tpr).')

    # Check the bounds of the receivers vs the database
    _database_bounds_checks(receivers, database)

    if "traction" in dumpfields:
        if "spherical" in f_in:
            normals = f_in['spherical/normals'][:, :]  # review in tpr
        elif "local" in f_in:
            normals = f_in['local/normals'][:, :]
            rotmat = f_in['local'].attrs['rotation_matrix']
            normals = np.dot(normals, rotmat)
        mu_all = f_in['elastic_params/mu']
        lbd_all = f_in['elastic_params/lambda']
        xi_all = f_in['elastic_params/xi']
        phi_all = f_in['elastic_params/phi']
        eta_all = f_in['elastic_params/eta']

    npoints = len(receivers)
    ntimesteps = _get_ntimesteps(database, source, receivers[0], dt,
                                 filter_freqs, remove_source_shift)

    if fileformat == "hdf5":
        if chunking == "points":
            chunks_vect = (npoints, 1, 3)
            chunks_tens = (npoints, 1, 6)
        elif chunking == "times":
            chunks_vect = (1, ntimesteps, 3)
            chunks_tens = (1, ntimesteps, 6)
        else:
            raise NotImplementedError("Unknown chunking flag.")

        grp = f_out.create_group("spherical")
        grp.attrs['points_number'] = npoints
        if dt is not None:
            grp.attrs['dt'] = dt
        else:
            grp.attrs['dt'] = database.info.dt

        if "velocity" in dumpfields:
            dset_vel = grp.create_dataset("velocity", (npoints, ntimesteps, 3),
                                          chunks=chunks_vect,
                                          compression="gzip",
                                          compression_opts=compression)
        if "displacement" in dumpfields:
            dset_disp = grp.create_dataset("displacement",
                                           (npoints, ntimesteps, 3),
                                           chunks=chunks_vect,
                                           compression="gzip",
                                           compression_opts=compression)
        if "strain" in dumpfields:
            dset_strn = grp.create_dataset("strain", (npoints, ntimesteps, 6),
                                           chunks=chunks_tens,
                                           compression="gzip",
                                           compression_opts=compression)
        if "traction" in dumpfields:
            dset_trac = grp.create_dataset("traction", (npoints, ntimesteps, 3),
                                           chunks=chunks_vect,
                                           compression="gzip",
                                           compression_opts=compression)
    else:

        if chunking == "points":
            chunks_vect = (npoints, 1, 3)
            chunks_tens = (npoints, 1, 6)
        elif chunking == "times":
            chunks_vect = (1, ntimesteps, 3)
            chunks_tens = (1, ntimesteps, 6)
        else:
            raise NotImplementedError("Unknown chunking flag.")

        grp = f_out.createGroup("spherical")
        grp.points_number = npoints
        if dt is not None:
            grp.dt = dt
        else:
            grp.dt = database.info.dt

        grp.createDimension("points", npoints)
        grp.createDimension("timesteps", ntimesteps)
        grp.createDimension("vector", 3)
        grp.createDimension("tensor", 6)

        if "velocity" in dumpfields:
            dset_vel = grp.createVariable("velocity", np.float64,
                                          ("points", "timesteps", "vector"),
                                          chunksizes=chunks_vect,
                                          zlib=True,
                                          complevel=compression)
        if "displacement" in dumpfields:
            dset_disp = grp.createVariable("displacement", np.float64,
                                           ("points", "timesteps", "vector"),
                                           chunksizes=chunks_vect,
                                           zlib=True,
                                           complevel=compression)
        if "strain" in dumpfields:
            dset_strn = grp.createVariable("strain", np.float64,
                                           ("points", "timesteps", "tensor"),
                                           chunksizes=chunks_tens,
                                           zlib=True,
                                           complevel=compression)
        if "traction" in dumpfields:
            dset_trac = grp.createVariable("traction", np.float64,
                                           ("points", "timesteps", "vector"),
                                           chunksizes=chunks_vect,
                                           zlib=True,
                                           complevel=compression)

    for i in np.arange(npoints):
        data = database.get_data_hybrid(source, receivers[i], dumpfields,
                                        remove_source_shift=remove_source_shift,
                                        reconvolve_stf=reconvolve_stf, dt=dt,
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
            # review only transverse isotropy in this case
            fa_ani_thetal = 0.0
            fa_ani_phil = 0.0

            c_11 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 0, 0, 0, 0)
            c_12 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 0, 0, 1, 1)
            c_13 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 0, 0, 2, 2)
            c_15 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 0, 0, 2, 0)
            c_22 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 1, 1, 1, 1)
            c_23 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 1, 1, 2, 2)
            c_25 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 1, 1, 2, 0)
            c_33 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 2, 2, 2, 2)
            c_35 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 2, 2, 2, 0)
            c_44 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 1, 2, 1, 2)
            c_46 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 1, 2, 0, 1)
            c_55 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
                              fa_ani_phil, 2, 0, 2, 0)
            c_66 = c_ijkl_ani(lbd, mu, xi, phi, eta, fa_ani_thetal,
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

    f_out.close()  # OK for netcdf too


def _make_receivers_from_spherical(inputfile):
    """
    Method to handle hybrid boundary input (in spherical coordinates) defined 
    by the mesh of a local solver.
    :param inputfile: path to a text or hdf5/netcdf file with spherical/local
        coordinates of points on the boundary of the local domain of the hybrid
        method. See more in hybrid_generate_output.
    :type inputfile: string
    """
    receivers = []
    if inputfile.endswith('.txt'):
        f = open(inputfile, "r")
        for line in f:
            rad, lat, lon = line.split()
            lat = float(lat)
            lon = float(lon)
            depth = (6371.0 - float(rad)) * 1000.0
            receivers.append(Receiver(
                latitude=lat,
                longitude=lon,
                depth_in_m=depth))
        f.close()

    elif inputfile.endswith('.hdf5') or inputfile.endswith('.nc'):
        f = h5py.File(inputfile, 'r')
        if "spherical/coordinates" not in f:
            raise ValueError('spherical/coordinates not found in file')
        coords = f['spherical/coordinates'][:, :]  # review tpr
        items = f['spherical'].attrs['points_number']
        if type(items) is np.ndarray:
            items = items[0]

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
        raise NotImplementedError('Provide input as txt, hdf5 or netcdf file.')

    return receivers


def _make_receivers_from_local(inputfile):
    """
    Method to handle hybrid boundary input (in local coordinates) defined 
    by the mesh of a local solver.
    :param inputfile: path to a hdf5/netcdf file with coordinates of points on 
        the boundary of the local domain of the hybrid method. See more in 
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
    rotmat = f['local'].attrs['rotation_matrix']
    items = f['local'].attrs['points_number']

    if type(items) is np.ndarray:
        items = items[0]

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


def _get_ntimesteps(database, source, receiver, dt, filter_freqs,
                    remove_source_shift):
    data = database.get_data_hybrid(source, receiver, dt=dt, dumpfields=(
        "displacement"), filter_freqs=filter_freqs,
                                    remove_source_shift=remove_source_shift)
    disp = data["displacement"][:, 0]

    return len(disp)


class HybridReceiversBoundaryInternalTest(object):
    """
    Instaseis Internal Test is a class to generate a network of receivers
    (recursive generation of a sphere). Also outputs a hdf5 file with 
    coordinates. 
    """
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
        grp.attrs['points_number'] = centroids.shape[0]
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
            normals[i, :] = np.cross(vect[0], vect[2])  # outward facing
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
