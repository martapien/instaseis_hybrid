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
from . import rotations
from .source import (Source, ForceSource, Receiver)
from scipy.integrate import cumtrapz


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
            n = normals[i, :]
            traction = np.zeros(ntimesteps, 3)
            # review is it correct: compute traction
            traction[:, 0] = strain[:, 0] * n[0] + strain[:, 3] * n[1] + \
                             strain[:, 4] * n[2]
            traction[:, 1] = strain[:, 3] * n[0] + strain[:, 1] * n[1] + \
                             strain[:, 5] * n[2]
            traction[:, 2] = strain[:, 4] * n[0] + strain[:, 5] * n[1] + \
                             strain[:, 2] * n[2]
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


class HybridSourcesInternalTest(object):
    """
    A class to handle hybrid sources represented by force and moment
    tensor point sources. In this test, we differentiate/integrate the 
    displacement/strain (respectively) and then differentiate/integrate the 
    final result in get_seismograms_hybrid_source.

    :param receivers: The list of receivers to define appropriate point sources.
    :type receivers: list of :class:`~instaseis.Receiver` objects
    :param database: The forward database to define point sources.
    :type database: :class: '~instaseis.InstaseisDB' object
    :param source: The source of the forward/background field
    :type source: :class: '~instaseis.source.Source' object
    """

    def __init__(self, receivers=None, database=None, source=None):
        self.receivers = receivers
        self.database = database
        self.pointsources = self._from_database(receivers, database, source)

    def __len__(self):
        return len(self.pointsources)

    def __getitem__(self, index):
        return self.pointsources[index]

    def _from_database(self, receivers, database, source):
        """generate point sources from database and receivers"""

        normals = receivers[1]
        areas = receivers[2]
        receivers = receivers[0]

        # Check the bounds of the hybrid source to make sure they can be
        # calculated with the database.

        min_depth = min(_i.depth_in_m for _i in receivers)
        max_depth = max(_i.depth_in_m for _i in receivers)

        db_min_depth = self.database.info.planet_radius - \
            self.database.info.max_radius
        db_max_depth = self.database.info.planet_radius - \
            self.database.info.min_radius

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
        pointsources = []
        rec_counter = 0

        dt_old = database.info.dt
        dt = dt_old / 10.0

        if database.info.stf == "errorf":
            dumpfields = ("velocity", "strain")
        else:
            dumpfields = ("displacement", "strain")

        for rec in receivers:
            normal = normals[rec_counter]
            area = areas[rec_counter]
            rec_counter += 1

            latitude = rec.latitude
            longitude = rec.longitude
            depth_in_m = rec.depth_in_m

            data = database.get_data_hybrid(
                source=source, receiver=rec, dt=dt,
                dumpfields=dumpfields)

            if database.info.stf == "errorf":
                displ = data["velocity"]
                strain = data["strain"]
            elif database.info.stf == "gauss_0":
                displ = data["displacement"]
                strain = data["strain"]
            else:
                raise NotImplementedError('Implemented only for forward '
                                          'gauss_0 and errorf stfs')

            mu = data["elastic_params"]["mu"]
            lbd = data["elastic_params"]["lambda"]

            normal = rotations.rotate_vector_xyz_earth_to_xyz_src(
                normal, rec.longitude_rad, rec.colatitude_rad)
            normal *= -1.0

            # append moment tensor sources
            # recall voigt in tpr: Mtt Mpp Mrr Mrp Mrt Mtp

            stf0 = -np.array(displ[:, 0])
            stf1 = -np.array(displ[:, 1])
            stf2 = -np.array(displ[:, 2])

            m_tt = (lbd + 2.0 * mu) * normal[0] * area
            m_pp = lbd * normal[0] * area
            m_rr = lbd * normal[0] * area
            m_rp = 0.0
            m_rt = mu * normal[2] * area
            m_tp = mu * normal[1] * area
            pointsources.append(Source(latitude, longitude,
                                       depth_in_m=depth_in_m,
                                       m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                       m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                                       sliprate=stf0, dt=dt))
            m_tt = lbd * normal[1] * area
            m_pp = (lbd + 2.0 * mu) * normal[1] * area
            m_rr = lbd * normal[1] * area
            m_rp = mu * normal[2] * area
            m_rt = 0.0
            m_tp = mu * normal[0] * area
            pointsources.append(Source(latitude, longitude,
                                       depth_in_m=depth_in_m,
                                       m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                       m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                                       sliprate=stf1, dt=dt))
            m_tt = lbd * normal[2] * area
            m_pp = lbd * normal[2] * area
            m_rr = (lbd + 2.0 * mu) * normal[2] * area
            m_rp = mu * normal[1] * area
            m_rt = mu * normal[0] * area
            m_tp = 0.0
            pointsources.append(Source(latitude, longitude,
                                       depth_in_m=depth_in_m,
                                       m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                       m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                                       sliprate=stf2, dt=dt))

            # append force sources
            # define forces f_r, f_t, f_p from strain
            # NOTE stf0 = traction_t, stf1 = traction_p, stf2 = traction_r
            stf0 = np.array(strain[:, 0]) * normal[0] * (lbd + 2.0 * mu) \
                 + np.array(strain[:, 1]) * normal[0] * lbd \
                 + np.array(strain[:, 2]) * normal[0] * lbd \
                 + 2.0 * normal[1] * mu * np.array(strain[:, 5]) \
                 + 2.0 * normal[2] * mu * np.array(strain[:, 4])

            stf1 = 2.0 * normal[0] * mu * np.array(strain[:, 5]) \
                 + normal[1] * lbd * np.array(strain[:, 0]) \
                 + normal[1] * (lbd + 2.0 * mu) * np.array(strain[:, 1]) \
                 + normal[1] * lbd * np.array(strain[:, 2]) \
                 + 2.0 * normal[2] * mu * np.array(strain[:, 3])

            stf2 = 2.0 * normal[0] * mu * np.array(strain[:, 4]) \
                 + 2.0 * normal[1] * mu * np.array(strain[:, 3]) \
                 + normal[2] * lbd * np.array(strain[:, 0]) \
                 + normal[2] * lbd * np.array(strain[:, 1]) \
                 + normal[2] * (lbd + 2.0 * mu) * np.array(strain[:, 2])

            if database.info.stf == "gauss_0":
                stf0 = cumtrapz(stf0, dx=dt, initial=0.0)
                stf1 = cumtrapz(stf1, dx=dt, initial=0.0)
                stf2 = cumtrapz(stf2, dx=dt, initial=0.0)

            f_t = area
            f_p = area
            f_r = area
            pointsources.append(ForceSource(latitude, longitude,
                                            depth_in_m=depth_in_m,
                                            f_r=0, f_t=f_t, f_p=0,
                                            sliprate=stf0, dt=dt))
            pointsources.append(ForceSource(latitude, longitude,
                                            depth_in_m=depth_in_m,
                                            f_r=0, f_t=0, f_p=f_p,
                                            sliprate=stf1, dt=dt))
            pointsources.append(ForceSource(latitude, longitude,
                                            depth_in_m=depth_in_m,
                                            f_r=f_r, f_t=0, f_p=0,
                                            sliprate=stf2, dt=dt))

        return pointsources


class HybridSourcesInternalTest2(object):
    """
    A class to handle hybrid sources represented by force and moment
    tensor point sources. In this test, we do not differentiate/integrate the 
    displacement/strain in the stfs of the sources. We then
    differentiate/integrate accordingly the final result in 
    get_seismograms_hybrid_source2.

    :param receivers: The list of receivers to define appropriate point sources.
    :type receivers: list of :class:`~instaseis.Receiver` objects
    :param database: The forward database to define point sources.
    :type database: :class: '~instaseis.InstaseisDB' object
    :param source: The source of the forward/background field
    :type source: :class: '~instaseis.source.Source' object
    """

    def __init__(self, receivers=None, database=None, source=None):
        self.receivers = receivers
        self.database = database
        self.pointsources = self._from_database(receivers, database, source)
        self.stftype = database.info.stf

    def __len__(self):
        return len(self.pointsources)

    def __getitem__(self, index):
        return self.pointsources[index]

    def _from_database(self, receivers, database, source):
        """generate point sources from database and receivers"""

        normals = receivers[1]
        areas = receivers[2]
        receivers = receivers[0]

        # Check the bounds of the hybrid source to make sure they can be
        # calculated with the database.
        min_depth = min(_i.depth_in_m for _i in receivers)
        max_depth = max(_i.depth_in_m for _i in receivers)

        db_min_depth = self.database.info.planet_radius - \
            self.database.info.max_radius
        db_max_depth = self.database.info.planet_radius - \
            self.database.info.min_radius

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
        pointsources = []
        rec_counter = 0

        for rec in receivers:
            normal = normals[rec_counter]
            area = areas[rec_counter]
            rec_counter += 1

            latitude = rec.latitude
            longitude = rec.longitude
            depth_in_m = rec.depth_in_m
            dumpfields = ("displacement", "strain")

            data = database.get_data_hybrid(
                source=source, receiver=rec, dt=database.info.dt,
                dumpfields=dumpfields)

            displ = data["displacement"]
            strain = data["strain"]
            mu = data["elastic_params"]["mu"]
            lbd = data["elastic_params"]["lambda"]
            dt = data["dt"]  # = database.info.dt

            normal = rotations.rotate_vector_xyz_earth_to_xyz_src(
                normal, rec.longitude_rad, rec.colatitude_rad)
            normal *= -1.0

            # append moment tensor sources
            # recall voigt in tpr: Mtt Mpp Mrr Mrp Mrt Mtp

            stf0 = -np.array(displ[:, 0])
            stf1 = -np.array(displ[:, 1])
            stf2 = -np.array(displ[:, 2])

            m_tt = (lbd + 2.0 * mu) * normal[0] * area
            m_pp = lbd * normal[0] * area
            m_rr = lbd * normal[0] * area
            m_rp = 0.0
            m_rt = mu * normal[2] * area
            m_tp = mu * normal[1] * area
            pointsources.append(Source(latitude, longitude,
                                       depth_in_m=depth_in_m,
                                       m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                       m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                                       sliprate=stf0, dt=dt))
            m_tt = lbd * normal[1] * area
            m_pp = (lbd + 2.0 * mu) * normal[1] * area
            m_rr = lbd * normal[1] * area
            m_rp = mu * normal[2] * area
            m_rt = 0.0
            m_tp = mu * normal[0] * area
            pointsources.append(Source(latitude, longitude,
                                       depth_in_m=depth_in_m,
                                       m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                       m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                                       sliprate=stf1, dt=dt))
            m_tt = lbd * normal[2] * area
            m_pp = lbd * normal[2] * area
            m_rr = (lbd + 2.0 * mu) * normal[2] * area
            m_rp = mu * normal[1] * area
            m_rt = mu * normal[0] * area
            m_tp = 0.0
            pointsources.append(Source(latitude, longitude,
                                       depth_in_m=depth_in_m,
                                       m_rr=m_rr, m_tt=m_tt, m_pp=m_pp,
                                       m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
                                       sliprate=stf2, dt=dt))

            # append force sources
            # define forces f_r, f_t, f_p from strain
            # NOTE stf0 = traction_t, stf1 = traction_p, stf2 = traction_r
            stf0 = np.array(strain[:, 0]) * normal[0] * (lbd + 2.0 * mu) \
                   + np.array(strain[:, 1]) * normal[0] * lbd \
                   + np.array(strain[:, 2]) * normal[0] * lbd \
                   + 2.0 * normal[1] * mu * np.array(strain[:, 5]) \
                   + 2.0 * normal[2] * mu * np.array(strain[:, 4])

            stf1 = 2.0 * normal[0] * mu * np.array(strain[:, 5]) \
                   + normal[1] * lbd * np.array(strain[:, 0]) \
                   + normal[1] * (lbd + 2.0 * mu) * np.array(strain[:, 1]) \
                   + normal[1] * lbd * np.array(strain[:, 2]) \
                   + 2.0 * normal[2] * mu * np.array(strain[:, 3])

            stf2 = 2.0 * normal[0] * mu * np.array(strain[:, 4]) \
                   + 2.0 * normal[1] * mu * np.array(strain[:, 3]) \
                   + normal[2] * lbd * np.array(strain[:, 0]) \
                   + normal[2] * lbd * np.array(strain[:, 1]) \
                   + normal[2] * (lbd + 2.0 * mu) * np.array(strain[:, 2])

            f_t = area
            f_p = area
            f_r = area
            pointsources.append(ForceSource(latitude, longitude,
                                            depth_in_m=depth_in_m,
                                            f_r=0, f_t=f_t, f_p=0,
                                            sliprate=stf0, dt=dt))
            pointsources.append(ForceSource(latitude, longitude,
                                            depth_in_m=depth_in_m,
                                            f_r=0, f_t=0, f_p=f_p,
                                            sliprate=stf1, dt=dt))
            pointsources.append(ForceSource(latitude, longitude,
                                            depth_in_m=depth_in_m,
                                            f_r=f_r, f_t=0, f_p=0,
                                            sliprate=stf2, dt=dt))

        return pointsources


class HybridReceiversBoundaryInternalTest(object):
    """Instaseis Internal Test
     a class to generate a network of receivers"""
    def __init__(self, latitude, longitude, depth_in_m, radius=45000,
                 recursion_level=3, save_hdf5=False, savepath=None):
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.depth_in_m = float(depth_in_m)
        self.radius = float(radius)
        self.network = self._network_on_sphere(latitude, longitude,
                                               depth_in_m, radius,
                                               recursion_level,
                                               save_hdf5, savepath)

    def _network_on_sphere(self, latitude, longitude,
                           depth_in_m, radius, recursion_level,
                           save_hdf5, savepath):
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
            if save_hdf5:
                tpr[counter, 0] = 90.0 - lat
                tpr[counter, 1] = lon
                tpr[counter, 2] = 6371000.0 - depth
                counter += 1

        normals, areas = self.sphere_surface_vars(vertices, triangles)

        if save_hdf5:
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
