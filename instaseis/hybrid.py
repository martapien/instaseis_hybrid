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
import warnings

from math import floor, ceil
from . import rotations
from .source import Source, Receiver
from .helpers import c_ijkl_ani
from . import open_db

from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()


def hybrid_prepare_inputs(inputfile, outputfile, fwd_db_path, dt,
                          time_window=None, remove_source_shift=True,
                          dumpcoords="spherical",
                          dumpfields=("velocity", "strain"),
                          precision='f4',
                          max_data_buffer_in_mb=1024):

    print("Instaseis: Starting the preparation of inputs for data "
          "generation....")

    inputs = {}
    max_data_in_bytes = max_data_buffer_in_mb * 1024 ** 2

    if dumpcoords != "spherical" and dumpcoords != "local":
        raise NotImplementedError("Can dump only in tpr (spherical) or xyz ("
                                  "(local) coordinates")

    f_in = h5py.File(inputfile, "r")
    if "spherical" in f_in:
        coordinates = _read_coordinates(inputfile)
        if dumpcoords == "local":
            coords_rotmat = f_in['spherical'].attrs['rotmat_xyz_glob_to_loc']
        else:
            coords_rotmat = None
        coordinates_local = False
    elif "local" in f_in:
        coordinates = _read_coordinates(inputfile)
        coords_rotmat = f_in['local'].attrs['rotmat_xyz_loc_to_glob']
        coordinates_local = True
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')

    if "traction" in dumpfields:
        if "spherical" in f_in:
            normals = f_in['spherical/normals'][:, :]  # in tpr
        elif "local" in f_in:
            normals = f_in['local/normals'][:, :]
            # ToDo normals =
    else:
        normals = None

    npoints = coordinates.shape[0]

    if time_window is not None:
        _time_window_bounds_checks(time_window, fwd_db_path)
        itmin = int(floor(time_window[0] / dt))
        itmax = int(ceil(time_window[1] / dt))
        ntimesteps = itmax - itmin
        dt_hdf5 = dt
        if (time_window[1] - time_window[0]) % dt > 1e-13:
            warnings.warn('The specified time window divided by the specified '
                          'dt is not an integer. The number of time steps '
                          'to output was rounded to %d' %(ntimesteps))
    else:
        ntimesteps, dt_hdf5 = _get_ntimesteps(fwd_db_path, dt,
                                              remove_source_shift)
        itmin = 0
        itmax = ntimesteps
        if dt is not None:
            dt_hdf5 = dt

    ncomp = len(dumpfields) * 3
    if "strain" or "stress" in dumpfields:
        ncomp += 3

    if precision == 'f4':
        npoints_buf = int(floor(((max_data_in_bytes / 4) / ncomp) /
                                   ntimesteps))
    elif precision == 'f8':
        npoints_buf = int(floor(((max_data_in_bytes / 8) / ncomp) /
                                   ntimesteps))
    else:
        raise NotImplementedError

    _prepare_outfile(dumpfields, dumpcoords, npoints, ntimesteps, precision,
                     dt_hdf5, outputfile)

    inputs["dt"] = dt
    inputs["dumpfields"] = dumpfields
    inputs["dumpcoords"] = dumpcoords
    inputs["coordinates_local"] = coordinates_local
    inputs["precision"] = precision
    inputs["normals"] = normals
    inputs["npoints"] = npoints
    inputs["itmin"] = itmin
    inputs["itmax"] = itmax
    inputs["ntimesteps"] = ntimesteps
    inputs["npoints_buf"] = npoints_buf
    inputs["coords_rotmat"] = coords_rotmat
    inputs["outputfile"] = outputfile
    inputs["fwd_db_path"] = fwd_db_path
    inputs["remove_source_shift"] = remove_source_shift

    print("Instaseis: Done preparing inputs for data generation!")

    return inputs, coordinates


def hybrid_generate_output(source, inputs, coordinates,
                           filter_freqs=None, reconvolve_stf=False,
                           npoints_rank=None, start_idx=0, comm=None):
    """
    A method to generate the hdf5 file with the input (background,
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
        method.
        Hdf5/netcdf file: group spherical with dataset coordinates
        ([npoints, 3], where the second dimension is tpr, and with attribute
        nb_points defining the total number of gll boundary points. If
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
    :param dumpcoords: Which coordinate system do we dump in. Local (
    cartesian) or global spherical (tpr). Possible options "local" or
    "spherical", defaults to "spherical".
    :type dumpcoords: string, optional

    """

    dt = inputs["dt"]
    dumpfields = inputs["dumpfields"]
    dumpcoords = inputs["dumpcoords"]
    coordinates_local = inputs["coordinates_local"]
    precision = inputs["precision"]
    if "traction" in dumpfields:
        normals = inputs["normals"]
    npoints = inputs["npoints"]
    itmax = inputs["itmax"]
    itmin = inputs["itmin"]
    npoints_buf = inputs["npoints_buf"]
    ntimesteps = inputs["ntimesteps"]
    coords_rotmat = inputs["coords_rotmat"]
    outputfile = inputs["outputfile"]
    fwd_db_path = inputs["fwd_db_path"]
    remove_source_shift = inputs["remove_source_shift"]

    database = open_db(fwd_db_path)
    if database.info.is_reciprocal:
        raise ValueError('Extraction of background wavefield requires a '
                         'forward Instaseis database.')

    if coordinates_local:
        receivers = _make_receivers(coordinates, coordinates_local,
                                    rotmat=coords_rotmat)
        if dumpcoords == "local":
            # we transpose it to have loc_to_glob, this rotmat is glob_to_loc
            coords_rotmat = coords_rotmat.T
        else:
            coords_rotmat = None
    else:
        receivers = _make_receivers(coordinates, coordinates_local)

    # Check the bounds of the receivers vs the database
    receivers = _database_bounds_checks(receivers, database)

    if comm is None:
        f_out = h5py.File(outputfile, "a")
        npoints_rank = npoints
    else:
        f_out = h5py.File(outputfile, "a", driver='mpio', comm=comm)

    grp_coords = f_out[dumpcoords]
    if "velocity" in dumpfields:
        velocity = np.zeros((npoints_buf, ntimesteps, 3), dtype=precision)
        dset_v = grp_coords["velocity"]
    if "displacement" in dumpfields:
        disp = np.zeros((npoints_buf, ntimesteps, 3), dtype=precision)
        dset_d = grp_coords["displacement"]
    if "strain" in dumpfields:
        strain = np.zeros((npoints_buf, ntimesteps, 6), dtype=precision)
        dset_strn = grp_coords["strain"]
    if "traction" in dumpfields:
        traction = np.zeros((npoints_buf, ntimesteps, 3), dtype=precision)
        dset_tr = grp_coords["traction"]
    if "stress" in dumpfields:
        stress = np.zeros((npoints_buf, ntimesteps, 6), dtype=precision)
        dset_strs = grp_coords["stress"]

    buf_idx = 0 
    for i in np.arange(npoints_rank):
        j = i - buf_idx * npoints_buf

        data = database.get_data_hybrid(source, receivers[i], dumpfields,
                                        dumpcoords=dumpcoords,
                                        coords_rotmat=coords_rotmat,
                                        remove_source_shift=remove_source_shift,
                                        reconvolve_stf=reconvolve_stf, dt=dt,
                                        filter_freqs=filter_freqs)

        if "velocity" in dumpfields:
            velocity[j, :, :] = np.array(data["velocity"][itmin:itmax, :],
                                    dtype=precision)

        if "displacement" in dumpfields:
            disp[j, :, :] = np.array(data["displacement"][itmin:itmax, :],
                                     dtype=precision)

        if "strain" in dumpfields:
            strain[j, :, :] = np.array(data["strain"][itmin:itmax, :],
                                     dtype=precision)

        if "stress" or "traction" in dumpfields:
            params = data["elastic_params"]

            # 123 = tpr or 123 = xyz
            e_11 = np.array(data["strain"][itmin:itmax, 0])
            e_22 = np.array(data["strain"][itmin:itmax, 1])
            e_33 = np.array(data["strain"][itmin:itmax, 2])
            e_32 = np.array(data["strain"][itmin:itmax, 3])
            e_31 = np.array(data["strain"][itmin:itmax, 4])
            e_12 = np.array(data["strain"][itmin:itmax, 5])

            mu = params["mu"]
            lbd = params["lambda"]
            xi = params["xi"]
            phi = params["phi"]
            eta = params["eta"]

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

            stress_tmp = np.zeros((ntimesteps, 6))

            stress_tmp[:, 0] = (c_11 * e_11 + 2.0 * c_15 * e_31 + c_12 * e_22
                                + c_13 * e_33)  # 11
            stress_tmp[:, 1] = (c_12 * e_11 + 2.0 * c_25 * e_31 + c_22 * e_22
                                + c_23 * e_33)  # 22
            stress_tmp[:, 2] = (c_13 * e_11 + 2.0 * c_35 * e_31 + c_23 * e_22
                                + c_33 * e_33)  # 33
            stress_tmp[:, 3] = 2.0 * (c_46 * e_12 + c_44 * e_32)  # 23
            stress_tmp[:, 4] = (c_15 * e_11 + c_25 * e_22 + c_35 * e_33 + 2.0
                                * c_55 * e_31)  # 13
            stress_tmp[:, 5] = 2.0 * (c_66 * e_12 + c_46 * e_32)  # 12

            if "stress" in dumpfields:
                stress[j, :, :] = np.array(stress_tmp[:, :], dtype=precision)

            if "traction" in dumpfields:
                n = normals[i, :]
                traction_tmp = np.zeros((ntimesteps, 3))
                traction_tmp[:, 0] = n[0] * stress_tmp[:, 0] + \
                                 n[1] * stress_tmp[:, 5] + \
                                 n[2] * stress_tmp[:, 4]
                traction_tmp[:, 1] = n[0] * stress_tmp[:, 5] + \
                                 n[1] * stress_tmp[:, 1] + \
                                 n[2] * stress_tmp[:, 3]
                traction_tmp[:, 2] = n[0] * stress_tmp[:, 4] + \
                                 n[1] * stress_tmp[:, 3] + \
                                 n[2] * stress_tmp[:, 2]

                traction[j, :, :] = np.array(traction_tmp[:, :],
                                             dtype=precision)

        if j == (npoints_buf - 1) or i == (npoints_rank - 1):

            if "velocity" in dumpfields:
                dset_v[start_idx+buf_idx*npoints_buf:start_idx+i+1, :, :] = \
                    velocity[:j + 1, :, :]
            if "displacement" in dumpfields:
                dset_d[start_idx+buf_idx*npoints_buf:start_idx+i+1, :, :] = \
                    disp[:j + 1, :, :]
            if "strain" in dumpfields:
                dset_strn[start_idx+buf_idx*npoints_buf:start_idx+i+1, :, :] = \
                    strain[:j + 1, :, :]
            if "traction" in dumpfields:
                dset_tr[start_idx+buf_idx*npoints_buf:start_idx+i+1, :, :] = \
                    traction[:j + 1, :, :]
            if "stress" in dumpfields:
                dset_strs[start_idx+buf_idx*npoints_buf:start_idx+i+1, :, :] = \
                    stress[:j + 1, :, :]

            buf_idx += 1

    f_out.close()


def hybrid_get_elastic_params(source, inputfile, outputfile, db_path):

    print("Instaseis: Extracting elastic parameters....")

    f_in = h5py.File(inputfile, "r")
    if "spherical" in f_in:
        coordinates = _read_coordinates(inputfile)
        coordinates_local = False
    elif "local" in f_in:
        coordinates = _read_coordinates(inputfile)
        coords_rotmat = f_in['local'].attrs['rotmat_xyz_loc_to_glob']
        coordinates_local = True
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')

    database = open_db(db_path)

    if coordinates_local:
        receivers = _make_receivers(coordinates, coordinates_local,
                                    rotmat=coords_rotmat)
    else:
        receivers = _make_receivers(coordinates, coordinates_local)

    # Check the bounds of the receivers vs the database
    receivers = _database_bounds_checks(receivers, database)

    database.get_elastic_params(source, receivers, outputfile)


def _prepare_outfile(dumpfields, dumpcoords, npoints, ntimesteps, precision,
                     dt, outputfile):

    f_out = h5py.File(outputfile, "w")

    grp = f_out.create_group(dumpcoords)

    if "velocity" in dumpfields:
        grp.create_dataset("velocity", (npoints, ntimesteps, 3),
                           dtype=precision)

    if "displacement" in dumpfields:
        grp.create_dataset("displacement", (npoints, ntimesteps, 3),
                           dtype=precision)
    if "strain" in dumpfields:
        grp.create_dataset("strain", (npoints, ntimesteps, 6),
                           dtype=precision)
    if "stress" in dumpfields:
        grp.create_dataset("stress", (npoints, ntimesteps, 6),
                           dtype=precision)

    if "traction" in dumpfields:
        grp.create_dataset("traction", (npoints, ntimesteps, 3),
                           dtype=precision)

    grp.attrs['nb_points'] = npoints
    grp.attrs['nb_timesteps'] = ntimesteps
    grp.attrs['dt'] = dt

    f_out.close()


def _read_coordinates(inputfile):
    if inputfile.endswith('.hdf5') or inputfile.endswith('.nc'):
        f = h5py.File(inputfile, 'r')
        if "spherical/coordinates" in f:
            coordinates = np.array(f['spherical/coordinates'][:, :])  # in tpr
        elif "local/coordinates" in f:
            coordinates = np.array(f['local/coordinates'][:, :])  # in tpr
        else:
            raise ValueError('spherical/coordinates or local/coordinates not '
                             'found in file')
        f.close()
    else:
        raise NotImplementedError('Provide input as hdf5 or netcdf file.')
    return coordinates


def _make_receivers(coordinates, coordinates_local=False, rotmat=None):
    """
    Method to handle hybrid boundary input (in spherical coordinates) defined
    by the mesh of a local solver.
    :param inputfile:
    :type inputfile: string
    """
    receivers = []
    items = coordinates.shape[0]

    # review if OK
    if coordinates_local:
        if rotmat is None:
            raise ValueError("Need a rotation matrix in local coordinates")
        coordinates[:, 2] += 6371000.  # radius of the Earth
        coordinates = rotations.hybrid_coord_transform_local_cartesian_to_tpr(
            coordinates, rotmat)

    for i in np.arange(items):
        lat = 90.0 - coordinates[i, 0]
        lon = coordinates[i, 1]
        dep = (6371000.0 - coordinates[i, 2])
        receivers.append(Receiver(
            latitude=lat,
            longitude=lon,
            depth_in_m=dep))

    return receivers


def _time_window_bounds_checks(time_window, path_to_db):
    database = open_db(path_to_db)
    length_in_s = database.info.length
    if time_window[0] > time_window[1]:
        raise ValueError("The time window must be specified via (tmin, tmax)"
                         " where tmin < tmax.")
    if time_window[0] < 0:
        raise ValueError("The tmin must be greater than 0.")
    if time_window[1] > length_in_s:
        raise ValueError("Specified tmax is %f seconds whereas the database "
                         "length is only %f seconds." %(time_window[1],
                                                        length_in_s))


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
        for rec in receivers:
            if db_min_depth <= rec.depth_in_m <= db_max_depth:
                pass
            elif -1.0 <= rec.depth_in_m <= 0.0:
                rec.depth_in_m = 0.0
            else:
                raise ValueError("The shallowest receiver to construct a hybrid"
                                 "src is %.1f km deep. The database only has a"
                                 "depth range from %.1f km to %.1f km." % (
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
    return receivers


def _get_ntimesteps(path_to_db, dt,
                    remove_source_shift):
    database = open_db(path_to_db)
    source = Source.from_strike_dip_rake(latitude=10, longitude=20, M0=1e+21,
                                         strike=32., dip=62., rake=90.,
                                         depth_in_m=100)
    receiver = Receiver(30, 40)
    data = database.get_data_hybrid(source, receiver, dt=dt, dumpfields=(
        "displacement"), remove_source_shift=remove_source_shift)
    disp = data["displacement"][:, 0]
    return len(disp), database.info.dt

