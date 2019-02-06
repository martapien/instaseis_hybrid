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

import os
import numpy as np
import h5py
import warnings
from math import floor, ceil
from obspy import Stream, Trace
from distutils.version import LooseVersion
from scipy.integrate import cumtrapz
from obspy.signal.interpolation import lanczos_interpolation

from . import rotations
from .source import Source, Receiver
from .helpers import c_ijkl_ani, get_band_code
from . import open_db

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    parallel_active = True
    warnings.warn("Running with MPI, so hybrid_extraction and "
                  "hybrid_repropagation work only in parallel mode. You can "
                  "use all other Instaseis features as usual.")
except:
    warnings.warn("Running without MPI, so hybrid_extraction and "
                  "hybrid_repropagation work only in serial mode. You can use "
                  "all other Instaseis features as usual.")
    rank = 0
    nprocs = 1
    parallel_active = False

WORK_DIR = os.getcwd()

KIND_MAP = {
    'displacement': 0,
    'velocity': 1,
    'acceleration': 2}


def _diff_and_integrate(n_derivative, data, comp, dt_out):
    for _ in np.arange(n_derivative):
        # In some numpy version there is an incompatibility here - 1.11
        # works for both so we branch here.
        if LooseVersion(np.__version__) >= LooseVersion("1.11.0"):
            data[comp] = np.gradient(data[comp], dt_out)
        else:  # pragma: no cover
            data[comp] = np.gradient(data[comp], [dt_out])

    # Cannot happen currently - maybe with other source time functions?
    for _ in np.arange(-n_derivative):  # pragma: no cover
        # adding a zero at the beginning to avoid phase shift
        data[comp] = cumtrapz(data[comp], dx=dt_out, initial=0.0)


def hybrid_extraction(input_path, output_path, fwd_db_path, dt, source,
                      time_window=None, filter_freqs=None, reconvolve_stf=False,
                      remove_source_shift=True, dumpcoords="spherical",
                      dumpfields=("velocity", "strain"), precision='f4',
                      max_data_buffer_in_mb=1024):

    if dumpcoords != "spherical" and dumpcoords != "local":
        raise NotImplementedError("Can dump only in tpr (spherical) or xyz ("
                                  "(local) coordinates")

    print("Instaseis: Launching output generation on proc %d..." % rank)

    # open the input file either in parallel or normally
    if input_path.endswith('.hdf5') or not input_path.endswith('.nc'):
        if parallel_active:
            inputfile = h5py.File(
                input_path, "a", driver='mpio', comm=MPI.COMM_WORLD)
        else:
            inputfile = h5py.File(input_path, "a")
    else:
        raise NotImplementedError('Provide input as hdf5 or netcdf file.')

    # get the total number of points to dump
    if "spherical" in inputfile:
        npoints = inputfile['spherical/coordinates'].shape[0]
    elif "local" in inputfile:
        npoints = inputfile['local/coordinates'].shape[0]
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')

    # define number of points to handle per process
    start_idx, points_per_process = _get_npoints_per_proc(npoints)

    # create group and datasets for medium parameters in coords file
    grp = inputfile.create_group("Instaseis_medium_params")
    grp.create_dataset("mu", (npoints,), dtype=precision)
    grp.create_dataset("rho", (npoints,), dtype=precision)
    grp.create_dataset("lambda", (npoints,), dtype=precision)
    grp.create_dataset("xi", (npoints,), dtype=precision)
    grp.create_dataset("phi", (npoints,), dtype=precision)
    grp.create_dataset("eta", (npoints,), dtype=precision)

    # create the output file
    if parallel_active:
        outputfile = h5py.File(output_path, "w", driver='mpio', comm=comm)
    else:
        outputfile = h5py.File(output_path, "w")

    _hybrid_generate_output(inputfile, outputfile, fwd_db_path, dt,
                            source, time_window=time_window,
                            filter_freqs=filter_freqs,
                            reconvolve_stf=reconvolve_stf,
                            remove_source_shift=remove_source_shift,
                            dumpcoords=dumpcoords,
                            dumpfields=dumpfields,
                            precision=precision,
                            max_data_buffer_in_mb=max_data_buffer_in_mb,
                            start_idx=start_idx,
                            npoints_rank=points_per_process,
                            npoints_tot=npoints)

    print("Instaseis: Done generating and writing output on proc %d!" % rank)


def hybrid_repropagation(fields_path, coords_path, receiver, bwd_db_path,
                         no_filter=True, bg_field_file=None, components=None,
                         dt=None, kind='displacement', kernelwidth=12,
                         return_obspy_stream=True):

    if coords_path.endswith('.hdf5') or not coords_path.endswith('.nc'):
        if parallel_active:
            f_coords = h5py.File(
                coords_path, "r", driver='mpio', comm=MPI.COMM_WORLD)
            f_fields_loc = h5py.File(
                fields_path, "r", driver='mpio', comm=MPI.COMM_WORLD)
            if bg_field_file is not None:
                f_fields_bg = h5py.File(
                    bg_field_file, "r", driver='mpio', comm=MPI.COMM_WORLD)
            else:
                f_fields_bg = None
        else:
            f_coords = h5py.File(coords_path, "r")
            f_fields_loc = h5py.File(fields_path, "r")
            if bg_field_file is not None:
                f_fields_bg = h5py.File(bg_field_file, "r")
            else:
                f_fields_bg = None
    else:
        raise NotImplementedError('Provide input as hdf5 or netcdf file.')

    # coordinate files and fields files in the same coordinates; bg always
    # in spherical (required):
    _files_coordinates_checks(f_coords, f_fields_loc, f_fields_bg)

    # get the total number of points to dump
    if "spherical" in f_coords:
        npoints = f_coords['spherical/coordinates'].shape[0]
    elif "local" in f_coords:
        npoints = f_coords['local/coordinates'].shape[0]
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')

    # define number of points to handle per process
    start_idx, points_per_process = _get_npoints_per_proc(npoints)

    bwd_db = open_db(bwd_db_path)

    # read the coordinates file
    coords_data = _read_coordinates_file(f_coords, start_idx,
                                         points_per_process,
                                         repropagation=True)
    # read the file with fields dumped by local solver
    local_fields_data = _read_local_fields_file(f_fields_loc, start_idx,
                                                points_per_process)

    # read the file with background fields if provided
    if f_fields_bg is not None:
        bg_fields_data = _read_bg_fields_file(f_fields_bg, start_idx,
                                              points_per_process)
    else:
        bg_fields_data = None

    data = bwd_db.get_seismograms_hybrid(
        receiver, coords_data, local_fields_data, no_filter=no_filter,
        bg_f_data=bg_fields_data, components=components)

    f_coords.close()
    f_fields_loc.close()

    if f_fields_bg is not None:
        f_fields_bg.close()

    if parallel_active:
        all_data = comm.gather(data, root=0)
    else:
        all_data = data

    if rank == 0:
        final_data = {}
        if components is None:
            components = bwd_db.default_components

        for i in np.arange(len(all_data)):
            for comp in components:
                if comp in final_data:
                    final_data[comp] += all_data[i][comp]
                else:
                    final_data[comp] = all_data[i][comp]

        # review moved this from get seismograms hybrid, make sure it's OK
        for comp in components:

            if dt is not None:
                # We don't need to align a sample to the peak of the source
                # time function here.
                new_npts = int(round((len(final_data[comp]) - 1) *
                                     bwd_db.info.dt / dt, 6) + 1)
                final_data[comp] = lanczos_interpolation(
                    data=np.require(final_data[comp], requirements=["C"]),
                    old_start=0, old_dt=bwd_db.info.dt, new_start=0, new_dt=dt,
                    new_npts=new_npts, a=kernelwidth, window="blackman")
                # The resampling assumes zeros outside the data range. This
                # does not introduce any errors at the beginning as the data is
                # actually zero there but it does affect the end. We will
                # remove all samples that are affected by the boundary
                # conditions here.
                if round(dt / bwd_db.info.dt, 6) != 1.0:
                    affected_area = kernelwidth * bwd_db.info.dt
                    final_data[comp] = \
                        final_data[comp][
                        :-int(np.ceil(affected_area / dt))]

            if dt is None:
                dt_out = bwd_db.info.dt
            else:
                dt_out = dt

            if type(dt_out) is np.ndarray:
                dt_out = dt_out[0]

            n_derivative = KIND_MAP[kind]

            if n_derivative:
                for comp in components:
                    _diff_and_integrate(n_derivative=n_derivative,
                                        data=final_data, comp=comp,
                                        dt_out=dt_out)

        if return_obspy_stream:
            # Convert to an ObsPy Stream object.
            st = Stream()
            band_code = get_band_code(dt_out)
            for comp in components:
                tr = Trace(data=final_data[comp],
                           header={"delta": dt_out,
                                   "station": receiver.station,
                                   "network": receiver.network,
                                   "location": receiver.location,
                                   "channel": "%sX%s" % (
                                       band_code, comp)})
                st += tr
            return st
        else:
            return final_data
    else:
        return None


def _hybrid_generate_output(inputfile, outputfile, fwd_db_path, dt,
                            source, start_idx, npoints_rank, npoints_tot,
                            time_window=None, filter_freqs=None,
                            reconvolve_stf=False, remove_source_shift=True,
                            dumpcoords="spherical",
                            dumpfields=("velocity", "strain"),
                            precision='f4', max_data_buffer_in_mb=1024):

    """
    A method to generate the hdf5 file with the input (background,
    injected) field for a local hybrid simulation. Dumps displacements,
    velocities, strains or tractions.

    :param outputfile: A path to the output hdf5/netcdf file that is
    created, e.g. "/home/user/hybrid_output.hdf5". The output file includes
    group 'spherical' with datasets corrseponding to dumpfields,
    e.g. (by default):
    ['spherical/velocity'] and ['spherical/strain']
    :type outputfile: hdf5 file
    :param inputfile: A path to a text or hdf5/netcdf file with spherical/local
    coordinates of points on the boundary of the local domain of the hybrid
    method.
    Hdf5/netcdf file: group spherical with dataset coordinates
    ([npoints, 3], where the second dimension is tpr, and with attribute
    nb_points defining the total number of gll boundary points. If
    coordinates are in the local frame of reference, dataset spherical
    requires an attribute rotation_matrix for right-multiplication to
    rotate to tpr.
    :type inputfile: hdf5 file
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

    database = open_db(fwd_db_path)
    if database.info.is_reciprocal:
        raise ValueError('Extraction of background wavefield requires a '
                         'forward Instaseis database.')

    # read the coordinates file
    coords_file_data = _read_coordinates_file(
        inputfile, start_idx, npoints_rank, dumpcoords=dumpcoords,
        dumpfields=dumpfields)

    # we define a new one, as dt=None is something we want later if that's
    # what it is!
    if dt is not None:
        dt_hdf5 = dt
    else:
        dt_hdf5 = database.info.dt

    # get the number of timesteps
    if time_window is not None:
        _time_window_bounds_checks(time_window, database)
        itmin = int(floor(time_window[0] / dt_hdf5))
        itmax = int(ceil(time_window[1] / dt_hdf5))
        ntimesteps = itmax - itmin
        if (time_window[1] - time_window[0]) % dt > 1e-13:
            warnings.warn('The specified time window divided by the specified '
                          'dt is not an integer. The number of time steps '
                          'to output was rounded to %d' % (ntimesteps))
    else:
        ntimesteps, dt_hdf5 = _get_ntimesteps(fwd_db_path, dt_hdf5,
                                              remove_source_shift)
        itmin = 0
        itmax = ntimesteps

    # prepares the group and datasets of the output file
    grp_coords = _prepare_output_file(
        dumpfields, dumpcoords, npoints_tot, ntimesteps, precision, dt,
        outputfile)

    # define how much we store in memory at once
    max_data_in_bytes = max_data_buffer_in_mb * 1024 ** 2
    ncomp = len(dumpfields) * 3
    if "strain" in dumpfields:
        ncomp += 3
    if "stress" in dumpfields:
        ncomp += 3
    if precision == 'f4':
        npoints_buf = int(floor(((max_data_in_bytes / 4) / ncomp) / ntimesteps))
    elif precision == 'f8':
        npoints_buf = int(floor(((max_data_in_bytes / 8) / ncomp) / ntimesteps))
    else:
        raise NotImplementedError

    # prepare a list of receivers from the coordinates
    receivers = _make_receivers(coords_file_data['coordinates'], database)

    if dumpcoords == "local":
        # we transpose it to have loc_to_glob, this rotmat is glob_to_loc
        coords_rotmat = coords_file_data['rotmat_xyz_glob_to_loc'].T
    else:
        coords_rotmat = None

    # Set the empty arrays for data, and get datasets of the output file
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

    # get the medium parameters group
    grp_params = inputfile["Instaseis_medium_params"]
    dset_mu = grp_params["mu"]
    dset_rho = grp_params["rho"]
    dset_lambda = grp_params["lambda"]
    dset_xi = grp_params["xi"]
    dset_phi = grp_params["phi"]
    dset_eta = grp_params["eta"]
    mu_all = np.zeros(npoints_buf, dtype=precision)
    rho_all = np.zeros(npoints_buf, dtype=precision)
    lbd_all = np.zeros(npoints_buf, dtype=precision)
    xi_all = np.zeros(npoints_buf, dtype=precision)
    phi_all = np.zeros(npoints_buf, dtype=precision)
    eta_all = np.zeros(npoints_buf, dtype=precision)

    buf_idx = 0
    for i in np.arange(npoints_rank):

        j = i - buf_idx * npoints_buf

        data = database.get_data_hybrid(source, receivers[i], dumpfields,
                                        dumpcoords=dumpcoords,
                                        coords_rotmat=coords_rotmat,
                                        remove_source_shift=remove_source_shift,
                                        reconvolve_stf=reconvolve_stf, dt=dt,
                                        filter_freqs=filter_freqs)

        mu_all[j] = data["elastic_params"]["mu"]
        rho_all[j] = data["elastic_params"]["rho"]
        lbd_all[j] = data["elastic_params"]["lambda"]
        xi_all[j] = data["elastic_params"]["xi"]
        phi_all[j] = data["elastic_params"]["phi"]
        eta_all[j] = data["elastic_params"]["eta"]

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

            c_11, c_12, c_13, c_15, c_22, c_23, c_25, c_33, c_35, c_44, c_46, \
            c_55, c_66 = _get_cijkl(data["elastic_params"])

            # 123 = tpr or 123 = xyz
            e_11 = np.array(data["strain"][itmin:itmax, 0])
            e_22 = np.array(data["strain"][itmin:itmax, 1])
            e_33 = np.array(data["strain"][itmin:itmax, 2])
            e_32 = np.array(data["strain"][itmin:itmax, 3])
            e_31 = np.array(data["strain"][itmin:itmax, 4])
            e_12 = np.array(data["strain"][itmin:itmax, 5])

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
                if dumpcoords == 'local' and \
                        coords_file_data['coordinate_system'] == 'local':
                    n = coords_file_data['normals'][i, :]
                elif dumpcoords == 'spherical' and \
                        coords_file_data['coordinate_system'] == 'spherical':
                    n = coords_file_data['normals'][i, :]
                else:
                    raise NotImplementedError("Sorry, not yet implemented")

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

            # after we have reached a full buffer, we dump into file
            # collective write is much faster for parallel setups
            idx1 = start_idx + buf_idx * npoints_buf
            idx2 = start_idx + i + 1

            if parallel_active:
                with dset_mu.collective:
                    dset_mu[idx1:idx2] = mu_all[:j + 1]
                with dset_rho.collective:
                    dset_rho[idx1:idx2] = rho_all[:j + 1]
                with dset_lambda.collective:
                    dset_lambda[idx1:idx2] = lbd_all[:j + 1]
                with dset_xi.collective:
                    dset_xi[idx1:idx2] = xi_all[:j + 1]
                with dset_phi.collective:
                    dset_phi[idx1:idx2] = phi_all[:j + 1]
                with dset_eta.collective:
                    dset_eta[idx1:idx2] = eta_all[:j + 1]

                if "velocity" in dumpfields:
                    with dset_v.collective:
                        dset_v[idx1:idx2, :, :] = velocity[:j + 1, :, :]
                if "displacement" in dumpfields:
                    with dset_d.collective:
                        dset_d[idx2:idx2, :, :] = disp[:j + 1, :, :]
                if "strain" in dumpfields:
                    with dset_strn.collective:
                        dset_strn[idx1:idx2, :, :] = strain[:j + 1, :, :]
                if "traction" in dumpfields:
                    with dset_tr.collective:
                        dset_tr[idx1:idx2, :, :] = traction[:j + 1, :, :]
                if "stress" in dumpfields:
                    with dset_strs.collective:
                        dset_strs[idx1:idx2, :, :] = stress[:j + 1, :, :]
            else:
                dset_mu[idx1:idx2] = mu_all[:j + 1]
                dset_rho[idx1:idx2] = rho_all[:j + 1]
                dset_lambda[idx1:idx2] = lbd_all[:j + 1]
                dset_xi[idx1:idx2] = xi_all[:j + 1]
                dset_phi[idx1:idx2] = phi_all[:j + 1]
                dset_eta[idx1:idx2] = eta_all[:j + 1]

                if "velocity" in dumpfields:
                    dset_v[idx1:idx2, :, :] = velocity[:j + 1, :, :]
                if "displacement" in dumpfields:
                    dset_d[idx2:idx2, :, :] = disp[:j + 1, :, :]
                if "strain" in dumpfields:
                    dset_strn[idx1:idx2, :, :] = strain[:j + 1, :, :]
                if "traction" in dumpfields:
                    dset_tr[idx1:idx2, :, :] = traction[:j + 1, :, :]
                if "stress" in dumpfields:
                    dset_strs[idx1:idx2, :, :] = stress[:j + 1, :, :]

            buf_idx += 1

    outputfile.close()
    inputfile.close()


def hybrid_get_elastic_params(input_path, db_path, source=None, precision='f4'):

    print("Instaseis: Extracting elastic parameters....")

    # open the input file either in parallel or normally
    if input_path.endswith('.hdf5') or not input_path.endswith('.nc'):
        if parallel_active:
            inputfile = h5py.File(
                input_path, "a", driver='mpio', comm=MPI.COMM_WORLD)
        else:
            inputfile = h5py.File(input_path, "a")
    else:
        raise NotImplementedError('Provide input as hdf5 or netcdf file.')

    # get the total number of points to dump
    if "spherical" in inputfile:
        npoints = inputfile['spherical/coordinates'].shape[0]
    elif "local" in inputfile:
        npoints = inputfile['local/coordinates'].shape[0]
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')

    # define number of points to handle per process
    start_idx, points_per_process = _get_npoints_per_proc(npoints)

    # read the coordinates file
    # ToDo verify if OK here
    coords_file_data = _read_coordinates_file(inputfile, start_idx,
                                              points_per_process)

    max_data_in_bytes = 1024
    ncomp = 6
    ntimesteps = 1
    if precision == 'f4':
        npoints_buf = int(floor(((max_data_in_bytes / 4) / ncomp) / ntimesteps))
    elif precision == 'f8':
        npoints_buf = int(floor(((max_data_in_bytes / 8) / ncomp) / ntimesteps))
    else:
        raise ValueError('Specify single or double precision only')

    grp = inputfile.create_group("Instaseis_medium_params")
    dset_mu = grp.create_dataset("mu", (npoints,), dtype=precision)
    dset_rho = grp.create_dataset("rho", (npoints,), dtype=precision)
    dset_lambda = grp.create_dataset("lambda", (npoints,), dtype=precision)
    dset_xi = grp.create_dataset("xi", (npoints,), dtype=precision)
    dset_phi = grp.create_dataset("phi", (npoints,), dtype=precision)
    dset_eta = grp.create_dataset("eta", (npoints,), dtype=precision)

    mu_all = np.zeros(npoints_buf, dtype=precision)
    rho_all = np.zeros(npoints_buf, dtype=precision)
    lbd_all = np.zeros(npoints_buf, dtype=precision)
    xi_all = np.zeros(npoints_buf, dtype=precision)
    phi_all = np.zeros(npoints_buf, dtype=precision)
    eta_all = np.zeros(npoints_buf, dtype=precision)

    database = open_db(db_path)

    if source is not None:
        receivers = _make_receivers(coords_file_data['coordinates'], database)

    else:
        rec = Receiver(latitude=10.0, longitude=10.0)
        sources = _make_sources(coords_file_data['coordinates'], database)

    buf_idx = 0
    for i in np.arange(points_per_process):
        j = i - buf_idx * npoints_buf

        if source is not None:
            data = database.get_elastic_params(source=source,
                                               receiver=receivers[i])
        else:
            data = database.get_elastic_params(source=sources[i],
                                               receiver=rec)

        mu_all[j] = data["mu"]
        rho_all[j] = data["rho"]
        lbd_all[j] = data["lambda"]
        xi_all[j] = data["xi"]
        phi_all[j] = data["phi"]
        eta_all[j] = data["eta"]

        if j == (npoints_buf - 1) or i == (points_per_process - 1):

            # after we have reached a full buffer, we dump into file
            # collective write is much faster for parallel setups
            idx1 = start_idx + buf_idx * npoints_buf
            idx2 = start_idx + i + 1

            if parallel_active:
                with dset_mu.collective:
                    dset_mu[idx1:idx2] = mu_all[:j + 1]
                with dset_rho.collective:
                    dset_rho[idx1:idx2] = rho_all[:j + 1]
                with dset_lambda.collective:
                    dset_lambda[idx1:idx2] = lbd_all[:j + 1]
                with dset_xi.collective:
                    dset_xi[idx1:idx2] = xi_all[:j + 1]
                with dset_phi.collective:
                    dset_phi[idx1:idx2] = phi_all[:j + 1]
                with dset_eta.collective:
                    dset_eta[idx1:idx2] = eta_all[:j + 1]

            else:
                dset_mu[idx1:idx2] = mu_all[:j + 1]
                dset_rho[idx1:idx2] = rho_all[:j + 1]
                dset_lambda[idx1:idx2] = lbd_all[:j + 1]
                dset_xi[idx1:idx2] = xi_all[:j + 1]
                dset_phi[idx1:idx2] = phi_all[:j + 1]
                dset_eta[idx1:idx2] = eta_all[:j + 1]

            buf_idx += 1

    inputfile.close()


def _prepare_output_file(dumpfields, dumpcoords, npoints, ntimesteps, precision,
                         dt, outputfile):

    grp = outputfile.create_group(dumpcoords)

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

    return grp


def _read_coordinates_file(inputfile, start_idx, npoints_rank,
                           dumpfields=(), dumpcoords=(),
                           repropagation=False):
    # ToDo can make it more compact here
    coords_file_data = {}

    end_idx = start_idx + npoints_rank

    if "spherical" in inputfile:
        coords_file_data['coordinate_system'] = 'spherical'

        dset_coords = inputfile['spherical/coordinates']
        with dset_coords.collective:
            coords_file_data["coordinates"] = \
                np.array(dset_coords[start_idx:end_idx, :])  # in tpr

        if dumpcoords == "local" and not repropagation:
            coords_file_data["rotmat_xyz_glob_to_loc"] = \
                inputfile['spherical'].attrs['rotmat_xyz_glob_to_loc']

        if "traction" in dumpfields or repropagation:
            dset_norm = inputfile['spherical/normals']
            with dset_norm.collective:
                coords_file_data["normals"] = \
                    np.array(dset_norm[start_idx:end_idx, :])  # in tpr

            if repropagation:
                dset_weight = inputfile['spherical/weights']
                with dset_weight.collective:
                    coords_file_data["weights"] = \
                        np.array(dset_weight[start_idx:end_idx])  # in tpr

    elif "local" in inputfile:
        coords_file_data['coordinate_system'] = 'local'

        dset_coords = inputfile['local/coordinates']
        with dset_coords.collective:
            coordinates = np.array(dset_coords[start_idx:end_idx, :])  # in xyz

        coords_file_data["rotmat_xyz_glob_to_loc"] = \
            inputfile['local'].attrs['rotmat_xyz_loc_to_glob']
        # radius of the Earth OR radius of box top if at depth
        radius_of_box_top = inputfile['local'].attrs['radius_of_box_top']

        if type(radius_of_box_top) is np.ndarray:
            radius_of_box_top = radius_of_box_top[0]

        coordinates[:, 2] += radius_of_box_top
        coordinates = rotations.hybrid_coord_transform_local_cartesian_to_tpr(
            coordinates, coords_file_data["rotmat_xyz_glob_to_loc"])

        coords_file_data["coordinates"] = coordinates

        if "traction" in dumpfields or repropagation:
            dset_norm = inputfile['local/normals']
            with dset_norm.collective:
                coords_file_data["normals"] = \
                    np.array(dset_norm[start_idx:end_idx, :])  # in xyz
                # Note that we don't rotate normals here, as it's a waste of
                # time and adds an extra loop; we do that later in the
                # get_seismograms_hybrid

            if repropagation:
                dset_weight = inputfile['local/weights']
                with dset_weight.collective:
                    coords_file_data["weights"] = \
                        np.array(dset_weight[start_idx:end_idx])  # in tpr

        coords_file_data["radius_of_box_top"] = radius_of_box_top
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')
    if repropagation:
        grp_medium_params = inputfile['Instaseis_medium_params']
        dset_mu = grp_medium_params['mu']
        with dset_mu.collective:
            mu = dset_mu[start_idx:end_idx]
        dset_lbd = grp_medium_params['lambda']
        with dset_lbd.collective:
            lbd = dset_lbd[start_idx:end_idx]
        dset_xi = grp_medium_params['xi']
        with dset_xi.collective:
            xi = dset_xi[start_idx:end_idx]
        dset_phi = grp_medium_params['phi']
        with dset_phi.collective:
            phi = dset_phi[start_idx:end_idx]
        dset_eta = grp_medium_params['eta']
        with dset_eta.collective:
            eta = dset_eta[start_idx:end_idx]
        coords_file_data["elastic_parameters"] = \
            np.array([mu, lbd, xi, phi, eta]).T

    return coords_file_data


def _read_local_fields_file(fieldsfile, start_idx, npoints_rank):
    fields_file_data = {}
    end_idx = start_idx + npoints_rank

    if "spherical" in fieldsfile:
        grp_fields = fieldsfile['spherical']
        fields_file_data['coordinate_system'] = 'spherical'
    elif "local" in fieldsfile:
        grp_fields = fieldsfile['local']
        fields_file_data['coordinate_system'] = 'local'
    else:
        raise NotImplementedError

    fields_file_data['displacement'] = \
        grp_fields['displacement'][start_idx:end_idx, :, :]

    fields_file_data['strain'] = \
        grp_fields['strain'][start_idx:end_idx, :, :]

    dt = grp_fields.attrs['dt']

    # When extracting from hdf5, dt is a float. When extracting from
    # netcdf, dt is a numpy array of length 1.
    if type(dt) is np.ndarray:
        dt = dt[0]

    fields_file_data['dt'] = dt

    return fields_file_data


def _read_bg_fields_file(fieldsfile, start_idx, npoints_rank):
    fields_file_data = {}
    end_idx = start_idx + npoints_rank

    if "spherical" not in fieldsfile:
        raise NotImplementedError("Background field need to be in spherical "
                                  "coordinates")

    grp_fields = fieldsfile['spherical']

    fields_file_data['velocity'] = \
        grp_fields['velocity'][start_idx:end_idx, :, :]
    fields_file_data['stress'] = \
        grp_fields['stress'][start_idx:end_idx, :, :]

    return fields_file_data


def _make_receivers(coordinates, database):
    """
    Method to handle hybrid boundary input (in spherical coordinates) defined
    by the mesh of a local solver.
    :param inputfile:
    :type inputfile: string
    """
    receivers = []
    items = coordinates.shape[0]

    for i in np.arange(items):
        lat = 90.0 - coordinates[i, 0]
        lon = coordinates[i, 1]
        dep = (6371000.0 - coordinates[i, 2])
        receivers.append(Receiver(
            latitude=lat,
            longitude=lon,
            depth_in_m=dep))

    # Check the bounds of the receivers vs the database
    receivers = _database_bounds_checks(receivers, database)

    return receivers


def _make_sources(coordinates, database):
    sources = []
    items = coordinates.shape[0]

    with open('coords_tpr_test.txt', 'w') as f:
        for i in np.arange(items):
            lat = 90.0 - coordinates[i, 0]
            lon = coordinates[i, 1]
            dep = (6371000.0 - coordinates[i, 2])
            # if dep < 0.0:
            # ToDo remove this
            f.write("point: %f, coord: (%f, %f, %f) \n"
                    % (i, lat, lon, dep))
            sources.append(Source(
                latitude=lat,
                longitude=lon,
                depth_in_m=dep))

    sources = _database_bounds_checks(sources, database)

    return sources


def _time_window_bounds_checks(time_window, database):
    length_in_s = database.info.length
    if time_window[0] > time_window[1]:
        raise ValueError("The time window must be specified via (tmin, tmax)"
                         " where tmin < tmax.")
    if time_window[0] < 0:
        raise ValueError("The tmin must be greater than 0.")
    if time_window[1] > length_in_s:
        raise ValueError("Specified tmax is %f seconds whereas the database "
                         "length is only %f seconds." % (time_window[1],
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
            elif rec.depth_in_m <= 0.0:
                rec.depth_in_m = 0.0
                warnings.warn("The shallowest receiver to construct a hybrid"
                              "src is %.1f km deep. The database only has a"
                              "depth range from %.1f km to %.1f km."
                              "Receiver depth set to 0." % (
                                  min_depth / 1000.0, db_min_depth / 1000.0,
                                  db_max_depth / 1000.0))
            else:
                raise NotImplementedError

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


def _files_coordinates_checks(f_coords, f_fields_loc, f_fields_bg=None):

    if (('spherical' in f_coords) and ('spherical' not in f_fields_loc)) \
            or (('local' in f_coords) and ('local' not in f_fields_loc)) \
            or (('spherical' in f_fields_loc) and ('spherical' not in f_coords)) \
            or (('local' in f_fields_loc) and ('local' not in f_coords)):
        raise NotImplementedError("Only spherical or local groups "
                                  "allowed. Both files must have the same "
                                  "groups, i.e. be in the same "
                                  "coordinates.")

    if f_fields_bg is not None:
        if 'spherical' not in f_fields_bg:
            raise NotImplementedError("Background field must be in spherical "
                                      "coordinates")


def _get_ntimesteps(database, dt,
                    remove_source_shift):

    if isinstance(database, str):
        database = open_db(database)

    source = Source.from_strike_dip_rake(latitude=10, longitude=20, M0=1e+21,
                                         strike=32., dip=62., rake=90.,
                                         depth_in_m=100)
    receiver = Receiver(30, 40)
    data = database.get_data_hybrid(source, receiver, dt=dt, dumpfields=(
        "displacement"), remove_source_shift=remove_source_shift)
    disp = data["displacement"][:, 0]
    return len(disp), database.info.dt


def _get_npoints_per_proc(npoints):

    if nprocs > 1:
        points_per_process = int(floor(npoints / nprocs))
        extra_points = npoints - (points_per_process * nprocs)

        if extra_points == 0:
            start_idx = points_per_process * rank
        else:
            if rank < extra_points:
                start_idx = (points_per_process + 1) * rank
                points_per_process += 1
            else:
                start_idx = (points_per_process * rank) + extra_points
    else:
        start_idx = 0
        points_per_process = npoints

    return start_idx, points_per_process


def _get_cijkl(elastic_params):
    mu = elastic_params["mu"]
    lbd = elastic_params["lambda"]
    xi = elastic_params["xi"]
    phi = elastic_params["phi"]
    eta = elastic_params["eta"]
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

    return c_11, c_12, c_13, c_15, c_22, c_23, c_25, c_33, c_35, c_44, \
           c_46, c_55, c_66

