from .hybrid import _hybrid_generate_output, hybrid_get_elastic_params, \
    _get_ntimesteps, _prepare_output_file
from . import open_db
from obspy.core import Stream, Trace


import numpy as np
from math import floor, ceil
import h5py
import warnings

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    parallel_feature = True
except:
    warnings.warn("Running without MPI, so hybrid_extraction_parallel won't "
                  "work! You can use all Instaseis features")
    parallel_feature = False


def hybrid_extraction_parallel(input_path, output_path, fwd_db_path, dt,
                               source, time_window=None,
                               filter_freqs=None, reconvolve_stf=False,
                               remove_source_shift=True,
                               dumpcoords="spherical",
                               dumpfields=("velocity", "strain"),
                               precision='f4',
                               max_data_buffer_in_mb=1024):

    print("Instaseis: Launching output generation on proc %d..." % rank)

    # open input file in parallel
    if input_path.endswith('.hdf5') or not input_path.endswith('.nc'):
        inputfile = h5py.File(
            input_path, "a", driver='mpio', comm=MPI.COMM_WORLD)
    else:
        raise NotImplementedError('Provide input as hdf5 or netcdf file.')

    # get total number of coordinate points
    if "spherical" in inputfile:
        npoints = inputfile['spherical/coordinates'].shape[0]
    elif "local" in inputfile:
        npoints = inputfile['local/coordinates'].shape[0]
    else:
        raise NotImplementedError('Input file must be either in spherical '
                                  'coordinates or in local coordinates of'
                                  'the 3D solver.')

    # define number of points to handle per process
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

    # collectively create group and datasets for medium parameters in the
    # coords file
    grp = inputfile.create_group("Instaseis_medium_params")
    grp.create_dataset("mu", (npoints,), dtype=precision)
    grp.create_dataset("rho", (npoints,), dtype=precision)
    grp.create_dataset("lambda", (npoints,), dtype=precision)
    grp.create_dataset("xi", (npoints,), dtype=precision)
    grp.create_dataset("phi", (npoints,), dtype=precision)
    grp.create_dataset("eta", (npoints,), dtype=precision)



    outputfile = h5py.File(output_path, "a", driver='mpio', comm=comm)


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
                            npoints_rank=points_per_process)

    print("Instaseis: Done generating and writing output on proc %d!" % rank)


def hybrid_get_seismograms_parallel(fieldsfile, coordsfile, receiver,
                                    bwd_db_path, bg_field_file=None,
                                    components=None,
                                    kind='displacement', dt=None,
                                    filter_freqs=None,
                                    kernelwidth=12, return_obspy_stream=True):

    if rank == 0:
        f_coords = h5py.File(coordsfile, "r")

        if "spherical" in f_coords:
            grp_coords = f_coords['spherical']
        elif "local" in f_coords:
            grp_coords = f_coords['local']
        else:
            raise NotImplementedError("Only spherical or local groups "
                                      "allowed.")
        npoints = grp_coords.attrs['nb_points']
        f_coords.close()

        npoints_rank = int(floor(npoints / nprocs))
        start_idx = 0
        for i in np.arange(1, nprocs):
            start_idx_send = i * npoints_rank
            tag2 = i + 20
            tag3 = i + 30
            if i < (nprocs - 1):
                comm.send(npoints_rank, dest=i, tag=tag2)
                comm.send(start_idx_send, dest=i, tag=tag3)
            else:
                npoints_rank_new = npoints_rank + (npoints % nprocs)
                comm.send(npoints_rank_new, dest=i, tag=tag2)
                comm.send(start_idx_send, dest=i, tag=tag3)
    else:
        tag2 = rank + 20
        tag3 = rank + 30
        npoints_rank = comm.recv(source=0, tag=tag2)
        start_idx = comm.recv(source=0, tag=tag3)

    bwd_db = open_db(bwd_db_path)

    data = bwd_db.get_seismograms_hybrid_NEW(
        fieldsfile, coordsfile, receiver, npoints_rank=npoints_rank,
        start_idx=start_idx, components=components, kind=kind, dt=dt,
        kernelwidth=kernelwidth, bg_field_file=bg_field_file)

    """
    hybrid_src = HybridSources(fieldsfile=fieldsfile, coordsfile=coordsfile,
                               filter_freqs=filter_freqs,
                               npoints_rank=npoints_rank, start_idx=start_idx, 
                               bg_field_file=bg_field_file)


    data = bwd_db.get_seismograms_hybrid_source(sources=hybrid_src,
                                                receiver=receiver, dt=dt,
                                                components=components,
                                                kind=kind,
                                                kernelwidth=kernelwidth)
    """

    all_data = comm.gather(data, root=0)

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

        final_data["delta"] = all_data[0]["delta"]
        final_data["band_code"] = all_data[0]["band_code"]

        if return_obspy_stream:
            # Convert to an ObsPy Stream object.
            st = Stream()
            for comp in components:
                tr = Trace(data=final_data[comp],
                           header={"delta": final_data["delta"],
                                   "station": receiver.station,
                                   "network": receiver.network,
                                   "location": receiver.location,
                                   "channel": "%sX%s" % (final_data["band_code"], comp)})
                st += tr
            return st
        else:
            return data
    else:
        return None


def hybrid_get_elastic_params_parallel(coordsfile, db_path, source=None):


    if rank == 0:

        f_in = h5py.File(coordsfile, "a")
        npoints = f_in['local'].attrs['nb_points']
        
        npoints_rank = int(floor(npoints / nprocs))
        start_idx = 0

        for i in np.arange(1, nprocs):
            start_idx_send = i * npoints_rank
            tag2 = i + 20
            tag3 = i + 30
            if i < (nprocs - 1):
                comm.send(npoints_rank, dest=i, tag=tag2)
                comm.send(start_idx_send, dest=i, tag=tag3) 
            else:
                npoints_rank_new = npoints_rank + (npoints % nprocs)
                comm.send(npoints_rank_new, dest=i, tag=tag2)
                comm.send(start_idx_send, dest=i, tag=tag3)
        f_in.close()

    else:
        tag2 = rank + 20
        tag3 = rank + 30
        npoints_rank = comm.recv(source=0, tag=tag2)
        start_idx = comm.recv(source=0, tag=tag3)
    
    hybrid_get_elastic_params(coordsfile, db_path, source=source,
                              npoints_rank=npoints_rank,
                              start_idx=start_idx, comm=comm)
