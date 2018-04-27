from mpi4py import MPI
from .hybrid import hybrid_prepare_inputs, hybrid_generate_output, hybrid_get_elastic_params
from .source import HybridSources
from . import open_db
from obspy.core import Stream, Trace


import numpy as np
from math import floor
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


def hybrid_generate_output_parallel(inputfile, outputfile, fwd_db_path, dt,
                                    source, time_window=None,
                                    filter_freqs=None, reconvolve_stf=False,
                                    remove_source_shift=True,
                                    dumpcoords="spherical",
                                    dumpfields=("velocity", "strain"),
                                    precision='f4',
                                    max_data_buffer_in_mb=1024):

    if rank == 0:
        inputs, coordinates_send = hybrid_prepare_inputs(
            inputfile, outputfile, fwd_db_path, dt, time_window,
            remove_source_shift, dumpcoords, dumpfields, precision,
            max_data_buffer_in_mb)

        npoints_rank = int(floor(inputs["npoints"] / nprocs))
        start_idx = 0
        coordinates = np.array(coordinates_send[:npoints_rank],
                               dtype=np.float32)
        print('Instaseis: Proc %d sending info to other procs...' % rank)
        for i in np.arange(1, nprocs):
            start_idx_send = i * npoints_rank
            tag1 = i + 10
            tag2 = i + 20
            tag3 = i + 30
            if i < (nprocs - 1):
                send = coordinates_send[start_idx_send:(i + 1) * npoints_rank]
                comm.send(npoints_rank, dest=i, tag=tag2)
                comm.send(start_idx_send, dest=i, tag=tag3)
                comm.Send(send, dest=i, tag=tag1)
            else:
                send = coordinates_send[start_idx_send:]
                npoints_rank_new = npoints_rank + (inputs["npoints"] % nprocs)
                comm.send(npoints_rank_new, dest=i, tag=tag2)
                comm.send(start_idx_send, dest=i, tag=tag3)
                comm.Send(send, dest=i, tag=tag1)
        print('Instaseis: Proc %d done sending info!' % rank)

    else:
        tag1 = rank + 10
        tag2 = rank + 20
        tag3 = rank + 30
        npoints_rank = comm.recv(source=0, tag=tag2)
        start_idx = comm.recv(source=0, tag=tag3)
        coordinates = np.empty((npoints_rank, 3), dtype=np.float32)
        comm.Recv(coordinates, source=0, tag=tag1)
        print('Instaseis: Proc %d received info!' % rank)
        inputs = None

    inputs = comm.bcast(inputs, root=0)

    print("Instaseis: Launching output generation on proc %d..." % rank)

    hybrid_generate_output(source, inputs, coordinates,
                           filter_freqs=filter_freqs,
                           reconvolve_stf=reconvolve_stf,
                           npoints_rank=npoints_rank,
                           start_idx=start_idx, comm=comm)

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
