from mpi4py import MPI
from .hybrid import hybrid_prepare_inputs, hybrid_generate_output
import numpy as np
from math import floor

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
        coordinates = coordinates_send[:npoints_rank]
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
        print('Instaseis: Proc %d about to receive info...' % rank)
        tag1 = rank + 10
        tag2 = rank + 20
        tag3 = rank + 30
        npoints_rank = comm.recv(source=0, tag=tag2)
        start_idx = comm.recv(source=0, tag=tag3)
        coordinates = np.empty((npoints_rank, 3))
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
