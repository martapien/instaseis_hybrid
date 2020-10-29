#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An example of how to use hybrid_repropagation in the parallel mode.
The paths to database and to files output by the local solver need
to be specified.
"""

import instaseis
import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# specify paths
bwd_db_path = "path/to/reciprocal_database"
fieldsfile = "path/to/hdf5_file_with_local_wavefields"
coordsfile = "path/to/hdf5_file_with_local_coordinates"
writefolder = "path/to/folder_to_dump_mseed_files"

# specify receiver coordinates
receiver_latitudes = [10., 20., 30.]
receiver_longitudes = [10., 20., 30.]

if rank == 0:
    if not os.path.exists(writefolder):
        os.mkdir(writefolder)

# generate seismograms
for i in np.arange(len(receiver_latitudes)):

    receiver = instaseis.Receiver(
        latitude=receiver_latitudes[i],
        longitude=receiver_longitudes[i]
    )

    st_hyb = instaseis.hybrid_repropagation(
        fields_path=fieldsfile,
        no_filter=True,
        coords_path=coordsfile,
        receiver=receiver,
        bwd_db_path=bwd_db_path,
        components='Z',
        dt=0.05
    )

    if rank == 0:
        # write seismograms to folder
        writepath = os.path.join(writefolder, "hybrid_seismogram_%d.mseed" %i)
        st_hyb.write(writepath, format="MSEED")

if rank == 0:
    print("Finished generating hybrid seismograms!")
