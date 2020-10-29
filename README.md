## Hybrid extension to Instaseis

The hybrid extenstion to [Instaseis](http://instaseis.net) provides the framework for coupling global 
Instaseis databases (generated with [AxiSEM](http://seis.earth.ox.ac.uk/axisem/)) with a local wave 
propagation solver of choice. It is a two-step procedure, i.e. there is no dynamic boundary condition,
and the coupling happens via static HDF5 files that serve as an interface between the solvers.

Three setups are possible:

1. The local domain is around receivers at the Earth's surface. In this case, we use Instaseis to generate
   HDF5 files that contain information about the global wavefields on the boundary of the local domain.
   The information from the HDF5 files needs to be then used by the local solver to impose background
   wavefields on the boundary of the local domain (wavefield _injection_).
   
   - First run Instaseis.
   - Then run a local solver.
   
   This requires a _forward_ Instaseis database.
   
2. The local domain is around the source at the Earth's surface. In this case, the local simulation 
   generates HDF5 files with information on the boundary of the local domain. They are then used by 
   Instaseis to extrapolate local wavefields to receivers at distance (wavefield _repropagation_ or 
   _extrapolation_).
   
   - First run a local solver.
   - Then run Instaseis.
   
   This requires a _reciprocal_ Instaseis database.
   
3. The local domain is at depth. This combines both injection and extrapolation.

   - First run Instaseis.
   - Then run a local solver.
   - And finally run Instaseis again.
   
   This requires both a _forward_ and a _reciprocal_ Instaseis database (for the same 1D model).
   
   
For more details look at the **How to run** section below and read https://doi.org/10.31223/X5HG65.
   
When using this work, please also cite https://doi.org/10.31223/X5HG65


### How to install

1. Clone the repository.
   
   ```
   git clone https://github.com/martapien/instaseis_hybrid.git
   ```
   
2. Create an environment and install dependencies 
(dependencies like in [Instaseis documentation](http://instaseis.net)).

   ```
   conda create --name instaseis_hybrid python=3.7
   ```
    
   ```
   conda install -c conda-forge h5py obspy requests tornado flake8 pytest mock basemap pyqt pip jsonschema responses pyqtgraph pytest-xdist 
   ```

3. Install instaseis (like in [Instaseis documentation](http://instaseis.net)).

   ```
   cd instaseis_hybrid
   pip install -v -e .
   ```
    
4. Add mpi4py and parallel h5py to the environment (optional, but recommended).

   Parallel h5py and mpi4py are not required, but it makes the code run uncomparably faster. 
   
   If you wish to install parallel h5py, you need to skip h5py in the conda install command above and 
   install it with pip, linking it to a parallel-enabled build of HDF5.
   
   The steps (see also [here](https://drtiresome.com/2016/08/23/build-and-install-mpi-parallel-hdf5-and-h5py-from-source-on-linux/)
   on hints how to install.)   

   - Install MPICH from source
     (see [here](https://www.mpich.org/static/downloads/3.3.1/mpich-3.3.1-README.txt) for mpich 3.3.1).
     
     [Download](https://www.mpich.org/downloads/) 
     and unpack mpich into the mpich-X.X folder (X.X is version number).
     Proceed with installing:
        ```
        mkdir /path/to/mpich-build
        mkdir /path/to/mpich-install

        cd /path/to/mpich-build

        /path/to/mpich-X.X/configure --enable-fortran=all --enable-shared --enable-romio --with-device=ch3:sock --prefix=/path/to/mpich-install |& tee c.txt

        make |& tee m.txt
        make check
        make install
        ```
     Now add to `.bashrc`:
     
        ```
        PATH=/home/marta/Numerics/mpich-install/bin:$PATH
        export PATH 
        ```
     and finally run `source ~/.bashrc` in terminal.
     
   - Install ZLIB from source 
     (see also [here](https://geeksww.com/tutorials/libraries/zlib/installation/installing_zlib_on_ubuntu_linux.php)).
     
     [Download](https://zlib.net/) 
     and unpack zlib into the zlib-X.X folder (X.X is version number).
     Proceed with installing:
        ```
        mkdir /path/to/zlib-build
        mkdir /path/to/zlib-install

        cd /path/to/zlib-build

        path/to/zlib-X.X/configure --prefix=/path/to/zlib-install

        make
        make install
        ```
     Now add to `.bashrc`:
     
        ```
        export ZLIB_HOME=/path/to/zlib-install
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ZLIB_HOME/lib
        export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$ZLIB_HOME/lib/pkgconfig
        ```
     and finally run `source ~/.bashrc`.
   
   - Install parallel-enabled HDF5 from source
     (linking to the MPICH and ZLIB installations above).
     
     [Download](https://www.hdfgroup.org/downloads/hdf5/) 
     and unpack hdf5 into the hdf5-X.X folder (X.X is version number).
     
     NOTE: At the time of writing (October 2020), h5py does not work with HDF5 version 1.12.0, but 
     it works with version 1.10.7.
     
     Proceed with installing:
        ```
        mkdir /path/to/hdf5-build
        mkdir /path/to/hdf5-install

        cd /path/to/hdf5-build

        CC=/path/to/mpich-install/bin/mpicc path/to/hdf5-X.X/configure --enable-parallel --enable-fortran --with-zlib=/path/to/zlib-install/include,/path/to/zlib-install/lib --prefix /path/to/hdf5-install 

        make
        make check
        make install
        ```
     Now add to `.bashrc`:
        ```
        PATH=/path/to/hdf5-install/bin:$PATH
        export PATH
        ```
     and finally run `source ~/.bashrc`.
     
   - In the `instaseis_hybrid` environment, 
     [install mpi4py with pip](https://mpi4py.readthedocs.io/en/stable/install.html) 
     (it will link to the local MPICH installation as long as `which mpicc` points to it).
     
     ```
     pip install mpi4py
     ```
     
   - In the `instaseis_hybrid` environment, 
     [install h5py with pip](https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5)
     (linking to the local HDP5 installation explicitly this time).
     
     ```
     CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/hdf5-install pip install --no-binary=h5py h5py
     ```
5. Installing [AxiSEM](http://seis.earth.ox.ac.uk/axisem/) (to generate Instaseis databases) requires 
   netcdf-fortran. In order to not end up with multiple installations of MPICH and HDF5, 
   we recommend installing netcdf from source and linking it to the locally installed versions 
   (to then point AxiSEM to this locally installed netcdf version).
   
   See also that to use repack_databases, Instaseis needs netcdf4 in python. netcdf4 can
   also be linked to the locally installed netcdf version.


### How to run

(this section is work in progress)

#### Wavefield injection

##### Basic concept: 

1. Run Instaseis to save information on the boundary of the local solver in HDF5.
2. Run a local solver imposing wavefield information from HDF5 files on the boundary.

The function `hybrid_extraction` extracts global wavefields to chosen locations (usually 
on the local mesh points). It then saves a HDF5 file with information required for injection.

See `instaseis/hybrid_example_scripts/using_parallel_hybrid_extraction.py` for an example script 
on how to extract wavefields for injection with mpi4py and parallel h5py. 

##### The format of the injection HDF5 interfacing files:

#### Wavefield repropagation/extrapolation

##### Basic concept: 

1. Run a local solver.
2. Save information on the boundary of the local solver in HDF5.
3. Run Instaseis to repropagate wavefields to receivers.

The function `hybrid_repropagation` repropagates local wavefields to chosen receiver locations. It outputs
the resulting seismogram as an 
[obspy stream object](https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html).
    
For example, for a local domain around the source, we need to provide HDF5 files with local wavefields and
with local coordinates, an Instaseis receiver object, and a path to a reciprocal Instaseis database:

    st_hyb = instaseis.hybrid_repropagation(
                fields_path=fieldsfile,
                coords_path=coordsfile, 
                receiver=instaseis_receiver,
                bwd_db_path=bwd_db_path, 
                components='Z',
                dt=0.05
    )

The above outputs only the Z component, and resamples the final seismogram to a desired time step.

For a local domain at depth, a HDF5 file with background wavefields is also required.

See `instaseis/hybrid_example_scripts/using_parallel_hybrid_repropagation.py` for an example script 
on how to use repropagation with mpi4py and parallel h5py. 

##### The format of the repropagation HDF5 interfacing files:


#### Final comments

Contact Marta Pienkowska (marta.pienkowska@erdw.ethz.ch) should you wish to have
some details on Specfem3D Cartesian that has been modified to work with both 
injection and extrapolation of the hybrid method.

Note that merged Instaseis databases are not implemented in the hybrid framework for the time being.