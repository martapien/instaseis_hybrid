import os
import sys
import h5py
import netCDF4
from math import ceil
import click

if sys.version_info.major == 2:
    str_type = (basestring, str, unicode)  # NOQA
else:
    str_type = (bytes, str)

__netcdf_version = tuple(int(i) for i in netCDF4.__version__.split("."))

@click.command()
@click.argument("input_filename")
@click.argument("output_filename")
@click.option('--output_filetype', type=click.Choice(["hdf5", "netcdf"]),
              default="hdf5",
              help="Choose if you wish the output to be a hdf5 or a netcdf "
                   "file")
@click.option('--chunking_type', type=click.Choice(["points", "times"]),
              default="times",
              help="Choose chunking type. If you intend to extract entire "
                   "time series per point, choose `times`. If you intend to "
                   "extract a specific time step for all points, choose "
                   "`points`.  Incorrect chunking slows down data access.")
@click.option('--chunking_size', type=int, default=10,
              help="Choose chunking size for the defined type. For example, if "
                   "=1, then data is chunked per single point or time step"
                   "(depending on selected type). If =10, then ten points or "
                   "time steps  are going to form a contiguous chunk."
                   "Incorrect chunking slows down data access.")
@click.option("--compression_level",
              type=click.IntRange(1, 9), default=4,
              help="Compression level from 1 (fast) to 9 (slow).")
def compress_output(input_filename, output_filename, output_filetype,
                    chunking_type, chunking_size, compression_level):
    """
    Transposes all data in the "/Snapshots" group.

    :param input_filename: The input filename.
    :param output_filename: The output filename.
    """
    assert os.path.exists(input_filename)
    assert not os.path.exists(output_filename)

    f_in = h5py.File(input_filename, 'r')

    pbar = click.progressbar

    if output_filetype == "hdf5":

        f_out = h5py.File(output_filename, 'w')

        # copy for every group
        for grp in f_in:
            grp_in = f_in[grp]
            grp_out = f_out.create_group(grp)

            # copy attributes
            for attr in grp_in.attrs:
                grp_out.attrs[attr] = grp_in.attrs[attr]

            # copy datasets
            for dset in grp_in:
                dset_in = grp_in[dset]
                shape = dset_in.shape
                dtype = dset_in.dtype

                if chunking_type == "points":
                    chunks = (shape[0], chunking_size, shape[2])
                    factor = int((8 * 1024 * 1024 / 4) / shape[0])  # divide by
                    # npoints
                    s = int(ceil(shape[1] / float(factor)))  # nb of 8mb blocks
                elif chunking_type == "times":
                    chunks = (chunking_size, shape[1], shape[2])
                    factor = int((8 * 1024 * 1024 / 4) / shape[1])  # divide by
                    # ntimesteps
                    s = int(ceil(shape[0] / float(factor)))  # nb of 8mb blocks
                else:
                    raise NotImplementedError("Unknown chunking flag.")

                dset_out = grp_out.create_dataset(
                    dset, shape, dtype=dtype, chunks=chunks,
                    compression="gzip", compression_opts=compression_level)

                click.echo(click.style("--> Writing dataset `%s` in group `%s`"
                                       "to new file." % (dset, grp),
                                       fg="green"))

                with pbar(range(s), length=s,
                          label="\r ") as idx:
                    for _i in idx:
                        if chunking_type == "points":
                            _s = slice(_i * factor, _i * factor + factor)
                            dset_out[:, _s, :] = dset_in[:, _s, :]
                        if chunking_type == "times":
                            _s = slice(_i * factor, _i * factor + factor)
                            dset_out[_s, :, :] = dset_in[_s, :, :]

    elif output_filetype == "netcdf":

        f_out = netCDF4.Dataset(output_filename, "w", format="NETCDF4")

        # copy for every group
        for grp in f_in:
            grp_in = f_in[grp]
            grp_out = f_out.createGroup(grp)

            # copy attributes
            for attr in grp_in.attrs:
                _s = grp_in.attrs[attr]
                if isinstance(_s, str_type):
                    # The setncattr_string() was added in version 1.2.3.
                    # Before that it was the default behavior.
                    if __netcdf_version >= (1, 2, 3):
                        grp_out.setncattr_string(attr, _s)
                    else:
                        grp_out.setncattr(attr, str(_s))
                else:
                    setattr(grp_out, attr, _s)

            # copy datasets
            for dset in grp_in:
                dset_in = grp_in[dset]
                shape = dset_in.shape
                dtype = dset_in.dtype

                if chunking_type == "points":
                    chunks = (shape[0], chunking_size, shape[2])
                    factor = int((8 * 1024 * 1024 / 4) / shape[0])  # divide by
                    # npoints
                    s = int(ceil(shape[1] / float(factor)))  # nb of 8mb blocks
                elif chunking_type == "times":
                    chunks = (chunking_size, shape[1], shape[2])
                    factor = int((8 * 1024 * 1024 / 4) / shape[1])  # divide by
                    # ntimesteps
                    s = int(ceil(shape[0] / float(factor)))  # nb of 8mb blocks
                else:
                    raise NotImplementedError("Unknown chunking flag.")

                grp_out.createDimension("points", shape[0])
                grp_out.createDimension("timesteps", shape[1])
                grp_out.createDimension("components", shape[2])

                dset_out = grp_out.createVariable(
                    dset, dtype, ("points", "timesteps", "components"),
                    chunksizes=chunks, zlib=True, complevel=compression_level)

                with pbar(range(s), length=s,
                          label="Writing group `%s` to file" % grp) as idx:
                    for _i in idx:
                        if chunking_type == "points":
                            _s = slice(_i * factor, _i * factor + factor)
                            dset_out[:, _s, :] = dset[:, _s, :]
                        if chunking_type == "times":
                            _s = slice(_i * factor, _i * factor + factor)
                            dset_out[_s, :, :] = dset[_s, :, :]
    else:
        NotImplementedError("Unknown output filetype. Set output filetype to "
                            "hdf5 or netcdf.")

    f_in.close()
    f_out.close()


if __name__ == "__main__":
    compress_output()
