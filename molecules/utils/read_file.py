import h5py
import tables 


def read_h5_contact_maps(filename): 
    data = tables.open_file(filename, 'r')
    contact_maps = data.root.contact_maps.read()
    data.close()
    return contact_maps


def read_h5_RMSD(filename):
    data = tables.open_file(filename, 'r')
    RMSD = data.root.RMSD.read()
    data.close()
    return RMSD


def open_h5(h5_file, mode='r', libver='latest', swmr=True, **kwargs):
    """
    Opens file in single write multiple reader mode
    libver specifies newest available version,
    may not be backwards compatable

    Parameters
    ----------
    h5_file : str
        name of h5 file to open

    mode : str
        mode to open the file in 'r' or 'w'

    libver : str
        version argument for h5py

    swmr : bool
        single writer multiple reader option for h5py

    Returns
    -------
    open h5py file to be used in a context manager

    """
    # TODO: bug in sparse_contact_map_from_matrices when libver and swmr
    # specified
    return h5py.File(h5_file, mode, **kwargs) #, libver=libver, swmr=swmr)

