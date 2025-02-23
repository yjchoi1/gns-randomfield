import numpy as np


def load_npz_data(path):
    """Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
            
    positions = data[0][0]
    particle_types = data[0][1]
    material_properties = data[0][2]
    
    return positions, particle_types, material_properties


if __name__ == "__main__":
    path = '/mnt/c/Users/baage/Documents/tmp/trajectory.npz'
    data = load_npz_data(path)
    a=1