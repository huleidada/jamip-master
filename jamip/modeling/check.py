import numpy as np

def is_symbol(value):
    """
    check the validity of given symbol.
    
    Arguments:
        symbol_of_element: symbol of element.
    """
    from jamip.structure.atomic_number import number
    
    return True if value in number else False
    
def is_species(value):
    """
    check the validity of given spceies.
    
    Arguments:
        name_of_species: name of species. i.e. 'Na+', 'Fe3+'
    """
    import re
    from jamip.structure.atomic_number import number
    
    result = re.match('([A-Z][a-z]?)([0-9.]*)[-+]$', value)
    if result:
        symbol, valence = result.groups()
        if symbol in number:
            return True

    return False

def is_list_or_array(value):

    return isinstance(value, (list, np.ndarray))

def formated_vector(value):

    vector = np.array(value, dtype=float)
    assert vector.shape == (3,)
    return vector

def formated_fraction_vector(value, lattice=None):
    if len(value) == 3:     # (3,) for molecule default cartesian
        vector = formated_vector(value)
    elif len(value) == 4:
        if value[3].lower() == 'direct':
            vector = formated_vector(value[:3]) @ lattice 
        elif value[3].lower() == 'cartesian':
            if lattice is None:
                raise ValueError("Miss lattice to convert fraction vector")
            vector = formated_vector(value[:3]) @ np.linalg.inv(lattice)
        else:
            raise ValueError("Unknown formated vector type.")
    return vector

def formated_cartesian_vector(value, lattice=None):
    if len(value) == 3:     # (3,) for molecule default cartesian
        vector = formated_vector(value)
    elif len(value) == 4:
        if value[3].lower() == 'direct':
            if lattice is None:
                raise ValueError("Miss lattice to convert fraction vector")
            vector = formated_vector(value[:3]) @ lattice 
        elif value[3].lower() == 'cartesian':
            vector = formated_vector(value[:3])
        else:
            raise ValueError(f"Unknown formated vector type. {value}")
    else:
        raise ValueError(f"Error value length. {len(value)}")

    return vector

def formated_rotation_angle(value):
    """_summary_

    Args:
        value (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    if isinstance(value, list) and len(value) == 2:
        if value[1].lower() == 'degree':
            theta = value[0] / 180 * np.pi
        elif value[1].lower() == 'radian':
            theta = value[0]
        else:
            raise ValueError("Unknown formated angle type.")
    else:
        theta = float(value)    
    return theta

def formated_rotation_matrix(value, axis):
    """_summary_

    Args:
        value (_type_): _description_
        axis (_type_): _description_

    Returns:
        _type_: _description_
    """    
    theta = formated_rotation_angle(value)
    C=np.math.cos(theta)
    S=np.math.sin(theta)
    ax,ay,az = np.array(axis[:3])/np.linalg.norm(axis[:3])
 
    t=1-C
    rotation_matrix=[[t*ax*ax+C,    t*ax*ay-S*az, t*ax*az+S*ay],
                     [t*ax*ay+S*az, t*ay*ay+C,    t*ay*az-S*ax],
                     [t*ax*az-S*ay, t*ay*az+S*ax, t*az*az+C]]

    return np.array(rotation_matrix)

def formated_positions(value, num=3):

    positions = np.array(value, dtype=float)
    shape = positions.shape
    if shape == (num,):
        positions = positions[None,:]
    elif len(shape) == 2 and shape[1] == num:
        pass
    else:
        raise ValueError('unrecognized positions shape: %s' %shape)
    return positions

def formated_matrix(value, num=3):
    matrix = np.around(value).astype(int)
    assert matrix.shape == (3,3)
    assert np.sum(np.abs(matrix-np.array(value))) < 1e-3
    return matrix

def formated_lattice(value):

    lattice = np.array(value, dtype=float)
    assert lattice.shape == (3,3)
    return lattice

def formated_elements(value, length:int=None):

    if is_list_or_array(value):
        for row in value:
            if not is_symbol(row):
                raise ValueError('unrecognized symbol: %s' %row) 
        if length and len(value) != length:
            raise ValueError('data shape not match')
        return value
 
    elif is_symbol(value):
        if length:
            return [value]*length
        else:
            return [value]
