# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import OrderedDict
import numpy as np


def enum2dict(enum):
    dicts=OrderedDict()
    for e in enum:
        dicts[e.name]=e.value
    return dicts
        
def cartesian2direct(lattice, position):
    """
    convert Cartesian to Direct.
    
    Arguments:
        lattice: lattice parameter of structure. i.e.
            [[x1,y1,z1],
             [x2,y2,z2],
             [x3,y3,z3]]
             
        position: atomic position (type: Cartesian). i.e. [5.234, 0, 0, 'Cartesian']
        
    Returns:
        atomic position (type: Direct). i.e. [0.1, 0, 0, 'Direct']
    """
    from .check import check_formated_position_only_cartesian
    
    inv=np.linalg.inv(lattice).T
    direct=np.dot(inv,position[:3])
    direct=direct.tolist()
    direct.append('Direct')
    
    return direct

def direct2cartesian(lattice, position):
    """
    convert Direct to Cartesian.
    
    Arguments:
        lattice: lattice parameter of structure. i.e.
            [[x1,y1,z1],
             [x2,y2,z2],
             [x3,y3,z3]]
             
        position: atomic position (type: Direct). The valid format is :
            [0.1, 0, 0, 'Direct']
            [0.1, 0, 0] (for Direct, can not be specify)
        
    Returns:
        atomic position (type: Cartesian). i.e. [5.234, 0, 0, 'Cartesian']
    """
    from .check import check_formated_position_only_direct
    
    if check_formated_position_only_direct(position):
        cartesian=np.dot(position[:3], lattice)
        cartesian=cartesian.tolist()
        cartesian.append('Cartesian')
        return cartesian
    else:
        raise ValueError('unknown position')

def any2direct(lattice, position):
    """
    convert it to Direct if type of position is Cartesian. do not anything if that is Direct.
    
    Arguments:
        lattice: lattice parameter of structure. i.e.
            [[x1,y1,z1],
             [x2,y2,z2],
             [x3,y3,z3]]
             
        position: atomic position. The valid format is :
            [0.1, 0, 0, 'Direct']
            [0.1, 0, 0] (for Direct, can not be specify)
            [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
    
    Returns:
        atomic position (type: Direct). i.e. [0.1, 0, 0, 'Direct']
    """
    from .check import check_formated_position_only_direct, check_formated_position_only_cartesian
    
    if check_formated_position_only_direct(position):
        return position if len(position) == 4 else position+['Direct']
    elif check_formated_position_only_cartesian(position):
        return cartesian2direct(lattice=lattice, position=position)
    else:
        raise ValueError('unknown position') 
        
def any2cartesian(lattice, position):
    """
    convert it to Cartesian if type of position is Direct. do not anything if that is Cartesian.
    
    Arguments:
        lattice: lattice parameter of structure. i.e.
            [[x1,y1,z1],
             [x2,y2,z2],
             [x3,y3,z3]]
             
        position: atomic position. The valid format is:
            [0.1, 0, 0, 'Direct']
            [0.1, 0, 0] (for Direct, can not be specify)
            [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
            
    Returns:
        atomic position (type: Cartesian). i.e. [5.234, 0, 0, 'Cartesian']
    """
    from .check import check_formated_position_only_direct, check_formated_position_only_cartesian
    
    if check_formated_position_only_direct(position):
        return direct2cartesian(lattice=lattice, position=position)
    elif check_formated_position_only_cartesian(position):
        return position
    else:
        raise ValueError('unknown position')   

def normalize_position(position, dtype, **kwargs):
    """
    remove the translation periodicty of atomic position, ensure coordinate's values are between 0 and 1.
    i.e. for Direct type, [1.1, 0, 0] -> [0.1, 0, 0]
    
    Arguments:
        position: atomic position. the valid format:
            [0.1, 0.0, 0.0, 'Direct']
            [0.1, 0.0, 0.0]
            [5.234, 0.0, 0.0, 'Cartesian']
        dtype: type of coordinate after translating ('Direct' or 'Cartesian').
            
        kwargs:
            lattice: lattice parameter of structure. need to be given when dtype is Cartesian. i.e.
                [[x1,y1,z1],
                 [x2,y2,z2],
                 [x3,y3,z3]]
            precision (default=1e-5): used to determine whether the component of position is very close to 1.
                    from other atoms.
                    
    Returns:
        converted positionincluded the type (list). i.e. [0.1, 0, 0, 'Direct'], [5.234, 0.0, 0.0, 'Cartesian']
    """
    from .check import check_formated_position_only_direct, check_formated_position_only_cartesian
    
    # check
    if check_formated_position_only_cartesian(position) or dtype.lower().startswith('c'):
        if not 'lattice' in kwargs: raise Exception("can't find the 'lattice' parameter")
        lattice=np.array(kwargs['lattice'])
        
    if check_formated_position_only_direct(position):
        pass
    elif check_formated_position_only_cartesian(position):
        position=any2direct(lattice=lattice, position=position)
    else:
        raise ValueError('uknown position')
            
    new_position=[]
    for v in position[:3]:
        new_v=v-int(v) # fraction of v
        if new_v < 0:
            new_v += 1
        new_position.append(new_v)
    new_position.append('Direct')
    
    # check
    precision=1e-5
    if 'precision' in kwargs:
        precision=kwargs['precision']
    for i in range(0, len(new_position)-1):
        if np.linalg.norm(1-new_position[i]) <= precision:
            new_position[i]=0.0
            
    if dtype.strip().lower().startswith('c'): # Cartesian
        lattice=np.array(kwargs['lattice'])
        position=new_position
        new_position=any2cartesian(lattice, position)
    elif not dtype.strip().lower().startswith('d'):
        raise ValueError('unknown dtype')
    return new_position


def cell2poscar(cell):
    """
    convert cell to poscar.
    
    Arguments:
        cell: cell-typed structure. The valid format is:
            {'lattice':lattice,
             'positions':positions,
             'numbers':numbers,
             'magmoms':magmoms(optional)}
             
    Returns:
        poscar-typed structure. The valid format is:
            {'lattice': lattice,
             'elements': elements,
             'numbers': numbers,
             'type': 'Direct',
             'positions': positions}
    """
    from collections import OrderedDict
    from .fetch import get_symbol_by_z
    
    lattice=cell['lattice']
    
    # elements, numbers and position
    atom_set=OrderedDict() # elements and its positions
    
    for i in range(0, len(cell['numbers'])):
        symbol=get_symbol_by_z(z=cell['numbers'][i])
        if symbol in atom_set:
            tmp=atom_set[symbol]
            tmp.append(cell['positions'][i].tolist())
            atom_set[symbol]=tmp
        else:
            atom_set[symbol]=[cell['positions'][i].tolist()]
    elements=[]
    numbers=[]
    positions=[]
    for k in atom_set.keys():
        elements.append(k)
        numbers.append(len(atom_set[k]))
        if positions ==[]:
            positions=atom_set[k]
        else:
            positions=np.vstack((positions, atom_set[k]))
    lattice=np.array(lattice)
    elements=np.array(elements)
    numbers=np.array(numbers)
    
    poscar={'lattice': lattice,
            'elements': elements,
            'numbers': numbers,
            'type': 'Direct',
            'positions': positions}
    return poscar

def poscar2cell(poscar):
    """
    convert poscar to cell.
    
    Arguments:
        poscar: poscar-typed structure. The valid format is:
            {'lattice': lattice,
             'elements': elements,
             'numbers': numbers,
             'type': 'Direct',
             'positions': positions}
             
    Returns:
        cell-typed structure. The valid format is:
            {'lattice':lattice,
             'positions':positions,
             'numbers':numbers,
             'magmoms':magmoms(optional)}
    """
    from .fetch import get_z_by_symbol
    
    lattice=poscar['lattice']
    positions=poscar['positions']
    numbers=[]
    for i in range(0, len(poscar['elements'])):
        for j in range(0, poscar['numbers'][i]):
            numbers.append(get_z_by_symbol(poscar['elements'][i]))
    cell={'lattice':lattice, 'positions':positions, 'numbers':numbers}
    return cell

def to_poscar5x(path_of_poscar4x, path_of_poscar5x):
    """
    convert poscar4.x to poscar5.x.
    
    Arguments:
        path_of_poscar4x: path of poscar4x.
        path_of_poscar5x: path of converted poscar5.x.
    Returns:
        output new poscar5.x file.
    """
    # check
    f=open(path_of_poscar4x, 'r')
    contents=f.readlines()
    f.close()
    
    contents.insert(5, contents[0])
    
    f=open(path_of_poscar5x, 'w')
    contents=''.join(contents)
    f.write(contents)
    f.close()
    

def raw2std_position(position, transformation_matrix, origin_shift):
    """
    convert non-conventional position (raw position from input structure in IO stream) to standardized position.
    
    position_std=transformation_matrix*position_input+origin_shif (mod 1).
    
    Arguments:
        position: input position of atom [1x3].
        transformation_matrix: array-like [3x3].
        origin_shift: array-like [1x3].
    
    Returns:
        standardized position.
    """
    I=np.identity(3) # unit matrix
    
    position=np.array(position)
    transformation_matrix=np.array(transformation_matrix)
    origin_shift=np.transpose(origin_shift)
    
    position_std=None
            
    for i in range(0,position.shape[0]):
        tmp=np.dot(transformation_matrix, position[i]*np.transpose(I[i]))
        if position_std is None:
            position_std=tmp
        else:
            position_std += tmp
    position_std += origin_shift
    position_std=normalize_position(position_std, dtype='Direct')[:-1]
    return position_std

def translation(position, direction):
    """
    rotation position.
    
    Arguments:
        position: atomic position. The valid format:
            [0.1, 0.0, 0.0, 'Direct']
            [0.1, 0.0, 0.0]
            [5.234, 0.0, 0.0, 'Cartesian']
        direction: direction vector to add the vacuum along lattice vector(a/b/c). The valid format is :
            [0.1, 0, 0, 'Direct']
            [0.1, 0, 0] (for Direct, can not be specify)
            [5.234, 0, 0, 'Cartesian'] (for Cartesian, must be given)
    
    Returns:
        new position.
    """
    from .check import check_formated_position_only_cartesian, check_formated_position_only_direct
    
    # check
    new=None
    if check_formated_position_only_direct(position) and check_formated_position_only_direct(direction):
        tmp=np.array(position[:3])+np.array(direction[:3])
        new=[tmp[0], tmp[1], tmp[2], 'Direct']
    elif check_formated_position_only_cartesian(position) and check_formated_position_only_cartesian(direction):
        tmp=np.array(position[:3])+np.array(direction[:3])
        new=[tmp[0], tmp[1], tmp[2], 'Cartesian']
    else:
        raise ValueError('unmatched type between position and direction')
    return new
    
def rotation(position, axis, theta, **kwargs):            
    """
    rotation position. if giving origin, rotation axis will not through the coordinate origin [0,0,0]. given origin is start point of axis.
    
    Arguments:
        position: atomic position. The valid format:
            [0.1, 0.0, 0.0, 'Direct']
            [0.1, 0.0, 0.0]
            [5.234, 0.0, 0.0, 'Cartesian']
        axis: rotation axis. The valid format:
            [0.1, 0.0, 0.0, 'Direct']
            [0.1, 0.0, 0.0]
            [5.234, 0.0, 0.0, 'Cartesian']
        theta: rotation angle. The valid format:
            [30, 'Degree']
            [0.2, 'Radian']
            
        kwargs:
            origin: rotation origin. Noth that it is the origin of the axis of rotation, not a point on the axis of rotation.
                The valid format:
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0]
                [5.234, 0.0, 0.0, 'Cartesian']
    
    Returns:
        new position.
    """        
    from .check import check_formated_position_only_cartesian, check_formated_position_only_direct, check_formated_angle
    
    # check
    tmpa=np.array(axis[:3])/np.linalg.norm(axis[:3])
    if len(axis) == 3:
        axis=[tmpa[0], tmpa[1], tmpa[2], 'Direct']
    else:
        axis=[tmpa[0], tmpa[1], tmpa[2], axis[-1]]
    
    origin=None
    if 'origin' in kwargs: origin=kwargs['origin']
    
    if not check_formated_angle(theta): raise ValueError('unrecognized theta')
    theta=any2radian(theta)[0]
    C=np.math.cos(theta)
    S=np.math.sin(theta)

    ax=float(axis[0])
    ay=float(axis[1])
    az=float(axis[2])
    t=1-C
    rotation_matrix=[[t*ax*ax+C,    t*ax*ay-S*az, t*ax*az+S*ay],
                     [t*ax*ay+S*az, t*ay*ay+C,    t*ay*az-S*ax],
                     [t*ax*az-S*ay, t*ay*az+S*ax, t*az*az+C]]

    # rotation_matrix=[[C+(1-C)*ax*ax, (1-C)*ax*ay+az*S, (1-C)*ax*az-ay*S],
    #              [(1-C)*ax*ay-az*S, C+(1-C)*ay*ay, (1-C)*ay*az+ax*S],
    #              [(1-C)*ax*az+ay*S, (1-C)*ay*az-ax*S, C+(1-C)*az*az]] 

    new=None
    if origin is None:
        if check_formated_position_only_cartesian(position) and check_formated_position_only_cartesian(axis):
            tmp=np.dot(position, rotation_matrix)
            new=[tmp[0], tmp[1], tmp[2], 'Cartesian']
        elif check_formated_position_only_direct(position) and check_formated_position_only_direct(axis):
            tmp=np.dot(position, rotation_matrix)
            new=[tmp[0], tmp[1], tmp[2], 'Direct']
    else:
        if check_formated_position_only_cartesian(position) and check_formated_position_only_cartesian(axis) and check_formated_position_only_cartesian(origin):
            tmp=np.dot(np.array(position[:3])-np.array(origin[:3]), rotation_matrix)+np.array(origin[:3])
            new=[tmp[0], tmp[1], tmp[2], 'Cartesian']
        elif check_formated_position_only_direct(position) and check_formated_position_only_direct(axis) and check_formated_position_only_direct(origin):
            tmp=np.dot(np.array(position[:3])-np.array(origin[:3]), rotation_matrix)+np.array(origin[:3])
            new=[tmp[0], tmp[1], tmp[2], 'Direct']
    return new

def degree2radian(theta):
    """
    convert degree to radian.
    """
    from .check import check_formated_angle_only_degree
    
    if check_formated_angle_only_degree(theta):
        return [np.pi*(float(theta[0])/180.0),'Radian']
    else:
        raise ValueError('unrecognized theta')
    
def radian2degree(theta):
    """
    convert radian to degree.
    """
    from .check import check_formated_angle_only_radian
    
    if check_formated_angle_only_radian(theta):
        return [theta*180.0/np.pi, 'Degree']
    else:
        raise ValueError('unrecognized theta')
    
def any2radian(theta):
    """
    convert to radian.
    """
    from .check import check_formated_angle, check_formated_angle_only_degree
    
    if not check_formated_angle(theta):
        raise ValueError('unrecognized theta')
    if check_formated_angle_only_degree(theta):
        return [np.pi*(float(theta[0])/180.0),'Radian']
    else:
        return theta
    
def any2degree(theta):
    """
    convert to degree.
    """
    from .check import check_formated_angle, check_formated_angle_only_radian
    
    if not check_formated_angle(theta):
        raise ValueError('unrecognized theta')
    if check_formated_angle_only_radian(theta):
        return [theta*180.0/np.pi, 'Degree']
    else:
        return theta
    
def formatting_species(name_of_species):
    """
    convert this to formated species. like, 'Na+' -> 'Na1+'.
    """
    from .check import check_species
    import re

    if check_species(name_of_species):
        result = re.match('([A-Z][a-z]?)([0-9.]*)([-+])$', name_of_species)
        symbol, valence, sign= result.groups()
        if valence == '': valence = '1'
        return symbol + valence + sign 

def formatting_position(formated_atom, dtype='Direct', **kwargs):
    """
    convert the position of given formated_atom to formated position.
    
    Argurments:
        formated_atom: formated atom. Note that type of coordinate is 'Direct',  you can not specify
            the type. The valid formation:
                ['Na', 0.1, 0.0, 0.0, 'Direct']
                ['Na', 0.1, 0.0, 0.0] # Direct-type
                ['Na', 5.234, 0.0, 0.0, 'Cartesian']
                
                contain species information:
                ['Na1+', 0.1, 0.0, 0.0, 'Direct']
                ['Na1+', 0.1, 0.0, 0.0]
                ['Na1+', 5.234, 0.0, 0.0, 'Cartesian']
                
                Also, it is support the following format (Ignore element's symbol).
                [0.1, 0.0, 0.0, 'Direct']
                [0.1, 0.0, 0.0] # Direct-type
                [5.234, 0.0, 0.0, 'Cartesian']
                
        dtype (default='Direct'): type of formated positon after convertion. 'Direct' or 'Cartesian'
            
        kwargs:
            for "Cartesian"-type position:
                lattice: lattice parameter of structure. i.e.
                    [[x1,y1,z1],
                     [x2,y2,z2],
                     [x3,y3,z3]]
            isNormalizingCoordinate (default=True): whether to remove the periodic boundary condition, 
                    ensure the value of atomic coordinate is between 0 and 1 (i.e. 1.3 -> 0.3).
            precision (default=1e-3): used to determine whether the component of position is very close to 1.
                    from other atoms.
    """
    
    from .check import check_formated_atom_only_direct, check_formated_atom_only_cartesian
    from .check import check_formated_position_only_direct, check_formated_position_only_cartesian
    from .convert import any2direct, any2cartesian, normalize_position
    from .variables import default_constants
    
    isNormalizingCoordinate=default_constants.isNormalizingCoordinate.value
    if 'isNormalizingCoordinate' in kwargs:
        isNormalizingCoordinate=kwargs['isNormalizingCoordinate']
    precision=default_constants.precision.value
    if 'precision' in kwargs:
        precision=kwargs['precision']
    
    lattice=None        
    if 'lattice' in kwargs:
        lattice=kwargs['lattice']
    
    position=None
    if check_formated_atom_only_direct(formated_atom):
        position=formated_atom[1:] if len(formated_atom) == 5 else formated_atom[1:]+['Direct']
    elif check_formated_atom_only_cartesian(formated_atom):
        if not('lattice' in kwargs): raise ValueError('Need lattice')
        position=any2direct(lattice, formated_atom[1:])   
    elif check_formated_position_only_direct(formated_atom):
        position=formated_atom if len(formated_atom) == 4 else formated_atom+['Direct']
    elif check_formated_position_only_cartesian(formated_atom):
        if not('lattice' in kwargs): raise ValueError('Need lattice')
        position=any2direct(lattice, formated_atom)
    else: 
        import warnings
        warnings.warn('wrong format of formated_atom')
        return None

    if isNormalizingCoordinate:
        position=normalize_position(position=position, dtype='Direct', precision=precision) # i.g. [0.1, 0.0, 0.0, 'Direct'], [5.234, 0.0, 0.0, 'Cartesian']
        
    if dtype.lower().startswith('d'):
        pass
    elif dtype.lower().startswith('c'):
        if not('lattice' in kwargs): raise ValueError('Need lattice')
        position=any2cartesian(lattice=lattice, position=position)
    else:
        import warnings
        warnings.warn('wrong format of dtype')
        return None
    return position
        
def mp2jamip(struct,mp_id=None,isPersist=True):
    from .variables import default_constants
    from ..materials.structure import Structure
    from ..materials.atom import Atom
    mp = Structure()
    mp.comment = struct.composition.reduced_formula
    if mp_id: mp.label = mp_id        
    mp.ntypes=len(struct.composition.elements)
    mp.natoms=struct.composition.num_atoms
    mp.lattice=struct.lattice.matrix
    mp.volume=struct.volume
    mp.volume_per_atom=mp.volume/mp.natoms
    
    if len(struct.sites) != mp.natoms: raise ValueError('inconsistent dimensions between species and atoms')
    dtype = 'direct'
    for site in struct.sites:
        if dtype[0].lower() == 'd':
            position=site.frac_coords
        elif dtype[0].lower() == 'c':
            position=site.coords

        formated_atom=[site.species_string, position[0], position[1], position[2], dtype]                
        atom=Atom().create(formated_atom=formated_atom,
                           isNormalizingCoordinate=True,
                           isPersist=isPersist)
        mp.add_atom(atom)

    symprec=default_constants.symprec.value
    angle_tolerance=default_constants.angle_tolerance.value
    mp.update(isPersist=isPersist, symprec=symprec, angle_tolerance=angle_tolerance)
    return mp
