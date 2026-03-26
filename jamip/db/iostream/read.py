# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import numpy as np
from .cif import *

class ReadError(Exception):
    pass

class Read(object):
    """
    read a structure file. At present, the code support the following file types: 
        for single structure:
            cif, poscar, xyz, mol.
        for multiple strucutre:
            VASP:
                xdatcar
            CALYPSO:
                calypso
            
    Note that:
        1. not case sensitive for file type.
        2. for 'poscar' type, it can recognize three kinds of files which end with '.vasp', 'poscar' and 'contcar'.
        
    Examples:
        >>> raw=Read('/home/examples/structures/TlCu3Se2.cif').run()
        >>> raw=Read('/home/examples/structures/CONTCAR4', dtype='poscar').run()
        
        >>> raw=Read('/home/fu/workspace/EinsteinRelation/Si/nve/XDATCAR', dtype='xdatcar', isChanged=False).run()
        >>> raw=Read('/home/fu/workspace/Thallide/opt/212/In2SnSe2/2/10_45/XDATCAR', dtype='xdatcar', isChanged=True, srange=[5,11]).run()
        
        >>> raw=Read('/home/fu/struct.dat', dtype='calypso', elements=['Sn', 'S'], srange=[1,200]).run()
    """
    
    def __init__(self, path, dtype=None, **kwargs):
        """
        Arguments:
            python: path of file. i.e. /home/xx/xx/POSCAR, POSCAR
            dtype: data type. 
                crystal: 
                    for single structure:
                        cif, poscar
                    for multiple structures:
                        VASP:
                            xdatcar# read XDATCAR file
                        CALYPSO:
                            calypso # read struct.dat file
                            
                    for absorption spectrum
                        DM or dipole_matrix: read MATRIX file
                        
                molecule: xyz, mol
                
            kwargs:
                srange: range of structure which need to fetch from file. i.e. [100, 200]
                
                for xdatcar:
                    isChange: whether the shape of cell is changing during the calculation (True or False). 
                        For example, for full relax structure, you should set the value to 'True'.
                        
                for poscar:
                    isContainedConstraints (default=False): whether to contain the atomic constraint information.
                    isContainedVelocities (default=False): whether to read the velocities of atoms.
                    isContainedElement (default=True): whether to contain element information.

        """
        self.path=path
        
        if dtype == None:
            # for crystal
            if os.path.basename(path).lower().endswith('.cif'):
                self.dtype='cif'
            elif os.path.basename(path).lower().endswith('.vasp') or \
                os.path.basename(path).lower().endswith('poscar') or \
                os.path.basename(path).lower().endswith('contcar'):
                self.dtype='poscar'
            # for molecule
            elif  os.path.basename(path).lower().endswith('.xyz'):
                self.dtype='xyz'
            elif os.path.basename(path).lower().endswith('.mol'):
                self.dtype='mol'
            else:
                raise TypeError('unrecognized type')
        elif dtype.strip().lower() == 'cif':
            self.dtype='cif'
        elif dtype.strip().lower() == 'vasp' or \
            dtype.lower() == 'poscar' or \
            dtype.lower() == 'contcar':
            self.dtype='poscar'
        
        # for molcule
        elif dtype.strip().lower() == 'xyz':
            self.dtype='xyz'
        elif dtype.strip().lower() == 'mol':
            self.dtype='mol'
        
        # for multiple structures    
        elif dtype.strip().lower() == 'xdatcar':
            self.dtype='xdatcar'
            if not 'isChanged' in kwargs:
                raise ValueError("can't find out isChanged")
            self.isChanged=kwargs['isChanged']
            self.srange=None
            if 'srange' in kwargs:
                self.srange=kwargs['srange']
        elif dtype.strip().lower() == 'calypso':
            self.dtype='calypso'
            if not 'elements' in kwargs:
                raise ValueError("can't find out elements")
            self.elements=kwargs['elements']
            self.srange=None
            if 'srange' in kwargs:
                self.srange=kwargs['srange']
        
        # for absorption
        elif (dtype.strip().lower() == 'dm') or (dtype.strip().lower() == 'dipole_matrix'):
            self.dtype='dipole_matrix'
            self.erange=None
            self.isRemoveReduplication=None
            self.kpoint=None
            if 'erange' in kwargs:
                self.erange=kwargs['erange']
            elif 'isRemoveReduplication' in kwargs:
                self.isRemoveReduplication=kwargs['isRemoveReduplication']
                self.dE=0.01
                if 'dE' in kwargs:
                    self.dE=kwargs['dE']
            elif 'kpoint' in kwargs:
                self.kpoint=kwargs['kpoint']
                self.nv=None
                if 'nv' in kwargs:
                    self.nv=kwargs['nv']
                self.nc=None
                if 'nc' in kwargs:
                    self.nc=kwargs['nc']
                self.precision=1e-3
                if 'precision' in kwargs:
                    self.precision=kwargs['precision']
        else:
            raise TypeError('unrecognized type')
        
        # for poscar
        if self.dtype == 'poscar':
            self.isContainedConstraints=False
            if 'isContainedConstraints' in kwargs:
                self.isContainedConstraints=kwargs['isContainedConstraint']
                if not isinstance(self.isContainedConstraints, bool):
                    raise ValueError('isContainedConstraints is not a boolean value')
                
            self.isContainedVelocities=False
            if 'isContainedVelocities'in kwargs:
                self.isContainedVelocities=kwargs['isContainedVelocities']
                if not isinstance(self.isContainedVelocities, bool):
                    raise ValueError('isContainedVelocities is not a boolean value')

            self.isContainedElement=True
            if 'isContainedElement' in kwargs:
                self.isContainedElement=kwargs['isContainedElement']
                if not isinstance(self.isContainedElement, bool):
                    raise ValueError('isContainedElement is not a boolean value')
                if not self.isContainedElement:
                    import warnings
                    warnings.warn('non-contained element information in returned data')

        
    def run(self):
        """
        read file, get the structure's data.
        
        Returns:
            python dictionary of structure.
            cif: 
                lattice=[[x1,y1,z1],
                         [x2,y2,z2],
                         [x3,y3,z3]]
                 elements=['Ca', 'Fe', 'Sb']
                 numbers=[2, 8, 24]
                 type= Direct
                 positions=[[a1_x,a1_y,a1_z],
                           [a2_x,a2_y,a2_z],
                           [a3_x,a3_y,a3_z],
                           ...]
            vasp:
                comment: comment of the first line
                lattice=[[x1,y1,z1],
                         [x2,y2,z2],
                         [x3,y3,z3]]
                elements=['Ca', 'Fe', 'Sb']
                numbers=[2, 8, 24]
                type= Direct or Cartesian
                positions=[[a1_x,a1_y,a1_z],
                          [a2_x,a2_y,a2_z],
                          [a3_x,a3_y,a3_z],
                          ...]
                constraints=[[T,T,T], # Selective dynamics (optional)
                            [F,F,F],
                            [T,F,T],
                            ...]
            ...
        """
        if self.dtype == 'cif':
            return self.__readCIF()    
        elif self.dtype == 'poscar':
            return self.__readPOSCAR()
        elif self.dtype == 'xyz':
            return self.__readXYZ()
        elif self.dtype == 'mol':
            return self.__readMOL()
        elif self.dtype == 'xdatcar':
            return self.__readXDATCAR(self.isChanged, self.srange)
        elif self.dtype == 'calypso':
            return self.__readCALYPSO(self.elements, self.srange)
        elif self.dtype == 'dipole_matrix':
            # type 3
            if not(self.kpoint is None):
                return self.__readDipoleMatrix(kpoint=self.kpoint, nv=self.nv, nc=self.nc, precision=self.precision)
            # type 2
            elif not(self.isRemoveReduplication is None):
                return self.__readDipoleMatrix(isRemoveReduplication=self.isRemoveReduplication, dE=self.dE)
            # type 1
            elif not(self.erange is None):
                return self.__readDipoleMatrix(erange=self.erange)
            else:
                return self.__readDipoleMatrix()
            
    def __readCIF(self):
        """
        read CIF file
        
        Returns:
            python dictionary of structure.
                lattice=[[x1,y1,z1],
                         [x2,y2,z2],
                         [x3,y3,z3]]
                 elements=['Ca', 'Fe', 'Sb']
                 numbers=[2, 8, 24]
                 type= Direct
                 positions=[[a1_x,a1_y,a1_z],
                           [a2_x,a2_y,a2_z],
                           [a3_x,a3_y,a3_z],
                           ...]
        """
        import re
        cf=parse_cif(self.path)
        cb=cf[0][1]

        # lattice parameters
        aa=float(cb['_cell_length_a'])
        bb=float(cb['_cell_length_b'])
        cc=float(cb['_cell_length_c'])
        alpha=float(cb['_cell_angle_alpha'])
        beta=float(cb['_cell_angle_beta'])
        gamma=float(cb['_cell_angle_gamma'])
        alpha=alpha*(np.pi/180)
        beta=beta*(np.pi/180)
        gamma=gamma*(np.pi/180)

        # lattice vector
        lattice=[]
        lattice=lattice_vector(aa, bb, cc, alpha, beta, gamma)

        # elements
        elements=[]
        for symbol in cb['_atom_site_type_symbol']:
            e = re.findall("[A-Z][a-z]?",symbol)
            if len(e) == 1:
                elements.append(e[0])
            else:
                raise ValueError("Unknown element symbol %s" %symbol)

        # space group number
        group_number=None
        if '_space_group.it_number' in cb:
            group_number=str(cb['_space_group.it_number'])
        elif '_space_group_it_number' in cb:
            group_number=str(cb['_space_group_it_number'])
        elif '_symmetry_int_tables_number' in cb:
            group_number=str(cb['_symmetry_int_tables_number'])
        else:
            group_number=None

        # space group H-M symbol
        symbolHM=None
        if '_space_group.Patterson_name_h-m' in cb:
            symbolHM=format_symbol(cb['_space_group.patterson_name_h-m'])
        elif '_symmetry_space_group_name_h-m' in cb:
            symbolHM=format_symbol(cb['_symmetry_space_group_name_h-m'])
        else:
            symbolHM=None


        # symmetry operations
        for name in ['_space_group_symop_operation_xyz',
                     '_space_group_symop.operation_xyz',
                     '_symmetry_equiv_pos_as_xyz']:
            if name in cb:
                sitesym=cb[name]
                break
            else:
                sitesym=None

        # positions
        positions=[]
        if sitesym:
            positions=equival_pos(sitesym, cb)
        elif symbolHM:
            if SG.get(symbolHM):
                sitesym = SG.get(symbolHM)
                positions=equival_pos(SG.get(symbolHM), cb)
            else:
                raise SpacegroupNotFoundError('invalid spacegroup %s, not found in data base' %
                                              (symbolHM,))
        elif group_number:
            sitesym = SG.get(group_number)
            positions=equival_pos(SG.get(group_number), cb)
        else:
            raise SpacegroupValueError('either *number* or *symbol* must be given for space group!')

        # numbers 
        numbers=[]
        if '_atom_site_symmetry_multiplicity' in cb:
            numbers=cb['_atom_site_symmetry_multiplicity']
        elif sitesym:
            numbers=numbers_cal(sitesym, cb)

        # compare numbers and positions.
        if len(positions) > sum(numbers):
            import warnings
            CIFError = "Total number of atom_coords beyond the CIF declaration in %s with accuracy of 1E-3 ! Reset the accuracy to 1E-2." %self.file
            warnings.warn(CIFError)
            positions=equival_pos(sitesym, cb, prec=1e-2)

        assert len(positions) == sum(numbers)

        # comment
        for name in ['_chemical_formula_structural',
                     '_chemical_formula_sum']:
            if name in cb:
                comment = cb[name]
                break
        else:
            comment = None


        # join elements
        if len(elements) != len(set(elements)):
            edict = {}
            index = 0
            for elm,num in zip(elements,numbers):
                if elm not in edict:
                    edict[elm] = list(range(index,index+num)) 
                else:
                    edict[elm].extend(list(range(index,index+num)))
                index += num

            elements=[]
            numbers = []
            new_positions = []
            for elm in edict:
                index = np.array(edict[elm])
                elements.append(elm)
                numbers.append(len(index))
                for i in index:
                    new_positions.append(positions[i])
            positions = new_positions

        # type = 'Direct'

        lattice=np.array(lattice)
        elements=np.array(elements)
        numbers=np.array(numbers)
        positions=np.array(positions)

        cif={'comment':'jamip-cif',
             'lattice': lattice,
             'elements': elements,
             'numbers': numbers,
             'type': 'Direct',
             'positions': positions,
             'constraints':[]}

        return cif 
    
    def __readPOSCAR(self):   
        """
        read POSCAR file
        Note that: the code only support the format of POSCAR for VASP-5.x.
            In POSCAR file, it must contain the element information.
            
        Returns:
            python dictionary of structure.
                comment: comment of the first line
                lattice=[[x1,y1,z1],
                         [x2,y2,z2],
                         [x3,y3,z3]]
                elements=['Ca', 'Fe', 'Sb']
                numbers=[2, 8, 24]
                type=Direct or Cartesian
                positions=[[a1_x,a1_y,a1_z],
                          [a2_x,a2_y,a2_z],
                          [a3_x,a3_y,a3_z],
                          ...]
                constraints=[[T,T,T], # Selective dynamics (optional)
                            [F,F,F],
                            [T,F,T],
                            ...]
                velocities=[[a1_x,a1_y,a1_z], # velocities of atoms (optional)
                            [a2_x,a2_y,a2_z],
                            [a3_x,a3_y,a3_z],
                            ...]
        """
        infile=open(self.path)
        
        # comment
        comment=''
        string=infile.readline()
        if string != "":
            comment=string.strip().split('\n')[0]
            
        scale=float(infile.readline())
        
        # lattice
        # ensure all structure's scale equal 1 inside the program     
        lattice=[]
        for i in range(0, 3):
            try:
                tmp=np.array([float(s0) for s0 in infile.readline().split()])
                if tmp.shape[0] == 3:
                    lattice.append(tmp*scale)
                else:
                    raise ValueError('dimension of lattice parameter is less than 3')
            except ValueError:
                raise ValueError("can't transfer literal to float type")
        lattice=np.array(lattice)
        
        # element VASP5.x
        # Note that we need check symbol of element is valid by comparing the element table in jamipdb.
        elements=[]
        if self.isContainedElement:
            tmp=np.array(infile.readline().split())
            for i in range(0, tmp.shape[0]):
                if not tmp[i].isalpha():
                    raise ValueError('elements contain non-alphabet')
        elements=tmp
        
        # numbers
        numbers=[]
        try:
            tmp=np.array([int(s0) for s0 in infile.readline().split()])
            if self.isContainedElement:
                if elements.shape[0] != tmp.shape[0]:
                    raise ValueError("length of numbers don't match with that of elements")
            numbers=tmp
        except ValueError:
            raise("can't transfer literal to integer type!")
            
        tmp=infile.readline()
        isConstraint=False
        type=''
        if tmp.strip().lower().startswith('s'): # Selective dynamics
            isConstraint=True
            # type
            tmp=infile.readline()
            if tmp.strip().lower().startswith('c'):
                type='Cartesian'
            elif tmp.strip().lower().startswith('d'):
                type='Direct'
            else:
                raise ValueError("invalid coordinate's type in file")
        # type    
        elif tmp.strip().lower().startswith('c'):
            type='Cartesian'
        elif tmp.strip().lower().startswith('d'):
            type='Direct'
        else:
            raise ValueError("invalid coordinate's type in file")
        
        # position
        natoms=sum(numbers)
        positions=[]
        constraints=[]
        for i in range(0, natoms):
            try:
                string=infile.readline().split()
                if (not isConstraint and len(string) < 3) or (isConstraint and len(string) != 6):
                    raise ValueError('dimension of atomic position is less than 3')
                tmp=np.array([float(s0) for s0 in string[:3]])
                positions.append(tmp)
                
                # constraint
                if isConstraint and self.isContainedConstraints:
                    tmp=np.array([False if s0.startswith('F') else True for s0 in string[3:6]])
                    constraints.append(tmp)
                    
            except ValueError:
                raise ValueError("can't transfer literal to float type")
            
        # read velocity
        velocities=[]
        if self.isContainedVelocities:
            infile.readline()
            for i in range(0, natoms):
                tmp=[float(s0) for s0 in infile.readline().split()]
                velocities.append(tmp)
        
        positions=np.array(positions)
        constraints=np.array((constraints))
        velocities=np.array(velocities)
        
        
        poscar={'comment': comment,
                'lattice': lattice,
                'elements': elements,
                'numbers': numbers,
                'type': type,
                'positions': positions}
        if self.isContainedConstraints and constraints != []:
            poscar['constraints']=constraints    
        if self.isContainedVelocities and velocities != []:
            poscar['velocities']=velocities

        infile.close()
        
        return poscar
    
    def __readXYZ(self):
        """
        read XYZ file.
        Note that the coordinate type of positions can only be Cartesian.
        
        Returns:
            python dictionary of structure.
                elements=['Ca', 'Fe', 'Sb']
                numbers=[2, 8, 24]
                positions=[[a1_x,a1_y,a1_z],
                          [a2_x,a2_y,a2_z],
                          [a3_x,a3_y,a3_z],
                          ...]
        """
        infile=open(self.path)
        
        # natoms
        try:
            natoms=int(infile.readline())
        except ValueError:
            return ValueError('invalid atomic numbers in XYZ file')
        
        infile.readline() # skip comment
        
        # atoms
        counter=0 # counter of atoms
        atoms={}
        string=infile.readline()
        while(string):
            if string.split() != []: # skip blank line
                element_symbol=string.split()[0] # atomic name
                try:
                    ptmp=np.array([float(s0) for s0 in string.split()[1:]]) # atomic position
                except ValueError:
                    raise ValueError('invalid atomic position in xyz file!')
                
                if element_symbol in atoms.keys():
                    value=atoms[element_symbol]
                    atoms[element_symbol]=np.vstack((value, ptmp))
                else:
                    atoms[element_symbol]=ptmp
                counter=counter+1
                string=infile.readline()
                
        if counter != natoms:
            raise ReadError("numbers of the atoms and positions are not consistent")
        
        # conversion format
        xyz={}
        xyz['elements']=np.array(list(atoms.keys()))
        numbers=[]
        positions=[]
        for e in atoms.keys():
            dim=atoms[e].shape
            if len(dim) == 1 and dim[0] == 3:
                numbers.append(1)
                positions.append(atoms[e])
            elif len(dim) == 2 and dim[1] == 3:
                numbers.append(dim[0])
                for p in range(0,dim[0]):
                    positions.append(atoms[e][p])
            else:
                raise ReadError('invalid atomic position')
            
        xyz['numbers']=np.array(numbers)
        xyz['positions']=np.array(positions)
        xyz['type']='Cartesian'
        infile.close()
        
        return xyz
    
    def __readMOL(self):
        """
        read MOL file.
        Note that the coordinate type of positions can only be Cartesian.
        
        Returns:
            python dictionary of structure.
                elements=['Ca', 'Fe', 'Sb']
                numbers=[2, 8, 24]
                positions=[[a1_x,a1_y,a1_z],
                           [a2_x,a2_y,a2_z],
                           [a3_x,a3_y,a3_z],
                           ...]
        """
        infile=open(self.path)

        # the number of all atoms
        infile.readline()  # skip formula
        infile.readline()  # skip comment
        infile.readline()  # skip blank line
        try:
            natoms=int(infile.readline().split()[0])
        except ValueError:
            return ValueError('invalid atomic numbers in MOL file')

        # atoms
        counter=0  # counter of atoms
        atoms={}
        string=infile.readline()
        while (counter < natoms):
            if string.split() != []:  # skip blank line
                element_symbol=string.split()[3]  # atomic name
                try:
                    ptmp=np.array([float(s0) for s0 in string.split()[0:3]])  # atomic position
                except ValueError:
                    raise ValueError('invalid atomic position in mol file!')

                if element_symbol in atoms.keys():
                    value=atoms[element_symbol]
                    atoms[element_symbol]=np.vstack((value, ptmp))
                else:
                    atoms[element_symbol]=ptmp
                counter += 1
                string=infile.readline()

        # conversion format
        molecule={}
        molecule['elements']=np.array(list(atoms.keys()))
        numbers=[]
        positions=[]
        for e in atoms.keys():
            dim=atoms[e].shape
            if len(dim) == 1 and dim[0] == 3:
                numbers.append(1)
                positions.append(atoms[e])
            elif len(dim) == 2 and dim[1] == 3:
                numbers.append(dim[0])
                for p in range(0, dim[0]):
                    positions.append(atoms[e][p])
            else:
                raise ReadError('invalid atomic position!')

        molecule['numbers']=np.array(numbers)
        molecule['positions']=np.array(positions)
        molecule['type']='Cartesian'
        infile.close()
        
        return molecule
                           
                           
    def __readXDATCAR(self, isChanged, srange=None):
        """
        read XDATCAR file.
        Note that: the code only support the format of POSCAR for VASP-5.x.
            In POSCAR file, it must contain the element information.
            
        Arguments:
            isChange: whether the shape of cell is changing during the calculation (True or False). 
                For example, for full relax structure, you should set the value to 'True'.
            srange: range of structure which need to fetch from file. i.e. [100, 200]
            
        Returns:
            collection of structure. i.e. [structure0, structure1, structure2,...]
                for each structure (dictionary-like). It contains:
                    comment: comment of the first line
                    lattice=[[x1,y1,z1],
                             [x2,y2,z2],
                             [x3,y3,z3]]
                    elements=['Ca', 'Fe', 'Sb']
                    numbers=[2, 8, 24]
                    type=Direct or Cartesian
                    positions=[[a1_x,a1_y,a1_z],
                              [a2_x,a2_y,a2_z],
                              [a3_x,a3_y,a3_z],
                              ...]
                    constraints=[[T,T,T], # Selective dynamics (optional)
                                [F,F,F],
                                [T,F,T],
                                ...]
        """
        import warnings
        
        infile=open(self.path)
        
        structures = []
        
        string = infile.readline()
        identifier=string.strip()
        counter=0 # counter of structure
        while string:
            if isChanged: # shape of cell is changing during this calculation.
                if string.startswith(identifier):
                    scale=float(infile.readline()) # scale of lattice parameter
                    # lattice parameter
                    lattice=[]
                    for i in range(0,3):
                        tmp=[float(s0) for s0 in infile.readline().split()]
                        lattice.append(tmp)
                        
                    lattice=np.array(lattice)*scale
                    
                    # type of element
                    elements=np.array(infile.readline().split())
                    numbers=np.array([int(s0) for s0 in infile.readline().split()])
                    # read atom coordinate
                    if infile.readline().startswith('Direct configuration=') or infile.readline().startswith(' '):
                        positions=[]
                        for i in range(0, np.sum(numbers)):
                            positions.append([float(s0) for s0 in infile.readline().split()])
                        structure={'lattice':lattice, 'elements':elements, 'numbers':numbers, 'type':'Direct', 'positions':positions}

                        if srange is None:
                            structures.append(structure)
                        elif counter >= srange[0] and counter < srange[1]:
                            structures.append(structure)

                    counter += 1
            else:
                if string.startswith(identifier):
                    scale=float(infile.readline()) # scale of lattice parameter
                    # lattice parameter
                    lattice=[]
                    for i in range(0,3):
                        tmp=[float(s0) for s0 in infile.readline().split()]
                        lattice.append(tmp)
                    lattice=np.array(lattice)*scale
                    
                    # type of element
                    elements=np.array(infile.readline().split())
                    numbers=np.array([int(s0) for s0 in infile.readline().split()])
                    
                # read atom coordinate
                if string.startswith('Direct configuration=') or string.startswith(' '):
                    positions=[]
                    for i in range(0, np.sum(numbers)):
                        positions.append([float(s0) for s0 in infile.readline().split()])
                    structure={'lattice':lattice, 'elements':elements, 'numbers':numbers, 'type':'Direct', 'positions':positions}
                    
                    if srange is None:
                        structures.append(structure)
                    elif counter >= srange[0] and counter < srange[1]:
                        structures.append(structure)

                    counter += 1
                                
            string=infile.readline()
            
        if (not srange is None) and (counter < srange[1]):
            warnings.warn("don't reach the maximum of the given range.\nactual: [%d, %d]; expect: [%d, %d]" %(srange[0], counter, srange[0], srange[1]))
        
        infile.close()
        structures=np.array(structures)            
        return structures
    
    def __readCALYPSO(self, elements, srange=None):
        """
        read struct.dat file.
        
        Arguments:
            elements: elements of structure. i.e. ['Sn', 'S']
            srange: range of structure which need to fetch from file. i.e. [100, 200]
            
        Returns:
            collection of structure. i.e. [structure0, structure1, structure2,...]
                for each structure (dictionary-like). It contains:
                    energy: calculated energy of structure
                    lattice=[[x1,y1,z1],
                             [x2,y2,z2],
                             [x3,y3,z3]]
                    elements=['Ca', 'Fe', 'Sb']
                    numbers=[2, 8, 24]
                    type=Direct or Cartesian
                    positions=[[a1_x,a1_y,a1_z],
                              [a2_x,a2_y,a2_z],
                              [a3_x,a3_y,a3_z],
                              ...]
        """
        import warnings
        
        elements=np.array(elements)
        
        infile=open(self.path)
        
        structures=[]
        string=infile.readline().strip()
        
        counter=0 # counter of structure
        while string:
            if string.strip().startswith('Optimized Structure'):
                energy=float(infile.readline().split('=')[1])
                volume=float(infile.readline().split('=')[1])
                infile.readline().split('=')[1] # number of element
                element_numbers=np.array([int(s0) for s0 in infile.readline().split()[1:]])
                infile.readline()
                infile.readline()
            
                # lattice
                lattice=[]
                for i in range(0, 3):
                    lattice.append([float(s0) for s0 in infile.readline().split()])
                lattice=np.array(lattice)
            
                for i in range(0, 4):
                    infile.readline()
            
                # atomic position
                positions=[]
                for i in range(0, np.sum(element_numbers)):
                    positions.append([float(s0) for s0 in infile.readline().split()])
                positions=np.array(positions)
            
                structure={'energy':energy, 'lattice':lattice, 'elements':elements, 'numbers':element_numbers, 'type':'Direct', 'positions':positions}
                if srange is None:
                    structures.append(structure)
                elif counter >= srange[0] and counter < srange[1]:
                    structures.append(structure)
                counter += 1
            string=infile.readline()
        
        if (not srange is None) and (counter < srange[1]):
            warnings.warn("don't reach the maximum of the given range.\nactual: [%d, %d]; expect: [%d, %d]" %(srange[0], counter, srange[0], srange[1]))
        
        infile.close()    
        structures=np.array(structures)  
        return structures
    
    def __readDipoleMatrix(self, **kwargs):
        """
        read dipole matrix file ('MATRIX') from absorption spectrum calculation.
        
        Argument:
            kwarge:
                
                kwargs:
                    There are three types:
                    1:
                        dipole matrix with different kpoint
                        erange (default=None): range of energy (unit: eV). i.e. [-5, 5]
                    
                    2:
                        isRemoveReduplication: only output the largest value at a energy.
                        dE (default=0.01): interval of energy.
                
                    3:
                        kpoint:
                        nv: index of valence band.
                        nc: index of conduct band.
                        precision (default=1e-3): used to determine whether the two values are overlapped.
                        Note that nv and nc both count from 1.
                
        Return:
            dipole matrix. 
        """
        import warnings
        from utils.check import exist
        
        def filter(dipole_matrix, dE):
            """
            remove the reduplicated and small values with same energy, only save the largest one.
            
            Argument:
                dipole_matrix: dipole matrix [enenrgy, amp].
                dE (default=0.01): interval of energy.
            
            Return:
                dipole matrix [energy, amp]
            """
            energy=dipole_matrix[:,0]
            amp=dipole_matrix[:,1]
            
            emin=np.min(energy)
            emiax=np.max(energy)
            
            ea={} # key: energy, value: amp
            for i in range(0, len(energy)):
                e=int(energy[i]/dE)*dE
                if not(e in ea.keys()):
                    ea[e]=amp[i]
                elif ea[e] < amp[i]:
                    ea[e]=amp[i]
            energy=ea.keys()
            amp=ea.values()
            return np.transpose(np.vstack((energy, amp))) # [energy, amp]
        
        infile=open(self.path)
        
        type=1
        dipole_matrix={} # {kpoint: [nv, nc, energy, amp]}
        if 'erange' in kwargs:
            erange=kwargs['erange']
        elif 'isRemoveReduplication' in kwargs:
            isRemoveReduplication=False
            isRemoveReduplication=kwargs['isRemoveReduplication']
            dE=0.01
            if 'dE' in kwargs:
                dE=kwargs['dE']
            
            if isRemoveReduplication:
                dipole_matrix=[] # [energy, amp]
                type=2
        elif 'kpoint' in kwargs:
            kpoint=None
            kpoint=kwargs['kpoint']
            nv=None
            if 'nv' in kwargs:
                nv=kwargs['nv']
            nc=None
            if 'nc' in kwargs:
                nc=kwargs['nc']
            precision=1e-3
            if 'precision' in kwargs:
                precision=kwargs['precision']
            
            if not(kpoint is None):
                if not(nv is None) and not(nc is None):
                    dipole_matrix=None # scale
                elif not (nv is None) or not (nc is None):
                    dipole_matrix=[] # [nv/nc, energy, amp]
                else:
                    dipole_matrix=[] # [nv, nc, energy, amp]
                type=3
                    
        string=infile.readline()
        
        while string:
            if string.startswith('NV') and not('*' in string):
                tmp=string.split('=')
                nv0=int(tmp[1].split('NC')[0])
                nc0=int(tmp[2].split('ISP')[0])
                string=infile.readline()
                if string.startswith('E') and not('*' in string):
                    tmp=string.split('=')
                    energy0=float(tmp[1].split('AMP')[0])
                    amp0=float(tmp[2].split('K')[0])
                    kpoint0=[float(s0) for s0 in tmp[-1].split(',')]
                    value=np.array([nv0, nc0, energy0, amp0])
                    
                    if type == 1:
                        kpoint0='[%.6f, %.6f, %.6f]' %(kpoint0[0], kpoint0[1], kpoint0[2])
                        if not(kpoint0 in dipole_matrix.keys()):
                            dipole_matrix[kpoint0]=value # [band[nv,nc], energy, amp]
                        else:
                            dipole_matrix[kpoint0]=np.vstack(([dipole_matrix[kpoint0], value]))
                    elif type == 2:
                        dipole_matrix.append([energy0, amp0])
                    elif type == 3:
                        if exist(kpoint, [kpoint0], entity_type='float', precision=precision):
                            if not(nv is None) and not(nc is None):
                                if (nv == nv0) and (nc == nc0):
                                    dipole_matrix=[energy0, amp0]
                            if (not(nv is None) and (nc is None)) and (nv == nv0):
                                dipole_matrix.append([nc0, energy0, amp0])
                            if ((nv is None) and not(nc is None)) and (nc == nc0):
                                dipole_matrix.append([nv0, energy0, amp0])
                            if (nv is None) and (nc is None):
                                dipole_matrix.append([nv0, nc0, energy0, amp0])
            string=infile.readline()
            
        if type == 2:
            dipole_matrix=np.array(dipole_matrix)
            dipole_matrix=filter(dipole_matrix, dE)
            dipole_matrix=sorted(dipole_matrix, key=lambda dp: dp[0])
        elif type == 3:
            dipole_matrix=np.array(dipole_matrix)
            
        return dipole_matrix
