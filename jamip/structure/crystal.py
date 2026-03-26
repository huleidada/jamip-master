__all__=['Read','convert_value','parse_multiline_string','parse_singletag',
'parse_loop','parse_items','parse_block','parse_cif','parse_in','format_symbol',
'equival_pos','numbers_cal','lattice_vector','SpacegroupError',
'SpacegroupNotFoundError','SpacegroupValueError']

import re
import numpy as np
import pathlib

pattern_float = re.compile(r'([-+]?)(\d*)(\.?)(\d*)')

class ReadError(Exception):
    pass
    
class Read(object):
    """
    reading structure
    
    arguments:
        file: path of structure. i.e. /home/xx/xx/POSCAR, POSCAR
        type: type of structure file. i.e. crystal: cif, poscar; molecule: xyz, mol....
    
    """
    
    def __init__(self, file, ftype=None, multi=False):
        self.file = file
        self.multi = multi
        self.pointer = None
        path = pathlib.Path(file)
        
        if ftype == None:
            if path.suffix == '.cif':
                self.ftype='cif'
            elif path.suffix == '.xyz':
                self.ftype='xyz'
            elif path.suffix == '.extxyz':
                self.ftype='extxyz'
            elif path.suffix == '.sdf':
                self.ftype='sdf'
            elif path.suffix == '.mol':
                self.ftype='mol'
            elif path.suffix == '.pdb':
                self.ftype='pdb'
            elif path.suffix == '.vasp' or path.suffix == '.poscar':
                self.ftype = 'poscar'
            elif path.suffix == '.xml':
                self.ftype = 'qe'
            elif path.suffix == '.in':
                self.ftype = 'qe'
            elif path.suffix == '.struct':
                self.ftype = 'wien2k'
            elif 'CONTCAR' in path.name:
                self.ftype='poscar'
            elif 'POSCAR' in path.name:
                self.ftype='poscar'
            else:
                raise ReadError(f'please specify the type of file! input path is {file}.')
            
        elif ftype == 'cif':
            self.ftype='cif'
        elif ftype.lower() == 'poscar' or ftype.lower() == 'vasp':
            self.ftype='poscar'
        elif ftype.lower() == 'xyz':
            self.ftype='xyz'
        elif ftype.lower() == 'extxyz':
            self.ftype='extxyz'
        elif ftype.lower() == 'sdf':
            self.ftype='sdf'
        elif ftype.lower() == 'mol':
            self.ftype='mol'
        elif ftype.lower() == 'qe':
            self.ftype='qe'
        elif path.suffix == '.pdb':
            self.ftype='pdb'
        elif ftype.lower() == 'wien2k':
            self.ftype='wien2k'
        elif ftype.lower() == 'gaussian':
            self.ftype='gaussian'
        else:
            raise ReadError('unknown type of file!')
                    
    def getStructure(self):
        """
        read structure
        
        returns:
            json's object of a structure
            
        """
        if self.ftype == 'cif':
            return self.__readCIF()
        elif self.ftype == 'poscar':
            return self.__readPOSCAR()
        elif self.ftype == 'xyz':
            return self.__readXYZ()
        elif self.ftype == 'extxyz':
            return self.__readExtXYZ()
        elif self.ftype == 'sdf':
            return self.__readMOL()
        elif self.ftype == 'mol':
            return self.__readMOL()
        elif self.ftype == 'qe':
            return self.__readQE()
        elif self.ftype == 'pdb':
            return self.__readPDB()
        elif self.ftype == 'gaussian':
            return self.__readGS()
        elif self.ftype == 'wien2k':
            return self.__readWien()

    def __readGS(self):
        """
        read gs.gjf file

        returns:
            cif: A dictionary including:
                 elements=['Ca', 'Fe', 'Sb']
                 numbers=[2, 8, 24]
                 type= Direct
                 positions=[[a1_x,a1_y,a1_z],
                           [a2_x,a2_y,a2_z],
                           [a3_x,a3_y,a3_z],
                           ...]
        """
        input=open(self.file)
        # gaussian settings
        for line in input:
            if line.strip() == '':
                break
        
        # comment
        comment = ''
        for line in input:
            if line.strip() != '':
                comment += line.strip()
            else:
                break

        # structure
        charge, mspin = input.readline().split()
        species = []
        positions = []
        for line in input:
            line = line.split()
            if len(line) == 4:
                species.append(line[0])
                positions.append(line[1:])
            else:
                break
        coords = np.array(positions, dtype=float)
        elements, indices, numbers = np.unique(species, return_inverse=True, return_counts=True)
        mask = np.argsort(indices)
        positions = coords[mask]
            
        # TODO
        # connection 
        #for line in input:
        #    line = line.split()

        molecule={'comment': comment,
                  'elements': elements,
                  'numbers': numbers,
                  'positions': positions,
        #          'connections': None,
        }
        return molecule
    
    def __readWien(self):
        """
        read case.struct file
        SYSTEM
        F   LATTICE,NONEQUIV.ATOMS:  2 225 Fm-3m
        MODE OF CALC=RELA unit=ang
        8.178738  8.178738  8.178738 90.000000 90.000000 90.000000
        ATOM  -1: X=0.00000000 Y=0.00000000 Z=0.00000000
                MULT= 1          ISPLIT= 8
        Ti1        NPT=  781  R0=0.00005000 RMT=    0.0000   Z: 22.000
        LOCAL ROT MATRIX:    1.0000000 0.0000000 0.0000000
                            0.0000000 1.0000000 0.0000000
                            0.0000000 0.0000000 1.0000000
        ATOM   2: X=0.50000000 Y=0.50000000 Z=0.50000000
                MULT= 1          ISPLIT= 8
        C          NPT=  781  R0=0.00010000 RMT=    0.0000   Z:  6.000
        LOCAL ROT MATRIX:    0.0000000 0.0000000 0.0000000
                            0.0000000 0.0000000 0.0000000
                            0.0000000 0.0000000 0.0000000
        0      NUMBER OF SYMMETRY OPERATIONS

        returns:
            cif: A dictionary including:
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
        from collections import defaultdict
        
        input=open(self.file)
        comment=input.readline().strip()
        line = input.readline()
        latticetype = line.split()[0]
        nunique = int(line[27:].split()[0])
        # spacegroup, hm_symbol = int(line[30:].split()[0])

        # lattice type: F, I, C, R, A, B
        if latticetype == 'P':
            cell = np.eye(3)
        elif latticetype == 'F':
            cell = np.array([[ 0.0, 0.5, 0.5],
                             [ 0.5, 0.0, 0.5],
                             [ 0.5, 0.5, 0.0]])
        elif latticetype == 'I':
            cell = np.array([[-0.5, 0.5, 0.5],
                             [ 0.5,-0.5, 0.5],
                             [ 0.5, 0.5,-0.5]])
        elif latticetype == 'R':
            cell = np.array([[ 2.0/3.0, 1.0/3.0, 1.0/3.0],
                             [-1.0/3.0, 1.0/3.0, 1.0/3.0],
                             [-1.0/3.0,-2.0/3.0, 2.0/3.0]])
        if latticetype == 'CXY' and latticetype == 'C':
            cell = np.array([[0.5, 0.5, 0.0],
                             [0.5,-0.5, 0.0],
                             [0.0, 0.0, 1.0]])
        elif latticetype == 'CXZ' and latticetype == 'B':
            cell = np.array([[ 0.5, 0.0, 0.5],
                             [ 0.0, 1.0, 0.0],
                             [ 0.5, 0.0,-0.5]])
        elif latticetype == 'CYZ' and latticetype == 'A':
            cell = np.array([[-1.0, 0.0, 0.0],
                             [ 0.0,-0.5, 0.5],
                             [ 0.0, 0.5, 0.5]])
        line = input.readline()
        calc = re.findall(r"CALC=([A-z]+)", line)
        # calc method: RELA（全相对论）, NREL（非相对论）
        unit = re.findall(r"unit=([A-z]+)", line)
        # unit: ang, bohr
        a,b,c,alpha,beta,gamma = np.array(input.readline().split(), dtype=float)
        ca = np.cos(np.radians(alpha))
        cb = np.cos(np.radians(beta))
        cg = np.cos(np.radians(gamma))
        sg = np.sin(np.radians(gamma))

        lattice = np.array([[a, b * cg, c * cb],
                            [0, b * sg, c * (ca - cb * cg) / sg],
                            [0, 0, c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg) / sg]])
        lattice = np.dot(cell, lattice) * 0.529177

        # someday the unit can be considered
        # if len(unit) == 0 or unit[0].lower() == 'bohr':
        #     lattice *= 0.529177
        # elif unit[0].lower() == 'ang':
        #     lattice *= 1.0
        # else:
        #     raise ReadError(f"Unsupported unit {unit[0]} in {self.file}!")

        atoms = [{} for i in range(nunique)]
        # info: species, multiplicity, and atomic number
        # NPT: number of points in the radial mesh
        # R0: starting radius of the radial mesh
        # RMT: muffin-tin radius
        # ISPLIT: number of angular momentum channels
        # Z: atomic number
        # LOCAL ROT MATRIX: local rotation matrix of the atom
        atom_idx = -1
        for line in input:
            if "ATOM" in line:
                atom_idx += 1
                x = re.findall(r"X\s*=\s*(-?\d+\.\d+)",line)[0]
                y = re.findall(r"Y\s*=\s*(-?\d+\.\d+)",line)[0]
                z = re.findall(r"Z\s*=\s*(-?\d+\.\d+)",line)[0]
                site = np.array([x,y,z], dtype=float)
                atoms[atom_idx]["site"] = site
            elif 'RMT' in line:
                specie = re.findall(r'[A-Z][a-z]?', line.split()[0])[0]
                Z = re.findall(r'Z\s*:\s*(\d+\.\d+)', line)[0]
                atoms[atom_idx]['specie'] = specie
                atoms[atom_idx]['Z'] = float(Z)
                # LOCAL ROT MATRIX
                vector1 = input.readline().split()[-3:]
                vector2 = input.readline().split()[-3:]
                vector3 = input.readline().split()[-3:]
                matrix = np.array([vector1, vector2, vector3], dtype=float)
                atoms[atom_idx]["LOCAL ROT MATRIX"] = matrix
            else:
                results = re.findall(r'[A-z0-9]\s*=\s*(\d+\.\d+)',line)
                for key,value in results:
                    atoms[atom_idx][key] = float(value)
        
        print(nunique)
        print(atoms)

        elements = []
        numbers = []
        positions = defaultdict(list)
        for value in atoms:
            specie = value.get('specie', 'Unknown')
            for i, elm in enumerate(elements):
                if elm == specie:
                    numbers[i] += 1
                    break
            else:
                elements.append(specie)
                numbers.append(1)
            positions[specie].append(value['site'])
        positions = np.array([pos for specie in elements for pos in positions[specie]], dtype=float)

        cif={'comment':comment,
             'lattice': lattice,
             'elements': elements,
             'numbers': numbers,
             'type': 'Direct',
             'positions': positions}

        return cif       

    def __readQE(self):
        """
        read QE.xml file

        returns:
            cif: A dictionary including:
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
        from math import sqrt, sin, cos
        if self.file.endswith('.xml'):
            return parse_qe_xml(self.file)

        elif self.file.endswith('.in'):
            params = parse_in(self.file)
            ibrav = int(params['ibrav'])
            natoms = int(params['nat'])
            cellunit = 'angstrom'

            celldm = [None]*6
            namelist = ['A','B','C','cosAB','cosAC','cosBC']
            for i, name in enumerate(namelist):
                if name in params:
                    celldm[i] = float(params[name])
                elif f'celldm({i+1})' in params:
                    celldm[i] = float(params[f'celldm({i+1})'])
                    if i in [1,2]: celldm[i] *= celldm[0]

            # lattice %
            
            if ibrav == 1:      # cubic P
                lattice = np.eyes(3) * celldm[0] 
            elif ibrav == 2:    # cubic F
                lattice = np.array([[-1, 0, 1],
                                    [ 0, 1, 1],
                                    [-1, 1, 0]]) * celldm[0]/2 
            elif ibrav == 3:    # cubic I
                lattice = np.array([[ 1, 1, 1],
                                    [-1, 1, 1],
                                    [-1,-1, 1]]) * celldm[0]/2
            elif ibrav == -3:   # cubic I
                lattice = np.array([[-1, 1, 1],
                                    [ 1,-1, 1],
                                    [ 1, 1,-1]]) * celldm[0]/2
            elif ibrav == 4:    # Hexagonal and Trigonal P   celldm(3)=c/a
                a = celldm[0];  c = celldm[2]
                lattice = np.array([[   1,         0, 0],
                                    [-1/2, sqrt(3)/2, 0],
                                    [   0,       0, c/a]]) * a
            elif ibrav == 5:    # Trigonal R, 3fold axis c   celldm(4)=cos(gamma)
                a = celldm[0];  c = celldm[3]
                tx=sqrt((1-c)/2);  ty=sqrt((1-c)/6);  tz=sqrt((1+2*c)/3)
                lattice = np.array([[ tx,  -ty, tz],
                                    [  0, 2*ty, tz],
                                    [-tx,  -ty, tz]]) * a    
            elif ibrav == -5:   # Trigonal R, 3fold axis <111>   celldm(4)=cos(gamma) 
                a = celldm[0];  c = celldm[3]
                tx=sqrt((1-c)/2);  ty=sqrt((1-c)/6);  tz=sqrt((1+2*c)/3)
                u = tz - 2*sqrt(2)*ty;  v = tz + sqrt(2)*ty
                lattice = np.array([[ u, v, v],
                                    [ v, u, v],
                                    [ v, v, u]]) * a / sqrt(3) 
            elif ibrav == 6:    # Tetragonal P  
                a = celldm[0];  c = celldm[2]
                lattice = np.array([[ a, 0, 0],
                                    [ 0, a, 0],
                                    [ 0, 0, c]]) 
            elif ibrav == 7:    # Tetragonal I
                a = celldm[0];  c = celldm[2]
                lattice = np.array([[  1, -1, c/a],
                                    [  1,  1, c/a],
                                    [ -1, -1, c/a]]) * a/2
            elif ibrav == 8:    # Orthorhombic P 
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                lattice = np.array([[ a, 0, 0],
                                    [ 0, b, 0],
                                    [ 0, 0, c]])
            elif ibrav == 9:    # Orthorhombic base-centered
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                lattice = np.array([[ a/2, b/2, 0],
                                    [-a/2, b/2, 0],
                                    [   0,   0, c]])
            elif ibrav == -9:   
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                lattice = np.array([[ a/2, -b/2, 0],
                                    [ a/2,  b/2, 0],
                                    [   0,    0, c]])
            elif ibrav == 91:    
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                lattice = np.array([[ a,   0,    0],
                                    [ 0, b/2, -c/2],
                                    [ 0, b/2,  c/2]])
            elif ibrav == 10:   # Orthorhombic face-centered
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                lattice = np.array([[ a/2,   0, c/2],
                                    [ a/2, b/2,   0],
                                    [   0, b/2, c/2]])
            elif ibrav == 11:   # Orthorhombic body-centered
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                lattice = np.array([[ a/2, b/2, c/2],
                                    [-a/2, b/2, c/2],
                                    [-a/2,-b/2, c/2]])
            elif ibrav == 12:   # Monoclinic P, unique axis c
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                gamma = np.arccos(celldm[3])
                lattice = np.array([[           a,            0, 0],
                                    [b*cos(gamma), b*sin(gamma), 0],
                                    [           0,            0, c]])
            elif ibrav == -12:  
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                beta = np.arccos(celldm[4])
                lattice = np.array([[          a, 0,           0],
                                    [          0, b,           0],
                                    [c*cos(beta), 0, c*sin(beta)]])
            elif ibrav == 13:   # Monoclinic base-centered
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                gamma = np.arccos(celldm[3])
                lattice = np.array([[         a/2,            0, -c/2],
                                    [b*cos(gamma), b*sin(gamma),    0],
                                    [         a/2,            0,  c/2]])
            elif ibrav == -13:  
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                beta = np.arccos(celldm[4])
                lattice = np.array([[        a/2, b/2,           0],
                                    [       -a/2, b/2,           0],
                                    [c*cos(beta),   0, c*sin(beta)]])
            elif ibrav == 14:   # Triclinic
                a = celldm[0];  b = celldm[1];  c = celldm[2]
                gamma = np.arccos(celldm[3])
                beta  = np.arccos(celldm[4])
                alpha = np.arccos(celldm[5])
                u = c*(cos(alpha)-cos(beta)*cos(gamma))/sin(gamma)
                v = c*sqrt( 1 + 2*cos(alpha)*cos(beta)*cos(gamma) - \
                    cos(alpha)**2-cos(beta)**2-cos(gamma)**2 )/sin(gamma) 
                lattice = np.array([[          a,             0, 0],
                                    [b*cos(gamma), b*sin(gamma), 0],
                                    [ c*cos(beta),            u, v]])

            elif ibrav == 0:   # free
                lattice = []
                with open(self.file,'r') as f:
                    for line in f:
                        if 'CELL_PARAMETERS' in line:
                            cellunit = line.split()[1]
                            for i in range(3):
                                lat = f.readline().split()
                                assert len(lat) == 3
                                lattice.append(lat)

            if cellunit == 'angstrom' or cellunit == 'alat':
                lattice = np.asarray(lattice,dtype=float)  
            elif cellunit == 'bohr':
                lattice = np.asarray(lattice,dtype=float) * 0.529177  
            else:
                raise KeyError("Unsupport CELL_PARAMETER Unit. ")

            # positions %
            with open(self.file,'r') as f:
                 for line in f:
                    if 'ATOMIC_POSITIONS' in line:
                        posunit = line.split()[1]
                        elmdict = {}
                        for i in range(natoms):
                            apos = f.readline().split()
                            elm = apos[0]
                            if elm not in elmdict:
                                elmdict[elm] = []
                            elmdict[elm].append(apos[1:4])
                        

            elements = []
            numbers = []
            positions = []
            for elm,position in elmdict.items():
                elements.append(elm)
                numbers.append(len(position))
                positions.extend(position)
            positions = np.array(positions,dtype=float)
            
            if posunit == 'angstrom' or posunit == 'alat':
                type = 'Cartesian'
            elif posunit == 'bohr':
                type = 'Cartesian'
                positions = positions*0.529177
            elif posunit == 'crystal':
                type = 'Direct'
            else:
                raise KeyError("Unsupport ATOMIC_POSITIONS Unit. ")
                
        espessro = {'comment': 'jamip',
                    'lattice':lattice,
                    'elements':elements,
                    'numbers':numbers,
                    'type': type,
                    'positions':positions}

        return espessro
    
    def __readCIF(self, p1=False):
        """
        read CIF file

        returns:
            cif: A dictionary including:
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
        from jamip.db.iostream.spaceGroupD3 import spacegroups as SG
        cf=parse_cif(self.file)
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
                if e[0] in ('D','T'):
                    elements.append('H')
                else:
                    elements.append(e[0])
            else:
                raise ValueError("Unknown element symbol %s" %symbol)
        
        # sitesym
        sitesym = get_sitesym(cb)
        if sitesym is None:
            if p1:
                sitesym = ['x,y,z']
            else:
                raise SpacegroupValueError('either *number* or *symbol* must be given for space group!')

        # positions
        positions=equival_pos(sitesym, cb)
   
        # numbers
        numbers=[]
        if '_atom_site_symmetry_multiplicity' in cb:
            numbers=np.array(cb['_atom_site_symmetry_multiplicity'], dtype=int)
        else:
            numbers=numbers_cal(sitesym, cb)

        # compare numbers and positions.
        if len(positions) > sum(numbers):
            import warnings
            CIFError = "Total number of atom_coords beyond the CIF declaration in %s with accuracy of 1E-3 ! Reset the accuracy to 1E-2." %self.file
            warnings.warn(CIFError) 
            positions=equival_pos(sitesym, cb, prec=1e-2)

        assert len(positions) == sum(numbers), ' positions %s != atoms number %s' %(len(positions), sum(numbers))

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

        # type
        type='Direct'

        lattice=np.array(lattice)
        elements=np.array(elements)
        numbers=np.array(numbers)
        positions=np.array(positions)
        

        cif={'comment':comment,
             'lattice': lattice,
             'elements': elements,
             'numbers': numbers,
             'type': type,
             'positions': positions}

        return cif       
    
    def __readPOSCAR(self): # only for VASP5.x (It means the file need to contain the element information)
        """
        read POSCAR file
        
        poscar:
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
        
        returns:
            json's object of a structure
            
        """
        input=open(self.file)
        
        # comment
        comment=input.readline().strip()            
        scale=float(input.readline())
        
        # lattice
        # ensure all structure's scale equal 1 inside the program     
        lattice=[]
        for i in range(0,3):
            try:
                tmp=np.array(input.readline().split(),dtype=float)
                assert tmp.shape[0] == 3, "Lattice vector length != 3 " 
                lattice.append(tmp*scale)
            except ValueError:
                raise ValueError("can't transfer literal to float type!")
        lattice=np.array(lattice)
        
        # element VASP5.x
        # Note that:
        #   need check symbol of element is valid by comparing the element table in jamipdb
        elements=np.array(input.readline().split())
        for i in elements:
            assert i.isalpha(), 'elements contain non-alphabet!'
        
        # numbers
        numbers=np.array(input.readline().split())
        assert len(numbers) == len(elements), "length of numbers don't match with that of elements"
        try:
            numbers = numbers.astype(int)
        except ValueError:
            raise ValueError("can't transfer literal to int type!")
        
        tmp=input.readline().lower()
        isConstraint=False
        type=''
        if tmp.startswith('s'): # Selective dynamics
            isConstraint=True
            # type
            tmp=input.readline().lstrip().lower()
            if tmp.startswith('c'):
                type='Cartesian'
            elif tmp.startswith('d'):
                type='Direct'
            else:
                raise ValueError('type of POSCAR is invalid')
        # type    
        elif tmp.lstrip().startswith('c'):
            type='Cartesian'
        elif tmp.lstrip().startswith('d'):
            type='Direct'
        else:
            raise ValueError('type of POSCAR is invalid')
        
        # position
        natoms=sum(numbers)
        positions=[]
        constraints=[]
        for i in range(0, natoms):
            try:
                string=input.readline().split()
                positions.append(np.array(string[:3],dtype='float'))

                # constraint
                if isConstraint :
                    assert len(string) >= 6, 'The number of columns needs to be greater than 6 !'
                    tmp=np.array([False if s0.startswith('F') else True for s0 in string[3:6]])
                    constraints.append(tmp)
            except ValueError:
                raise ValueError("can't transfer literal to float type!")
        positions=np.array(positions)
        if type == 'Cartesian':
            positions = positions*scale
        constraints=np.array((constraints))

        # velocity
        velocities = []
        for line in input:
            if line.strip():
                v = line.split()
                if len(v) != 3: break
                velocities.append(v)
        if len(velocities) == natoms:
            velocities = np.array(velocities, dtype=float)
        else: 
            velocities = []
        
        input.close()
        poscar={'comment':comment,
                'lattice':lattice,
                'elements':elements,
                'numbers':numbers,
                'type':type,
                'positions':positions,
                'velocities':velocities,
                'constraints':constraints}
        return poscar

    def __readPDB(self):
        """
        Read atom line from pdb format
        HETATM    1  H14 ORTE    0       6.301   0.693   1.919  1.00  0.00        H

        returns:
            object of a structure
        """
        from collections import defaultdict
        input=open(self.file)

        lattice = None
        title = "Built with JAMIP"
        elements = []
        positions = []

        for line in input:
            row = line.split()
            if row[0] == "ATOM" or row[0] == "HETATM":
  
                name = line[12:16].strip()
                #altloc = line[16]
                resname = line[17:21].strip()
                # chainid = line[21]        # Not used
                #seq = line[22:26].split()
                #if len(seq) == 0:
                #    resseq = 1
                #else:
                #    resseq = int(seq[0])  # sequence identifier
                # icode = line[26]          # insertion code, not used
  
                # atomic coordinates
                try:
                    coord = np.array([float(line[30:38]),
                                      float(line[38:46]),
                                      float(line[46:54])], dtype=np.float64)
                except ValueError:
                    raise ValueError("Invalid or missing coordinate(s)")
  
                # occupancy & B factor
                try:
                    occupancy = float(line[54:60])
                except ValueError:
                    occupancy = None  # Rather than arbitrary zero or one
  
                if occupancy is not None and occupancy < 0:
                    warnings.warn("Negative occupancy in one or more atoms")
  
                try:
                    bfactor = float(line[60:66])
                except ValueError:
                    # The PDB use a default of zero if the data is missing
                    bfactor = 0.0
  
                # segid = line[72:76] # not used
                symbol = line[76:78].strip().capitalize()
                if symbol == '': 
                    symbol = re.findall(r"[A-Z][a-z]?", name)[0]
                elements.append(symbol)
                positions.append(coord)
  
            elif row[0] == 'CRYST1':

                aa = float(line[6:15])   # a
                bb = float(line[15:24])  # b
                cc = float(line[24:33])  # c
                alpha = float(line[33:40])*np.pi/180  # alpha
                beta = float(line[40:47])*np.pi/180  # beta
                gamma = float(line[47:54])*np.pi/180  # gamma

                # lattice vector
                lattice=lattice_vector(aa, bb, cc, alpha, beta, gamma)

            elif row[0] == 'END':
                break

            elif row[0] == "TITLE":
                title = line[6:].strip()

        #species = np.array([v for v in atoms])
        #numbers = np.array([len(v) for v in atoms.values()])
        #positions=[]
        #for e,v in atoms.items():
        #    positions.extend(v)
        
        input.close()
        # conversion format
        poscar = {'comment': title, 
                  'elements': elements,
                  'positions': np.array(positions),
                  'type': 'Cartesian'
        }
        if lattice != None:
            poscar['mol_lattice'] = lattice
        
        return poscar

    def __readXYZ(self):
        """
        read xyz file
            
        poscar:
            elements=['Ca', 'Fe', 'Sb']
            numbers=[2, 8, 24]
            positions=[[a1_x,a1_y,a1_z],
                      [a2_x,a2_y,a2_z],
                      [a3_x,a3_y,a3_z],
                      ...]
        Note: coordinate type of positions can only be Cartesian.
        
        returns:
            object of a structure
        """
        from collections import defaultdict
        input=open(self.file)
        
        # natoms
        try:
            natoms=int(input.readline())
        except ValueError:
            return ValueError('invalid natoms in xyz file!')
        # comment
        comment=input.readline().strip()
        pat = re.compile(r'''
            (?P<e>[A-Z][a-z]?)\s+
            (?P<x>-?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+
            (?P<y>-?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+
            (?P<z>-?\d*\.?\d+(?:[eE][+-]?\d+)?)
        ''', re.X)
        
        # atoms
        elements = []
        positions = []
        for line in input:
            m = pat.search(line)
            try:
                if m:
                    specie = m.group('e')
                    coord = np.array([m.group('x'), m.group('y'), m.group('z')], dtype=float)
                    elements.append(specie)
                    positions.append(coord)
                else:
                    print(line)
            except:
                print(line)
                print(result)
                raise

        if len(positions) != natoms:
            raise ReadError(f"number of atoms doesn't match! Declaration {natoms} but find {len(positions)}.")
        
        input.close()
        # conversion format
        poscar = {'comment': comment, 
                  'elements': elements,
                  'positions': np.array(positions),
        }
        
        return poscar

    def __readExtXYZ(self):
        """
        read ase xyz file
        Lattice="8.355470657348633 0.0 -1.9041740894317627 -3.0704608474743504 7.758936610760726 -1.9074528217315674 0.0 0.0 11.653740882873535" Properties=species:S:1:pos:R:3 pbc="T T T"
            
        poscar:
            elements=['Ca', 'Fe', 'Sb']
            numbers=[2, 8, 24]
            positions=[[a1_x,a1_y,a1_z],
                      [a2_x,a2_y,a2_z],
                      [a3_x,a3_y,a3_z],
                      ...]
        Note: coordinate type of positions can only be Cartesian.
        
        returns:
            object of a structure
        """
        from collections import defaultdict
        input=open(self.file)

        if self.multi and self.pointer != None:
            input.seek(self.pointer,0)
        
        # natoms
        try:
            natoms=int(input.readline())
        except ValueError:
            return ValueError('invalid natoms in xyz file!')
        # comment
        comment=input.readline().strip()
        #Lattice="8.355470657348633 0.0 -1.9041740894317627 -3.0704608474743504 7.758936610760726 -1.9074528217315674 0.0 0.0 11.653740882873535" Properties=species:S:1:pos:R:3 pbc="T T T"
        try:
            lattice = re.findall(r'Lattice=\"([-e\d\.\s]+)\"', comment)[0].split()
        except:
            print(comment)
        lattice = np.array(lattice, dtype='float').reshape(3,3)
        pbc = re.findall(r'pbc=\"([TF\s]+)\"', comment)[0].split()
        
        # atoms
        atoms=defaultdict(list)
        for i in range(natoms):
            line = input.readline()
            result = re.findall(r'\s*([A-Z][a-z]?)\s*(-?\d*\.\d+)\s*(-?\d*\.\d+)\s*(-?\d*\.\d+)', line)
            if len(result):
                specie = result[0][0] # atomic name
                coord = np.array(result[0][1:], dtype=float) # atomic position
                atoms[specie].append(coord)

        # next natoms
        if self.multi:
            pointer = input.tell()
            for line in input:
                result = line.split()
                if len(result) == 1 and re.match(r'^\d+$', result[0]):
                    break
                pointer = input.tell()
            else:
                pointer = None
            self.pointer = pointer

        elements = np.array([v for v in atoms])
        numbers = np.array([len(v) for v in atoms.values()])
        positions=[]
        for e,v in atoms.items():
            positions.extend(v)
        if len(positions) != natoms:
            raise ReadError("number of atoms doesn't match!")
        
        input.close()
        # conversion format
        molecule={'comment': comment, 
                  'lattice': lattice,
                  'elements': elements,
                  'numbers': numbers,
                  'positions': np.array(positions),
                  'type': 'Cartesian'
        }
        
        return molecule

    def __readSDF(self):
        """
        tmp from ase
        """
        with open(self.file, 'r') as f:
            lines = f.readlines()
        # first three lines header
        natoms = int(lines[3].split())
        positions = []
        symbols = []
        for line in lines[4:4+natoms]:
            x, y, z, symbol = line.split()[:4]
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        elements, indices, numbers = np.unique(symbols, return_inverse=True, return_counts=True)
        mask = np.argsort(indices)
        positions = positions[mask]
        raise ValueError("SDF == MOL?")
                
    def __readMOL(self):
        """
        tmp from ase
        """
        with open(self.file, 'r') as f:
            lines = f.readlines()
        comment = lines[2].strip()
        natoms = int(lines[3][:3].strip())
        nbonds = int(lines[3][3:6].strip())
        coords = []
        symbols = []
        for line in lines[4:4 + natoms]:
            x, y, z, symbol = line.split()[:4]
            coords.append([x,y,z])
            symbols.append(symbol)
        coords = np.array(coords, dtype=float)
        elements, indices, numbers = np.unique(symbols, return_inverse=True, return_counts=True)
        mask = np.argsort(indices)
        positions = coords[mask]
        # connections
        connects = []
        for line in lines[4+natoms:]:
            if 'CHG' in line: break
            if 'END' in line: break
            if 'M' in line: break
            idx1 = line[:3].strip()
            idx2 = line[3:6].strip()
            num = line[6:9].strip()
            idx1 = mask[int(idx1)-1]
            idx2 = mask[int(idx2)-1]
            connects.append([idx1,idx2,float(num)])
        
        molecule={'comment': comment,
                  'elements': elements,
                  'numbers': numbers,
                  'positions': positions,
                  'connections': connects
        }
        return molecule

# coding: utf-8
# Copyright (c) JAMIP Development Team.
# Distributed under the terms of the JLU License.


#=================================================================
# This file is part of JAMIP.
#
# Copyright (C) 2021 Jilin University
#
#  JAMIP is a platform for high throughput calculation. It aims to 
#  make simple to organize and run large numbers of tasks on the 
#  superclusters and post-process the calculated results.
#  
#  JAMIP is a useful packages integrated the interfaces for ab initio 
#  programs, such as, VASP, Guassian, QE, Abinit and 
#  comprehensive workflows for automatically calculating by using 
#  simple parameters. Lots of methods to organize the structures 
#  for high throughput calculation are provided, such as alloy,
#  heterostructures, etc.The large number of data are appended in
#  the MySQL databases for further analysis by using machine 
#  learning.
#
#  JAMIP is free software. You can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published 
#  by the Free sofware Foundation, either version 3 of the License,
#  or (at your option) and later version.
# 
#  You should have recieved a copy of the GNU General Pulbic Lincense
#  along with JAMIP. If not, see <https://www.gnu.org/licenses/>.
#=================================================================

"""
Module to read cif file and return a dictionary of POSCAR.

    poscar={'lattice':lattice,
            'elements':elements,
            'numbers':numbers,
            'type':type,
            'positions':positions,
           }

"""

class SpacegroupError(Exception):
    """Base exception for the spacegroup module."""
    pass

class SpacegroupNotFoundError(SpacegroupError):
    """Raised when given space group cannot be found in data base."""
    pass

class SpacegroupValueError(SpacegroupError):
    """Raised when arguments have invalid value."""
    pass

def convert_value(value):
    """
    Convert CIF value string to corresponding python type.

    Arguments:
        value: A number string which needs to be translated to float value.

    Returns:
        value: Object of a float value.

    """
    import warnings
    value=value.strip()
    if re.match('(".*")|(\'.*\')$', value):
        return value[1:-1]
    elif re.match(r'[+-]?\d+$', value):
        return int(value)
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$', value):
        return float(value)
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\(\d+\)$',
                  value):
        return float(value[:value.index('(')])  # strip off uncertainties
    elif re.match(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\(\d+$',
                  value):
        warnings.warn('Badly formed number: "{0}"'.format(value))
        return float(value[:value.index('(')])  # strip off uncertainties
    else:
        return value


def parse_multiline_string(lines, line):
    """
    Parse semicolon-enclosed multiline string and return it.

    """
    assert line[0] == ';'
    strings = [line[1:].lstrip()]
    while True:
        line = lines.pop().strip()
        if line[:1] == ';':
            break
        strings.append(line)
    return '\n'.join(strings).strip()


def parse_singletag(lines, line):
    """
    Parse a CIF tag(entries starting with underscore). Returns
    a key-value pair.

    Arguments:
        lines: All lines.
        line: A single line starts with '_'.

    Return:
        key: The single tag(entries starting with underscore) as key.
        convert_value(value): The single value corresponded to the tag.

    Examples:
        The string '_symmetry_Int_Tables_number       62' will
        be translated to a key-value pair: {'_symmetry_Int_Tables_number': 62}.

    """
    kv = line.split(None, 1)
    if len(kv) == 1:
        key = line
        line = lines.pop().strip()
        while not line or line[0] == '#':
            line = lines.pop().strip()
        if line[0] == ';':
            value=parse_multiline_string(lines, line)
        else:
            value=line
    else:
        key, value=kv
    return key, convert_value(value)


def parse_loop(lines):
    """
    Parse a CIF loop. Returns a dict with column tag names as keys
    and a lists of the column content as values.

    Arguments:
        lines: The all lines in cif file.

    Return:
        column: A column based dictionary about the tags and
        corresponding values in a loop.

    """
    import shlex
    import warnings
    header = []
    line = lines.pop().strip()
    while line.startswith('_'):
        header.append(line.lower())
        line = lines.pop().strip()
    columns = dict([(h, []) for h in header])

    tokens = []
    while True:
        lowerline = line.lower()
        if (not line or
            line.startswith('_') or
            lowerline.startswith('data_') or
            lowerline.startswith('loop_')):
            break
        if line.startswith('#'):
            line = lines.pop().strip()
            continue
        if line.startswith(';'):
            t = [parse_multiline_string(lines, line)]
        else:
            if len(header) == 1:
                t = [line]
            elif line.count("'") > 2 and "' '" not in line:
                start = line.index("'")
                end = line.rindex("'")+1
                t = line[:start].split() + [line[start:end]] + line[end:].split()
            else:
                t = shlex.split(line, posix=False)

        line = lines.pop().strip()

        tokens.extend(t)
        if len(tokens) < len(columns):
            continue
        if len(tokens) == len(header):
            for h, t in zip(header, tokens):
                columns[h].append(convert_value(t))
        else:
            raise
            warnings.warn('Wrong number of tokens: {0}'.format(tokens))
        tokens = []
    if line:
        lines.append(line)
    return columns


def parse_items(lines, line):
    """
    Parse a CIF data items and return a dict with all tags.

    Arguments:
        lines: The all lines in cif file.
        line: A single line which will be translated to a key-value pair
        or just be a single tag.

    Return:
        tags: The all key-value pairs obtained from parse_singletag
        and parse_loop.

    """
    tags = {}
    while True:
        if not lines:
            break
        line = lines.pop()
        if not line:
            break
        line = line.strip()
        lowerline = line.lower()
        if not line or line.startswith('#'):
            continue
        elif line.startswith('_'):
            key, value = parse_singletag(lines, line)
            tags[key.lower()] = value
        elif lowerline.startswith('loop_'):
            tags.update(parse_loop(lines))
        elif lowerline.startswith('data_'):
            if line:
                lines.append(line)
            break
        elif line.startswith(';'):
            parse_multiline_string(lines, line)
        else:
            raise ValueError('Unexpected CIF file entry: "{0}"'.format(line))
    return tags


def parse_block(lines, line):
    """
    Parse a CIF data block and return a tuple with the block name
    and a dict with all tags.

    Arguments:
        lines: The all lines in cif file.
        line: A single line which will be a single tag.

    Return:
        blockname: The name of a block which starts with 'data_'.
        tags: The all tags.

    """
    assert line.lower().startswith('data_')
    blockname = line.split('_', 1)[1].rstrip()
    tags = parse_items(lines, line)
    return blockname, tags

def parse_cif(fileobj):
    """
    Parse a CIF file. Returns a list of blockname and tag pairs.
    All tag names are converted to lower case.

    Arguments:
        fileobj: The cif file name.

    Return:
        blocks:The all blocks obtained from parse_block. (The number
        of the blocks is usually 2)

    """
    import pathlib
    if isinstance(fileobj, (str, pathlib.PosixPath)):
        fileobj = open(fileobj)
    lines = [''] + fileobj.readlines()[::-1]  # all lines (reversed)
    blocks = []
    while True:
        if not lines:
            break
        line = lines.pop()
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        blocks.append(parse_block(lines, line))
    return blocks

def format_symbol(symbol):
    """
    Returns well formatted Hermann-Mauguin symbol as extected by
    the database, by correcting the case and adding missing or
    removing duplicated spaces.

    """
    fixed = []
    s = symbol.strip()
    s = s[0].upper() + s[1:].lower()
    for c in s:
        if c.isalpha():
            if len(fixed) and fixed[-1] == '/':
                fixed.append(c)
            else:
                fixed.append(' ' + c + ' ')
        elif c.isspace():
            fixed.append(' ')
        elif c.isdigit():
            fixed.append(c)
        elif c == '-':
            fixed.append(' ' + c)
        elif c == '/':
            fixed.append(c)
    s = ''.join(fixed).strip()
    return ' '.join(s.split())

def equival_pos(equival, ci, prec=1e-3):
    """
    Translate the initial position coordinates to
    symmetry equivalent position coordinates.

    Arguments:
        equival: The tag (no, symbolHM or  sitesym)corresponding to
        symmetry operations.
        ci: The cif file resource.
        sum_pos: The total number of atoms in crystal. 

    Return:
        symXYZ: A list contains the equivalent position coordinates.

    """

    allXYZ = []
    for i, (X, Y, Z) in enumerate(zip(ci['_atom_site_fract_x'],
                                      ci['_atom_site_fract_y'],
                                      ci['_atom_site_fract_z'])):
        atommap = {'x':tofloat(X),'y':tofloat(Y),'z':tofloat(Z)}

        #i0 = len(allXYZ)
        for operation in equival:
            temp = operation.split(',')
            atomL = []
            for strs in temp:
                value = eval(strs,atommap)
                fraction = re.findall(r'\d+\.?\d?/\d+\.?\d?',strs)
                if len(fraction) > 0:
                    denominator = float(fraction[0].split('/')[1])
                    norm = np.around(value*denominator)
                    if np.abs(norm-value*denominator) < prec:
                        value = norm/denominator
                elif np.abs(np.around(value)-value) < prec:
                    value = np.around(value)
                atomL.append(value)
            atomL = np.array(atomL) - np.floor(atomL)
            allXYZ.append(atomL)

        #i1 = len(allXYZ)
        #if '_atom_site_symmetry_multiplicity' in ci:
        #    num = len(unique_sites(allXYZ[i0:i1], prec))
        #    print(ci['_atom_site_symmetry_multiplicity'][i], i1-i0, num)

    return unique_sites(allXYZ, prec)

def unique_sites(sites, tol=1e-3):

    uniqs = []
    for site in sites:
        if len(uniqs) == 0:
            uniqs = site[None, :]
        elif np.min(np.abs(uniqs-site).sum(axis=1)) > 3*tol:
            uniqs = np.vstack((uniqs,site))

    return uniqs

def numbers_cal(equival, ci):
    """
    Calculate the number of atoms of each type.

    Arguments:
        equival: The tag (no, symbolHM or  sitesym)corresponding to
        symmetry operations.
        ci: The cif file resource.

    Return:
        numbers: The number of atoms of each type.

    """
    numXYZ = []
    atomN = []
    numbers = []

    for X, Y, Z in zip(ci['_atom_site_fract_x'],
                       ci['_atom_site_fract_y'],
                       ci['_atom_site_fract_z']):

        atommap = {'x':tofloat(X),'y':tofloat(Y),'z':tofloat(Z)}

        for operation in equival:
            temp = operation.split(',')

            XX = temp[0].replace('1/2', '1./2.').replace('1/4', '1./4.')
            XX = XX.replace('3/4', '3./4.').replace('1/6', '1./6.')
            XX = XX.replace('1/3', '1./3.').replace('2/3', '2./3.')
            XXX = XX.replace('5/6', '5./6.')

            YY = temp[1].replace('1/2', '1./2.').replace('1/4', '1./4.')
            YY = YY.replace('3/4', '3./4.').replace('1/6', '1./6.')
            YY = YY.replace('1/3', '1./3.').replace('2/3', '2./3.')
            YYY = YY.replace('5/6', '5./6.')

            ZZ = temp[2].replace('1/2', '1./2.').replace('1/4', '1./4.')
            ZZ = ZZ.replace('3/4', '3./4.').replace('1/6', '1./6.')
            ZZ = ZZ.replace('1/3', '1./3.').replace('2/3', '2./3.')
            ZZZ = ZZ.replace('5/6', '5./6.')

            XXXX = eval(XXX, atommap)
            YYYY = eval(YYY, atommap)
            ZZZZ = eval(ZZZ, atommap)

            if XXXX < 0:
                XXXX = 1.0 + XXXX

            if YYYY < 0:
                YYYY = 1.0 + YYYY

            if ZZZZ < 0:
                ZZZZ = 1.0 + ZZZZ

            if XXXX >= 1.0:
                XXXX = XXXX - 1.0

            if YYYY >= 1.0:
                YYYY = YYYY - 1.0

            if ZZZZ >= 1.0:
                ZZZZ = ZZZZ - 1.0

            atomL = [XXXX, YYYY, ZZZZ]
            numXYZ.append(atomL)

        # calculate the number of atoms of each type.
        for i in numXYZ:
            if not i in atomN:
                atomN.append(i)
        temp = len(atomN)
        numbers.append(temp)
        numXYZ = []
        atomN = []

    return numbers

def lattice_vector(a, b, c, alpha, beta, gamma):
    """
    Translate lattice parameters to lattice vector.

    Arguments:
        a: The module of lattice parameter a.
        b: The module of lattice parameter b.
        c: The module of lattice parameter c.
        alpha: The included angle between vector b and c.
        beta: The included angle between vector a and c.
        gamma: The included angle between vector a and b.

    Return:
        latticeV: A list about lattice vector expressed by direct coordinate.

    """
    ax = a
    ay = 0.
    az = 0.
    bx = b * np.cos(gamma)
    by = b * np.sin(gamma)
    bz = 0.
    cx = c * np.cos(beta)
    cy = c * ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))
    cz = c * (np.power(1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
                        - np.power(np.cos(alpha), 2) - np.power(np.cos(beta), 2)
                        - np.power(np.cos(gamma), 2), 0.5) / np.sin(gamma))
    latticeV = [[ax, ay, az],
                [bx, by, bz],
                [cx, cy, cz]]

    return latticeV

def parse_qe_xml(xmlfile):
    ''' 
        input:
            qe: A xml including:
            atomic_postions: atom name, [x1,x2,x3], ...
            type= Direct
            cell: [[x1,y1,z1],
                   [x2,y2,y3],
                   [x3,y3,z3]] # unit: bohr
            1 bohr = 0.529177 ang 
    ''' 
    import xml.etree.cElementTree as ET

    for event, elem in ET.iterparse(xmlfile,events=('start', 'end')):
        if event =='start':
            if elem.tag == 'output':
                atoms = elem.findall('./atomic_structure/atomic_positions/atom')
                posdict = {}
                for atom in atoms:
                    element = atom.get('name')
                    if element not in posdict:
                        posdict[element] = []                
                    posdict[element].append(atom.text.split())
                cell = elem.find('./atomic_structure/cell')
                lattice = []
                for child in cell:
                    lattice.append(np.array(child.text.split(),dtype=float))
                break
        elem.clear()

    # returns %
    elements = []
    numbers = []
    positions = []
    for elm,position in posdict.items():
        elements.append(elm)
        numbers.append(len(position))
        positions.extend(position)
    lattice = np.asarray(lattice)*0.529177
    positions = np.array(positions,dtype=float)*0.529177

    espessro = {'comment': 'jamip',
                'lattice':lattice,
                'elements':elements,
                'numbers':numbers,
                'type': 'Cartesian',
                'positions':positions}
    return espessro

def parse_in(fileobj):
    """
    Parse a QE input file. Returns a list of blockname and tag pairs.
    All tag names are converted to lower case.

    Arguments:
        fileobj: The qe input file name.

    Return:
       dict 

    """
    if isinstance(fileobj, str):
        fileobj = open(fileobj)

    pairs = []
    for line in fileobj:
        pairs.extend(re.findall(r'(\S+)\s*=\s*(\d+\.?\d*)',line))

    return dict(pairs)

def tofloat(value):
    if isinstance(value, str):
        data = list(pattern_float.match(value).groups())
        if data[1] == '':
            data[1] = '0'
        return float(''.join(data))
    else:
        return value

def get_sitesym(cb):
    from jamip.db.iostream.spaceGroupD3 import spacegroups as SG
    
    # symmetry operations
    for name in ['_space_group_symop_operation_xyz',
                 '_space_group_symop.operation_xyz',
                 '_symmetry_equiv_pos_as_xyz']:
        if name in cb:
            return cb[name]

    # space group H-M symbol
    for name in ['_space_group.Patterson_name_h-m',
                 '_symmetry_space_group_name_h-m',
                 '_space_group_name_h-m_alt']:
        if name in cb:
            symbolHM=format_symbol(cb[name])
            sitesym = SG.get(symbolHM)
            #print(symbolHM, name, len(sitesym))
            if sitesym:
                return sitesym
            else:
                raise SpacegroupNotFoundError(f'invalid spacegroup {symbolHM}, not found in data base')
    
    # space group number
    for name in ['_space_group.it_number', 
                 '_space_group_it_number', 
                 '_symmetry_int_tables_number']:
        if name in cb:
            group_number = str(cb[name])
            sitesym = SG.get(group_number)
            #print(group_number, name, len(sitesym))
            if sitesym:
                return sitesym
            else:
                raise SpacegroupNotFoundError(f'invalid spacegroup {symbolHM}, not found in data base')
