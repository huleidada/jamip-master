# -*- coding: utf-8 -*-
import __future__ 
from .atom import Atom, Cell, Composition
import numpy as np

class Structure:
    """
    define a crystal structure, including,
    
    Attributes:
        species_of_elements: list, sepcies of elements;
        number_of_atoms:     list, numbers of species;
	    atomic_positions:    list, atomic position defined by Atom Object;
	    comment_line:        string, introduction of structure;
        scale_factor:        float, default, 1.0;
        lattice:             numpy.array(3x3), default, None;   	 
	    select_dynamic:      bool, default, False;  
        initial_velocity:    bool, default, False;
        format:              string, direct or cartesian
        direct:              bool, default, True;
    """

    magnetic = None 

    def __init__(self, *args, **kwargs):
        self._cell = None
        self.__positions = None
        self.__format = 'direct'
        self._comment = None
        self.__scale = 1.0
        self.__number = None
        self.__species = None
        self._frozen = False
        self._velocities = False
	

    @property
    def lattice(self):
        """
        Return:
            lattice parameters [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]].
        """
        return self._cell.vectors

    @lattice.setter
    def lattice(self, value):
        self._cell = Cell(value)

    @property
    def lattice_parameters(self):
        """
        Return:
            lattice parameters [a, b, c, alpha, beta, gamma].
        """
        return self._cell.parameters

    @property
    def direct(self):
        """Set the output format for atomic coordinates to 'direct' (fractional) or 'cartesian'.
        
        Returns:
            bool: Returns True if the atomic coordinates are in direct (fractional) format, False otherwise.
        """
        return True if self.__format == 'direct' else False
     
    @property
    def atomic_coord_format(self):
        return self.__format
     
    @atomic_coord_format.setter 
    def atomic_coord_format(self, value:str):
        if value.lower() in ('direct','cartesian'):
            self.__format = value.lower()
        else:
            raise ValueError(f'Unknown atomic coord format: {value}')

    @property 
    def atomic_positions(self):
        """Get the atomic positions in the structure.

        Returns:
            list: List of atomic positions.
        """
        return self.__positions

    def set_positions(self, elements, value=None):
        if len(elements) != len(value):
            raise ValueError(f'''
            Atoms NUMBER NOT CONSISTENT:
                  NATOMS -> {len(elements)}
                 NCOORDS -> {len(value)}
            ''')

        atoms =  []
        for i, coord in enumerate(value):
            atom = Atom(elements[i], coord, direct=self.direct, cell=self._cell)
            atoms.append(atom)

        self.__positions = atoms

    @property
    def composition(self):
        """
        Returns:
            int: The formula units Z of the structure.
        """
        return Composition.from_elements(self.get_elements(type='symbol'))
    
    def get_formula(self, reduced=False, split='', sort=False):
        """ Get the formula of the structure.
        
        Args:
            reduced (bool): Whether to divide the number of atoms by the formula units Z. Defaults to False.
            split (str): The split character. Defaults to ''.
            sort (bool): Whether to sort the positions by element. Defaults to True.

        Returns:
            str: The formula of the structure.
        """
        return self.composition.get_formula(reduced=reduced, split=split, sort=sort)

    def get_positions(self, type='direct'):
        """
        Retrieve the positions of the atoms in the structure.

        Args:
            type (str): The coordinate system for the positions. Defaults to 'direct'.
                - 'direct': Returns the positions in fractional coordinates.
                - 'cartesian': Returns the positions in Cartesian coordinates.

        Returns:
            np.ndarray: The positions of the atoms in the specified coordinate system.
        """
        if type == 'direct':
            return np.array([atom.scale_coord for atom in self.__positions])
        elif type == 'cartesian':
            return np.array([atom.coord for atom in self.__positions])
        else:
            raise TypeError("Unknown position type %s!" % type)

    def to_dict(self, json=False):
        """ Convert the structure to a dictionary.    
        
        Args:
            json (bool): Convert the data into a format supported by JSON.
        """
        if json:
            value = {'lattice': self.lattice.tolist(),
                     'positions': self.get_positions(type='direct').tolist(),
                     'elements':self.get_elements(type='symbol').tolist()}
            return value #json.dump(value)
        else:
            value = {'lattice': self.lattice,
                     'positions': self.get_positions(type='direct'),
                     'elements':self.get_elements(type='number')}
            return value

    def to_cell(self):
        """ Convert the structure to a cell.    """
        return (self.lattice, self.get_positions(type='direct'), self.get_elements(type='number'))

    def get_all_distances(self, pbc=True, diag=True):
        """  Calculate the distances between all atoms in the structure.

        Args:
            pbc (bool): Whether to consider periodic boundary conditions. Defaults to True.
            diag (bool): Whether to fill the diagonal of the distance matrix with the minimum distance. Defaults to True.

        Returns:
            np.ndarray: The distance matrix.
        """
        tot = sum(self.__number)
        distances = np.zeros((tot,tot))
        positions = self.get_positions(type='direct')
        vectors = positions[None,::] - positions[:,None,:] 
        if pbc: 
            vectors = vectors - np.floor(vectors)
            grid = np.mgrid[-1:1, -1:1, -1:1].reshape(3,-1).T
            vectors = vectors[None,:,:] + grid[:,None,None]
        else:
            vectors = vectors[None,:,:]
        distances = np.linalg.norm(np.dot(vectors, self._cell.vectors), axis=-1)
        distances = np.min(distances, axis=0)
        if diag: 
            mingvec = min(np.linalg.norm(self._cell.vectors, axis=1))
            np.fill_diagonal(distances, mingvec)
        return distances
     
    def __len__(self):
        return len(self.__positions)
	
    @property
    def comment_line(self):
        """ 
        Returns:
            Structure Title string
        """
        return self._comment 

    @comment_line.setter
    def comment_line(self, value=None):
        self._comment = value

    @property
    def scale_factor(self):
         """
         The simple setting parameters for VASP structure files, do not apply to lattice.

         Returns:
             float: lattice scale 
         """
         return self.__scale 

    @scale_factor.setter
    def scale_factor(self, value=1.0):
        self.__scale = value

    @property
    def species_of_elements(self):
        """
        List of element types. The type of element that allows repetition.

        Returns:
            list: list of atom species. 
        """
        return self.__species
    
    @species_of_elements.setter
    def species_of_elements(self, value):
        self.__species = value

    @property
    def number_of_atoms(self):
        """List of the number of atoms corresponding to speices.

        Returns:
            list: list of atom numbers.
        """
        return self.__number

    @number_of_atoms.setter
    def number_of_atoms(self, value):
        self.__number = value

    @property 
    def select_dynamic(self):
        """Switch to activate/deactivate the fixed atom positions functionality.

        Returns:
            bool: Returns True if the fixed atom positions functionality is active, False otherwise.
        """
        return False if self._frozen is False else True

    @select_dynamic.setter
    def select_dynamic(self, value=False):
        if self.__positions == None:
            raise ValueError("The position property is not set.")
        
        if isinstance(value, bool):
            self._frozen = value
        elif len(self.__positions) == len(value):
            for i,atom in enumerate(self.__positions):
                atom.freeze = value[i]
            self._frozen = True
        else:
            raise ValueError("Invalid select_dynamic setting.")

    @property
    def initial_velocity(self):
        """Switch to activate/deactivate the atom velocity functionality.

        Returns:
            bool: Returns True if the fixed atom positions functionality is active, False otherwise.
        """
        return False if self._velocities is False else True

    @initial_velocity.setter
    def initial_velocity(self, value=False):
        if self.__positions == None:
            raise ValueError("The position property is not set.")

        if isinstance(value, bool):
            self._velocities = value
        elif len(self.__positions) == len(value):
            for i,atom in enumerate(self.__positions):
                atom.velocity = value[i]
            self._velocities = True
        else:
            raise ValueError("Invalid initial_velocity setting.")

    @atomic_positions.setter
    def atomic_positions(self, value=None):
        elements = np.repeat(self.__species, self.__number)
        self.set_positions(elements, value)
        self._velocities = False

    def add_atom(self, atom):
        """Add an atom to the structure.

        Args:
            atom (Atom): Atom object to be added.
        """
        assert isinstance(atom, Atom) == True, "add_atom need jamip.structure.atom.Atom"
        self.__positions.append(atom)

    def get_atom(self, position, specie=None, tol=1e-3, is_exist=True):
        """Get the index of the atom closest to the given position.

        Args:
            position (list): Position of the atom to be found.
            specie (str, optional): Element symbol of the atom to be found. Defaults to None.
            tol (float, optional): Tolerance for finding the atom. Defaults to 1e-3.
            is_exist (bool, optional): Whether to raise an error if the atom is not found. Defaults to True.

        Returns:
            list: List of indices of the atoms closest to the given position.
        """
        positions = self.get_positions(type='direct')
        vectors = positions - position
        vectors = vectors - np.round(vectors)
        distances = np.linalg.norm(vectors@self.lattice, axis=1)
        assert len(distances) == len(positions) #! check
        if specie is None:
            indices = np.where(distances < tol)[0]
        else:
            elements = self.get_elements(type='symbol', safe=True)
            indices = np.where((distances < tol) & (elements == specie))[0]

        if len(indices) == 0:
            if is_exist: 
                raise RuntimeError("not find atom.")
        return indices

    def del_atom(self, position, specie=None, **kwargs):
        """ Delete an atom from the structure.

        Args:
            position (list): Position of the atom to be deleted.
            specie (str, optional): Element symbol of the atom to be deleted. Defaults to None.
            **kwargs: Additional arguments for get_atom method.
        """
        indices = self.get_atom(position, specie, **kwargs)
        if len(indices):
            atoms = []
            for i,atom in enumerate(self.__positions):
                if i not in indices:
                    atoms.append(atom)
            self.__positions = atoms

    def substitute_atom(self, position, specie, **kwargs): 
        """ Substitute an atom in the structure.

        Args:
            position (list): Position of the atom to be substituted.
            specie (str): Element symbol of the new atom.
            **kwargs: Additional arguments for get_atom method.
        """
        indices = self.get_atom(position, specie=None, **kwargs)
        if len(indices):
            for i,atom in enumerate(self.__positions):
                if i in indices:
                    atom.specie = specie

    def to_row(self):
        """ Convert the structure to a dictionary in POSCAR format.    """
        poscar={'comment': self.comment_line,
                'lattice': self.lattice,
                'elements':self.species_of_elements,
                'numbers': self.number_of_atoms,
                'type': self.atomic_coord_format,
                'positions': self.get_positions(type=self.__format)}
        return poscar

    @classmethod
    def from_cell(cls, cell, comment='jamip', direct=True, sort=False):
        """ Create a structure from a cell.

        Args:
            cell (tuple): A tuple containing the lattice, positions, and elements.
            comment (str): Comment line for the structure. Defaults to 'jamip'.
            direct (bool): Whether the positions are in direct or cartesian coordinates. Defaults to True.
            sort (bool): Whether to sort the positions by element. Defaults to True.

        Returns:
            Structure: A structure object.
        """
        assert len(cell) == 3
        lattice, positions, elements = cell
        obj = cls()
        obj.comment_line = comment
        obj.lattice = lattice
        obj.atomic_coord_format = 'direct' if direct else 'cartesian'
        species,numbers,positions = cls.elements2species(elements, positions, sort=sort)
        obj.species_of_elements = species
        obj.number_of_atoms = numbers
        obj.atomic_positions = positions
        return obj

    @classmethod
    def elements2species(cls, elements, positions=None, sort=False):
        """ Convert elements list to species groups & sort positions by elements

        Args:
            elements (list/array): List of atomic numbers (int) or element symbols (str)
            positions (list/array): List of atomic positions corresponding to elements
            sort (bool, optional): Whether to sort elements by atomic number. 
                                If False, preserves original order of first occurrence.
                                Defaults to True.

        Returns:
            tuple: Contains three arrays:
                - unique_elements: Array of unique element symbols
                - counts: Counts of each element type
                - grouped_positions: List of position arrays grouped by element type
        """      
        from .atomic_number import atomic

        if positions is None:
            active = None
            species = []
            numbers = []
            for i in elements:
                if i != active:
                    active = i
                    species.append(i)
                    numbers.append(1)
                else:
                    numbers[-1] += 1
            return np.array(species), np.array(numbers)

        else:
            # Process unique elements
            values, ids, indices, counts = np.unique(elements, return_index=True, 
                                                     return_inverse=True, return_counts=True)
            unique_elements = np.array([atomic[v] if isinstance(v, (int, np.integer)) else v 
                                        for v in values])
            
            # Handle sorting if disabled
            if not sort:
                sort_order = np.argsort(ids)
                unique_elements = unique_elements[sort_order]
                counts = counts[sort_order]
                indices = np.argsort(sort_order)[indices]
         
            # Group positions by element
            positions = np.array(positions)
            grouped_positions = [positions[indices == i] for i in range(len(unique_elements))]
            grouped_positions = np.concatenate(grouped_positions, axis=0)
            
            return unique_elements, counts, grouped_positions

    @property
    def composition(self):
        """
        Returns:
            int: The formula units Z of the structure.
        """
        return Composition(self.__species, self.__number)
    
    def get_species(self): 
        """  Return the valence state of elements in the structure.

        Returns:
            dict: The valence state of elements.
        """
        return self.composition.best_valence

    def get_elements(self, type='number', safe=False):
        """ 
        Args:
            type (str): The type of the element. Defaults to 'number'.
            safe (bool): Whether to return the element in the safe way. Defaults to False.

        Returns:
            np.ndarray: The elements of the structure.
        """
        from .atomic_number import number
        if not safe:
            if type == 'symbol':
                species = self.__species
            elif type == 'number':
                species = [number[i] for i in self.__species]
            return np.repeat(species, self.__number)
        else:
            if type == 'symbol':
                return np.array([atom.specie for atom in self.__positions])
            elif type == 'number':
                return np.array([number[atom.specie] for atom in self.__positions])

    def get_molecule(self, stable=False):
        """Convert to Structure object with optional vacuum padding and centering.
        
        Args:
            vaccum (float): Vacuum padding size (in Angstroms) when no cell exists (default: 20)
            center (bool): Whether to center positions in the cell (default: True)
        
        Returns:
            Structure: New structure object with specified configuration
        
        Note:
            For molecular systems (no cell), creates cubic box with given vacuum size
            Centering places atoms at cell center by adding half of lattice vectors
        """
        from .molecule import Molecule

        obj = Molecule()
        obj.comment_line = self.comment_line
        obj.species_of_elements = self.species_of_elements
        obj.number_of_atoms = self.number_of_atoms 
        if not stable or len(self) == 1:
            obj.atomic_positions = self.get_positions(type="cartesian")
        else:
            from .bonding import Bonding
            # unwrap_molecules_bfs(self):
            bd = Bonding(self, offset=0.3, factor=1, constraint="auto")
            unwrapped_coords = self.get_positions(type="cartesian")
            ref_idx = 0

            # 使用BFS遍历分子
            #visited = set([ref_idx])
            queue = [ref_idx]
            atom_translations = {idx: (np.zeros(3), False) for idx in range(len(self))}
            atom_translations[ref_idx] = (np.zeros(3), True)  # 参考原子无平移，已处理
 
            while queue:
                current = queue.pop(0)
                current_translation, _ = atom_translations[current]
 
                # 找出所有与current成键的原子
                for row in bd.data.get_neighbors(current):
                    neighbor = row[0]
                    neighbor_translation, neighbor_processed = atom_translations[neighbor]
                    bond_translation = row[2]

                    if not neighbor_processed:
                        new_translation = current_translation - bond_translation
                        unwrapped_coords[neighbor] += new_translation @ self.lattice
                        atom_translations[neighbor] = (new_translation, True)
                        queue.append(neighbor)
 
                    #visited.add(neighbor)

            obj.atomic_positions = unwrapped_coords

        return obj


    @property
    def volume(self):
        """
        Returns:
            float: The volume of the structure.
        """
        return self._cell.volume

    @property
    def density(self):
        """
        Returns:
            float: The density of the structure.
        """
        from .elementInfo import ElementDict
        masses = [ElementDict[i]['mass'] for i in self.__species]
        mass = sum(np.repeat(masses, self.__number))
        return mass / self.volume

    @property
    def packing_factor(self):
        """
        Returns:
            float: The packing factor of the structure.
        """
        from .elementInfo import ElementDict
        atom_volumes = [(ElementDict[i]['atomic_radius']/100)**3 for i in self.__species]
        atom_volumes = sum(np.repeat(atom_volumes, self.__number)) * 4/3 * np.pi
        return atom_volumes / self.volume

    def update(self):
        """
        Update the structure by recalculating the atomic positions and species.

        This method updates the structure's atomic positions and species based on the current state of the structure.
        It ensures that the atomic positions are sorted according to the species and updates the internal state of the structure.
        """
        elements = self.get_elements(type='symbol', safe=True)
        positions = self.get_positions(type='direct')
        species, numbers, positions = self.elements2species(elements, positions, sort=False)
        self.species_of_elements = species
        self.number_of_atoms = numbers
        self.atomic_coord_format = 'direct'
        self.atomic_positions = positions
        positions = self.get_positions(type='direct')

    def clear(self):
        """
        Clear the structure by resetting the atomic positions and species.

        This method clears the structure by resetting the atomic positions and species to empty lists.
        It also resets the internal state of the structure.
        """
        self.__positions = []
        self.__species = []
        self.__number = []

    def get_aflow_prototype(self, symprec=1e-3, dataset=None, mini=True, with_chemsys=True):
        # example A2B2C3_hP7_187_h_g_ai:C-F-Hf
        """
        • 为每个 Wyckoff 字母分配一个数字分数：a=1、b=2、等，并取所有 Wyckoff 位置的总和。将仅考虑总和最小的标注。
        • 如果此和仍存在多个 Wyckoff 位置的排列，请选择第一个原子种类具有最小 Wyckoff 字母的那个。
        """
        from jamip.structure.symmetry import SpaceSymmetry, LayerSymmetry
        from jamip.structure.atomic_number import atomic
        import pandas as pd
        import spglib
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        if dataset is None:
            dataset = spglib.get_symmetry_dataset(self.to_cell(), symprec=symprec)
 
        #print(dataset['number'], dataset['wyckoffs'])
        if mini and dataset['number'] > 142:
            wycs = [ord(i) for i in dataset['wyckoffs']]
            data = [(min(wycs), sum(wycs), dataset)]

            #shifts = np.array([[0,0,0],[1/2,0,0],[0,1/2,0],[1/2,1/2,0]])
            if dataset['number'] > 192:
                shifts = np.array([[1/2,1/2,0]])
            elif dataset['number'] > 142:
                shifts =  np.array([[1/3,2/3,0],[2/3,1/3,0]])

            for shift in shifts:
                lattice, positions, elements = self.to_cell()
                positions += shift 
                dataset = spglib.get_symmetry_dataset((lattice, positions, elements), symprec=symprec)
                wycs = [ord(i) for i in dataset['wyckoffs']]
                raw = (min(wycs), sum(wycs), dataset)
                data.append(raw) 

            # get best dataset
            sort_indices = np.lexsort(([i[0] for i in data], [i[1] for i in data]))
            dataset = data[sort_indices[0]][2]

        def wyckoff2formula(wyckoffs):
            species,numbers = np.unique(wyckoffs, return_counts=True)
            formula = ''
            for i,j in zip(species, numbers):
                if j == 1: j = ''
                formula += f'{j}{i}'
            return formula

        spg_num = dataset['number']
        species = [atomic[i] for i in dataset['std_types'][dataset['mapping_to_primitive']]]
        comp = Composition.from_elements(species)

        # pearson_symbol
        num_sites_conventional = len(species)
        if dataset['hall_number'] > 0:
            space = SpaceSymmetry.from_hall_number(dataset['hall_number'])
            pearson_symbol = space.pearson_symbol(num_sites_conventional)
        else:
            layer = LayerSymmetry.from_number(dataset['number'])
            pearson_symbol = layer.pearson_symbol(num_sites_conventional)
 
        sites = pd.DataFrame({'wyckoffs': dataset['wyckoffs'],
                              'equivalent_atoms': dataset['equivalent_atoms'],
                              'species': species,})

        unique_sites = sites.drop_duplicates()
        unique_species = []
        element_wyckoffs = []
        for key,grp in unique_sites.groupby('species'):
            wycs = wyckoff2formula(grp['wyckoffs'].values)
            element_wyckoffs.append(wycs)
            unique_species.append(key)

        indices = np.argsort(element_wyckoffs)
        all_wyckoffs = "_".join([element_wyckoffs[i] for i in indices])
        chemsys = '-'.join([unique_species[i] for i in indices])
 
        if with_chemsys:
            protostructure_label = f"{comp.ABformula}_{pearson_symbol}_{spg_num}_{all_wyckoffs}:{chemsys}"
        else:
            protostructure_label = f"{comp.ABformula}_{pearson_symbol}_{spg_num}_{all_wyckoffs}"
        return protostructure_label
