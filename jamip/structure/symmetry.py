import numpy as np
import pandas as pd
from typing import NamedTuple
from jamip.utils.logger import load_yaml
import spglib
from importlib.resources import files
from functools import cached_property
import threading
from pathlib import Path

class Dataset:

    _cache = {}  # 类变量缓存字典 {csv_path: data}
    _lock = threading.Lock()  # 保证线程安全

    def __init__(self):
        pass

    def _load_data(self, data_class, data_file):
        """线程安全的共享数据加载"""
        
        with self._lock:
            if data_class not in self._cache:
                print(f"首次加载 {data_class} 数据...")
                if Path(data_file).suffix == '.csv':
                    self._cache[data_class] = pd.read_csv(data_file)
                elif Path(data_file).suffix == '.json':
                    self._cache[data_class] = pd.read_json(data_file)
            return self._cache[data_class]        

    @cached_property
    def layerdata(self):
        datafile = files(__package__).joinpath("LayerGroup.csv")
        return self._load_data('layerdata', datafile)

    @property
    def layermaps(self):
        return self.layerdata.set_index('number', drop=False).to_dict('index')

    @cached_property
    def ramandata(self):
        datafile = files(__package__).joinpath('raman_active_modes.csv')
        return self._load_data('ramandata', datafile)
    
    @cached_property
    def arithmetic2d(self):
        datafile = files(__package__).joinpath('arithmetic2d.json')
        return self._load_data('arithmetic2d', datafile)
    
    @cached_property
    def arithmetic3d(self):
        datafile = files(__package__).joinpath('arithmetic.json')
        return self._load_data('arithmetic3d', datafile)
    
    @cached_property
    def symmetry_operation(self):
        datafile = files(__package__).joinpath('pointgroup.yaml')
        Symmetry_operation = load_yaml(datafile)
        for key, value in Symmetry_operation.items():
            Symmetry_operation[key] = np.array(value)
        return Symmetry_operation

    @classmethod
    def clear_cache(cls):
        with cls._lock:
            cls.cache.clear()

def normalized(vectors):
    vectors -= np.floor(vectors)
    vectors = np.around(vectors,8)
    vectors -= np.around(vectors)
    return vectors

def fast_cluster(indices, diff, lattice, tol): 
    vectors = []
    maps = []
    natom = len(indices)
    for i in np.arange(natom):
        for j in np.arange(natom):
            for m,v in enumerate(vectors):
                vec = normalized(diff[i,j,:]-v) @ lattice
                if np.max(np.abs(vec)) < tol:
                    maps.append([indices[i], indices[j], m])
                    break
            else:
                m = len(vectors)
                maps.append([indices[i],indices[j], m])
                vectors.append(normalized(diff[i,j,:]))
    return vectors, maps

def unique_vectors(vectors1, vectors2, lattice, tol):
    allvectors = list(vectors1)
    maps = np.arange(len(vectors2)) 
    for i,v1 in enumerate(vectors2):
        for j,v2 in enumerate(allvectors):
            vec = normalized(v1-v2) @ lattice
            if np.max(np.abs(vec)) < tol:
                maps[i] = j
                break
        else:
            maps[i] = len(allvectors)
            allvectors.append(v1)
    return allvectors, maps

class SymmetryError(Exception):
    pass

class SpaceSymmetry(NamedTuple):
    number: int
    hall_number: int
    symbol: str
    full_symbol: str
    pointgroup_international: str
    arithmetic_crystal_class_number: int

    @classmethod
    def from_number(cls, number:int):
        hall_number = [1, 2, 3, 6, 9, 18, 21, 30, 39, 57, 60, 63, 72, 81, 90, 
                       108, 109, 112, 115, 116, 119, 122, 123, 124, 125, 128, 
                       134, 137, 143, 149, 155, 161, 164, 170, 173, 176, 182,
                       185, 191, 197, 203, 209, 212, 215, 218, 221, 227, 228,
                       230, 233, 239, 245, 251, 257, 263, 266, 269, 275, 278,
                       284, 290, 292, 298, 304, 310, 313, 316, 322, 334, 335,
                       337, 338, 341, 343, 349, 350, 351, 352, 353, 354, 355,
                       356, 357, 358, 359, 361, 363, 364, 366, 367, 368, 369,
                       370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380,
                       381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
                       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
                       404, 406, 407, 408, 410, 412, 413, 414, 416, 418, 419,
                       420, 422, 424, 425, 426, 428, 430, 431, 432, 433, 435,
                       436, 438, 439, 440, 441, 442, 443, 444, 446, 447, 448,
                       449, 450, 452, 454, 455, 456, 457, 458, 460, 462, 463,
                       464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
                       475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485,
                       486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 497,
                       498, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
                       510, 511, 512, 513, 514, 515, 516, 517, 518, 520, 521,
                       523, 524, 525, 527, 529, 530]
        assert 0 < number <= 230, "space group index out of range."
        return cls.from_hall_number(hall_number[number-1])

    @classmethod
    def from_hall_number(cls, number:int):
        assert 0 < number <= 530, "hall number index out of range."
        dataset = spglib.get_spacegroup_type(number)
        return cls(dataset['number'], 
                   number, 
                   dataset['international'], 
                   dataset['international_full'], 
                   dataset['pointgroup_international'],
                   dataset['arithmetic_crystal_class_number'])

    @property
    def international(self):
        return self.symbol

    @property
    def crystal_system(self):
        if self.number <= 2:
            return 'triclinic'
        elif self.number <= 15:
            return 'monoclinic'
        elif self.number <= 74:
            return 'orthorhombic'
        elif self.number <= 142:
            return 'tetragonal'
        elif self.number <= 167:
            return 'trigonal'
        elif self.number <= 194:
            return 'hexagonal'
        elif self.number <= 230:
            return 'cubic'
        else:
            raise

    def pearson_symbol(self, num_sites:int=1):
        """Generates the Pearson symbol for the crystal structure.
        
        The Pearson symbol is a compact notation that describes:
        1. The crystal system (first lowercase letter)
        2. The Bravais lattice centering (uppercase letter)
        3. The number of atoms in the unit cell (integer)

        Args:
            num_sites: Number of atoms in the unit cell. Defaults to 1.

        Returns:
            str: Pearson symbol in the format "{system}{centering}{num_sites}",
            where:
            - system: One of 'a' (triclinic), 'm' (monoclinic), 'o' (orthorhombic),
                    't' (tetragonal), 'h' (trigonal/hexagonal), 'c' (cubic)
            - centering: Lattice centering ('P', 'C', 'I', 'F', 'R')
            - num_sites: Provided argument value

        Example:
            >>> crystal = CrystalStructure(spacegroup=225)  # FCC cubic
            >>> crystal.pearson_symbol(4)
            'cF4'

        Note:
            - Converts non-primitive centering types (A/B/S) to 'C'
            - Uses 'h' for both trigonal and hexagonal systems
            Follows the notation defined in:
            International Tables for Crystallography, Volume A (2016)
        """        
        cry_sys_dict = {
            "triclinic": "a",
            "monoclinic": "m",
            "orthorhombic": "o",
            "tetragonal": "t",
            "trigonal": "h",
            "hexagonal": "h",
            "cubic": "c",
        }
        symbol = cry_sys_dict[self.crystal_system]
        centering = self.international[0]
        if centering in ("A", "B", "C", "S"):
            centering = "C"
        return f"{symbol}{centering}{num_sites}"

    @property
    def multiplicity(self):
        """ Return the symmetry degeneracy of the space group. """
        df = Dataset().arithmetic3d
        return df[df['arithmetic_number']==self.arithmetic_crystal_class_number]['multiplicity'].values[0]

class LayerSymmetry(NamedTuple):
    number: int
    spacegroup_number: int
    symbol: str
    full_symbol: str
    pointgroup_international: str
    arithmetic_crystal_class_number: int

    @classmethod
    def from_number(cls, number:int):
        """Creates a LayerGroup instance from layergroup number (1-80).
        
        Initializes symmetry information corresponding to the 80 two-dimensional layer groups
        defined by layergroup  numbers in the International Tables for Crystallography.

        Args:
            number: Layergroup  number of the layer group (1-80 inclusive).

        Returns:
            LayerGroup: Initialized layer group instance with these attributes:
                - spacegroup: Corresponding 2D layer group number
                - symbol: Short Hermann-Mauguin symbol
                - full_symbol: Full international symbol
                - pointgroup: Associated point group
                - arithmetic_number: Layer group arithmetic index

        Raises:
            AssertionError: If number is outside valid range (1-80)

        Example:
            >>> layer = LayerGroup.from_number(7)
            >>> layer.symbol
            'p2'
            >>> layer.pointgroup
            '2'

        Note:
            The arithmetic numbering follows the standard scheme in:
            International Tables for Crystallography (Vol. E), Chapter 1.4
        """  
        assert 0 < number <= 80, "layer group index out of range."
        maps = Dataset().layermaps[number]
        return cls(number, 
                   maps['spacegroup'], 
                   maps['symbol'], 
                   maps['full_symbol'], 
                   maps['pointgroup'],
                   maps['arithmetic_number'])
    
    @classmethod
    def from_hall_number(cls, number:int):
        # TODO
        raise

    @property
    def multiplicity(self):
        """ Return the symmetry degeneracy of the layer group. """
        df = Dataset().arithmetic2d
        #return df[df['arithmetic_number']==self.arithmetic_crystal_class_number]['multiplicity'].values[0]
        return df[df['pointgroup']==self.pointgroup_international]['multiplicity'].values[0]

    @property
    def international(self):
        return self.symbol

    def pearson_symbol(self, num_sites=1):
        """Generates the Pearson symbol for the crystal structure.
        
        The Pearson symbol is a compact notation that describes:
        1. The crystal system (first lowercase letter)
        2. The Bravais lattice centering (uppercase letter)
        3. The number of atoms in the unit cell (integer)

        Args:
            num_sites: Number of atoms in the unit cell. Defaults to 1.

        Returns:
            str: Pearson symbol in the format "{system}{centering}{num_sites}",
            where:
            - system: One of 'a' (triclinic), 'm' (monoclinic), 'o' (orthorhombic),
                    't' (tetragonal), 'h' (trigonal/hexagonal), 'c' (cubic)
            - centering: Lattice centering ('P', 'C', 'I', 'F', 'R')
            - num_sites: Provided argument value

        Example:
            >>> crystal = CrystalStructure(spacegroup=225)  # FCC cubic
            >>> crystal.pearson_symbol(4)
            'cF4'

        Note:
            - Converts non-primitive centering types (A/B/S) to 'C'
            - Uses 'h' for both trigonal and hexagonal systems
            Follows the notation defined in:
            International Tables for Crystallography, Volume A (2016)
        """
        cry_sys_dict = {
            "triclinic": "a",
            "monoclinic": "m",
            "orthorhombic": "o",
            "tetragonal": "t",
            "trigonal": "h",
            "hexagonal": "h",
            "cubic": "c",
        }
        symbol = cry_sys_dict[self.crystal_system]
        centering = self.international[0]
        if centering in ("A", "B", "C", "S"):
            centering = "C"
        return f"{symbol}{centering}{num_sites}"

    @property
    def crystal_system(self):
        """Determines the crystal system based on the layer group number.

        The crystal system is derived from the arithmetic numbering scheme defined in
        International Tables for Crystallography, Volume E (2010), where layer groups
        are categorized by their symmetry properties:

        Number Range | Crystal System
        -------------|--------------
        1-2          | Triclinic
        3-18         | Monoclinic
        19-48        | Orthorhombic
        49-64        | Tetragonal
        65-72        | Trigonal
        73-80        | Hexagonal

        Returns:
            str: The crystal system name in lowercase (e.g., 'monoclinic').

        Raises:
            ValueError: If the layer group number is outside the valid range (1-80).

        Example:
            >>> group = LayerGroup(number=25)
            >>> group.crystal_system
            'orthorhombic'

        Note:
            This classification follows the standard 2D layer group numbering system.
            The trigonal and hexagonal systems are treated as distinct categories.
        """        
        if self.number <= 2:
            return 'triclinic'
        elif self.number <= 18:
            return 'monoclinic'
        elif self.number <= 48:
            return 'orthorhombic'
        elif self.number <= 64:
            return 'tetragonal'
        elif self.number <= 72:
            return 'trigonal'
        elif self.number <= 80:
            return 'hexagonal'
        else:
            raise

class EqualTools:

    def __init__(self, number:int, group='space'):
        self.arithmetic_number = number
        self.group = group
        if group == 'space':
            df = Dataset().arithmetic3d
            self.dataset = df.iloc[number-1].to_dict()
        elif group == 'layer':
            df = Dataset().arithmetic2d
            self.dataset = df.iloc[number-1].to_dict()
        self._lattice_symmetry = None

    @property
    def lattice_symmetry(self):
        if self._lattice_symmetry == None:
            self._lattice_symmetry = self.get_lattice_symmetry()
        return self._lattice_symmetry

    @property
    def noperation(self):
        return len(self.dataset['all_generators'])

    @property
    def operations(self):
        return {symbol: self.get_operation(symbol) for symbol in self.dataset['all_generators']}
        
    def get_operation(self, symbol):
        maps = {'2_100':'2+100', '2_010':'2+010','m_100':'m+100','m_010':'m+010'}
        if symbol in maps:
            if self.group == 'layer' and (30 <= self.arithmetic_number <= 45):
                symbol = maps[symbol] 
            elif self.group == 'layer' and ( 38 < self.arithmetic_number <=58):
                symbol = maps[symbol] 
        return Dataset().symmetry_operation[symbol]    

    @property
    def generators(self):
        gens = ['1'] + self.dataset['generators']
        return {symbol: self.get_operation(symbol) for symbol in gens}    

    @property
    def unique_generators(self):
        gens = self.dataset['generators']
        allgens = self.lattice_symmetry.dataset['generators']
        if self.group == 'layer':
            if self.arithmetic_number in (8,9,):
                unique = ['2_001','-1']
            elif self.arithmetic_number in (16,):
                unique = ['4+001','-1']
            elif self.arithmetic_number in (23,):
                unique = ['2_010','-1']
            elif self.arithmetic_number in (26,27,28,42,43,44):
                unique = ['-1']
            elif self.arithmetic_number in (32,34,35):
                unique = ['2_001','-1']
            elif self.arithmetic_number in (36,):
                unique = ['2_001']
            elif self.arithmetic_number in (39,):
                unique = ['2_110','-1']
            else:
                if len(set(gens) - set(allgens)) != 0:
                    print(self.arithmetic_number, self.lattice_symmetry.arithmetic_number)
                    print(self.dataset)
                    print(self.lattice_symmetry.dataset)
                    raise ValueError("unique failed")
                unique = set(allgens) - set(gens)
        return {symbol: self.get_operation(symbol) for symbol in unique}

    @property
    def unique_shifts(self):
        number = self.lattice_symmetry.arithmetic_number
        if self.group == 'space':
            raise
        elif self.group == 'layer':
            if number in (27,29,):
                return np.array([[0,0,0],[1/2,0,0],[0,1/2,0],[1/2,1/2,0]])
            elif number in (34, 35, 43, 45):
                return np.array([[0,0,0],[1/3,2/3,0],[2/3,1/3,0]])
            else:
                raise ValueError(f"Unsupport layergeoup {number}.")

    def get_lattice_symmetry(self):
        map2s = [2,2,                       # T   -1    1-2
                 6,6,6,6,6,                 # Mo  2/m   3-7
                 #12,12,12,12,12,12,        # Mr  2/m   8-13
                 20,20,20,20,20,20,         # Mr  2/m   8-13
                 #20,20,20,20,20,20,20,20,  # Ort mmm   14-21
                 29,29,29,29,29,29,29,29,   # Ort mmm   14-21
                 29,29,29,29,29,29,29,29,   # Tet 4/mmm 22-29
                 #37,37,36,37,37,36,36,37,  # Tri -3m   30-37
                 45,45,45,45,45,45,45,45,   # Tri -3m   30-37
                 45,45,45,45,45,45,45,45]   # Hex 6/mmm 38-45
        map3s = [2, 2,                                # -1       1-2
                 7, 7, 7, 7, 7, 7,                    # 2/m      3-8
                 18, 18, 18, 18, 18, 18, 18, 18,      # mmm     9-21 
                 18, 18, 18, 18, 18,
                 36, 36, 36, 36, 36, 36, 36, 36,      # 4/mmm   22-37
                 36, 36, 36, 36, 36, 36, 36, 36,                                 
                 58, 58, 58, 58, 58, 58, 58, 58,      # 6/mmm  38-58
                 58, 58, 58, 58, 58, 58, 58, 58, 
                 58, 58, 58, 58, 58,
                 72, 72, 72, 72, 72, 72, 72, 72,      # m-3m  59-73
                 72, 72, 72, 72, 72, 72, 72]

        if self.group == 'layer': 
            number = map2s[self.arithmetic_number-1]
        elif self.group == 'space': 
            assert len(map3s) == 73, "map3s out of range"
            number = map3s[self.arithmetic_number-1]
        if number == 12:
            print(self.arithmetic_number, map2s[self.arithmetic_number-1])
        return EqualTools(number, group=self.group) 

    def set_lattice_symmetry(self, lattice):
        self._lattice_symmetry = EqualTools.from_lattice(lattice, group=self.group).get_lattice_symmetry()

    @classmethod
    def from_lattice(cls, lattice, group='layer'):
        cell = (lattice, [[0,0,0]], [0])
        if group == 'layer':
            from spglib import spglib
            dataset = spglib.get_symmetry_layerdataset(cell, symprec=1e-2)
            number = dataset['number']
            assert number in [2,6,14,18,37,47,61,80], f"number is {number}"
            assert 0 < number <= 80, "layergroup index out of range."
            return cls.from_layergroup(number)
        elif group == 'space':
            import spglib
            cell = (lattice, [[0,0,0]], [0])
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-2)
            number = dataset['number']
            assert number in [2,10,12,47,65,123,166,191,221], f"number is {number}"
            assert 0 < number <= 230, "layergroup index out of range."
            return cls.from_spacegroup(number)
        else:
            raise

    @classmethod
    def from_layergroup(cls, number:int):
        maps = [1, 2,                                                             # -1     1-2   
                3, 4, 5, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 12, 12, 13,           # 2/m    3-18
                14, 14, 14, 15, 16, 16, 16, 17, 18, 18, 18, 18, 18, 18, 18,        
                18, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21,       # mmm   19-48 
                22, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 29, 29,   # 4/mmm 49-64
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]   # 6/mmm 65-80

        assert 0 < number <= 80, "layergroup index out of range."
        return cls(maps[number-1], group='layer')

    @classmethod
    def from_spacegroup(cls, number:int):
        """ Arithmetic crystal classes 
        https://onlinelibrary.wiley.com/iucr/itc/Cb/ch1o4v0001/ """
        maps = [1, 2,                                                             # -1       1-2
                3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 7, 7, 8,                            # 2/m      3-15
                9, 9, 9, 9, 10, 10, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,   # mmm     16-74 
                13, 13, 14, 14, 14, 15, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 
                18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19,
                19, 19, 19, 19, 20, 20, 21, 21, 21, 21, 
                22, 22, 22, 22, 23, 23, 24, 25, 26, 26, 26, 26, 27, 27, 28, 28,   # 4/mmm   75-142
                28, 28, 28, 28, 28, 28, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 
                31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 35, 35,
                36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
                37, 37, 37, 37,                                                
                38, 38, 38, 39, 40, 41, 42, 43, 42, 43, 42, 43, 44, 45, 46, 45,   # 6/mmm  143-194
                46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 51, 51, 51, 51, 52,
                53, 53, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 57, 57, 56, 56,
                58, 58, 58, 58, 
                59, 60, 61, 59, 61, 62, 62, 63, 63, 64, 62, 64, 65, 65, 66, 66,   # m-3m   195-230 
                67, 65, 65, 67, 68, 69, 70, 68, 69, 70, 71, 71, 71, 71, 72, 72,
                72, 72, 73, 73]
        assert 0 < number <= 230, "spacegroup index out of range."
        return cls(maps[number-1], group='space')
    
    @classmethod
    def is_lattice_equal(self, lattice_parameters1, lattice_parameters2, dim=2, tol=0.01):
        """ Determine whether the lattice constants are consistent.
            if consistent, return the lattice transition matrix.

        Args:
            atoms1 (Structure): input structure 1
            atoms2 (Structure): input structure 2
            dim (int, optional): structure dimension. Defaults to 2.
            tol (float, optional): lattice tolerance. Defaults to 0.01.

        Returns:
            (bool, list): (lattice consistent or not, lattice transition matrix)
        """        
        # 晶格常数一致
        a1, b1, c1, alpha1, beta1, gamma1 = lattice_parameters1
        a2, b2, c2, alpha2, beta2, gamma2 = lattice_parameters2

        def lattice_diff(a,b):
            return abs(a-b)/(a+b)

        if dim == 2:
            # c1 = c2
            if lattice_diff(c1,c2) > tol:
                return False, []
            # gamma1 = gamma2
            if lattice_diff(gamma1,gamma2) > tol:
                return False, []
            # if a1 != a2 or b1 != b2, try a1 = b2 and a2 = b1
            if lattice_diff(a1,a2) > tol or lattice_diff(b1,b2) > tol:
                if lattice_diff(a1,b2) < tol and lattice_diff(a2,b1) < tol:
                    if lattice_diff(alpha1,beta2) > tol and lattice_diff(alpha1,180-beta2) < tol:
                        return True, [[0,1,0],[-1,0,0],[0,0,1]]
                    elif lattice_diff(alpha2,beta1) > tol and lattice_diff(alpha2,180-beta1) < tol:
                        return True, [[0,-1,0],[1,0,0],[0,0,1]]
                    else:
                        return True, [[0,1,0],[1,0,0],[0,0,1]]
                else:
                    return False,[]
            else:
                return True, [[1,0,0],[0,1,0],[0,0,1]]
                
        return False,[]
    
    @classmethod
    def is_element_equal(self, elements1, elements2):        
        """
        Generate element-wise mapping between two atomic structures and check for species equality.
        
        This function compares two atomic structures by:
        1. Checking if they contain the same elements in the same proportions
        2. Generating mappings between atoms of the same element
        
        Parameters
        ----------
        atoms1 : Atoms
            First atomic structure to compare
        atoms2 : Atoms
            Second atomic structure to compare
            
        Returns
        -------
        tuple[bool, list[tuple[np.ndarray, np.ndarray]]]
            A tuple containing:
            - bool: True if structures have identical elements and counts, False otherwise
            - list: Mappings between atom indices for each element (empty if not equal)
            
        Examples
        --------
        >>> equal, maps = is_element_equal(elements1, elements2)
        >>> if equal:
        ...     for elem_map in maps:
        ...         print(f"Element mapping: {elem_map[0]} -> {elem_map[1]}")
        """
        # Get unique elements and their indices for both structures
        species1, indices1 = np.unique(elements1, return_inverse=True)
        species2, indices2 = np.unique(elements2, return_inverse=True)

        # check species and indices
        equal = True
        maps = []

        # Early return if species counts or atom counts differ
        if len(species1) != len(species2) or (len(indices1) != len(indices2)):
            return False, []

        # Check each element and generate mappings        
        for i in range(len(species1)):
            idx1 = np.where(indices1 == i)[0]
            idx2 = np.where(indices2 == i)[0]
            if species1[i] != species2[i] or len(idx1) != len(idx2):
                equal = False
                break
            maps.append([idx1, idx2])

        return equal, maps
        
    @classmethod
    def is_position_equal(
        cls,
        positions1: np.ndarray,
        positions2: np.ndarray,
        maps: list[tuple[np.ndarray, np.ndarray]] | None = None,
        lattice: np.ndarray | None = None,
        tol: float = 0.1
    ) -> np.ndarray | None:
        """
        Determine if positions1 can be translated to positions2 within a given tolerance.
        
        This method checks for translational symmetry between two sets of atomic positions by:
        1. Calculating possible shift vectors between equivalent positions
        2. Validating the shift vectors against symmetry operations
        3. Confirming the optimal shift minimizes the position differences
        
        Parameters
        ----------
        positions1 : np.ndarray
            (N,3) array of atomic positions for structure 1
        positions2 : np.ndarray
            (M,3) array of atomic positions for structure 2
        maps : list[tuple[np.ndarray, np.ndarray]], optional
            List of index mappings between equivalent atoms. Each tuple contains
            (indices_in_positions1, indices_in_positions2). If None, assumes direct 1:1 mapping.
        lattice : np.ndarray, optional
            (3,3) array representing the lattice vectors. Used for periodic boundary handling.
            If None, assumes Cartesian coordinates.
        tol : float, optional
            Maximum allowed RMSD for considering positions equivalent (in Angstroms)
        
        Returns
        -------
        np.ndarray or None
            The optimal shift vector that aligns positions1 to positions2, or None if no valid
            translation found within tolerance.
        
        Raises
        ------
        AssertionError
            If the final calculated shift doesn't properly align the structures within tolerance
        
        Notes
        -----
        The algorithm proceeds in two main steps:
        1. Candidate shift generation by comparing equivalent position pairs
        2. Shift validation against common Wyckoff positions and final optimization
        
        Examples
        --------
        >>> pos1 = np.array([[0,0,0], [0.5,0.5,0.5]])
        >>> pos2 = np.array([[0.1,0.1,0.1], [0.6,0.6,0.6]])
        >>> shift = is_position_squal(pos1, pos2, tol=0.15)
        >>> print(shift)  # Should return True, [0.1, 0.1, 0.1]
        """  
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import KDTree        
        from scipy.cluster.hierarchy import DisjointSet

        def remove_duplicate_vectors(data, tolerance):
            """ Remove duplicate vectors within given tolerance using hierarchical clustering.

            Args:
                data (array): shift vectors
                tolerance (float): tolerance of shift

            Returns:
                array: shift vectors without duplicates
            """            
            tree = KDTree(data)
            pairs = tree.query_pairs(tolerance)
            graph = DisjointSet(np.arange(len(data)))

            for i, j in pairs:
                graph.merge(i,j)
            parents = [graph[i] for i in graph]
            return data[np.unique(parents, return_index=True)[1]]

        def get_shift_vectors(vectors, alls=None, tol=0.05):
            """ get shift vectors from positions difference

            Args:
                vectors (array): positions difference
                alls (array, optional): Candidate shift vector. Defaults to None.
                tol (float, optional): tolerance of shift. Defaults to 0.05.

            Returns:
                newalls (array): shift vectors that make linear_sum_assignment value less than tol
            """
                   
            if len(alls) == 0:
                alls = np.r_[vectors[0,:], vectors[:,0]][1:]
                alls = normalized(alls)
                if len(alls) > 1:
                    alls = remove_duplicate_vectors(alls, tol)

            newalls = []
            for vec in alls:
                diff = vectors - vec
                #diff2 = np.abs(np.sum(normalized(diff), axis=-1))
                diff2 = np.sum(np.abs(normalized(diff)), axis=-1)
                row_ind, col_ind = linear_sum_assignment(diff2)
                davg = np.mean(diff2[row_ind, col_ind])
                #print(diff2, davg)
                if davg < tol:
                    newalls.append(vec)

            return newalls

        # Initialize lattice if not provided
        lattice = np.eye(3) if lattice is None else lattice

        # Prepare position mappings - sort by mapping size for efficiency
        if maps is None:
            maps = [[np.arange(len(positions1)), np.arange(len(positions2))]]
        else:
            maps = sorted(maps, key=lambda x: len(x[0]))

        # Step 1: Generate candidate shift vectors
        candidate_shifts = []
        for idx1, idx2 in maps:
            position_diffs = positions1[idx1, None] - positions2[None, idx2]
            candidate_shifts = get_shift_vectors(position_diffs, candidate_shifts, tol)            
            if not candidate_shifts:  # Early termination if no valid shifts
                return False, None

        # Step 2: Validate against common Wyckoff positions
        common_wyckoff = np.array([
            [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0],
            [0.5, 0.5, 0], [1/3, 2/3, 0], [2/3, 1/3, 0]
        ])

        final_shift = None
        for shift in candidate_shifts:
            # Check if shift matches common fractional coordinates
            wyckoff_diffs = np.sum(np.abs(normalized(shift[:2] - common_wyckoff[:, :2])), axis=1)
            if np.any(wyckoff_diffs < 1e-3):
                final_shift = shift
                break

        # Fallback to first candidate if no Wyckoff match found
        final_shift = final_shift or candidate_shifts[0]

        # Final validation
        diff = positions1[:, None] - positions2[None, :] - final_shift
        cost_matrix = np.abs(np.sum(normalized(diff), axis=-1))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        avg_deviation = np.mean(cost_matrix[row_ind, col_ind])
        
        if avg_deviation >= tol:
            raise AssertionError(
                f"Invalid shift vector (avg deviation {avg_deviation:.3f} > tolerance {tol})"
            )
        
        return True, final_shift

    def is_structure_equal(self, atoms1, atoms2, all_operations=False, tol=0.03):
        """calculate the transformation matrix from atoms1 to atoms2

        Args:
            atoms1 (Structure): input structure
            atoms2 (Structure): target structure
            all_operations (bool, optional): use symmetry operations or lattice_symmetry operations. Defaults to False.
            tol (float, optional): tolerance. Defaults to 0.03.

        Returns:
            (bool, str, np.ndarray): transformation matrix, shift
        """
        # element check %
        equal1, maps = self.is_element_equal(atoms1.get_elements(),
                                             atoms2.get_elements())

        # lattice check %
        equal2 = self.is_lattice_equal(atoms1.lattice_parameters, atoms2.lattice_parameters)
        if equal1 is False or equal2 is False:
            return False, None, None
        
        lattice = atoms1.lattice
        coord1 = atoms1.get_positions()
        coord2 = atoms2.get_positions()
        if isinstance(equal2, list):
            coord2 = coord2 @ np.array(equal2)
        # print(coord1)
        # print(coord2)
        # print()

        operations = self.lattice_symmetry.generators if all_operations else self.generators
        for symbol,matrix in operations.items():
            coord = coord1 @ matrix.T
            #print('unitcell', coord)
            #print('stdcell', coord2)
            equal3, shift = self.is_position_equal(coord, coord2, maps, lattice, tol=tol*10)
            # print(symbol, shift)
            # print(coord)
            # print(coord2)
            # print()
            if equal3:
                return True, symbol, shift
        else:
            return False, None, None


    def get_unique_structure(self, cell):
        """
        Generate symmetrically unique structures by applying symmetry operations from the generating set.
        
        This method applies each symmetry operation from the unique generators to the input structure,
        then checks if the transformed structure is symmetrically equivalent to any previously found
        unique structures through translation. Only genuinely new structures are added to the result set.

        Parameters
        ----------
        cell : Structure
            The input atomic structure to analyze. Should contain lattice, atomic positions, and elements.

        Returns
        -------
        dict[str, Structure]
            A dictionary mapping symmetry operation symbols to their corresponding unique structures.
            Always includes the identity operation ('1') as the first entry.

        Notes
        -----
        - The method only uses the generating set of symmetry operations rather than the full group,
        which significantly reduces computational cost while still finding all unique structures.
        - Symmetry equivalence is determined by checking for translational symmetry between
        transformed structures after applying each symmetry operation.
        - The input structure is always included as the identity operation ('1').
        """
        from jamip.structure import Structure

        elements = cell.get_elements()
        lattice = cell.lattice
        positions = cell.get_positions(type='direct')
        
        # Create element mapping for translation checks
        element_maps = []
        for element in np.unique(elements):
            element_indices = np.where(elements == element)[0]
            element_maps.append((element_indices, element_indices))

        # Initialize with identity operation
        unique_cells = {'1':cell}

        # Check each generator (skipping identity)
        for symbol,matrix in self.unique_generators.items():
            if symbol == '1': continue

            positions1 = positions @ matrix.T
            for cell2 in unique_cells.values():
                positions2 = cell2.get_positions(type='direct')
                equal, idx = self.is_position_equal(positions1, positions2, element_maps, lattice)
                if not equal:
                    break
            else:
                new_cell = Structure.from_cell((lattice, positions1, elements))
                unique_cells[symbol] = new_cell

        return unique_cells
