import pkg_resources
import numpy as np
import pandas as pd
from typing import NamedTuple
from jamip.utils.utils import lazy_property
from jamip.utils.logger import load_yaml
from jamip.structure import write
import spglib
#from numba import njit

datafile = pkg_resources.resource_filename(__name__, 'LayerGroup.csv')
LayerData = pd.read_csv(datafile)
layermaps = LayerData.set_index('number', drop=False).to_dict('index')
#datafile = pkg_resources.resource_filename(__name__, 'symm_data.yaml')
#subgroup = load_yaml(datafile)['maximal_subgroups']

datafile = pkg_resources.resource_filename(__name__, 'arithmetic2d.json')
Arithmetic2d = pd.read_json(datafile)
datafile = pkg_resources.resource_filename(__name__, 'arithmetic.json')
Arithmetic3d = pd.read_json(datafile)
datafile = pkg_resources.resource_filename(__name__, 'pointgroup.yaml')
Symmetry_operation = load_yaml(datafile)
for key, value in Symmetry_operation.items():
    Symmetry_operation[key] = np.array(value)

#@njit
def normalized(vectors):
    vectors -= np.floor(vectors)
    vectors = np.around(vectors,8)
    vectors -= np.around(vectors)
    return vectors

#@njit
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

#@njit
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

    @property
    def multiplicity(self):
        return Arithmetic3d[Arithmetic3d['pointgroup']==self.pointgroup_international, 'multiplicity'].values[0]

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
        dataset = spglib.get_spacegroup_type(i)
        return cls(dataset['number'], 
                   number, 
                   dataset['international'], 
                   dataset['international_full'], 
                   dataset['pointgroup_international'],
                   dataset['arithmetic_crystal_class_number'])

class LayerSymmetry(NamedTuple):
    number: int
    spacegroup_number: int
    symbol: str
    full_symbol: str
    pointgroup_international: str
    arithmetic_crystal_class_number: int

    @property
    def multiplicity(self):
        return Arithmetic2d[Arithmetic2d['pointgroup']==self.pointgroup_international, 'multiplicity'].values[0]

    @classmethod
    def from_number(cls, number:int):
        assert 0 < number <= 80, "layer group index out of range."
        maps = layermaps[number]['arithmetic_number']
        return cls(number, 
                   maps['spacegroup'], 
                   maps['symbol'], 
                   maps['full_symbol'], 
                   maps['pointgroup'],
                   maps['arithmetic_number'])

class EqualTools:

    def __init__(self, number:int, group='space'):
        self.arithmetic_number = number
        self.group = group
        if group == 'space':
            self.dataset = Arithmetic3d.iloc[number-1].to_dict()
        elif group == 'layer':
            self.dataset = Arithmetic2d.iloc[number-1].to_dict()
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

    @property
    def all_operations(self):
        return self.lattice_symmetry.operations

    @property
    def generators(self):
        gens = ['1'] + self.dataset['generators']
        return {symbol: self.get_operation(symbol) for symbol in gens}

    @property
    def unique_generators(self):
        gens = self.dataset['generators']
        allgens = self.lattice_symmetry.dataset['generators']
        if self.arithmetic_number in (26,27,28,42,43,44) and self.group == 'layer':
            unique = ['-1']
        elif self.arithmetic_number in (16,) and self.group == 'layer':
            unique = ['4+001','-1']
        elif self.arithmetic_number in (23,) and self.group == 'layer':
            unique = ['2_010','-1']
        elif self.arithmetic_number in (32,34,35) and self.group == 'layer':
            unique = ['2_001','-1']
        elif self.arithmetic_number in (8,9,) and self.group == 'layer':
            unique = ['2_001','-1']
        elif self.arithmetic_number in (36,) and self.group == 'layer':
            unique = ['2_001']
        elif self.arithmetic_number in (39,) and self.group == 'layer':
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
            if number == 27:
                return np.array([[0,0,0],[1/2,0,0],[0,1/2,0],[1/2,1/2,0]])
            elif number in (34, 35, 43):
                return np.array([[0,0,0],[1/3,2/3,0],[2/3,1/3,0]])
            else:
                raise ValueError("Unsupport spacegeoup.")

    def get_lattice_symmetry(self):
        map2s = [2,2,                      # T   -1    1-2
                 6,6,6,6,6,                # Mo  2/m   3-7
                 #12,12,12,12,12,12,        # Mr  2/m   8-13
                 20,20,20,20,20,20,        # Mr  2/m   8-13
                 #20,20,20,20,20,20,20,20,  # Ort mmm   14-21
                 29,29,29,29,29,29,29,29,  # Ort mmm   14-21
                 29,29,29,29,29,29,29,29,  # Tet 4/mmm 22-29
                 #37,37,36,37,37,36,36,37,  # Tri -3m   30-37
                 45,45,45,45,45,45,45,45,  # Tri -3m   30-37
                 45,45,45,45,45,45,45,45]  # Hex 6/mmm 38-45
        map3s = []

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
            assert number in [2,6,14,37,61,80]
            assert 0 < number <= 80, "layergroup index out of range."
            return cls.from_layergroup(number)
        elif group == 'space':
            cell = (lattice, [0,0,0], [0])
            dataset = spglib.get_symmetry_dataset(cell, symprec=1e-2)
            number = dataset['number']
            assert number in [2,10,12,47,65,123,191,221]
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

    #def convert_positions(self, unitcell, stdcell):
        # 计算在晶格变化后的原子坐标,满足
    def check_lattice(self, atoms1, atoms2):
        # 晶格常数一致
        lattice1 = atoms1.lattice_parameters
        lattice2 = atoms2.lattice_parameters
        if sum(abs(lattice1-lattice2)) > 1e-4:
            return False

        return True
    
    def is_lattice_equal(self, atoms1, atoms2, dim=2, tol=0.01):
        # 晶格常数一致
        a1, b1, c1, alpha1, beta1, gamma1 = atoms1.lattice_parameters
        a2, b2, c2, alpha2, beta2, gamma2 = atoms2.lattice_parameters
        print(a1, b1, c1, alpha1, beta1, gamma1)
        print(a2, b2, c2, alpha2, beta2, gamma2)

        def lattice_diff(a,b):
            return abs(a-b)/(a+b)

        if dim == 2:
            # c1 = c2
            if lattice_diff(c1,c2) > tol:
                return False
            # gamma1 = gamma2
            if lattice_diff(gamma1,gamma2) > tol:
                return False
            # if a1 != a2 or b1 != b2, try a1 = b2 and a2 = b1
            if lattice_diff(a1,a2) > tol or lattice_diff(b1,b2) > tol:
                if lattice_diff(a1,b2) < tol and lattice_diff(a2,b1) < tol:
                    if lattice_diff(alpha1,beta2) > tol and lattice_diff(alpha1,180-beta2) < tol:
                        return [[0,1,0],[-1,0,0],[0,0,1]]
                    elif lattice_diff(alpha2,beta1) > tol and lattice_diff(alpha2,180-beta1) < tol:
                        return [[0,-1,0],[1,0,0],[0,0,1]]
                    else:
                        return [[0,1,0],[1,0,0],[0,0,1]]
                else:
                    return False
            else:
                return False
                
        return False

    def is_structure_equal(self, atoms1, atoms2, all_operations=False, tol=0.03):
        # 计算atoms1向atoms2的转换矩阵
        # lattice1 @ n1 = lattice2
        # position1[indices] + shift = position2

        # get position mapping by elements %
        species1, indices1 = np.unique(atoms1.get_elements(), return_inverse=True)
        species2, indices2 = np.unique(atoms2.get_elements(), return_inverse=True)

        # check species and indices
        equal = True
        if len(species1) != len(species2) or len(indices1) != len(indices2):
            equal = False
        maps = []
        for i in range(len(species1)):
            idx1 = np.where(indices1==i)[0]
            idx2 = np.where(indices2==i)[0]
            if species1[i] != species2[i] or len(idx1) != len(idx2):
                equal = False
                break
            maps.append([idx1, idx2])

        # lattice check %
        equal2 = self.is_lattice_equal(atoms1, atoms2)
        if equal is False or equal2 is False:
            return None, None, None
        
        print('============ %s =================' %atoms1.get_formula())
        print(equal2, self.lattice_symmetry.arithmetic_number, self.arithmetic_number)
        # if self.arithmetic_number == 8 and self.lattice_symmetry.arithmetic_number == 12:
        #     print(self.group)
        #     print(self.get_lattice_symmetry().arithmetic_number)
        #     raise RuntimeError("find operation failed.")
        print()
        
        lattice = atoms1.lattice
        coord1 = atoms1.get_positions()
        coord2 = atoms2.get_positions()
        if isinstance(equal2, list):
            coord2 = coord2 @ np.array(equal2).T          
        # if len(equal2) and len(indices1) == 18:
        #     coord1 = coord1[:,[1,2,0]]
        #     coord2 = coord2[:,[1,2,0]]
        print(coord1)
        print(coord2)

#        operations = self.all_operations if all_operations else self.operations
        operations = self.lattice_symmetry.generators if all_operations else self.generators
        for symbol,matrix in operations.items():
            coord = coord1 @ matrix.T
            # print(matrix)
            shift = self.istranslate(coord, coord2, maps, lattice, tol=tol*10)
            print(symbol, shift)
            if not (shift is None):
                print(symbol)
                print()
                return symbol, shift, None

        else:
            return None,None,None
            # raise RuntimeError("find operation failed.")
        
    # def is_structure_symmetry_(self, atoms1, atoms2, all_operations=False, tol=0.03):

    #     symbol, shift, _ = self.is_structure_equal(atoms1, atoms2, all_operations, tol)
    #     return symbol is not None
        
    @classmethod
    def istranslate(self, positions1, positions2, maps=None, lattice=None, tol=0.1):
        """ 
        for elements [a,a,b,b,b,b]
        vectors = [a - a1] & [b - b1]
        positions1 = sorted(positions1)
        positions2 = sorted(positions2 - vectors)
        allclose(positions1, positions2)
        """        
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import KDTree

        # def lexsort(vectors):
        #     sorted_indices = np.lexsort((vectors[:, 2], vectors[:, 1], vectors[:, 0]))
        #     return vectors[sorted_indices]

        def remove_duplicates(data, tolerance):
            tree = KDTree(data)
            pairs = tree.query_pairs(tolerance)
            unique_indices = [i for i, j in pairs if i != j]
            return data[unique_indices]


        def get_shift(vectors, alls=None, tol=0.05):
            if len(alls) == 0:
                alls = np.r_[vectors[0,:], vectors[:,0]][1:]
                alls = normalized(alls)
                if len(alls) > 1:
                    alls = remove_duplicates(alls, tol)

            newalls = []
            for vec in alls:
                diff = vectors - vec
                diff2 = np.abs(np.sum(normalized(diff), axis=-1))
                row_ind, col_ind = linear_sum_assignment(diff2)
                davg = np.mean(diff2[row_ind, col_ind])
                print(vec, davg)
                if davg < tol:
                    newalls.append(vec)

            return newalls

        if lattice is None:
            lattice = np.eye(3)

        if maps is None:
            maps = [[np.arange(len(positions1)), np.arange(len(positions2))]]
        else:
            # sort maps by length
            maps = sorted(maps, key=lambda x: len(x[0]))

        # print(maps)

        # step 1: get shift vectors
        allvectors = []
        for idx1, idx2 in maps:
            diff = positions1[idx1, None] - positions2[None, idx2]
            allvectors = get_shift(diff, allvectors)
            # print(len(allvectors))
            # print(allvectors)
            if len(allvectors) == 0:
                return None
        
        shift = allvectors[0]
        # valid
        diff = positions1[:,None] - positions2[None, :] - shift
        diff2 = np.abs(np.sum(normalized(diff), axis=-1))
        row_ind, col_ind = linear_sum_assignment(diff2)
        davg = np.mean(diff2[row_ind, col_ind])
        assert davg < tol, f"shift vector is not valid, davg={davg}"
        return shift

        # if len(allvectors) == 1:
        #     return allvectors[0]
        # else:
        #     print('len allvectors > 1 !', len(allvectors))
        #     raise RuntimeError('len allvectors > 1 !')

        # # step 2: valid shift vecotrs 
        # if len(allvectors) == 1:
        #     vectors = allvectors[0]
        # else:
        #     vectors = allvectors[0]
        #     for vec in allvectors[1:]:
        #         diff = np.abs(vectors[:, None, :] - vec)  # a[:, None, :] 表示将 a 扩展为 (100, 1, 3) 的数组
        #         indices = np.where(np.all(diff < 0.05, axis=-1))
        #         vectors = vectors[indices[0]]
        #         if len(vectors) == 0:
        #             return None
                
        # print(vectors)
        # print(allvectors)

        # print(positions1-vectors[0] - np.floor((positions1-vectors[0])))
        # print(positions2)
        # print('===================')

        # # step 3: valid vectors on all vectors
        # for vec in vectors:
        #     status = True
        #     for idx1, idx2 in maps:

        #         diff = positions1[idx1,None] - positions2[None,idx2] - vec
        #         diff2 = np.abs(np.sum(normalized(diff), axis=-1))
        #         row_ind, col_ind = linear_sum_assignment(diff2)
                
        #         coord1 = lexsort(normalized(positions1[idx1]))
        #         coord2 = lexsort(normalized(positions2[idx2] + vec))

                
        #         print('vd2vs', np.mean(diff2[row_ind, col_ind]), np.sum(abs(coord1-coord2)), tol)

        #         if np.sum(abs(coord1-coord2)) > tol:
        #             status = False
        #             break
        #     if status:
        #         return vec
        
        # return None

    @classmethod
    def istranslate2(self, positions1, positions2, elements=None, lattice=None, tol=0.1):
        """
        由于数组是乱序的，处理起来很麻烦，
        这里按元素从少到多的顺序排序, 依次计算是否存在公共的平移向量
        """
        if elements is None:
            elements = np.ones(len(positions1), dtype=int)
        if lattice is None:
            lattice = np.eye(3)

        values,counts = np.unique(elements, return_counts=True)
        new_indices = np.arange(len(elements))
        #lattice[2] *= 0.8
        # natom, natom, 3
        vectors = []
        maps = []
        for idx in np.argsort(counts):
            indices = np.where(elements==values[idx])
            diff = positions1[indices][:,None,:]-positions2[indices][None,:,:]
            indices = np.where(elements==values[idx])[0]
            natom = len(indices)
            for i in range(natom):
                for j in range(natom):
                    for m,v in enumerate(vectors):
                        vec = normalized(diff[i,j,:]-v) @ lattice
                        if np.max(np.abs(vec)) < tol:
                            maps.append([indices[i], indices[j], m])
                            break
                    else:
                        m = len(vectors)
                        maps.append([indices[i],indices[j], m])
                        vectors.append(normalized(diff[i,j,:]))

        maps = np.array(maps)
        # match permutations
        natom = len(elements)
        for idx in np.unique(maps[:,2]):
            parts = maps[np.where(maps[:,2]==idx)]
            if len(parts) > natom: raise ValueError("Too much maps.")
            if len(np.unique(parts[:,0]))==natom and len(np.unique(parts[:,1])) == natom:
                    return parts[np.argsort(parts[:,0])][:,1].tolist()
                    
        return None

    def get_operation(self, symbol):
        maps = {'2_100':'2+100', '2_010':'2+010','m_100':'m+100','m_010':'m+010'}
        if symbol in maps:
            if self.group == 'layer' and (30 <= self.arithmetic_number <= 45):
                symbol = maps[symbol] 
            elif self.group == 'layer' and ( 38 < self.arithmetic_number <=58):
                symbol = maps[symbol] 
        return Symmetry_operation[symbol]

