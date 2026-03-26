import numpy as np
from jamip.structure import Structure, write
from .bonding import Bonding, BondCore
from collections import defaultdict, namedtuple, deque
import pandas as pd
import spglib
import numpy as np

# initialize element dict
from jamip.structure.symmetry import EqualTools
from jamip.structure.elementInfo import ElementData
from typing import NamedTuple
pauling = ElementData['X'].to_dict()
covalent_radius = ElementData['covalent_radius'].to_dict()

LayerSite = namedtuple('LayerSite', ['index','symop','shift'])
'''
LayerShift = namedtuple('LayerShift', ['index','symop','shift','spacing','spacing_in_cartesian','distance_area_rate'])
FullLayerShift = namedtuple('FullLayerShift', ['index','symop','shift','spacing','spacing_in_cartesian','distance',
                            'distance_area_rate','distance_cov_rate','ratetype','charge1','charge2',
                            'specie1','specie2', 'coordination_number1','coordination_number2',
                            'specie_s1','charge_s1','spacing_s1','distance_s1','distance_area_rate_s1','distance_cov_rate_s1',
                            'specie_s2','charge_s2','spacing_s2','distance_s2','distance_area_rate_s2','distance_cov_rate_s2'])
'''
Bondtype = namedtuple('Bondtype', ['key','rcov','meandist','r_r','coordnum','pauling','score'])
UnitGraph = namedtuple('UnitGraph', ['graph', 'vistted', 'ranks'])

class SymmetryError(Exception):
    pass

class AtomsError(Exception):
    pass


class LayerShift(NamedTuple):

    index: tuple
    operation: str
    shift: list
    spacing: float
    area_distance: float
    distance: float
    specie1: str
    specie2: str
    a: float
    c: float

    @property
    def spacing_in_cartesian(self):
        return self.spacing 

    @property
    def spacing_in_fraction(self):
        return self.spacing / self.c

    @property
    def convalent_radius_sum(self):
        return covalent_radius[self.specie1] + covalent_radius[self.specie2]

    @property
    def distance_area_rate(self):
        return self.area_distance / self.a

    @property
    def distance_cov_rate(self):
        return self.distance / self.convalent_radius_sum

class FullLayerShift(NamedTuple):

    index: tuple
    operation: str
    shift: list
    spacing: float
    area_distance: float
    distance: float
    specie1: str
    specie2: str
    a: float
    c: float
    charge1: int
    charge2: int
    coordination_number1: int
    coordination_number2: int
    u2: tuple
    d2: tuple

    @property
    def spacing_in_cartesian(self):
        return self.spacing 

    @property
    def spacing_in_fraction(self):
        return self.spacing / self.c

    @property
    def convalent_radius_sum(self):
        return covalent_radius[self.specie1] + covalent_radius[self.specie2]

    @property
    def distance_area_rate(self):
        return self.area_distance / self.a

    @property
    def distance_cov_rate(self):
        return self.distance / self.convalent_radius_sum


class LayerShiftCreator:

    def __init__(self, uatom, datom, spacing, distance, allvectors, indices, a, c):
        self.uatom = uatom
        self.datom = datom
        self.allvectors = allvectors
        self.indices = indices
        self.a = a
        self.c = c
        self.area_distance = distance
        self.spacing = spacing
        '''
        self.distance_area_rate = self.min_distance / a
        self.distance_cov_rate = self.distance / self.distance_cov
        self.spacing_in_fraction = self.spacing / c
        '''

    @property
    def convalent_radius_sum(self):
        return covalent_radius[self.uatom.specie] + covalent_radius[self.datom.specie]

    @property
    def distance(self):
        return np.sqrt(self.area_distance**2 + self.spacing**2)
    
    def reset_spacing(self, spacing:float=2):
        # 如果self.convalent_radius_sum大于self.area_distance，则计算cov_spacing
        if self.convalent_radius_sum > self.area_distance:
            cov_spacing = np.sqrt(self.convalent_radius_sum**2 - self.area_distance**2)
            # 将cov_spacing和spacing中较大的值赋给spacing_in_cartesian
            spacing_in_cartesian = max(cov_spacing, spacing)
        # 将spacing赋值给self.spacing
        self.spacing = spacing

    def get_unique_indices(self):
        # 获取唯一的索引
        indices = self.indices
        vectors = []
        linkset = np.unique(indices[:,1:], axis=0)
        # 遍历linkset中的每一对索引
        for i,j in linkset:
            # 获取与这对索引对应的原子ID
            ids = indices[:,0][np.where((indices[:,1]==i) & (indices[:,2]==j))]
            # 获取ID为4的原子对应的向量
            gid = ids[np.argmin(abs(ids-4))]
            vectors.append(self.allvectors[gid, i, j])
        return vectors

    def get_solid_angle(self):
        # 计算固角
        from .bonding import solid_angle
        # 获取索引1和索引2的唯一值及其出现次数
        idx1,num1 = np.unique(self.indices[:,1],return_counts=True)
        idx2,num2 = np.unique(self.indices[:,2],return_counts=True)

        angles = []
        # 遍历索引1的唯一值
        for idx in idx1:
            # 获取与索引1对应的索引
            indices = self.indices[np.where(self.indices[:,1]==idx)] 
            vectors = []
            # 遍历索引
            for i,j,k in indices:
                # 获取对应的向量
                vectors.append(self.vectors[i,j,k])
            # 计算固角
            center = np.array([0,0,0])
            angle = solid_angle(center, vectors)
            angles.append(angle)
        solid_angle_1 = np.mean(angles)
            
        angles = []
        # 遍历索引2的唯一值
        for idx in idx2:
            # 获取与索引2对应的索引
            indices = self.indices[np.where(self.indices[:,2]==idx)] 
            vectors = []
            # 遍历索引
            for i,j,k in indices:
                # 获取对应的向量
                vectors.append(self.vectors[i,j,k])
            # 计算固角
            center = np.array([0,0,0])
            angle = solid_angle(center, vectors*-1)
            angles.append(angle)
        solid_angle_2 = np.mean(angles)
        
        return solid_angle_1, solid_angle_2
        
    def check_symmetry(self):
        # 检查对称性
        idx1,num1 = np.unique(self.indices[:,1],return_counts=True)
        idx2,num2 = np.unique(self.indices[:,2],return_counts=True)
        # 如果索引1和索引2的出现次数都为1，则返回True，否则返回False
        return (len(set(num1)) == 1 and len(set(num2)) == 1)
 
    def get_coordination_number(self):
        # 获取配位数
        idx1,num1 = np.unique(self.indices[:,1],return_counts=True)
        idx2,num2 = np.unique(self.indices[:,2],return_counts=True)
        return num1, num2

    def minicopy(self, key, operation):
        # 获取唯一的索引
        vectors = self.get_unique_indices()
        # 创建LayerShift对象
        base = [key, operation, vectors, self.spacing, self.area_distance, self.distance, 
                self.uatom.specie, self.datom.specie, self.a, self.c]
        return LayerShift(*base)

    def fullcopy(self, key, operation, d2=None, u2=None):
        # 获取唯一的索引
        vectors = self.get_unique_indices()
        # 创建FullLayerShift对象
        base = [key, operation, vectors, self.spacing, self.area_distance, self.distance, 
                self.uatom.specie, self.datom.specie, self.a, self.c]
        # 获取配位数
        coords = self.get_coordination_number()
        # 创建site列表
        site = [self.uatom.charge, self.datom.charge, coords[0], coords[1]]
        # 如果u2不为None，则创建u2的fullcopy
        if u2 != None: u2 = u2.fullcopy(key, operation)
        # 如果d2不为None，则创建d2的fullcopy
        if d2 != None: d2 = d2.fullcopy(key, operation)
        # 创建addition列表
        addition = [u2, d2]

        # 创建FullLayerShift对象
        params = base + site + addition
        return FullLayerShift(*params)

class Unit:

    def __init__(self, structure:Structure, raw_structure:Structure, atom_indices:np.ndarray):
        self.unitcell = structure
        self.raw_structure = raw_structure
        self.atom_indices = np.array(atom_indices)
        self._stdcell = None
        self._dataset = None
        self._unique_cell = None
        self._atomdf = None
        self._mainaxis = None
        self.valences = None

    def set_valence(self, dataset):
        self.valences = {i:dataset[i] for i in self.unitcell.species_of_elements}

    @property
    def stdcell(self):
        if self._stdcell == None:
            self.get_stdcell()
        return self._stdcell

    @property
    def dataset(self):
        if self._dataset == None:
            self.get_stdcell()
        return self._dataset
    
    @property
    def mainaxis(self):
        if self._mainaxis == None:
            self.get_stdcell()
        return self._mainaxis

    @classmethod
    def from_structure(cls, structure, atom_indices=None, pbc_vectors=None):

        if atom_indices is None:
            atom_indices = np.arange(len(structure))
        else:
            atom_indices = np.array(list(atom_indices))

        # create unit structure
        lattice = structure.lattice
        positions = structure.get_positions()[atom_indices]
        elements = structure.get_elements()[atom_indices]
        
        if not (pbc_vectors is None):
            positions += np.array(pbc_vectors)

        unitcell = Structure.from_cell((lattice, positions, elements))
        return cls(unitcell, structure, atom_indices)

    @classmethod
    def merge(cls, units):
        # 获取第一个单元的原始结构
        structure = units[0].raw_structure
        # 获取第一个单元的晶格
        lattice = structure.lattice

        # 初始化原子索引、位置和元素列表
        atom_indices = []
        positions = []
        elements = []
        # 遍历所有单元
        for unit in units:
            # 将每个单元的原子索引添加到原子索引列表中
            atom_indices.extend(unit.atom_indices)
            # 将每个单元的位置添加到位置列表中
            positions.extend(unit.unitcell.get_positions())
            # 将每个单元的元素添加到元素列表中
            elements.extend(unit.unitcell.get_elements())

        # 根据晶格、位置和元素创建一个新的结构
        unitcell = Structure.from_cell((lattice, positions, elements))
        # 创建一个新的对象，传入新的结构、原始结构和原子索引
        obj = cls(unitcell, structure, atom_indices)
        # 设置新对象的维度为所有单元的最大维度
        obj.dim = max([unit.dim for unit in units])
        # 返回新对象
        return obj

    def get_stdcell(self):
        """Get the standard cell of the unit cell using spglib."""

        from spglib import spglib

        # get stdcell by spglib
        cell = self.unitcell.to_cell()
        dataset = spglib.get_symmetry_layerdataset(cell, symprec=1e-1)
        if dataset is None:
            dataset = spglib.get_symmetry_layerdataset(cell, symprec=1e-2)
        self._stdcell = redefine_by_spglib(dataset)
        self._dataset = dataset
        # write(self.unitcell, 'unitcell.vasp')
        # write(self._stdcell, 'stdcell.vasp')

        # check structure symmetry
        assert abs(self.unitcell.volume - self._stdcell.volume)/self._stdcell.volume < 1e-2, \
            "stdcell volume changed, %f != %f" %(self.unitcell.volume, self._stdcell.volume)
        # self.check_pointgroup_symmetry()
        # self.check_unique_pointgroup_symmetry()

        # get_unitcell_symmetry_operations  
        # equal = EqualTools.from_layergroup(self.dataset['number'])
        # equal.set_lattice_symmetry(self.stdcell.lattice)
        # symmetry, shift = equal.is_structure_equal(self.unitcell, self.stdcell, all_operations=True)
        # assert symmetry != None, "unitcell structure not equal to stdcell"

        # from jamip.modeling.structureAnalyser import StrcutureMath
        # sm = StrcutureMath(self.unitcell, self.stdcell, attempt_supercell=True, allow_subset=True)
        # assert sm.fit() == True, "unitcell structure not equal to stdcell"

    @classmethod
    def get_pbc_vectors(self, positions, axis_index=2):
        """
        将位点移动到原子中心，适用于未考虑坐标连续的结构
        """
        coords = np.array(positions[:,axis_index])
        sort_coords = np.sort(coords)
        pbc_vectors = np.zeros((len(coords), 3), dtype=int)

        atom_seps = np.append(np.diff(sort_coords), 1+sort_coords[0]-sort_coords[-1])
        itop = np.argmax(atom_seps)
        if itop < len(coords)-1:
            cmin = sort_coords[itop+1]
            pbc_vectors[:,2] = np.where((coords-cmin)>-1e-8, 0, -1)
        return pbc_vectors

    def reset_pbc_vectors(self, axis_index=2):

        lattice, positions, elements = self.unitcell.to_cell()
        self.pbc_vectors = self.get_pbc_vectors(positions, axis_index)
        positions -= self.pbc_vectors
        self.unitcell = Structure.from_cell((lattice, positions, elements))

        return self

    @property
    def atomdf(self):
        if self._atomdf is None:
            self.get_atoms_data()
        return self._atomdf

    def get_atoms_data(self, axis_index=2, tol=0.1, pbc=False):
        """
        获得结构在中间位置的原子索引
        """
        from scipy.cluster.hierarchy import fcluster, single

        positions = self.unitcell.get_positions(type='direct')
        if pbc:
            # 将结构中的z方向最大间隔移动到晶胞两侧 %
            pbc_vectors = self.get_pbc_vectors(positions, axis_index)
            positions -= pbc_vectors

        elements = self.unitcell.get_elements(type='symbol')
        coords = positions[:,axis_index]
        charges = [self.valences[i] for i in elements]
        indices = np.arange(len(elements))

        df = pd.DataFrame({'specie':elements, 'index': indices, 'charge':charges, 
                           #'site_symmetry_symbols': self.dataset['site_symmetry_symbols'],
                           'coord':coords, 'wyckoff':self.dataset['wyckoffs']})
        
        ds = coords * np.linalg.norm(self.unitcell.lattice[axis_index])
        if len(ds) > 1:
            Z = single(ds[:,None])
            fc = fcluster(Z, t=tol, criterion='distance')
            u, indices = np.unique(fc, return_inverse=True)
         
            # 基于实空间z坐标进行聚类
            cartesians = np.zeros_like(coords)
            for idx,grp in enumerate(u):
                h = np.around(np.mean(ds[np.where(indices==idx)]), 4)
                cartesians[np.where(indices==idx)] = h
            df['cartesian'] = cartesians
        else:
            df['cartesian'] = np.around(ds, 4)
            

        self._atomdf = df

    def get_edge_indices(self, charge=True, **kwargs):
        # 基于z坐标 + 多重度 + 元素种类排序，选择最早出现的结构单元
        # if charge: 保存表面和次表面原子索引, 表面原子价态与满足价态要求
        # if not charge: 仅保存表面原子索引

        df = self.atomdf
        #df = pd.concat([df,df])
        #dmax = df['cartesian'].max() - tol
        #dmin = df['cartesian'].min() + tol
        tcharge = df['charge'].sum()
        results = {}

        if charge is False:
            for row in df.sort_values(by=['cartesian','wyckoff','specie'], ascending=[False,False,True]).itertuples():
                results['u1'] = row
                break
            for row in df.sort_values(by=['cartesian','wyckoff','specie'], ascending=[True,False,True]).itertuples():
                results['d1'] = row
                break

        else:
            # upper %
            edge = None
            keys = []
            for row in df.sort_values(by=['cartesian','wyckoff','specie'], ascending=[False,False,True]).itertuples():
                for key in keys:
                    if key.specie == row.specie and key.cartesian == row.cartesian and key.wyckoff == row.wyckoff:
                        break
                else:
                    keys.append(row)
                    if edge is None and row.charge*tcharge>=0:
                        edge = len(keys)-1
                if len(keys) >= 2 and edge != None:
                    break
                
            results['u1'] = keys[edge]
            results['u2'] = None
            for i in range(len(keys)):
                if i != edge:
                    results['u2'] = keys[i] 
                    break
                 
            # lower %
            edge = None
            keys = []
            for row in df.sort_values(by=['cartesian','wyckoff','specie'], ascending=[True,False,True]).itertuples():
                for key in keys:
                    if key.specie == row.specie and key.cartesian == row.cartesian and key.wyckoff == row.wyckoff:
                        break
                else:
                    keys.append(row)
                    if edge is None and row.charge*tcharge>=0:
                        edge = len(keys)-1
                if len(keys) >= 2 and edge != None:
                    break
                
            results['d1'] = keys[edge]
            results['d2'] = None
            for i in range(len(keys)):
                if i != edge:
                    results['d2'] = keys[i] 
                    break

        self.sites = results
        return self.sites

    def get_indices_by_site(self, site):
        # TODO: for low symmetry unit (For example As2O6), same site with different wyckoff
        df = self.atomdf
        #pdf = df[(df['specie']==site.specie) & (df['cartesian']==site.cartesian) & (df['wyckoff']==site.wyckoff)]
        pdf = df[(df['specie']==site.specie) & (df['cartesian']==site.cartesian)]
        return pdf

    def get_positions_by_key(self, key):
        assert key in ('u1','u2','d1','d2'), "unknown key"
        site = self.sites[key]
        indices = self.get_indices_by_site(site)['index'].values
        positions = self.unitcell.get_positions()
        return positions[indices]

    def get_layer_site(self, idx, reference=None, ignore_symop=False):
        if ignore_symop is True:
            site = LayerSite(idx, "none", np.zeros(3))
        else:
            stdcell = self.stdcell if reference is None else reference

            equal = EqualTools.from_layergroup(self.dataset['number'])
            #print(equal.arithmetic_number, equal.dataset['generators'])
            equal.set_lattice_symmetry(stdcell.lattice)
            #print(stdcell.lattice_parameters, equal.lattice_symmetry.arithmetic_number)
            symop, shift = equal.is_structure_equal(self.unitcell, stdcell, all_operations=True)
            if symop != None:
                site = LayerSite(idx, symop, shift)
            else:
                print('get_layer_site find failed!')
                site = LayerSite(idx, "none", np.zeros(3))
            '''
            print('+++++++++++++++++++++++')
            print(symop, shift)
            print(stdcell.get_positions())
            print('')
            #print(self.apply_layer_site(site, apply_shift=True).get_positions())
            print(self.unitcell.get_positions())
            print('')
            print(self.apply_layer_site(site, stdcell, apply_shift=True).get_positions())
            print('+++++++++++++++++++++++')
            '''

        return site

    def apply_layer_site(self, site, structure=None, apply_shift=False):

        if structure is None: 
            structure = self.stdcell
        lattice, positions, elements = structure.to_cell()
        equal = EqualTools.from_layergroup(self.dataset['number'])
        if isinstance(site, str):
            operation = equal.get_operation(site).T
        else:
            operation = equal.get_operation(site.symop).T
        shift = site.shift if apply_shift else np.zeros(3) 
        #print('before operation')
        #print(positions)
        positions = (positions + shift) @ operation 
        #print('after operation')
        #print(positions)

        return Structure.from_cell((lattice, positions, elements))

    @property
    def unique_cells(self):
        """
        生成当前晶格下所有不具有平移对称性的结构单元
        返回字典 {对称性符号：结构}
        """
        equal = EqualTools.from_layergroup(self.dataset['number'])
        cells = equal.get_unique_structure(self.unitcell)

        units = {'1': self}
        #self.get_atoms_data(axis_index=2, tol=0.1, pbc=False)
        #self.get_edge_indices(charge=True, tol=0.3)

        for symbol,cell in cells.items():
            unit = Unit.from_structure(cell)
            unit.set_valence(self.valences)
            unit.get_stdcell()
            unit.get_atoms_data(axis_index=2, tol=0.1, pbc=False)
            unit.get_edge_indices(charge=True, tol=0.3)
            units[symbol] = unit
        return units

    def check_pointgroup_symmetry(self):
        equal = EqualTools.from_layergroup(self.dataset['number'])
        #print(self.dataset['number'], equal.operations)
        for symbol in equal.operations: 
            # symmetry, shift = equal.is_structure_equal(self.unitcell, self.stdcell)
            # if symmetry != None:
            #     raise ValueError(f"Error unique pointgroup symmetry: {symbol}")
            cell = self.apply_layer_site(symbol)
            # print(cell)
            symmetry, shift = equal.is_structure_equal(cell, self.stdcell)
            # print('xxxxxxxxxx',symbol, symmetry, shift)
            assert symmetry == '1', "Error pointgroup symmetry."
            if symmetry is None:
                write(cell, 'd1.vasp')
                write(self.stdcell, 'd2.vasp')
                raise

    def check_unique_pointgroup_symmetry(self):
        import warnings
        equal = EqualTools.from_layergroup(self.dataset['number'])
        if len(equal.unique_generators) == 0: return True
        operations = []
        for symbol in equal.unique_generators:
            cell = self.apply_layer_site(symbol)
            symmetry, shift = equal.is_structure_equal(cell, self.stdcell)
            if symmetry != None:
                operations.append(symbol)
        if len(operations) != 0: 
            warnings.warn(f"Error unique pointgroup symmetry: {operations}")
            # if self.dataset['number'] == 47 and symbol == '4+001': continue
            # #elif self.dataset['number'] == 23 and symbol == '4+001': continue
            # elif self.dataset['number'] == 48 and symbol == '4+001': continue
            # elif self.dataset['number'] == 1 and symbol == '-1': continue
            # elif self.dataset['number'] == 10 and symbol == '-1': continue
            # elif self.dataset['number'] == 55 and symbol == '-1': continue
            # elif self.dataset['number'] == 57 and symbol == '-1': continue
            # elif self.dataset['number'] == 58 and symbol == '-1': continue
            # elif self.dataset['number'] == 69 and symbol == '-1': continue
            # elif self.dataset['number'] == 67 and symbol == '-1': continue
            # elif self.dataset['number'] == 78 and symbol == '-1': continue
            # elif self.dataset['number'] == 72 and symbol == '2_001': continue

    @property
    def spacing_in_cartesian(self): 
        """
        求结构单元的层厚度
        """
        return self.atomdf['cartesian'].max() - self.atomdf['cartesian'].min()

    @property
    def thickness(self): 
        """
        求结构单元的层厚度
        """
        return self.spacing_in_cartesian

    @property
    def area(self):
        lattice = self.unitcell.lattice
        return np.linalg.norm(np.cross(lattice[0], lattice[1]))

    @property
    def c(self):
        lattice = self.unitcell.lattice
        return np.linalg.norm(lattice[2])

    @property
    def natom(self):
        return len(self.unitcell)

    @property
    def nsite(self):
        symbols, numbers = np.unique(self.dataset['equivalent_atoms'], return_counts=True)
        return len(numbers)

    @property
    def total_charge(self):
        return self.atomdf['charge'].sum()

    @property
    def mean_multiplicity(self):
        return round(self.nsite / self.natom , 4)

    @property
    def unique_sites(self):
        data = {}
        for key,grp in self.atomdf.groupby('symmetry'):
            data[key] = len(grp)
        return data

    @property
    def unique_formula(self):
        data = ''
        for key,grp in self.atomdf.groupby(['specie','symmetry']):
            s,m = key
            data += '%s%d' %(s, len(grp))
        return data
    
    @property
    def specie_feature(self):
        """
        columns: charge, ratom, n, mass, rcov, ea, x, ie
        """
        return [self.total_charge] + list(get_specie_features(self.unitcell.species_of_elements))

    @property
    def surface_specie_feature(self):
        """
        columns: charge, ratom, n, mass, rcov, ea, x, ie
        """
        specie = self.sites['d1'].specie
        charge = self.valences[specie]
        return [charge] + list(get_specie_features([specie]))

    @property
    def structure_feature(self):
        """
        columns: min_dist, avg_dist, avg_coordination_number, thickness
        """
        return list(get_bonding_descriptor(self.unitcell)) + [self.spacing_in_cartesian]

    @property
    def lattice_symmetry(self):
        return EqualTools.from_lattice(self.stdcell.lattice).arithmetic_number

    def __getstate__(self):
        self._atomdf = None
        self.sites = None
        state = self.__dict__.copy()
        return state
        
    def __eq__(self, other):

        if isinstance(other, Unit):
            s1 = self.stdcell
            s2 = other.stdcell
            d1 = bool(sorted(s1.get_elements()) == sorted(s2.get_elements()))
            # 允许空间群不一致，但是这里没有用子群
            #d2 = self.dataset['number'] == other.dataset['number']
            d2 = True
            #d2 = bool(self.lattice_symmetry == other.lattice_symmetry)
            # 可以暴力验证两个结构单元的原子坐标可以重叠，但是贵
            # equal = EqualTools.from_layergroup(self.dataset['number'])
            # equal.set_lattice_symmetry(s1.lattice)
            # d3 = equal.is_structure_equal(s1, s2, all_operations=True)[0]
            if d1 and d2:
                return True
        elif other is None:
            return False
        else:
            raise ValueError("unsupport type.")

        return False 

    def __str__(self):
        if self._stdcell == None: 
            self.get_stdcell()
        string = 'JAMIP UNIT\n'
        string += 'Formula: %s\n' %self.unitcell.get_formula()
        string += 'LayerGroup: %d\n' %self.dataset['number']
        string += 'Atom indices: %s\n' %list(self.atom_indices)
        string += 'Site symmetry: %s\n' %list(self.dataset['site_symmetry_symbols'])
        string += 'Wyckoffs: %s\n' %list(self.dataset['wyckoffs'])
        string += 'Center: %.4f\n' %self.unitcell.get_positions()[:,2].mean() 
        return string

    def __repr__(self):
        return self.unitcell.get_formula()

    def to_dict(self):
        """ save necessary information in dict """
        """ formula, layergroup, valence, usite, dsite """
        """ natom, species, sites """
        values, numbers = np.unique(self.dataset['site_symmetry_symbols'], return_counts=True)

        data = {'formula': self.unitcell.get_formula(sort=True),
                'layergroup': self.dataset['number'],
                'valence': self.valences,
                'total_charge': self.total_charge,
                'usite': self.sites['u1'].specie,
                'dsite': self.sites['d1'].specie,
                'natom': len(self.unitcell),
                'species': list(self.unitcell.species_of_elements),
                'sites': dict(zip(values, numbers)),
                'nequal': len(np.unique(self.dataset['equivalent_atoms'])),
                }
        return data

class DimensionAnalysis:

    debug = False

    def __init__(self, structure):

        self.structure = structure
        self.valences = None
        self.cutoff = 1.1
        self.cache = None
        self._initialize = False

    @classmethod
    def from_structure(cls, structure):
        lattice, positions, elements = structure.to_cell()
        positions = positions - np.floor(positions)
        s = Structure.from_cell((lattice, positions, elements))
        return cls(s)

    @classmethod
    def from_stdcell(cls, structure, symprec=0.1):
        stdcell = spglib.refine_cell(structure.to_cell(), symprec=symprec)
        s = Structure.from_cell(stdcell)
        return cls(s)

    def run(self):
        if not self._initialize:
            self.set_valence()
            self.set_bonding()

    def set_valence(self, value=None):
        if value != None:
            if isinstance(self.valences, dict):
                self.valences.update(value)
            else:
                self.valences = value
        else:
            self.valences = self.structure.get_species()

    def set_bonding(self, species=None, ions=None, **kwargs):
        """
        初始化成键信息，根据键长和元素类型进行分组
        """
        bonding = Bonding(self.structure, method='rdf', cutoff=4).data
        if not (ions is None):
            bonding.remove_bond_by_units(ions)
        groups = bonding.classify(tol=0.1)
        bondtypes = []

        pairs = get_pairs(species) if not (species is None) else []

        for key, value in groups.items():
            a,b,c = key
            if len(pairs) and [a,b] not in pairs:
                continue
            # mean distances 
            rmean = np.mean(bonding.distances[value[:,2]])
            # covalent_radius_distances
            rcov = round(covalent_radius[a] + covalent_radius[b], 4)
            # electronegativity
            p = max( np.abs(pauling[a]-pauling[b]), np.sqrt(pauling[a] * pauling[b]) )

            # connectivity weight 1.1 -> 0.95
            values, counts = np.unique(value[:,0], return_counts=True)

            # 从连通性的角度，希望保留高配位数的成键. 配位数高会提高键长，这里也算一点补偿
            cw = 0.9 + 0.2/ np.clip(np.mean(counts), 1, 4) #  1~4 -> 1.05~0.95

            # oxistates
            ox = 1
            if self.valences[a] > 0 and self.valences[b] > 0:
                ox = 0.2
            elif self.valences[a] <= 0 and self.valences[b] <= 0:
                ox = 0.8
            # final score
            score = round(-np.log2(cw*rmean/rcov)*rmean + p*ox, 4)
            #score = round(-np.log2(cw*rmean/rcov) + p*ox, 4)
            r_r = round(rmean/rcov, 4)
            #scores[key] = (rmean/rcov*(1+rminconv/50), score)
            #scores[key] = (rmean/rcov*(0.9+rmean/20), score)
            #scores[key] = (round(rmean/rcov,4), round(score,4))

            # 键长得分小于1或键长比大于1.3的基本不会成键
            if r_r < 1.3 and score > 1: 
                bt = Bondtype(key,rcov,rmean,r_r,round(np.mean(counts),4),p,score)
                bondtypes.append(bt)

            #self.debug = True
            if self.debug:
                print(f'{str(key)}\t', 'score:%6.3f' %score, 'rfrac:%6.3f' %(rmean/rcov), 'r:%6.3f' %rmean, 
                      'Pau:%6.3f' %p, 'oxi:%6.3f' %ox, 'cw:%4.1f' %cw)
            #print(key, -np.log2(rmean/rcov)*2, -np.log2(rmean/rcov)*np.sqrt(rmean), np.sqrt(rmean))
             
        self.bonding = bonding
        self.groups = groups
        self.bondtypes = bondtypes

    def get_boundary(self):
        """
        搜索结构中是否有位于顶点的原子，如果有，返回其序数
        """
        positions = self.structure.get_positions(type='direct')
        vertex = np.where(positions==False)[0]
        maps = {i:positions[i] for i in vertex}
        return maps

    def search_cutoff(self):
        from copy import deepcopy
        from scipy.cluster.hierarchy import DisjointSet

        graph = DisjointSet(np.arange(len(self.structure)))
        shift_indices = self.bonding.shift_indices
        boundary = self.get_boundary()

        bonds = []              # 用于构建外键维度信息，形式为 [(a,b,shift), (a,b,shift), ...]
        bond_indices = []       # 用于存储全部的内键，形式为 [i1,i2,i3,...] (i1,i2,i3,...表示成键在bonding类中的序号)
        outbond_indices = []    # 用于存储全部的外键，形式为 [i1,i2,i3,...] (i1,i2,i3,...表示成键在bonding类中的序号)
        actives = []            # 用于存储已激活的成键索引
        cache = None            # 缓存上一次成键分析的数据类
        distances = []
        bvstep = 0

        # 按得分顺序遍历成键，直至所有原子连接；建立新的成键类
        scores = [i.score for i in self.bondtypes]
        indices = np.argsort(scores)[::-1]
        for idx in indices:

            bts = self.bondtypes[idx]
            if self.debug:  print(bts)

            active_axis = None
            # 添加成键并进行维度识别
            matrix = self.groups[bts.key]
            for a,b,i,j in matrix:
                # 晶胞内成键
                if shift_indices[i]==0:    
                    graph.merge(a,b)

                # 晶胞外成键，但包含边界时保留其中的一半: a=border, shift>0
                else:
                    shift = self.bonding.shifts[shift_indices[i]] * j
                    ainb = (a in boundary and sum(abs(boundary[a]*shift))<1e-8) # a in boundary
                    binb = (b in boundary and sum(abs(boundary[b]*shift))<1e-8) # b in boundary
                    axis = shift[np.flatnonzero(shift)[0]]
                    if active_axis == None and (ainb or binb):
                        active_axis = -axis if ainb else axis

                    if (ainb and axis==active_axis) or (binb and axis == -active_axis):
                        graph.merge(a,b)
                    else:
                        bonds.append((a,b,tuple(shift)))

            #print(bonds)
            parents = [graph[i] for i in graph]
            adjacency = build_adjacency_list(parents, bonds)
            visited, ranks = traverse_component_graphs(adjacency)
            merged, visited, ranks = merge_mutual_visits(visited, ranks, graph)
            dplus = [bts.meandist] * len(matrix)

            # 搜索截至条件
            # 断键需要满足两个条件
            # 1. 共价半径比值 > 1.1
            # 2. 键长大于平均键长
            # 3. 键长评分出现足够大的跃变
            value, number = np.unique(parents, return_counts=True)
            #condition = (max(number) == len(parents)/2 and min(number) > 1 and max(number) > min(number))
            condition = False #(max(number) == len(parents)/2 and min(number) > 1 and max(number) > min(number))
            if condition or (max(ranks.values()) == 3 and (graph.n_subsets==1 or bts.r_r > 1.25)):
            #if max(ranks.values()) == 3 and (graph.n_subsets==1 or bts.r_r > 1.25):

                # 修正项，为成键加权近邻距离
                if len(actives) > 0:
                    s0, s1, s2 = scores[actives[0]], scores[actives[-1]], bts.score
                    bvstep = ((s0-s2)**2 - (s1-s2)**2) / s0

                # 如果终止键长<1.1，则断键无效, 当前的键将添加到成键中
                connect = np.sum(distances) / len(self.structure) / np.power(self.structure.volume, 1/3)
                connect = (np.sqrt(connect) + bts.r_r**2) / 2 
#                connect = (np.sqrt(connect) * bts.r_r**2) 
                #print(np.sum(distances)/ len(self.structure) / self.structure.volume)
                #print(bts.r_r, d_d, bvstep)
                # bvp = (bvs[0]-value[1])/bvs[0] if len(bvs) > 0 else 1
                #if value[0] < 1.1 and bvstep < 0.1:
                #if value[0] + bvstep*bvp < 1.05:
                #if bts.r_r < 1.05 and d_d < 1.05:
                if self.debug: 
                    d_d = bts.meandist / np.mean(distances) 
                    print('R:',bts.r_r,'\n','bvs:',bvstep,'\n','dd:',d_d,'\n','conn',connect,'\n')
                    print(f'Condition 1: {bts.r_r} + {bvstep} < {self.cutoff}')
                    print(f'Condition 2: {connect} < {self.cutoff}')
                    print(f'Condition 0: {condition} = number: {number}')
                if bts.r_r + bvstep < self.cutoff and connect < self.cutoff:
                    cache = deepcopy(graph), visited, ranks
                    bond_indices.extend(matrix[:,2])
                    # print('end bond search. save last')
                #else:
                    #bond_cutoff = bts
                    # print('end bond search. cutoff last')
                break

            cache = deepcopy(graph), visited, ranks
            actives.append(idx)
            bond_indices.extend(matrix[:,2])
            distances += dplus #[bts.meandist] * len(matrix)

        if len(actives) > 0:
            self.bond_actives = [self.bondtypes[i].key for i in actives]
            self.max_rcov = np.max([self.bondtypes[i].r_r for i in actives]) 
            self.max_length = np.max([self.bondtypes[i].meandist for i in actives])
        else:
            self.bond_actives = None
            self.max_rcov = self.max_length = None
        # graph: 晶格内连接；all_units: 晶格内组的跨晶格连接；
        # ranks: 晶格内组的维度
        if cache == None: 
            dim = 3 if len(bonds) > 0 else 0
        else:
            ranks = cache[-1]
            dim = max(ranks.values())
            self.cache = cache
        return dim

    @property
    def active_bond_matrix(self):

        bond_matrix = []
        for key in self.bond_actives:
            bond_matrix.append(self.groups[key])

        return np.concatenate(bond_matrix, axis=0)

    def get_nunit(self):
        graph, visited, ranks = self.cache
        allkeys = set()
        num = 0
        for key,value in visited.items():
            indices = set([v[0] for v in value])
            assert key in indices,  '不期待的报错 - get_nunit'
            if key not in allkeys:
                allkeys.update(indices)
                num += 1        
        return num

    def get_units_with_composition(self, type='formula'):
        from jamip.structure.atom import Composition

        graph, visited, ranks = self.cache
        elements = self.structure.get_elements(type='symbol')

        comps = []
        usedset = set()
        for key,value in visited.items():
            # 获取该unit中包含的原子索引
            indices = set([v[0] for v in value])
            atom_indices = []
            for i in graph:
                if graph[i] in indices:
                    atom_indices.append(i)

            # Verify that the current structural unit has been considered
            atomset = set(atom_indices)
            if len(atomset & usedset):
                raise ValueError('atoms group error!')

            c = Composition.from_elements([elements[i] for i in atomset] )
            c.dim = ranks[key]
            usedset.update(atomset)
            comps.append(c)

        return comps

    def get_units(self, merge=True):
        '''
        仅基于维度识别结果建立Unit类，高维结构单元会合并被包含的低维结构单元
        步骤1. 从graph建立消除PBC条件的结构单元(0.9,0.1) -> (0.9,1.1)
        步骤2. 基于结构单元的边界排序、合并低维结构单元
        '''
        from scipy.cluster.hierarchy import DisjointSet

        graph, visited, ranks = self.cache

        # Divide atoms into different structural units base on Graph, with each atom being utilized only once.
        unitcells = []
        usedset = set()
        for key,value in visited.items():
            # 获取该unit中包含的原子索引
            indices = set([v[0] for v in value])
            # 获取该unit中包含的原子的pbc向量
            pbcs = []
            atom_indices = []
            for i in graph:
                if graph[i] in indices:
                    atom_indices.append(i)
                    if graph[i] == key:
                        pbcs.append([0,0,0])
                    else:
                        for j,v in value:
                            if graph[i] == j:
                                pbcs.append(v)
                                break
                        else:
                            raise ValueError('visited error')

            # Verify that the current structural unit has been considered
            atomset = set(atom_indices)
            diffset = atomset - usedset
            if diffset == atomset:
                unit = Unit.from_structure(self.structure, atom_indices, pbcs)
                unit.dim = ranks[key]
                unitcells.append(unit)
                usedset.update(atomset)
            elif len(diffset) > 0:
                raise ValueError('atoms group error!')

        # Order the based on the average z coordinates of the atoms
        centers = [unit.unitcell.get_positions()[:,2].mean() for unit in unitcells]
        graph = DisjointSet(np.arange(len(centers)))
        # Merge units with the same coordinates
        sort_indices = np.argsort(centers)
        for i in range(len(centers)-1):
            j = sort_indices[i]
            k = sort_indices[i+1]
            if (centers[k] - centers[j]) < 1e-4:
                graph.merge(j,k)
        if len(centers) > 2:
            j = sort_indices[0]
            k = sort_indices[-1]
            if abs(centers[j] + 1 - centers[k]) < 1e-4:
                graph.merge(j,k)

        # Update units
        new_units = []
        used_indices = []
        for i in graph:
            if i in used_indices: continue
            indices = []
            for j in graph:
                if graph[i] == graph[j]:
                    indices.append(j)
            used_indices.extend(indices)
            if len(indices) == 1:
                unit = unitcells[indices[0]]
            else:
                unit = Unit.merge([unitcells[i] for i in indices])
            unit.set_valence(self.valences)
            new_units.append(unit)

        # update charge
        # TODO 
        '''
        from .valence import get_best_valence
        charges = [unit.total_charge for unit in new_units]
        chgsum = sum(charges)
        if chgsum != 0:
            print(self.structure.get_formula())
            print(self.valences)
            print(charges)
            print([unit.unitcell.get_formula() for unit in new_units])
            charge = -min(charges) if (chgsum < 0) else -max(charge)
            for unit in new_units:
                if abs(unit.total_charge) < abs(charge):
                    species = unit.unitcell.composition.species
                    numbers = unit.unitcell.composition.numbers
                    valence = get_best_valence(species, numbers, charge)
                    unit.set_valence(valence)
                else:
                    print(abs(unit.total_charge), abs(charge))

            charges = [unit.total_charge for unit in new_units]
            print(charges)
        '''
        self.units = new_units
        return new_units

class UnitAttach:
    
    @classmethod
    def attach_with_shifts(self, units, layersites, **kwargs):
        """
        这种方法完全基于逆运算获得，不经过interface类
        """
        lattice = units[0].stdcell.lattice

        species = []
        coords = []
        for site in layersites:
            cell = units[site.index].apply_layer_site(site, apply_shift=True)
            lat, positions, elements = cell.to_cell()
            species.extend(elements)
            coords.extend(positions)

        return Structure.from_cell((lattice, coords, species))

    @classmethod
    def attach_with_sites(self, units, layersites, layershifts, weights=None, **kwargs):
        """
        units: base units, List(Unit)
        layersites: unit operations, List(namedtuple)
        layershifts: layer bondings, List(BondCore)
        weights: lattice weights, List(float) or None

        这个函数的主要目的是调整层间距并生成结构，流程如下:
        1. 标准结构单元
        2. 结构单元的出现顺序和对称性操作
        3. 结构单元界面匹配，计算平移向量
        """

        #write(units[0].unitcell, 'u1.vasp')
        #write(units[1].unitcell, 'u2.vasp')
          
        def rescale(lattice, a, b, c=None):
            a1,b1,c1 = np.linalg.norm(lattice, axis=1)
            lattice[0] *= a/a1
            lattice[1] *= b/b1
            if c != None:
                lattice[2] *= c/c1
            return lattice

        # 此时晶格是需要重新生成的, 其中xy方向基于权重的缩放，z方向基于直接累加
        # 由于这里处理的都是四方相或六方向，a/b长度相图，不需要对a/b排序
        # 直接提取a1, a2, a' = w1a1 + w2a2, x = x / a1 * a' , y = y / a1 / a'

        # step 1: create lattice
        if weights is None:
            weights = np.ones(len(layersites))/len(layersites)

        va = []
        vb = []
        vc = []
        for site in layersites:
            # TODO 如果stdcell与原结构平移不等价怎么办?
            unit = units[site.index]
            lattice = unit.stdcell.lattice
            va.append(np.linalg.norm(lattice[0]))
            vb.append(np.linalg.norm(lattice[1]))
            vc.append(unit.spacing_in_cartesian)

        for layer in layershifts:
            vc.append(abs(layer.spacing)) # * np.linalg.norm(layer.lattice[2]))

        #print(vc)
        mva = np.sum(np.array(va)*weights)
        mvb = np.sum(np.array(vb)*weights)
        mvc = np.sum(vc)
        std_lattice = rescale(units[0].unitcell.lattice, mva, mvb, mvc)
        #print('Lattice', va, vb, vc)

        # create units with operation
        equal = EqualTools.from_lattice(std_lattice)
        units_before_shift = []
        for site in layersites:
            stdcell = units[site.index].unitcell
            cell = stdcell.to_cell()
            pbc_vectors = Unit.get_pbc_vectors(cell[1], axis_index=2)
            operation = np.linalg.inv(equal.get_operation(site.symop))
            lattice = rescale(cell[0], mva, mvb)
            matrix = get_lattice_transport(lattice, std_lattice)
            positions = (cell[1] - pbc_vectors) @ operation @ matrix
            positions[:,2] -= np.floor(np.min(positions[:,2]))

            # 重建Unit类,以获得上下界面的原子
            unitcell = Structure.from_cell((std_lattice, positions, cell[2]))
            unit = Unit.from_structure(unitcell, np.arange(len(positions)))
            unit.valences = units[site.index].valences
            if matrix[2,2] < 1: 
                unit.reset_pbc_vectors()
            unit.get_stdcell()
            unit.get_atoms_data(axis_index=2, tol=0.3, pbc=False)
            unit.get_edge_indices(charge=True)#, tol=0.1)
            units_before_shift.append(unit)
            #print(site)

        # find shift vector base on layershifts
        layer_shifts = []
        for i,layer in enumerate(layershifts):
            j = i                                    
            k = 0 if i+1 == len(layershifts) else i+1
            unit1 = units_before_shift[j]                # unit in lower layer
            unit2 = units_before_shift[k]                # unit in upper layer
            dsites = unit1.get_positions_by_key('u1')    # sites in lower layer
            usites = unit2.get_positions_by_key('d1')    # sites in upper layer
            #print(unit1.sites['u1'])
            #print(dsites)
            #print(unit1.atomdf)#unitcell.get_positions())
            #print(unit2.atomdf)

            # get bond vectors
            matrix = layer.to_matrix(full=False)
            vectors = layer.get_scaled_vectors(matrix)
            vectors[:,2] = 0.0
            #distance_area_rate = np.linalg.norm(vectors, axis=1)
            #print(i,'vectors',vectors)

            #shift = calculate_translation_with_matching(dsites, usites, vectors)#point_matrix)
            #layer_shifts.append(shift)
            #print(f'coord_shift = {coord_shift.tolist()}')
            #print(f'us = {usites}')

            # sites_in_lower_layer + bond_vectors = sites_in_upper_layer + shift_vectors
            #coord_shift = dsites[None,:,:] + vectors[:,None,:]
            coord_shift = dsites[None,:,:] + vectors[:,None,:]
            #coord_shift = dsites[:,None,:] + vectors[None,:,:]

            unique_shifts = []
            numbers = defaultdict(int)
            for i in coord_shift.reshape(-1,3):
                for n,j in enumerate(unique_shifts):
                    if np.sum(abs(normalized(i-j))) < 1e-2:
                        numbers[n] += 1
                        break
                else:
                    unique_shifts.append(i)
                    numbers[len(unique_shifts)-1] = 1
            unique_shifts = np.array(unique_shifts)
            numbers = [numbers[i] for i in range(len(numbers))]

            if len(unique_shifts) > len(usites):
                #print(unique_shifts)
                indices = np.argsort(numbers)[::-1][:len(usites)]
                unique_shifts = unique_shifts[indices]
                #print("cutoff")

            #print()
            #print(f'unishift = {unique_shifts+np.array([-1/3,1/3,0])}')
            #print(f'usites = {usites}')
            #print(f'numbers = {numbers}')
            #print(len(usites), len(dsites), len(coord_shift))

            # 基于上层位点，求解平移向量
            shift = EqualTools.istranslate(unique_shifts, usites)
            if shift is None:
                '''
                print(layer)
                print("===1===")
                print(coords1)
                print("===2===")
                print(unique_shifts)# - unique_shifts[2])
                print("===3===")
                coords2 = coords2 # *np.array([-1,-1,1])
                print(coords2)# - coords2[2])
                print("get shifts failed.")
                '''
                raise RuntimeError("get shifts failed.")

            #print('shift', shift)
            #print()

            layer_shift = normalized(shift)
            layer_shift[2] += layer.spacing / mvc 
            layer_shifts.append(layer_shift)

            # 验证一下成键距离
            #vectors = normalized(dsites[:,None,:] - usites[None,:,:] - layer_shift)
            #grid = np.mgrid[-1:2, -1:2, 0:1].reshape(3,-1).T # x,3
            #grid_vectors = grid[:,None,None,:] + vectors[None,:,:,:]
            #distances = np.min(np.linalg.norm(grid_vectors @ std_lattice, axis=3), axis=0)
            #print('ds', distances, distance_in_layer)
            #if layer.distance - np.min(distances) > 0.1:
            #    # 原则上应该完全一样，但这里我们只修正距离近的情况
            #    print("distance set error! layer_distance: %.4f, min_distance: %.4f" %(layer.distance, np.min(distances)))
            #    # 如果希望修正，需要重设mvc, 还是交给m3gnet吧

        # 累加每层的平移向量，一个周期的平移应为0
        #print(np.array(layer_shifts))
        layer_shifts_sum = np.roll(np.cumsum(layer_shifts, axis=0),1, axis=0)
        error = False
        # TODO
        if np.sum(np.abs(normalized(layer_shifts_sum[0]))) > 3e-3:
            #print(layer_shifts)
            #print(layer_shifts_sum)
            #print(np.sum(np.abs(normalized(layer_shifts_sum[0]))))
            #raise OSError("PBC condition failed")
            error = True

        for i,site in enumerate(layersites):
            j = i
            k = 0 if i+1 == len(layershifts) else i+1
            d = layer_shifts_sum[k]-layer_shifts_sum[j]
            d[2] -= np.floor(d[2])
#            print(d)

        # 拼接结构单元，需要结构单元的对称和平移操作
        # 在结构上挂载原子归属的结构单元
        species = []
        coords = []
        atoms_from_units = []
        for i,site in enumerate(layersites):
            unit = units_before_shift[i]
            cell = unit.unitcell.to_cell()
            positions = cell[1] + layer_shifts_sum[i] 
            species.extend(cell[2])
            coords.extend(positions)
            atoms_from_units.extend([site.index]*len(cell[2]))
            #print(i,cell[1])
            #print(np.array(positions))
            #print()

        structure = Structure.from_cell((std_lattice, coords, species))
        # update atom_from_units
        atoms_from_units = np.array(atoms_from_units)
        sorted_atoms_from_units = []
        species,numbers,indices = Structure.elements2species(species, return_inverse=True, sort=False)
        for i in range(len(species)):
            sorted_atoms_from_units.extend(atoms_from_units[np.where(indices==i)])
        structure.atoms_from_units = np.array(sorted_atoms_from_units)

        #d1 = np.min(structure.get_all_distances())
        #d2 = np.min([layer.spacing for layer in layershifts])
        #print(units, d1, d2)
        #if (d1+1 < d2 and d1 < 2) or d1 < 1.5:
        #    print(d1,d2)
        #    write(structure, 'test.vasp')
        #    print(layershifts)
        #    exit()
        if np.sum(np.abs(normalized(layer_shifts_sum[0]))) > 5e-3:
            #print('layer_shifts', layer_shifts)
            #print('layer_shifts_sum', layer_shifts_sum)
            #print(np.sum(np.abs(normalized(layer_shifts_sum[0]))))
            #for unit in units:
            #    print(unit.valences)
            write(structure, 'error.vasp') 
            raise OSError("PBC condition failed")
        
        return structure

    @classmethod
    def get_unique_shifts(cls, unit1, unit2):
        """
        生成结构单元所有可能的界面
        """
        from jamip.structure.elementInfo import ElementData
        #pauling = ElementData['X'].to_dict()
        covalent_radius = ElementData['covalent_radius'].to_dict()

        assert unit1.lattice_symmetry == unit2.lattice_symmetry, """
               not in same crystal system."""
        equal = EqualTools.from_layergroup(unit1.dataset['number'])
        #print(equal.unique_shifts)

        # unique_cell -> units
        def get_features(lsc):
            matrix = lsc.to_matrix(full=False) 
            vectors = lsc.get_vectors(matrix)
            #distance = np.linalg.norm(vectors, axis=1)
            distance_area = min(np.linalg.norm(vectors[:,:2], axis=1))
            a = np.linalg.norm(lsc.lattice[0])
            covalent_distance = sum([lsc.covalent_radius[i] for i in lsc.atoms])
            
            spacing = 2
            distance = np.sqrt(spacing ** 2 + distance_area ** 2) 
            if covalent_distance > distance:
                spacing = np.sqrt(covalent_distance**2 - distance_area**2)

            lsc.spacing = spacing
            lsc.distance_area = distance_area / a
            return lsc

        # 计算结构单元间的平移向量
        #lattice = unit1.stdcell.lattice
        lattice = unit1.unitcell.lattice
        #print('u1',unit1.unitcell.lattice_parameters)
        #print('u2',unit2.unitcell.lattice_parameters)
        #write(unit1.unitcell,'u1.vasp')
        #write(unit2.unitcell,'u2.vasp')

        layershifts = []
        for xi,xshift in enumerate(equal.unique_shifts):
            for s1,u1 in unit1.unique_cells.items():
                #print('U1-T',u1.thickness)
                for s2,u2 in unit2.unique_cells.items():
                    #print('U2-T',u1.thickness)
                    #print(u1.unitcell.get_positions())
                    #print(u2.unitcell.get_positions())
                    lsc = get_interface_descriptor(lattice, u1, u2, shift=xshift)#, reset_spacing=True) 
                    lsc = get_features(lsc)
                    lsc.symop = (s1,s2)
                    #print(lsc)
                    '''
                    matrix = lsc.to_matrix(full=False)
                    vectors = lsc.get_scaled_vectors(matrix)
                    print(matrix)
                    print(vectors)
                    exit()
                    '''
                    layershifts.append(lsc)

        return layershifts

def dot_product(A, B):
    return sum([a * b for a, b in zip(A, B)])

def cross_product(a, b):
    return [a[i] * b[j] - a[j] * b[i] for i, j in [(1, 2), (2, 0), (0, 1)]]

def subtract(A, B):
    return [a - b for a, b in zip(A, B)]

def build_adjacency_list(parents, bonds):
    graph = np.unique(parents)
    adjacency = {e: set() for e in graph}
    for (i, j, offset) in bonds:
        component_a = parents[i]
        component_b = parents[j]
        adjacency[component_a].add((component_b, offset))
    return adjacency

def rank_increase(a, b):
    if len(a) == 0:
        return True
    elif len(a) == 1:
        return a[0] != b
    elif len(a) == 4:
        return False

    l = a + [b]
    w = cross_product(subtract(l[1], l[0]), subtract(l[2], l[0]))
    if len(a) == 2:
        return any(w)
    elif len(a) == 3:
        return dot_product(w, subtract(l[3], l[0])) != 0
    else:
        raise Exception("This shouldn't be possible.")


def bfs(adjacency, start):
    """
    Traverse the component graph using BFS.

    The graph is traversed until the matrix rank of the subspace spanned by
    the visited components no longer increases.
    """
    visited = set()
    cvisited = defaultdict(list)
    queue = deque()
    queue.append((start, (0, 0, 0)))

    while queue:
        vertex = queue.popleft()
        if vertex in visited:
            continue

        visited.add(vertex)
        c, p = vertex
        if not rank_increase(cvisited[c], p):
            continue

        cvisited[c].append(p)

        for nc, offset in adjacency[c]:

            nbrpos = (p[0] + offset[0], p[1] + offset[1], p[2] + offset[2])
            nbrnode = (nc, nbrpos)
            if nbrnode in visited:
                continue

            if rank_increase(cvisited[nc], nbrpos):
                queue.append(nbrnode)

    return visited, len(cvisited[start]) - 1

def traverse_component_graphs(adjacency):
    vertices = adjacency.keys()
    all_visited = {}
    ranks = {}
    for v in vertices:
        visited, rank = bfs(adjacency, v)
        all_visited[v] = visited
        ranks[v] = rank

    return all_visited, ranks

def merge_mutual_visits(all_visited, ranks, graph):
    """Find components with mutual visits and merge them."""
    merged = False
    common = defaultdict(list)
    for b, visited in all_visited.items():
        for offset in visited:
            for a in common[offset]:
                assert ranks[a] == ranks[b]
                if not graph.connected(a,b):
                    graph.merge(a,b)
                    merged = True 
            common[offset].append(b)

    if not merged:
        return merged, all_visited, ranks

    merged_visits = defaultdict(set)
    merged_ranks = {}
    parents = [v for i,v in graph._parents.items()]
    for k, v in all_visited.items():
        key = parents[k]
        merged_visits[key].update(v)
        merged_ranks[key] = ranks[key]
    return merged, merged_visits, merged_ranks

def normalized(vectors):
    """Normalize vectors and return them."""
    vectors = vectors - np.floor(vectors)
    vectors = np.around(vectors,8)
    vectors = vectors - np.around(vectors)
    return vectors

def vec_angle(vector1, vector2):
    cos_ = np.dot(vector1,vector2)
    sin_ = np.linalg.norm(np.cross(vector1,vector2))
    return np.arctan2(sin_, cos_)

class LayerFeature:

    def __init__(self, structure, units:list):
        self.structure = structure
        self.all_units = units
        self.unique_units = None
        self.equivalent_units = None
        self.layershifts = None
        self.second_layershifts = None

    @classmethod
    def from_dimanalysis(cls, dim):
        structure = dim.structure
        units = dim.units
        obj = cls(structure, units)
        obj.set_unit_bonding(dim.bonding, dim.active_bond_matrix)

        return obj

    def set_unit_bonding(self, bonddata, unitdata):
        '''
        distances -> unit_id -> remove same id %
        '''
        indices = np.zeros(len(self.structure), dtype=int)
        for i, unit in enumerate(self.all_units):
            indices[unit.atom_indices] = i

        unit_indices = indices[bonddata.pair_indices]
        remove_indices = np.where(unit_indices[0]==unit_indices[1], False, True)       
        self.interface_bonding = bonddata.section(remove_indices)
        self.unit_bonding = unitdata

    def set_unique_units(self, ignore_symop=False):
        """
        get unique units and their equivalent indices and operations
        """
        unique = []
        equivalent_units = []
        for unit in self.all_units:
            for i,iunit in enumerate(unique):
                # exist equal unit %
                if unit == iunit:
                    #print(iunit, unique[i].stdcell.get_positions())
                    stdcell = unique[i].stdcell 
                    site = unit.get_layer_site(i, stdcell, ignore_symop=ignore_symop)
                    if site != None:
                        break
            else:
                # unique unit %
                unique.append(unit)
                idx = len(unique)-1
                site = unit.get_layer_site(idx, ignore_symop=ignore_symop)
                if site is None:
                    raise OSError("get unit symetry operation failed!")
                
            equivalent_units.append(site)

        #print(self.all_units)
        #print(unique)
        #print(2,equivalent_units)

        self.unique_units = unique
        # sort units 
        centers = [unit.atomdf['coord'].mean() for unit in self.all_units]
        sort_indices = np.argsort(centers)
        # equal informations
        self.equivalent_units = [equivalent_units[i] for i in sort_indices]

    def set_interface(self, full=True):
        # 本函数期望实现以下目标
        # 1. 识别结构单元是否等价，并计算其向不可约结构单元的转换方法
        # 2. 识别结构单元的层间环境，提取界面特征

        # 计算原子层之间的表面位置
        for i,unit in enumerate(self.all_units):
            unit.get_atoms_data(axis_index=2, tol=0.1, pbc=False)
            unit.get_edge_indices(charge=True, tol=0.3)

        # 计算相邻原子层表面位点的相对位置
        lattice = self.structure.lattice
        centers = [unit.atomdf['coord'].mean() for unit in self.all_units]
        sort_indices = np.argsort(centers)
        next_indices = np.roll(sort_indices,-1)
        layershifts = []
        for j,k in zip(sort_indices, next_indices):
            u1 = self.all_units[j] # 下层结构单元
            u2 = self.all_units[k] # 上层结构单元
            lsc = get_interface_descriptor(lattice, u1, u2)
            layershifts.append(lsc)
        self.layershifts = layershifts

        if full is True:
            layershifts = []
            for j,k in zip(sort_indices, next_indices):
                u1 = self.all_units[j] # 下层结构单元
                u2 = self.all_units[k] # 上层结构单元
                lsc1,lsc2 = get_interface_descriptor(lattice, u1, u2, second=True)
                lscs = [lsc1, lsc2] if u1.total_charge > u2.total_charge else [lsc2, lsc1]
                layershifts.append(lscs)
            self.second_layershifts = layershifts

        return layershifts

    def get_unit_species_descriptor(self):
        # 获取结构单元的平均元素特征和结构特征
        # charge, ratom, n, mass, rcov, ea, x, ie
        data = []
        for unit in self.unique_units:
            data.append(unit.specie_feature)
        data = np.array(data)

        #print(self.structure.get_formula())
        #print(self.unique_units)
        assert data[0,0] * data[1,0] < 0, \
            f"interface charge not match. atom1: {data[0,2]}@{data[0,0]} != atom2: {data[1,2]}@{data[1,0]}"

        # condition %
        indices = np.argsort(data[:,0])
        dataset = {}
        for j,idx in enumerate(indices):
            for i,key in enumerate(['charge','ratoms', 'n', 'mass', 'rcovs', 'eas', 'xs', 'ies']):
                dataset['unit%d_%s' %(j+1,key)] = np.mean(data[idx][i])
        return dataset

    def get_interface_species_descriptor(self):
        # 获取结构单元之间的界面元素特征
        # charge, ratom, n, mass, rcov, ea, x, ie
        from .elementInfo import get_shannon_radius

        def get_ion_radius(unit):
            natom = len(self.structure)
            coord_matrix = np.zeros((natom, natom))
            for a,b,i,j in self.unit_bonding:
                coord_matrix[a,b] += 1
                coord_matrix[b,a] += 1

            coord_map = defaultdict(list)
            for i,j in enumerate(self.structure.get_elements(type='symbol')):
                coord_map[j].append(sum(coord_matrix[i]))
            coord_map = {i:int(np.round(np.mean(v),0)) for i,v in coord_map.items()}
  
            site = unit.sites['d1']
            ion_radius = get_shannon_radius(site.specie, site.charge, coord_map[site.specie])
            return ion_radius

        data = []
        for unit in self.unique_units:
            data.append(unit.surface_specie_feature)
            data[-1].append(get_ion_radius(unit))
        data = np.array(data)

        assert data[0,0] * data[1,0] < 0, "interface charge not match."

        # condition %
        indices = np.argsort(data[:,0])
        dataset = {}
        for j,idx in enumerate(indices):
            for i,key in enumerate(['charge','ratoms', 'n', 'mass', 'rcovs', 'eas', 'xs', 'ies', 'rions']):
                dataset['unit%d_surface_%s' %(j+1,key)] = np.mean(data[idx][i])
        return dataset
    
    def get_interface_bonding_descriptor(self):
        # 获取结构单元之间的界面结构特征
        def get_rcov(lsc):
            radius = []
            for i in lsc.atoms:
                radius.append(lsc.covalent_radius[i])
            return sum(radius)

        def get_features(lsc):
            matrix = lsc.to_matrix(full=False) 
            vectors = lsc.get_vectors(matrix)
            # spacing
            spacing = np.mean(vectors[:,2])
            if spacing < 0:
                spacing += spacing # - np.floor(spacing)
            # distance_area
            #print('debug')
            distance_area = np.linalg.norm(vectors[:,:2], axis=1) / np.linalg.norm(lsc.lattice[0])
            
            #print(1,distance_area)
            #distance_area = np.linalg.norm((vectors@np.linalg.inv(lsc.lattice))[:,:2], axis=1)
            #print(2,distance_area)
            #print(3,vectors)
            #print(4,vectors@np.linalg.inv(lsc.lattice))
            # distance
            distance = np.linalg.norm(vectors, axis=1)
            return np.mean(spacing), np.mean(distance), np.mean(distance_area)

        data = []
        uniques = []
        for lsc in self.layershifts:
            if lsc.atoms[0] == lsc.atoms[1]: continue
            spacing, distance, distance_area = get_features(lsc)
            spacing = abs(lsc.spacing*np.linalg.norm(lsc.lattice[2]))
            data.append([spacing, distance, distance_area])

            for row in uniques:
                # is same interface ?
                if abs(row[0]-spacing)<1e-4 and abs(row[1]-distance)<1e-4 and abs(row[2]-distance_area)<1e-4:
                    break
            else:
                uniques.append([spacing, distance, distance_area])

        data = np.array(data)
        dataset = {}
        for i,key in enumerate(['spacing', 'distance', 'distance_area_rate']):
            dataset[key] = np.mean(data[:,i])
        dataset['interface_raw_data'] = uniques

        if self.second_layershifts:
            data = []
            for lsc1, lsc2 in self.second_layershifts:
                
                distance1 = distance2 = 20
                if lsc1 != None:# and lsc1.atoms[0] == lsc1.atoms[1]:
                    spacing1, distance1, distance_area1 = get_features(lsc1)
                    spacing1 = abs(lsc1.spacing*np.linalg.norm(lsc1.lattice[2]))
                    distance_cov_rate1 = distance1 / get_rcov(lsc1)
                if lsc2 != None:# and lsc2.atoms[0] == lsc2.atoms[1]:
                    spacing2, distance2, distance_area2 = get_features(lsc2)
                    spacing2 = abs(lsc2.spacing*np.linalg.norm(lsc2.lattice[2]))
                    distance_cov_rate2 = distance2 / get_rcov(lsc2)

                # all invalid %
                if distance1 > 10 and distance2 > 10:
                    continue
                elif distance1 <= distance2:
                    data.append([spacing1, distance1, distance_area1, distance_cov_rate1])
                elif distance1 > distance2:
                    data.append([spacing2, distance2, distance_area2, distance_cov_rate2])
 
            uniques = []
            for (spacing, distance, distance_area, _) in data:
                for row in uniques:
                    # is same interface ?
                    if abs(row[0]-spacing)<1e-4 and abs(row[1]-distance)<1e-4 and abs(row[2]-distance_area)<1e-4:
                        break
                else:
                    uniques.append([spacing, distance, distance_area])
 
            if len(data): 
                data = np.array(data)
                for i,key in enumerate(['spacing_s0', 'distance_s0', 'distance_area_rate_s0', 'distance_cov_rate_s0']):
                    dataset[key] = np.mean(data[:,i])
                dataset['interface_raw_data_second'] = uniques
            else:
                dataset['spacing_s0'] = dataset['spacing'] 
                dataset['distance_s0'] = dataset['distance']
                dataset['distance_area_rate_s0'] = dataset['distance_area_rate']
                dataset['distance_cov_rate_s0'] = 2 
                dataset['interface_raw_data_second'] = dataset['interface_raw_data'] 

        return dataset
    
    def get_structure_descriptor(self):
        s = self.structure
        a = s._cell.a
        c = s._cell.c
        density = s.density
        packing_factor = s.packing_factor
        volume_per_atom = s.volume / len(s)
        area = np.linalg.norm(np.cross(s.lattice[0], s.lattice[1]))
        min_dist, avg_dist, avg_coordination_number = get_bonding_descriptor(s)
        data = {'a':a, 'c':c, 'area':area, 'density':density, 'packing_factor':packing_factor, 'volume_per_atom':volume_per_atom,
                'min_dist':min_dist, 'avg_dist':avg_dist, 'avg_coordination_number':avg_coordination_number,} 
        return data

    def get_ions_descriptor(self):
        from jamip.structure import Structure, read
        from jamip.analysis.vasp.ewald import Ewald3D

        s = self.structure
        elements = s.get_elements(type='symbol')
        positions = s.get_positions()
        valences = np.zeros(len(s), dtype=int)

        for unit in self.all_units:
            for i in unit.atom_indices:
                valences[i] = unit.valences[elements[i]]

        pcoord = positions[np.where(valences>0)]
        pe = elements[np.where(valences>0)]
        ps = Structure.from_cell((s.lattice, pcoord, pe) )
        dp = ps.get_all_distances()
        dpsum = np.mean(np.min(dp, axis=0))
 
        ncoord = positions[np.where(valences<0)]
        ne = elements[np.where(valences<0)]
        ns = Structure.from_cell((s.lattice, ncoord, ne) )
        dn = ns.get_all_distances()
        dnsum = np.mean(np.min(dn, axis=0))
 
        ew = Ewald3D(s)
        ew.charges = valences

        unit1_thickness = []
        unit2_thickness = []
        for unit in self.unique_units:
            if unit.total_charge < 0:
                unit1_thickness.append(unit.spacing_in_cartesian)
            else:
                unit2_thickness.append(unit.spacing_in_cartesian)
        

        data = {'mean_cation_distance': dpsum,
                'mean_anion_distance': dnsum,
                'ewald_energy': ew.total_energy() / len(s),
                'unit1_thickness': np.mean(unit1_thickness),
                'unit2_thickness': np.mean(unit2_thickness),
               }       
 
        return data


    def get_symmetry_descriptor(self):
        s = self.structure
        dataset = spglib.get_symmetry_dataset(s.to_cell())
        nsym = EqualTools.from_spacegroup(dataset['number']).noperation
        return {'symmetry': nsym}

    def get_homo_descriptor(self):
        from pymatgen.core.molecular_orbitals import MolecularOrbitals
        s = self.structure

        mo = MolecularOrbitals(s.get_formula())
        homo = mo.band_edges['HOMO'][-1]
        lumo = mo.band_edges['LUMO'][-1]

        return {'homo':homo, 'lumo':lumo}

def get_interface_descriptor(lattice, u1, u2, shift=None, second=False):
    # u1: down layer atoms + up sites
    # u2: up layer atoms + down sites

    from .bonding import BondCore, solid_angle

    if shift is None: shift = np.zeros(3)
    sites1 = u1.unitcell.get_positions()           # 下层原子坐标
    sites2 = u2.unitcell.get_positions() + shift   # 上层原子坐标
    a = np.linalg.norm(lattice[0])
    c = np.linalg.norm(lattice[2])
    # usites + shift = dsites + vectors
    '''
    print('shift')
    print(shift)
    print('sites1')
    print(sites1)
    print('sites2')
    print(sites2)
    '''

    def get_bond(usites, dsites, full=False):

        usites[:,:2] -= np.floor(usites[:,:2])
        dsites[:,:2] -= np.floor(dsites[:,:2])

        # 计算平移矢量和层间距，为此把z改为0(认为z值应当相近)
        vectors = usites[:,None,:] - dsites[None,:,:]
        spacings = vectors[:,:,2].reshape(-1)
        tmp = []
        zshift = 0
        for i in spacings:
            if i + 0.5 < 1e-2:
                i += 1
                zshift = 1
            tmp.append(i)        
        vectors[:,:,2] = -zshift 
        # range -> (-0.5 - 0.5]
 
        # 考虑周期性, 对于指向原子简并的情况，应当消除简并
        grid = np.mgrid[-1:2, -1:2, zshift:zshift+1].reshape(3,-1).T # x,3
        grid_vectors = grid[:,None,None,:] + vectors[None,:,:,:]
        distances = np.linalg.norm(grid_vectors @ lattice, axis=3)
        min_distance = np.min(distances)
        indices = np.array(np.where(distances - min_distance < 1e-2))
        area_distances = distances[np.where(distances - min_distance < 1e-2)]

        # reset structure %
        positions = np.r_[usites,dsites] @ lattice
        elements = ['A'] * len(usites) + ['B'] * len(dsites)
        cell = (lattice, positions, elements)
        indices[2] += len(usites)
        # return spacing, min_distance, grid_vectors, indices
        bd = BondCore(area_distances, indices[1:], indices[0], cell, grid)

        bd.spacing = np.mean(tmp)
        bd.area_distance = min_distance
        return bd

    datom = u1.sites['u1']                   # 下层原子的上表面
    dsites = u1.get_positions_by_key('u1')  
    uatom = u2.sites['d1']                   # 上层原子的下表面
    usites = u2.get_positions_by_key('d1') + shift 

    # 提取最近邻特征
    #spacing, min_distance, allvectors, indices = get_bond(usites, dsites)
    #lsc = LayerShiftCreator(uatom, datom, spacing, min_distance, allvectors, indices, a, c)
    # if not lsc.check_symmetry():#self, indices):
    #    raise AtomsError('num sites check failed.')
    if second == False:
        lsc = get_bond(usites, dsites)
        lsc.atoms = (uatom.specie, datom.specie)
        return lsc

    elif second == True:
        lsc2d = None
        lsc2u = None

        datom2 = u1.sites['u2']                  # 下层原子的次表面
        if datom2 != None:
            dsites2 = u1.get_positions_by_key('u2')  
            lsc2d = get_bond(usites, dsites2)
            lsc2d.atoms = (uatom.specie, datom2.specie)

        uatom2 = u2.sites['d2']                  # 上层原子的次表面
        if uatom2 != None:
            usites2 = u2.get_positions_by_key('d2') + shift 
            lsc2u = get_bond(usites2, dsites)
            lsc2u.atoms = (uatom2.specie, datom.specie)

        return lsc2d, lsc2u
        #return lsc, lsc2d, lsc2u

def get_interface_descriptor_test(lattice, u1, u2, key=None, shift=None, reset_spacing=False, full=False, second=False):
    # u1: down layer atoms + up sites
    # u2: up layer atoms + down sites

    from .bonding import BondCore, solid_angle

    if shift is None: shift = np.zeros(3)
    #sites1 = u1.unitcell.get_positions()           # 下层原子坐标
    #sites2 = u2.unitcell.get_positions() + shift   # 上层原子坐标
    #a = np.linalg.norm(lattice[0])
    #c = np.linalg.norm(lattice[2])

    # usites + shift = dsites + vectors

    def get_bond(usites, dsites, full=False):

        usites[:,:2] -= np.floor(usites[:,:2])
        dsites[:,:2] -= np.floor(dsites[:,:2])

        # 计算平移矢量和层间距，为此把z改为0(认为z值应当相近)
        vectors = usites[:,None,:] - dsites[None,:,:]
        spacings = vectors[:,:,2].reshape(-1)
        tmp = []
        zshift = 0
        for i in spacings:
            if i + 0.5 < 1e-2:
                i += 1
                zshift = 1
            tmp.append(i)        
        vectors[:,:,2] = -zshift 
        # range -> (-0.5 - 0.5]
 
        # 考虑周期性, 对于指向原子简并的情况，应当消除简并
        grid = np.mgrid[-1:2, -1:2, zshift:zshift+1].reshape(3,-1).T # x,3
        grid_vectors = grid[:,None,None,:] + vectors[None,:,:,:]
        distances = np.linalg.norm(grid_vectors @ lattice, axis=3)
        min_distance = np.min(distances)
        indices = np.array(np.where(distances - min_distance < 1e-2))
        area_distances = distances[np.where(distances - min_distance < 1e-2)]

        #print(indices, len(usites), len(dsites), grid.shape)

        # reset structure %
        positions = np.r_[usites,dsites] @ lattice
        elements = ['A'] * len(usites) + ['B'] * len(dsites)
        cell = (lattice, positions, elements)
        indices[2] += len(usites)
        # return spacing, min_distance, grid_vectors, indices
        bd = BondCore(area_distances, indices[1:], indices[0], cell, grid)

        bd.spacing = np.mean(tmp)
        return bd

    datom = u1.sites['u1']                   # 下层原子的上表面
    dsites = u1.get_positions_by_key('u1')  
    uatom = u2.sites['d1']                   # 上层原子的下表面
    usites = u2.get_positions_by_key('d1') + shift 

    # 提取最近邻特征
    lsc = get_bond(usites, dsites)
    #spacing, min_distance, allvectors, indices = get_bond(usites, dsites)
    #lsc = LayerShiftCreator(uatom, datom, spacing, min_distance, allvectors, indices, a, c)
    # if not lsc.check_symmetry():#self, indices):
    #    raise AtomsError('num sites check failed.')

    lsc2d = lsc2u = None
    if second == True:

        datom2 = u1.sites['u2']                  # 下层原子的次表面
        if datom2 != None:
            dsites2 = u1.get_positions_by_key('u2')  
            lsc2d = get_bond(usites, dsites2)
            # spacing, min_distance, allvectors, indices = get_bond(usites, dsites2)
            # lsc2d = LayerShiftCreator(uatom, datom2, spacing, min_distance, allvectors, indices, a, c)

        uatom2 = u2.sites['d2']                  # 上层原子的次表面
        if uatom2 != None:
            usites2 = u2.get_positions_by_key('d2') + shift 
            lsc2u = get_bond(usites2, dsites)
            # spacing, min_distance, allvectors, indices = get_bond(usites2, dsites)
            # lsc2u = LayerShiftCreator(uatom2, datom, spacing, min_distance, allvectors, indices, a, c)

        return lsc, lsc2d, lsc2u
    
    return lsc

def get_lattice_transport(lattice1, lattice2):
    # transport positions from lattice1 to lattice2
    # 因为变换限制前后原子数不变，我们认为xy方向的原子随晶格发生缩放，z方向的原子实空间坐标不变
    # 检查 x1/y1 = x2/y2，如果 x1/y2 * x2/y2 = 1，交换xy轴，否则报错
    # 检查 三键角不发生明显改变 (且alpha=beta=90)
    from .atom import Cell

    cell1 = Cell(lattice1)
    cell2 = Cell(lattice2)

    matrix = np.eye(3)

    if abs(cell1.a/cell1.b - cell2.a/cell2.b) < 1e-2:
        pass
    elif abs(cell1.a/cell1.b*cell2.a/cell2.b - 1) < 1e-2:
        matrix[:2,:2] = np.array([[0,1],[1,0]])
    else:
        raise ValueError("GET lattice transport failed!")
    # 应当尝试比较键角，但没必要

    matrix[2,2] = cell1.c / cell2.c
    return matrix

def redefine_by_spglib(dataset):
    # lattice = std_lattice * T_mat -> lattice1.T @ dataset['transformation_matrix'] == lattice2.T
    # lattice1 = np.linalg.inv(T_mat.T) @ lattice2
    # with_same_volume: matrix = np.linalg.inv(T_mat.T)
    # idae_lattice = T_rot * std_lattice
    from jamip.modeling.structureFactory import remove_duplicates_with_tolerance 

    if np.abs(np.linalg.det(dataset['transformation_matrix'])) == 1:
        stdcell = Structure.from_cell((dataset['std_lattice'], dataset['std_positions'], dataset['std_types']))
        return stdcell

    lattice = dataset['std_lattice']
    positions = dataset['std_positions']
    elements = dataset['std_types']
    matrix = dataset['transformation_matrix']
    positions = positions - np.floor(positions)

    # suerpcell
    def get_range(array):
        dmin = sum(np.where(array<0, array, 0))
        dmax = sum(np.where(array>0, array, 0))
        return int(np.ceil(max(-dmin, dmax)))
    
    dim1 = get_range(matrix[:,0])
    dim2 = get_range(matrix[:,1])
    dim3 = get_range(matrix[:,2])
    grid = np.mgrid[-dim1:dim1+1, -dim2:dim2+1, -dim3:dim3+1].reshape(3,-1).T # x,3
    grid_positions = positions[None,:,:] + grid[:,None,:] # ngrid, natom, 3
#    cart_positions = normalized(grid_positions @ lattice @ np.linalg.inv(matrix.T@lattice))
    cart_positions = grid_positions @ np.linalg.inv(matrix.T)
    cart_positions = np.around(cart_positions,8)
    cart_positions -= np.floor(cart_positions)
#    print(cart_positions.shape)
#    print(np.max(cart_positions))
#    print(np.min(cart_positions))

    ndim = np.abs(np.around(np.linalg.det(matrix)).astype(int))
    indices = np.where((np.max(cart_positions, axis=2) < (1-1e-7) ) & (np.min(cart_positions, axis=2) > 1e-7))
    border_indices = np.where((np.max(cart_positions, axis=2) > (1-1e-7) ) | (np.min(cart_positions, axis=2) < 1e-7))

    # get border sites
    borders = []
    for i,j in zip(*border_indices):
        borders.append(cart_positions[i,j])
    if len(borders):
        borders = np.array(borders) - np.array([0.5,0.5,0.5])
        borders -= np.floor(borders)
        duplicates = remove_duplicates_with_tolerance(borders, tolerance=1e-3)
        ids = np.r_[indices[0], border_indices[0][~duplicates]]
        jds = np.r_[indices[1], border_indices[1][~duplicates]]
    else:
        ids, jds = indices
 
    # get all sites
    new_elements = []
    new_positions = []
 
    for i,j in zip(ids,jds):
        coord = cart_positions[i,j]
        new_elements.append(elements[j])
        new_positions.append(coord)

    duplicates = remove_duplicates_with_tolerance(new_positions, tolerance=1e-3)
    new_positions = np.array(new_positions)[~duplicates]
    new_elements = np.array(new_elements)[~duplicates]

    if abs(np.abs(np.linalg.det(matrix)) - len(new_elements)/len(elements)) > 1e-2:
        raise RuntimeError("search atoms failed. ndim is %s, need %d atom, find %d atom" %(ndim, ndim*len(elements), len(new_elements)))

    new_lattice = spglib.niggli_reduce(matrix.T @ lattice)
    stdcell = Structure.from_cell((new_lattice, new_positions, new_elements))

    return stdcell  

def get_specie_features(species):
    """ Calculate the atomic features of a species list.

    Args:
        species (list): The species list.

    Returns:
        features (list): The atomic features of the species list. 
                         Include [atomic_radius, n, mass, rcov, ea, x, ie].
    """    
    from jamip.structure.elementInfo import Element
    elements = [Element.from_symbol(e) for e in species]
    ratom = np.mean([e.atomic_radius for e in elements])
    n = np.mean([e.Z for e in elements])
    x = np.mean([e.X for e in elements])
    mass = np.mean([e.mass for e in elements])
    rcov = np.mean([e.covalent_radius for e in elements])
    ie = np.mean([e.first_ionization_energy for e in elements])
    ea = np.mean([0 if pd.isna(ele) else ele for ele in [e.electron_affinity for e in elements]])
    atom_features = [ratom, n, mass, rcov, ea, x, ie]
    return atom_features

def get_bonding_descriptor(structure):
    """ Calculate the bonding descriptor of a structure.

    Args:
        structure (Structure): The structure to calculate the bonding descriptor.

    Returns:
        min_dist (float): The minimum distance between atoms.
        avg_dist (float): The average distance between atoms.
        avg_coordination_number (float): The average coordination number of atoms.
    """    
    from jamip.structure.elementInfo import Element

    # 计算每个原子的最短键长
    bd = Bonding(structure, method='min', factor=1, offset=0.5)
    if len(bd.data.distances) == 0:
        rcov = Element.from_symbol(structure.species_of_elements[0]).covalent_radius
        min_dist = rcov*2
        avg_dist = rcov*2
        avg_coordination_number = 0
        # atom_volume = min_r**3 * len(bd.data.elements) * 4/3 * np.pi

    else:
        min_dist = min(bd.data.distances)
        avg_dist = np.mean(bd.data.distances)
        avg_coordination_number = len(bd.data.distances) * 2 / len(bd.data.elements)
        # voro_r = np.min(bd.distances, axis=0)
        # atom_volume = np.sum(voro_r**3) * 4/3 * np.pi
    # dist_packing_factor = atom_volume/structure.volume

    return min_dist, avg_dist, avg_coordination_number

def get_ion_connectivity(self, iontype="cation"):
    # 计算阳离子的连通性
    # 计算方法为: 计算仅通过阳离子键穿过晶格c方向所需的最短周期 / c方向晶格常数
    from scipy.sparse.csgraph import dijkstra
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import depth_first_order

    def get_distances(coords, lattice):
        """
        将coords的所有坐标移动到[0,1)区间，应用网格后坐标变为[-1,1), [-1,1), (0,1]
        """
        grid = np.mgrid[-1:1, -1:1, 0:1].reshape(3,-1).T # x,3
        vectors = coords[None,:,:] - coords[:,None,:] 
        vectors = vectors - np.floor(vectors)
        vectors[:,:,2] = np.where(vectors[:,:,2]>1e-8, vectors[:,:,2], 1)
        grid_vectors = vectors[None,:,:,:] + grid[:,None,None,:] 
        distances = np.linalg.norm(grid_vectors @ lattice, axis=3)
        distances = np.min(distances, axis=0)

        return distances

    def get_shortest_path(predecessors, i, j):        
        indices = []
        while True:
            indices.append(j)
            if predecessors[i,j] != i:
                j = predecessors[i,j]
            else:
                break
        return indices[::-1]

    lattice = self.structure.lattice
    charges = np.array([self.valences[i] for i in self.structure.get_elements(type='symbol')])
    if iontype == "cation":
        indices = np.where(charges>0)
    elif iontype == "anion":
        indices = np.where(charges<0)
    else:
        raise ValueError("Unknown iontype value: %s" %iontype)
    ions = self.structure.get_positions()[indices]
    distances = get_distances(ions, lattice)
    if distances.size == 1 or abs(np.min(distances)-distances[0,0]) < 1e-8:
        return 1
    else:
        dist_matrix, predecessors = dijkstra(distances**3, return_predecessors=True)
        loop = dist_matrix + dist_matrix.T
        row, col = np.diag_indices_from(loop)
        loop[row,col] = np.inf
        i,j = np.unravel_index(np.argmin(loop), loop.shape)
        idx1 = get_shortest_path(predecessors, i, j)
        idx2 = get_shortest_path(predecessors, j, i)
        x1 = np.array(idx1 + idx2)
        x2 = np.roll(x1, -1)
        assert abs(sum([distances[i,j]**3 for i,j in zip(x1,x2)]) - np.min(loop)) < 1e-8
        d = sum([distances[i,j] for i,j in zip(x1,x2)])
        connectivity_rate = d / np.linalg.norm(lattice[2])

    return connectivity_rate


def get_2d_axis(structure, visited, ranks, max_value=6):
    """
    计算二维材料的层方向,并将结构的z轴旋转到层方向
    返回 3x3的矩阵，满足 lattice_z = lattice @ matrix
    如果矩阵获取失败，返回None
    """        
    lattice = structure.lattice
    if max(ranks.values()) != 2: 
        print("no 2d unit!")
        return None

    # 遍历全部结构单元，对于维度=2 & 原子数>1的结构单元，根据其连通方向确定z轴
    # 在类似单斜晶胞中，结构层方向存在绝对垂直(通过笛卡尔坐标计算，满足alpha=beta=90)
    # 和相对垂直(通过分数坐标计算，得到类似001的标准向量，alpha!=90 or beta!=90)
    # 在本轮筛选中，原则上不包含相对垂直的结构
    zaxes = []
    allvectors = []
    for key,value in visited.items():
        if ranks[key] == 2 and len(value) > 1:
            vectors = []
            for i,v in value:
                if i == key:
                    vectors.append(v)
                    allvectors.append(v)
                
            # get absolute vector (cartesian) 
            vectors = np.array(vectors) @ lattice
            U,S,Vh = np.linalg.svd(vectors.T)
            for i,s in enumerate(S):
                if abs(s) <= 1e-8:
                    vector = U[:,i]
                    break
            else:
                return None
            
            # cartesian to direct
            vector = vector @ np.linalg.inv(lattice)
            vmin = np.min(np.abs([i for i in vector if abs(i) > 1e-4]))
            vector = np.around(vector / vmin, 8)
            zaxes.append(vector)

    # zaxes rank should be 1
    rank = 3
    if len(zaxes) == 1:
        rank = 1
    elif len(zaxes) > 1:
        U,S,Vh = np.linalg.svd(zaxes)
        rank = len(np.where(abs(S)>1e-8)[0])
    if rank != 1: 
        print('get z-axes failed. ')
        """
        print(rank, S)
        print(zaxes)
        print(lattice)
        raise
        """
        return None

    # 确定z轴没有问题，从前面的向量里选一些晶格常数小的补足矩阵
    allvectors = np.unique(allvectors, axis=0)
    cartvectors = allvectors @ lattice
    dists = np.linalg.norm(cartvectors, axis=1)

    vectors = [zaxes[0]]
    for i in np.argsort(dists):
        vector = allvectors[i]
        if dists[i] < 1e-8: continue
        if len(vectors) < 2:
            vectors.append(vector)
        else:
            matrix = np.r_[vectors, [vector]]
            if np.linalg.matrix_rank(matrix) == 3:
                break
    else:
        raise RuntimeError("Search lattice failed")

    matrix = np.clip(matrix[::-1], -max_value, max_value)

    # 接下来开始尝试晶胞重定义，使z轴为层方向
    '''
    lattice = self.structure.lattice
    positions = self.structure.get_positions(type='direct')
    elements = self.structure.get_elements(type='symbol')

    # 第一步，判断转换后晶胞体积是否一致(原子数一致)
    volume1 = abs(np.linalg.det(lattice))
    volume2 = abs(np.linalg.det(matrix@lattice))
    if volume2 < 1e-4:
        raise RuntimeError("matrix det != 0")

    # 1. 如果晶胞体积一致，直接转换即可
    if abs(volume1-volume2)/volume1 < 1e-2:
        matrix = np.around(matrix).astype(int)
        lattice = matrix @ lattice 
        positions = positions @ np.linalg.inv(matrix)
        self.structure = Structure.from_cell((lattice, positions, elements))

    # 2. 如果可以通过晶格转换获得垂直的c轴，尝试晶格重定义，再重做一次维度识别
    elif np.sum(np.abs(matrix-np.around(matrix))) < 0.1 and volume2 / volume1 < 5:
        matrix = np.around(matrix).astype(int)
        self.structure = redefine(self.structure, matrix)
        self.set_bonding()
        self.search_cutoff()

    # 3. 如果现有坐标轴与层方向正交(未必垂直)，将其作为z轴
    else:
        vector = np.cross(matrix[0],matrix[1])
        indices = np.where(abs(vector)>1e-4)[0]
        if len(indices) == 1:
            zaxis = indices[0]
            matrix = np.roll(np.eye(3, dtype=int), zaxis+1, axis=1)
            
            lattice = matrix @ lattice 
            positions = positions @ np.linalg.inv(matrix)
            self.structure = Structure.from_cell((lattice, positions, elements))

        # 无计可施，选择放弃
        else:
            matrix = None
            pass
    '''

    return matrix

def get_pairs(species):
    
    pairs = []
    for row in species:
        for i in row:
            for j in row:
                 pairs.append(sorted((i,j)))
    return pairs

def get_unique_sorting(matrix):

    value = np.linalg.norm(matrix, axis=-1)
    idx1 = np.argsort(np.sum(value, axis=1))
    idx2 = np.argsort(np.sum(value, axis=0))
    # 根据 matrix 的内容排序
    return idx1, idx2

def apply_sort(matrix):
    idx1, idx2 = get_unique_sorting(matrix)
    sorted_matrix = matrix[idx1][:,idx2]
    return sorted_matrix

def periodic_distance(vectors):
    """
    计算向量的周期性边界条件距离。
    """
    vectors = vectors - np.floor(vectors)
    return vectors

def cost_function_with_matching(x, A, B, point_matrix):
    """
    优化目标函数：动态匹配点对，比较 point_matrix 和 A 到 B+x 的周期性距离矩阵的误差。
    """
    from scipy.optimize import linear_sum_assignment  # 匈牙利算法

    # 平移后的点集 C
    C = B + x

    # 构造 A 到 C 的周期性距离矩阵
    calculated_matrix = A[:, None, :] - C[None, :, :]
    calculated_matrix = periodic_distance(calculated_matrix)

    calculated_matrix = apply_sort(calculated_matrix)
    point_matrix = apply_sort(point_matrix)

    # 使用匈牙利算法动态匹配点对关系
    n, m = calculated_matrix.shape[:2]
    reshaped_distances = np.linalg.norm(calculated_matrix, axis=-1)  # 距离矩阵，形状为 (n, m)
    row_ind, col_ind = linear_sum_assignment(reshaped_distances)  # 最优点对匹配

    # 按匹配关系重排矩阵
    matched_calculated_matrix = calculated_matrix[row_ind, col_ind, :]

    # 计算 point_matrix 和 matched_calculated_matrix 的平方误差
    diff = matched_calculated_matrix - point_matrix[row_ind, col_ind, :]
    return np.sum(diff**2)

def calculate_translation_with_matching(A, B, point_matrix):
    """
    使用全局优化求解平移向量，并动态匹配点对。
    """
    from scipy.optimize import differential_evolution
    # 定义搜索范围（周期性条件下的平移向量范围）
    bounds = [(0, 1)] * 3

    # 使用差分进化算法进行全局优化
    result = differential_evolution(cost_function_with_matching, bounds, args=(A, B, point_matrix), strategy='best1bin', tol=1e-6)

    return result.x.reshape(1, 3)
