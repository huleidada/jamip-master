from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import threading
import pathlib
import logging
import shutil
import json
import six
import sys
import re

__all__ = ['ICSDFilter', 'DimensionFilter']

class globalvar:
    """
    global variables for different plot types
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = object.__new__(cls)  
        return cls._instance

class ICSDFilter:

    def __init__(self, source):
        self.source = pathlib.Path(source)
        self.filter_occupy = True
        self.filter_distance = False
        self.data = pd.DataFrame()

    def generator(self, prefix=None):

        if self.source.is_dir():
            for path in self.source.iterdir():
                yield path
        elif self.source.suffix == '.csv':
            df = pd.read_csv(self.source)
            for row in df.itertuples():                
                yield pathlib.Path(prefix)/row.cif if prefix != None else pathlib.Path(row.cif)

    @staticmethod
    def hash(cifdata):
        # from jamip.structure.symmetry import SpaceSymmetry
        # from jamip.structure.atomic_number import atomic
        import pandas as pd
        # import spglib
 
        #print(dataset['number'], dataset['wyckoffs'])
        # if mini and dataset['number'] > 142:
        #     wycs = [ord(i) for i in dataset['wyckoffs']]
        #     data = [(min(wycs), sum(wycs), dataset)]

        #     #shifts = np.array([[0,0,0],[1/2,0,0],[0,1/2,0],[1/2,1/2,0]])
        #     if dataset['number'] > 192:
        #         shifts = np.array([[1/2,1/2,0]])
        #     elif dataset['number'] > 142:
        #         shifts =  np.array([[1/3,2/3,0],[2/3,1/3,0]])

        #     for shift in shifts:
        #         lattice, positions, elements = self.to_cell()
        #         positions += shift 
        #         dataset = spglib.get_symmetry_dataset((lattice, positions, elements), symprec=symprec)
        #         wycs = [ord(i) for i in dataset['wyckoffs']]
        #         raw = (min(wycs), sum(wycs), dataset)
        #         data.append(raw) 

        #     # get best dataset
        #     sort_indices = np.lexsort(([i[0] for i in data], [i[1] for i in data]))
        #     dataset = data[sort_indices[0]][2]

        # def wyckoff2formula(wyckoffs):
        #     species,numbers = np.unique(wyckoffs, return_counts=True)
        #     formula = ''
        #     for i,j in zip(species, numbers):
        #         if j == 1: j = ''
        #         formula += f'{j}{i}'
        #     return formula

        # spg_num = dataset['number']
        # species = [atomic[i] for i in dataset['std_types'][dataset['mapping_to_primitive']]]
        # comp = Composition.from_elements(species)

        # # pearson_symbol
        # num_sites_conventional = len(species)
        # space = SpaceSymmetry.from_hall_number(dataset['hall_number'])
        # pearson_symbol = space.pearson_symbol(num_sites_conventional)
 
        # sites = pd.DataFrame({'wyckoffs': dataset['wyckoffs'],
        #                       'equivalent_atoms': dataset['equivalent_atoms'],
        #                       'species': species,})

        # unique_sites = sites.drop_duplicates()
        # unique_species = []
        # element_wyckoffs = []
        # for key,grp in unique_sites.groupby('species'):
        #     wycs = wyckoff2formula(grp['wyckoffs'].values)
        #     element_wyckoffs.append(wycs)
        #     unique_species.append(key)

        # indices = np.argsort(element_wyckoffs)
        # all_wyckoffs = "_".join([element_wyckoffs[i] for i in indices])
        # chemsys = '-'.join([unique_species[i] for i in indices])
 
        # if with_chemsys:
        #     protostructure_label = f"{comp.ABformula}_{pearson_symbol}_{spg_num}_{all_wyckoffs}:{chemsys}"
        # else:
        #     protostructure_label = f"{comp.ABformula}_{pearson_symbol}_{spg_num}_{all_wyckoffs}"
        # return protostructure_label

        elements = []
        for specie in cifdata['_atom_site_label']:
            element = re.findall('[A-Z][a-z]?',specie)[0]
            if element == 'D': element = 'H'
            elements.append(element)

        df = pd.DataFrame({'specie': elements,
                           'multi':  cifdata['_atom_site_symmetry_multiplicity'],
                           'wyckoff': cifdata['_atom_site_wyckoff_symbol']})
        formula = ''
        for key,grp in df.groupby(['wyckoff', 'specie', 'multi']):
            wyckoff, specie, multi = key
            formula += '(%s%d%s)%d ' %(specie,multi,wyckoff,len(grp))
        return formula.rstrip() 

    @staticmethod
    def format(cifdata):
        from jamip.structure.crystal import format_symbol

        # space group H-M symbol
        symbolHM = None
        for name in ['_space_group.Patterson_name_h-m',
                    '_symmetry_space_group_name_h-m',
                    '_space_group_name_h-m_alt']:
            if name in cifdata:
                symbolHM=format_symbol(cifdata[name])
                break
        
        # space group number
        symbolSG = None
        for name in ['_space_group.it_number', 
                    '_space_group_it_number', 
                    '_symmetry_int_tables_number']:
            if name in cifdata:
                symbolSG=int(cifdata[name])
                break

        # min occ number        
        minocc = np.array(cifdata['_atom_site_occupancy'], dtype=float).min()

        info = {'hm_symbol': symbolHM, 'sp_number': symbolSG, 'minocc': minocc}
        return info

    @staticmethod
    def run(path):
        from jamip.structure.crystal import parse_cif
        from jamip.structure import read
        from jamip.structure.atomic_number import number

        info = {'cif': path.name, 'status': False}

        try:
            cif = parse_cif(path)[0][1]
        except:
            info['reason'] = 'cif format error'

        info.update(ICSDFilter.format(cif))
        formula = cif['_chemical_formula_sum']
        species = cif['_atom_site_label']
        info['formula'] = formula
 
        # filter1: occupyation
        if 1 - info['minocc'] > 1e-4:
            info['reason'] = 'partial occupation'
            return info
 
        # filter2: atoms & symbol
        try:
            named_elements = re.findall(r"[A-Z][a-z]?", formula)
            elements = []
            for specie in species:
                element = re.findall(r'[A-Z][a-z]?',specie)[0]
                assert element in number, f"Error element {element}"
                elements.append(element)
            assert set(named_elements) == set(elements)            
        except:
            info['reason'] = 'Element mismatch'
            return info
 
        # filter3: positions %
        try:
            structure = read(path)
        except:
            info['reason'] = 'read structure error'
            return info
 
        # info-site 
        info['status'] = True
        info['sites'] = ICSDFilter.hash(cif) 
        info['prototype'] = structure.get_aflow_prototype(with_chemsys=False)
 
        # info-formula
        info['formula'] = structure.composition.get_formula(sort=True, reduced=True, split='')
        info['ABformula'] = structure.composition.ABformula
        info['nspecie'] = len(structure.number_of_atoms)
        info['natom'] = sum(structure.number_of_atoms)
        info['Z'] = structure.composition.Z
 
        return info
    
    def mpirun(self, max_workers=30, max_iter=None):
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm

        file_paths = list(self.generator())
        if isinstance(max_iter, int):
            file_paths = file_paths[:max_iter]

        if max_workers > 1:

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 使用tqdm显示进度条
                results = list(tqdm(
                    executor.map(self.run, file_paths),
                    total=len(file_paths),
                    desc="Processing files"
                ))

        else:
            results = []
            # 使用普通for循环处理文件，用tqdm显示进度条
            for file_path in tqdm(file_paths, desc="Processing files"):
                results.append(self.run(file_path))
        
        return pd.DataFrame(results)

    
    def filter_distance(self):
        #minbond = np.min(structure.get_all_distances(pbc=True, diag=True))
        #if minbond < 0.5:  # 1.2*min_element_radius
        pass

    def filter_neutral(self, df, skipH=True):
        from jamip.structure.atom import Composition

        neutral = []
        for i,formula in enumerate(df['formula']):
            composition = Composition.from_formula(formula)
            try:
                valence = composition.best_valence
                total = 0
                for i,j in composition.as_dict().items():
                    total += valence[i] * j
                if skipH and 'H' in valence:
                    neutral.append(False)
                else:
                    neutral.append(total == 0)
            except:
                neutral.append(False)
            
        df['neutral'] = neutral
        return df


    @staticmethod
    def dimension(path):
        from jamip.structure import read as jpread
        from jamip.structure.convert import jamip2ase
        from ase.io import read
        from ase.geometry.dimensionality import analyze_dimensionality
        # try:     
        #     structure = read(path)
        # except:
        js = jpread(path)
        structure = jamip2ase(js)
        
        max_natom=200
        dimension = -1
        if max_natom is None or len(structure) < max_natom:
            try:
                data = analyze_dimensionality(structure, method='RDA')
                if len(data) and data[0].dimtype != None:
                    for i in ['3','2','1','0']:
                        if i in data[0].dimtype:
                            dimension = i
                            break
            except:
                pass
        return dimension

    def filter_dimension(self, df, max_workers=32):
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm

        file_paths = []
        for cif in df['cif']:
            file_paths.append(self.source/cif)

        # results = []
        # # 使用普通for循环处理文件，用tqdm显示进度条
        # for file_path in tqdm(file_paths, desc="Processing files"):
        #     print(file_path)
        #     results.append(self.dimension(file_path))
            
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 使用tqdm显示进度条
            dimensions = list(tqdm(
                executor.map(self.dimension, file_paths),
                total=len(file_paths),
                desc="Processing files"
            ))       

        df['dimension'] = dimensions
        return df

    def unique_by_sites(self, df):

        df = df[df['status']]

        indices = []
        for key, grp in df.groupby(['formula','sp_number','hm_symbol','sites']):
            if len(grp) == 1:
                indices.append(grp.index[0])
            else:
                indices.append(grp.index[0])

        indices = np.sort(indices)
        unique = df.loc[indices,:]
        return unique

    def unique_by_prototype(self, df):

        df = df[df['status']]

        indices = []
        for key, grp in df.groupby(['formula','prototype']):
            if len(grp) == 1:
                indices.append(grp.index[0])
            else:
                indices.append(grp.index[0])

        indices = np.sort(indices)
        unique = df.loc[indices,:]
        return unique    

    def get_similarity(cell1, cell2):
        # 导入计算距离的函数和MBTR模型
        from scipy.spatial.distance import pdist, squareform
        from dscribe.descriptors import MBTR
        from ase.atoms import Atoms
 
        # 获取两个晶胞中的原子种类
        species = np.unique(cell1[2])
        # 创建MBTR模型
        model = MBTR(
        species=species,
        k3 = {
            "geometry": {"function": "cosine"},
            "grid": {"min": -1, "max": 1, "n": 50, "sigma": 0.08},
            "weighting" : {"function": "exp", "scale": 0.3, "cutoff": 1e-3}
        },
        periodic=True,
        normalization="l2_each",
        flatten=True
        )
 
        # 获取第一个晶胞的晶格、位置和元素
        lattice, positions, elements = cell1
        # 创建ASE原子对象
        atoms = Atoms(symbols=elements,scaled_positions=positions,cell=lattice,pbc=True)
        # 计算第一个晶胞的描述符
        descriptor1 = model.create(atoms)[0]
        # 获取第二个晶胞的晶格、位置和元素
        lattice, positions, elements = cell2
        # 创建ASE原子对象
        atoms = Atoms(symbols=elements,scaled_positions=positions,cell=lattice,pbc=True)
        # 计算第二个晶胞的描述符
        descriptor2 = model.create(atoms)[0]
        # 计算两个描述符之间的距离
        Y = pdist([descriptor1, descriptor2])
        # 返回距离
        return Y
 

class DimensionFilter:

    active = False

    def __init__(self, source, output):
        self.source = pathlib.Path(source)
        self.output = pathlib.Path(output)
        if not self.output.exists():
            self.output.mkdir(exist_ok=True, parents=True)
        self.active = True
        self.write_unit = True

    def generator(self, cifs):
        from jamip.structure import read, write, Structure
        from jamip.structure.dimension import DimensionAnalysis
        import spglib

        globalvar.output = self.output
        globalvar.write_unit = self.write_unit
 
        for cif in cifs:
            s = read(self.source / cif)
            print(cif)
            # refine cell. low symprec may failed.
            stdcell = spglib.refine_cell(s.to_cell(), symprec=1e-1)
            if stdcell == None:
                stdcell = spglib.refine_cell(s.to_cell(), symprec=1e-2)

            s = Structure.from_cell(stdcell)
            dim = DimensionAnalysis(s)
            dim.cif = cif
            yield dim

    @staticmethod
    def run(dim):
        from jamip.structure import read, write, Structure
        from jamip.structure.dimension import DimensionAnalysis, AtomsError, LayerFeature, get_2d_axis
        from jamip.modeling.structureFactory import StructureFactory
        """ 提取结构中的二维单元，记录其化学式、空间群、价态 """
 
        cif = dim.cif
        dim.cutoff = 1.1
        dim.set_valence()
        dim.set_bonding()
        ndim = dim.search_cutoff()
        data = {'cif': cif, 'dim':ndim}
        if ndim != 2:
            return data

        # 如果包含2D结构单元,验证层方向是否与晶格方向正交
        graph, visited, ranks = dim.cache
        matrix = get_2d_axis(dim.structure, visited, ranks)
        if not (matrix is None) and matrix.size == 9 and abs(np.linalg.det(matrix)) > 1:
            sf = StructureFactory(dim.structure)
            sf.redefine(np.around(matrix))
            dim = DimensionAnalysis(sf.structure)
            dim.cutoff = 1.1
            dim.set_valence()
            dim.set_bonding()
            #dim.debug = True
            dim.search_cutoff()
            units = dim.get_units()
        else:
            units = dim.get_units()

#        for unit in units:
#            unit.get_stdcell()

        # 仅保留结构单元,不保存额外信息 
        try:
            #units = dim.get_units()
            lft = LayerFeature(dim)
            lft.set_unique_units(ignore_symop=False)
            #lft.set_interface(full=True)
        except AtomsError:
            return data
        except Exception as err:
            with open("error.log", 'a') as f:
                f.write(cif+'\n')
            value = sys.exc_info()
            six.reraise(*value)
            exit()

        allunit = []
        for i,unit in enumerate(lft.units):
            if unit.dim == 2:
                idx = cif.split('.')[0]
                filename = globalvar.output / ('%s-%d.vasp' %(idx, i))
                if globalvar.write_unit:
                    write(unit.stdcell, filename)
                unit.get_atoms_data(axis_index=2, tol=0.1, pbc=False)
                unit.get_edge_indices(charge=True, tol=0.3)
                dataset = unit.to_dict()
                dataset.update(data)
                dataset['path'] = filename.name
                allunit.append(dataset)
        #data['unit'] = json.dumps(unitlist)
        #print('inter',lft.get_interface_bonding_descriptor())
        #print('ou',unitlist)
        #print('uni',lft.units)
        #print('eqs',lft.equivalent_units)
        
        #data['unit_indices'] = [unit.index for unit in lft.equivalent_units]
        #data['operations'] = [unit.symop for unit in lft.equivalent_units]
        #data['distances'] = 
        #data['arearates'] = arearates 
        #data['spacings'] = [layer.spacing for layer in lft.layershifts] 
        #print(data)
 
        #logging.info("Finsih | "+cif)
 
        return allunit

    def filter_by_valence(self, df):
        from jamip.structure.atom import Composition

        indices = []
        for i,formula in enumerate(df['formula']):
            composition = Composition.from_formula(formula)
            # valence = best_valence
             

class UnitData:

    active = False

    def __init__(self, source:str, maps:dict, debug:bool=False):
        self.source = pathlib.Path(source)
        if not self.output.exists():
            self.output.mkdir(exist_ok=True, parent=True)
        self.active = True
        self.maps = maps
        self.debug = debug

    def generator(self, cifs):
        from jamip.structure import read, write, Structure
        from jamip.structure.dimension import Unit
 
        for cif in cifs:
            cif = self.source / cif
            data = {'cif': cif.name}
            valence = self.maps[cif.name]
            
            s = read(cif)
            unit = Unit.from_structure(s)
            unit.set_valence(valence)
            unit.debug = self.debug
            yield unit

    @staticmethod
    def run(unit):
        from jamip.structure import read, write, Structure
 
        #total_val = unit.total_val
        #if min(unit.valence.values()) > 0: total_val = 0
        if unit.debug == True:
            #print(unit)
            #unit.check_pointgroup_symmetry()
            #print("Check pointgroup symmetry success!")
            #unit.check_unique_pointgroup_symmetry()
            #print("Check unique pointgroup symmetry success!")
            pass
 
        unit.get_edge_indices()
        data.update(unit.to_dict())
 
        u1 = unit.sites['u1']
        d1 = unit.sites['d1']
 
        # 界面原子判据
        # 1. 元素种类一致 
        surface = True
        if u1.specie != d1.specie:
            surface = False
        # 2. 价态与总价态一致 
        # if surface and valence[u1.specie] * unit.total_charge <= 0:
        #     surface = False
        # 3. 次近邻原子未明显高出表面
        if surface and unit.sites['d2'] != None: 
            if d1.cartesian - unit.sites['d2'].cartesian > 0.2:
                surface = False
        if surface and unit.sites['u2'] != None: 
            if unit.sites['u2'].cartesian - u1.cartesian > 0.2:
                surface = False
        data['valid'] = surface
 
        # logging.info("Finsih | %s" %cif.name)
 
        return data


if __name__ == "__main__":

    # add dimension
    exit()

    # dim analysis
    jsons = run(get_dimension, df2['cif'])#.values[:10])
    with open('unit.json', 'a') as f:
        for data in jsons:
            if data != None:
                print(data)
                f.write(json.dumps(data, default=default_dump))
                f.write('\n')
