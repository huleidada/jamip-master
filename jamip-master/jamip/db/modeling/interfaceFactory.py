# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as nlg
from ..utils.variables import default_constants
from .structureFactory import StructureFactory
from scipy.spatial.distance import pdist
import pandas as pd

tolerate_angle_mismatch = 0.01
tolerate_vector_mismatch = 0.04

def vec_area(vector12):
    return np.linalg.norm(np.cross(vector12[0], vector12[1]))

def vec_angle(vector1, vector2):
    cos_ = np.dot(vector1,vector2)
    sin_ = np.linalg.norm(np.cross(vector1,vector2))
    return np.arctan2(sin_, cos_)

def get_2dmask(axis:int):
    mesh = np.mgrid[0:3, 0:3]
    mask = np.where((mesh[0]!=axis)&(mesh[1]!=axis))
    return (mask[0].reshape(2,2), mask[1].reshape(2,2))

def get_factors(n):
    """
    Generate all factors of n
    """
    for x in range(1, n + 1):
        if n % x == 0:
            yield x

def gen_sl_transform_matricies(area_multiple):
    return [
        np.array(((i, j), (0, area_multiple / i)))
        for i in get_factors(area_multiple)
        for j in range(area_multiple // i)
    ]

def get_match_transformation(vector_set1, vector_set2):
    """The tranformation matrix to conver set2 to set1"""
    # Generate 3D lattice vectors for film super lattice
    axis1 = np.cross(vector_set1[0], vector_set1[1])
    axis2 = np.cross(vector_set2[0], vector_set2[1])
    matrix1 = np.r_[vector_set1, [axis1]]
    matrix2 = np.r_[vector_set2, [axis2]]
    # scale %
    matrix2 = matrix2 * (np.linalg.norm(matrix1,axis=1) / np.linalg.norm(matrix2,axis=1))[:,None]
    # solve %
    #transform_matrix = np.transpose(np.linalg.solve(matrix1, matrix2))
    transform_matrix = np.linalg.solve(matrix1, matrix2)
    return transform_matrix


def reduce_vectors(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill
    """
    from numpy.linalg import norm as fast_norm

    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)

    if fast_norm(a) > fast_norm(b):
        return reduce_vectors(b, a)

    if fast_norm(b) > fast_norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))

    if fast_norm(b) > fast_norm(np.subtract(b, a)):
        return reduce_vectors(a, np.subtract(b, a))

    return [a, b]


def shape_supercell_matrix(smat):
    if smat is None:          
        _smat = np.eye(3, dtype='intc', order='C')
    elif len(np.ravel(smat)) == 3:
        _smat = np.diag(smat)
    elif len(np.ravel(smat)) == 4:
        _smat = np.eye(3, dtype='intc', order='C')
        _smat[:2,:2] = np.around(smat).reshape(2,2)
    elif len(np.ravel(smat)) == 9:
        _smat = np.array(smat, dtype='intc').reshape(3,3)
    else:
        msg = "supercell_matrix shape has to be (3,) or (3, 3)"
        raise RuntimeError(msg)
    return _smat

class Supercell(StructureFactory):

    def __init__(self, structure, supercell_matrix, **kwargs):
        super().__init__(structure)
        self.supercell_matrix = supercell_matrix
        self._hkl = np.array([0,0,1], dtype='intc', order='C')
        self._surface_matrix = np.eye(3, dtype='intc', order='C')
        self._terminal = None

    @property
    def hkl(self):
        return self._hkl

    @hkl.setter
    def hkl(self, value):
        hkl = np.array(value, dtype=int).reshape(3,)
        smat = self._surface(self.raw_structure, hkl)
        self._hkl = hkl
        self._surface_matrix = shape_supercell_matrix(smat)

    @property
    def surface_matrix(self):
        return self._surface_matrix

    @property
    def supercell_matrix(self):
        return self._supercell_matrix

    @property
    def redefine_matrix(self):
        return np.dot(self._surface_matrix, self._supercell_matrix)

    @supercell_matrix.setter
    def supercell_matrix(self, smat):
        self._supercell_matrix = shape_supercell_matrix(smat)

    def equivalent_vectors(self, nmiller=10, tol=1e-3):

        base_lattice = np.dot(self.surface_matrix, self.raw_structure.lattice)
        super_lattice = np.dot(self.supercell_matrix, base_lattice)

        r1 = np.linalg.norm(super_lattice[0])
        r2 = np.linalg.norm(super_lattice[1])
        angle = vec_angle(super_lattice[0], super_lattice[1])
        area = vec_area(super_lattice[:2])
#        print(r1, r2, angle, area)

        nmiller = np.array(nmiller, dtype=int).ravel()
        # 遍历向量的米勒指数
        if nmiller.size == 1:
            n = nmiller[0]
            N = np.mgrid[-n:n+1,-n:n+1].reshape(2,-1).T
        elif nmiller.size == 2:
            n, m = nmiller
            N = np.mgrid[-n:n+1,-m:m+1].reshape(2,-1).T
        vectors = N @ base_lattice[:2,:2]                           
        length = np.linalg.norm(vectors, axis=1)
        mask1 = np.where(np.abs(length-r1)<tol)[0]       # 向量1的等长向量
        mask2 = np.where(np.abs(length-r2)<tol)[0]       # 向量2的等长向量
        # 计算向量的角坐标，获得夹角
        angle_coord1 = np.arctan2(vectors[mask1,1], vectors[mask1,0])
        angle_coord2 = np.arctan2(vectors[mask2,1], vectors[mask2,0])
        angles = angle_coord1[:,None]-angle_coord2[None,:]
        angles = np.where(angles>=np.pi, angles-2*np.pi, angles)
        angles = np.where(angles<-np.pi, angles+2*np.pi, angles)
        mesh1, mesh2 = np.where(np.abs(angles-angle)<1e-6)
        angle_coord0 = np.arctan2(super_lattice[0,1], super_lattice[0,0])
        N1 = N[mask1][mesh1]
        N2 = N[mask2][mesh2]
        angles = angle_coord1[mesh1]-angle_coord0
        angles = np.where(angles>=np.pi, angles-2*np.pi, angles)
        angles = np.where(angles<-np.pi, angles+2*np.pi, angles)
        angles = angles/np.pi*180
        results = {}
        for i in range(len(N1)):
            results[angles[i]] = [N1[i],N2[i]]
        return results

    def multiple(self, vectors):
        """
        基于平移操作，重复结构中的原子
        parameters:
            axis_index: 垂直层的方向
            tol: 检查最大层间距是否够大
        """
        atoms = []
        for atom in self.structure.atoms:
            element=atom.element.symbol
            for vector in vectors:
                coord = np.array(atom.position) + vector
                formated_atom = [element] + coord.tolist()
                atoms.append(formated_atom)
        self.add_atoms(atoms)
        self._raw_structure = self._structure

    def symmetry_operation(self, symop, shift=None, axis_index=2, **kwargs):
        '''
        site: direct position (3,)
        theta: radian
        '''
        from jamip.db.materials.structure import Structure
        from jamip.structure.symmetry import Symmetry_operation

        # get rotate_matrix
        if axis_index != 2:            
            raise ValueError('Come soon ....')        
        poscar = self.structure.formatting('poscar')
        operation = Symmetry_operation[symop]
        shift = np.array(shift) if not (shift is None) else np.zeros(3)

        positions = (positions['positions'] + shift) @ operation 
        # shift z if minus in positions
        if min(poscar['positions'][:,2]) < -1e-8 and max(poscar['positions'][:,2]) > 1e-8:
            poscar['positions'][:,2] += min(poscar['positions'][:,2]) * -1 + 1e-4

        # create new structure 
        self._raw_structure = Structure().create(poscar)

    def fixed(self, axis_index=2):
        """
        对于已经预移动的晶胞，直接确定边界
        """
        cell=self.structure.formatting('cell')
        positions = cell['positions']
        coords = positions[:, axis_index]
        self._terminal = (min(coords), max(coords))

    def center(self, axis_index=2, tol=0):
        """
        基于最大层间距，将结构移至晶胞中心
        parameters:
            axis_index: 垂直层的方向
            tol: 检查最大层间距是否够大
        """
        # get structure
        cell=self.structure.formatting('cell')
        lattice_parameters=self.structure.lattice_parameters
        positions = cell['positions']
        coords = positions[:, axis_index]

        # get atom seps
        sort_coords = np.sort(coords)
        atom_seps = np.append(np.diff(sort_coords), 1+sort_coords[0]-sort_coords[-1])
        if max(atom_seps)*lattice_parameters[axis_index] < tol:
            raise RuntimeError('Warning! max separation in structure if less than tol.')

        # set atom terminal
        # coord: 1,2,3,4,5 , r=7
        # seps: 1,1,1,1,(1+7-5), max_seps=3, bottom=1, top=5
        itop = np.argmax(atom_seps)
        if itop == len(coords)-1:
            self._terminal = (sort_coords[0], sort_coords[-1])
        else:
            self._terminal = (sort_coords[itop+1], sort_coords[itop]+1)

    def set_terminal(self, thickness:int, loweratom=None, upperatom=None, tol=1, axis_index=2, z0=0):
        """
        对于体相结构，需要根据原子类型及厚度设置终端
        """
        from jamip.utils.utils import is_periodic
        from jamip.structure.atomic_number import number
        from jamip.utils.convert import counter
        import json

        # get structure
        cell=self.structure.formatting('cell')
        lattice_parameters=self.structure.lattice_parameters
        lattice = cell['lattice']
        elements = cell['numbers']
        positions = cell['positions']
        coords = positions[:, axis_index]

        # 将结构移至晶胞中心，进行分层操作
        self.center(axis_index, tol)
        cmin, cmax = self._terminal
        shift_coords = np.where((coords-cmin)>-1e-8, coords, coords+1)
        sort_indices = np.argsort(shift_coords)

        atom_layers = []
        layer = [sort_indices[0]]
        for index in np.arange(1,len(coords)):
            i,j = sort_indices[index-1], sort_indices[index]
            d = (shift_coords[j] - shift_coords[i]) * lattice_parameters[axis_index]
            if d < tol:
                layer.append(j)
            else:
                atom_layers.append(layer)
                layer = [j]
        atom_layers.append(layer)

        # 统计每层的原子类型，确定层周期
        layer_species = [json.dumps(counter([elements[i] for i in layer])) for layer in atom_layers]
        species, layer_index = np.unique(layer_species, return_inverse=True)

        # 层状结构单元至少要包含一个周期。如果在晶胞内层有更小的重复周期，则取那个周期
        unit = is_periodic(layer_index)
        mini_periodic = normal_periodic = len(layer_index)
        if unit != None and normal_periodic % len(unit) == 0:
            mini_periodic = len(unit)
        
        # find first layer
        i = 0
        if loweratom != None:
            for i,species in enumerate(layer_species):
                cmin = np.min(shift_coords[atom_layers[i]])
                if number[loweratom] in json.loads(species) and cmin - z0 > -1e-8:
                    break

        # find end layer
        nlayer = len(layer_species)
        for i in range(i, 20*nlayer):
            if i < mini_periodic: continue
            Z = i // normal_periodic 
            i = i % normal_periodic
            cmax = np.max(shift_coords[atom_layers[i]]) + Z
            if (cmax-cmin)*lattice_parameters[axis_index] > thickness:
                if upperatom == None or number[upperatom] in json.loads(layer_species[layer_index[i]]):
                    break

        shift = tol/lattice_parameters[axis_index]/10
        atomshift = self._terminal[0]
        cmin = np.around(cmin+atomshift-shift,4)
        cmax = np.around(cmax+atomshift+shift,4)
        self._terminal = (cmin-np.floor(cmin), cmax-np.floor(cmin))

    @property
    def structure(self):
        """
        structure object after operation.
        """
        self._structure = self.raw_structure

        if self._terminal is None:
            self.redefine(self._supercell_matrix)
        else:
            Z = int(np.ceil(self._terminal[1]))
            if Z == 0:
                self.redefine(self._supercell_matrix)
            else:
                matrix = np.diag([1,1,Z])
                supercell_matrix = np.dot(self.redefine_matrix, matrix)
                self.redefine(supercell_matrix)
                unit_atoms=[]
                for atom in list(self._structure.atoms):
                    if atom.position[2] - self._terminal[0]/Z < -1e-8 or atom.position[2] - self._terminal[1]/Z > 1e-8:
                        unit_atoms.append(atom.to_formated_atom())
                self.del_atoms(unit_atoms)

        return self._structure 

class InterfaceFactory(object):

    axis = {'x':0, 'y':1, 'z':2}
    
    def __init__(self, layers, isPersist=False, **kwargs):
        """
        Arguments:
            layers: structure's object. tuple
            isOperateOnSelf: Whether to operate itself.
            isPersist (default=False): whether to save to the database.
            
            kwargs:
                isCloneFullInfo (default=False): whether to clone all information of structure.
        """
        assert len(layers) >= 2
        self._layers = []
        for layer in layers:
            self._layers.append(Supercell(layer, supercell_matrix=None))
        self._structure = None
        # Check that the lattice vectors are perpendicular to the direction 

    @property
    def raw_layers(self):
        """
        raw structure.
        """
        return [layer.raw_structure for layer in self._layers]

    @property
    def structure(self):
        """
        structure object after operation.
        """

    def match_zsl(self, max_area=400, area_tol=0.1, length_tol=0.05, angle_tol=0.08):
        """
        https://www.sciencedirect.com/science/article/pii/S0927025615003365#b0050
        """
        vec_base1 = self._layers[0].structure.lattice[:2]
        vec_base2 = self._layers[1].structure.lattice[:2]
        area1 = vec_area(vec_base1)
        area2 = vec_area(vec_base2)
      
        # create mesh %
        area_size1 = np.arange(1, int(max_area / area1))
        area_size2 = np.arange(1, int(max_area / area2))
        mesh1,mesh2 = np.meshgrid(area_size1, area_size2)
        mesh1 = mesh1.ravel()
        mesh2 = mesh2.ravel()

        mask1 = np.where(np.abs(mesh1/mesh2-area2/area1)<area_tol)
        mask2 = np.where(np.abs(mesh2/mesh1-area1/area2)<area_tol)
        mask = np.unique(np.r_[mask1[0], mask2[0]])
        area12 = np.c_[mesh1[mask],mesh2[mask]]
        
        for i,j in np.sort(area12, axis=0):

            vector1 = gen_sl_transform_matricies(i) @ vec_base1 
            vector2 = gen_sl_transform_matricies(j) @ vec_base2
            vector1 = [reduce_vectors(v1,v2) for v1,v2 in vector1]
            vector2 = [reduce_vectors(v1,v2) for v1,v2 in vector2]
            length1 = np.linalg.norm(vector1, axis=-1) # [n,2]
            length2 = np.linalg.norm(vector2, axis=-1) # [m,2]
            length12 = np.abs( length2[None,:,:] / length1[:,None,:] - 1 )
            # distances search %
            mask12 = np.where((length12[:,:,0] < length_tol) & (length12[:,:,1] < length_tol))

            for i,j in zip(*mask12):
                angle1 = vec_angle(vector1[i][0], vector1[i][1]) 
                angle2 = vec_angle(vector2[j][0], vector2[j][1]) 
                if np.abs(angle2/angle1 - 1) < angle_tol:
                    miller1 = np.dot(vector1[i], np.linalg.pinv(vec_base1))
                    miller2 = np.dot(vector2[j], np.linalg.pinv(vec_base2))
                    yield (miller1, miller2, vector1[i], vector2[j])


    @classmethod
    def match(self, lattice1, lattice2, nmiller=10, fix_angle=None,
              tolerate_vector_mismatch=0.04, tolerate_angle_mismatch=0.01, **kwargs):
        """
        以手动旋转的方式匹配两个平面格子
        """
        assert lattice1.shape==(2,2)
        assert lattice2.shape==(2,2)

        # create miller grid
        nmiller = np.array(nmiller, dtype=int).ravel()
        if nmiller.size == 1:
            n1 = n2 = m1 = m2 = nmiller[0]
        elif nmiller.size == 2:
            n1 = n2 = nmiller[0]
            m1 = m2 = nmiller[1]
        elif nmiller.size == 4:
            n1, n2, m1, m2 = nmiller
        N = np.mgrid[-n1:n1+1,-n2:n2+1,-m1:m1+1,-m2:m2+1].reshape(4,-1).T
        N = N[N.any(axis=1)]  # exclude (0,0,0,0)
        df = pd.DataFrame(N, columns=['n1', 'n2', 'm1', 'm2'])
        df['idx1'] = np.arange(len(N))

        vector1 = N[:,:2] @ lattice1                           # 向量1 
        vector2 = N[:,2:] @ lattice2                           # 向量2
        # 由于使用的是向量差的模长，向量对长度相同，方向相同，无需考虑夹角问题
        #df['dist1'] = np.linalg.norm(vector1-vector2, axis=1)
        #df = df[df['dist1']<tolerate_vector_mismatch]
        df = df[np.linalg.norm(vector1-vector2, axis=1)<tolerate_vector_mismatch]
        nmatch = len(df)

        if nmatch > 1:

            # 从向量对中选取非同向的对(行列式不为0)，当然也可以计算面积并进行截断
            pair_mask = np.triu_indices(nmatch, k=1)
            data1 = df.values[pair_mask[0]]
            data2 = df.values[pair_mask[1]]
            data = np.c_[data1, data2] 
            adf = pd.DataFrame(data, columns=['n1', 'n2', 'm1', 'm2', 'idx1', 'n3', 'n4', 'm3', 'm4', 'idx2', ])

            pair_vector11 = vector1[adf['idx1']]
            pair_vector12 = vector1[adf['idx2']]

            if fix_angle != None:
                len11 = np.linalg.norm(pair_vector11, axis=1)
                len12 = np.linalg.norm(pair_vector12, axis=1)
                cosine=np.divide(np.einsum('ij, ij->i', pair_vector11, pair_vector12),len11*len12) #基矢A1和A2之间的余弦值
                fix_angle_radian = fix_angle*np.pi/180
                adf['angle_mismatch'] = np.abs(cosine-np.cos(fix_angle_radian))
                adf = adf[adf['angle_mismatch']<tolerate_angle_mismatch] #与固定的夹角之间余弦值的差值

            else:
                pair_area = np.abs(np.cross(pair_vector11,pair_vector12,axis=1))
                unitarea = max(vec_area(lattice1), vec_area(lattice2))
                adf = adf[pair_area >= unitarea]

            if len(adf) > 0:
                # sort result by mismatch and vector length
                adf['vector_mismatch'] = np.linalg.norm(vector1[adf['idx1']]-vector2[adf['idx1']], axis=1) + \
                                         np.linalg.norm(vector1[adf['idx2']]-vector2[adf['idx2']], axis=1)
                adf['vector_length'] = np.linalg.norm(vector1[adf['idx1']], axis=1) + np.linalg.norm(vector1[adf['idx2']], axis=1)
                adf = adf.sort_values(by=['vector_mismatch','vector_length'])
                result = adf[['n1','n2','n3','n4', 'm1','m2','m3','m4', 'vector_mismatch','vector_length']].copy()
                if 'angle_mismatch' in adf.columns:
                    result['angle_mismatch'] = adf['angle_mismatch']

                return result 


    def twister_match(self, angles, nmiller, fix_angle=None, axis_index=2, 
                      tolerate_vector_mismatch=0.04, tolerate_angle_mismatch=0.01, **kwargs):
        """
        angle: Upper layer cell rotation Angle, unit degree
        nmiller: Maximum allowable Miller index
        fix_angle: Fixed Angle of the matching lattice, unit degree
        attach_axis: The lattice direction for interface matching
        tolerate_vector_mismatch: Lattice vector mismatch tolerance, assert |a-a'| < tol 
        tolerate_angle_mismatch: Lattice vector mismatch tolerance, assert |cos(alpha) - cos(alpha')| < tol 
        """
        layer1 = self._layers[0].structure
        layer2 = self._layers[1].structure

        for angle in angles:
            # get rotate matrix
            initial_angle = angle*np.pi/180
            R = np.array([[np.cos(initial_angle),-np.sin(initial_angle)],
                          [np.sin(initial_angle),np.cos(initial_angle)]]) 
         
            # get lattice 
            mask = get_2dmask(axis_index)
            lattice1 = layer1.lattice[mask]
            lattice2 = layer2.lattice[mask] @ R
            df = self.match(lattice1, lattice2, nmiller, fix_angle, tolerate_vector_mismatch, tolerate_angle_mismatch) 
            if not (df is None): 
                # create trans matrix base first return
                row = df.values[0]
                matrix1 = np.array([[row[0], row[1], 0],
                                    [row[2], row[3], 0],
                                    [     0,      0, 1]]).astype(int) 
                matrix2 = np.array([[row[4], row[5], 0],
                                    [row[6], row[7], 0],
                                    [     0,      0, 1]]).astype(int)
                yield round(angle,6), matrix1, matrix2, row[8:]
        

    def attach(self, weights=None, shift=None, spacing=None, axis_index:int=2, isPersist=False, **kwargs):
        """
        direction: [x, y, z, dtype]
        """
        # from ..materials.atom import Atom
        from ..materials.structure import Structure
        from ..utils.convert import any2radian, any2direct, cell2poscar

        unioncell = {'lattice':[], 'positions':[], 'numbers':[]}
        for layer in self._layers:
            cell = layer.structure.formatting('cell')
            unioncell['lattice'].append(cell['lattice'])
            unioncell['positions'].append(cell['positions'])
            unioncell['numbers'].append(cell['numbers'])

        # check lattice parameters
        if axis_index not in (0,1,2):
            raise ValueError('Only axis directions are supported')

        # set wetght % 
        if weights is None:
            #weights = np.ones(len(self._layers)) / len(self._layers) 
            weights = np.zeros(len(self._layers))
            weights[0] = 1
        elif len(weights) == len(self._layers):
            weights = np.array(weights) / sum(weights)
        else:
            raise ValueError('Invalid lattice weight!')

        # set spacing & len(spacing) >= 2 %
        spacing = np.atleast_1d(spacing)
        if len(spacing) == 1:
            spacing = np.repeat(spacing, len(self._layers))
        elif len(spacing) != len(self._layers):
            raise ValueError('Invalid layer spacing!')

        # set shifts %
        if shift is None:
            shift = np.zeros((len(self._layers),3))
        else:
            shift = np.atleast_2d(shift)
            if len(self._layers) != len(shift):
                raise ValueError('Invalid layer shift!')

        # rotate
        # 在晶格匹配后，虽然晶格矢量的长度和夹角一致，向量的方向未必一致，通过自动或手动旋转使晶格一致
        # 对于晶格大小不一致的情况，旋转比坐标转换稳定
        # 这里需要验证一下
        if len(self._layers) == 2:
            lattice1 = unioncell['lattice'][0]
            lattice2 = unioncell['lattice'][1]
            if 'angle' in kwargs and kwargs['angle'][1].lower() in ['radian','degree']:
                angle = any2radian(kwargs['angle'])[0]
                R = np.array([[np.cos(angle),-np.sin(angle)],
                              [np.sin(angle),np.cos(angle)]])
                mask = get_2dmask(axis_index)
                unioncell['lattice'][0][mask] = np.dot(lattice1[mask],R)
            else:  # auto matrix
                trans = get_match_transformation(lattice2[:2], lattice1[:2])
                unioncell['lattice'][1] = np.dot(trans, lattice2)

        # shift positions
        for i,vec in enumerate(shift):
            unioncell['positions'][i] += vec

        # set lattice
        lattice = np.eye(3)
        for i in range(3):
            # angle match %
            vectors = np.array([lattice[i] for lattice in unioncell['lattice']])
            d = pdist(vectors, 'cosine')
            assert max(d) < 0.01, 'angle mismatch in axis %s !' %i

            if axis_index != i:
                #rmatch = nlg.norm(a-b)  # lattice_tolerance < 1e-2
                lattice[i] = weights @ vectors
            else:
                # 计算每个区间的原子层厚度，中间用 vaccum/2 , spacing1, spacing2, ... , vaccum/2 填充
                #amatch = vec_angle(a,b)  # angle_tolerance < 1/180*np.pi
                hs = [np.linalg.norm(lattice[i]) for lattice in unioncell['lattice']]
                pmin = [positions[:,i].min() for positions in unioncell['positions']]
                pmax = [positions[:,i].max() for positions in unioncell['positions']]
                h = sum([(pmax[j]-pmin[j])*hs[j] for j in range(len(hs))]) + sum(spacing) 
                lattice[i] = unioncell['lattice'][0][i] / hs[0] * h
                
                # shift positions %
                height = spacing[-1]/2
                for j,positions in enumerate(unioncell['positions']):
                    positions[:,i] = ((positions[:,i] - pmin[j]) * hs[j] + height) / h 
                    height += (pmax[j] -pmin[j]) * hs[j] + spacing[j]
                    #if j < len(spacing):
                    #    height += spacing[j]

        # attach atoms %
        numbers = np.r_[tuple(unioncell['numbers'])]
        positions = np.r_[tuple(unioncell['positions'])][np.argsort(numbers)]
        numbers = np.sort(numbers)
        
        cell_new={'lattice': lattice,
                  'positions': positions,
                  'numbers': numbers}
        
        return Structure().create(raw_structure=cell2poscar(cell_new))
