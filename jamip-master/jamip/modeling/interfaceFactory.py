# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as nlg
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

def get_match_rotate_angle(vector_set1, vector_set2, angle_tolerance=5):
    """
    The tranformation matrix to conver set2 to set1
    part1: assert gamma angle 1 = gamma angle 2
    part2: get_rotation_matrix by angle vec1[0] and vec2[0]
    """
    # Generate 3D lattice vectors for film super lattice
    # axis1 = np.cross(vector_set1[0], vector_set1[1])
    # axis2 = np.cross(vector_set2[0], vector_set2[1])
    # matrix1 = np.r_[vector_set1, [axis1]]
    # matrix2 = np.r_[vector_set2, [axis2]]
    # # scale %
    # matrix2 = matrix2 * (np.linalg.norm(matrix1,axis=1) / np.linalg.norm(matrix2,axis=1))[:,None]
    # # solve %
    # #transform_matrix = np.transpose(np.linalg.solve(matrix1, matrix2))
    # transform_matrix = np.linalg.solve(matrix1, matrix2)
    # return transform_matrix
    gamma1 = vec_angle(vector_set1[0], vector_set1[1])
    gamma2 = vec_angle(vector_set2[0], vector_set2[1])
    assert abs(gamma1 - gamma2) < angle_tolerance, "gamma angle not match"
    angle = vec_angle(vector_set1[0], vector_set2[0])
    # print(gamma1, gamma2)
    # print("rotate angle: ", angle)
    return angle

def reduce_vectors(a, b):
    """
    Generate independent and unique basis vectors based on the
    methodology of Zur and McGill
    """
    if np.dot(a, b) < 0:
        return reduce_vectors(a, -b)

    if np.linalg.norm(a) > np.linalg.norm(b):
        return reduce_vectors(b, a)

    if np.linalg.norm(b) > np.linalg.norm(np.add(b, a)):
        return reduce_vectors(a, np.add(b, a))

    if np.linalg.norm(b) > np.linalg.norm(np.subtract(b, a)):
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

class LayerFactory(StructureFactory):

    def __init__(self, structure, supercell_matrix, **kwargs):
        super().__init__(structure)
        self.supercell_matrix = supercell_matrix
        self._hkl = np.array([0,0,1], dtype='intc', order='C')
        self._surface_matrix = np.eye(3, dtype='intc', order='C')
        self._terminal = None
        self._axis = 2

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

    def set_fixed_center(self):
        """
        对于已经预移动的晶胞，直接确定边界
        """
        coords = self.structure.get_positions(type='direct')[:, self._axis]
        self._terminal = (coords.min(), coords.max())

    def set_center(self, tol=0):
        """
        基于最大层间距，将结构移至晶胞中心
        parameters:
            tol: 检查最大层间距是否够大
        """
        # get structure
        lattice_parameters=self.structure.lattice_parameters
        coords = self.structure.get_positions(type='direct')[:, self._axis]

        # get atom seps
        sort_coords = np.sort(coords)
        atom_seps = np.append(np.diff(sort_coords), 1+sort_coords[0]-sort_coords[-1])
        if max(atom_seps)*lattice_parameters[self._axis] < tol:
            raise RuntimeError('Warning! max separation in structure if less than tol.')

        # set atom terminal
        # coord: 1,2,3,4,5 , r=7
        # seps: 1,1,1,1,(1+7-5), max_seps=3, bottom=1, top=5
        itop = np.argmax(atom_seps)
        if itop == len(coords)-1:
            self._terminal = (sort_coords[0], sort_coords[-1])
        else:
            self._terminal = (sort_coords[itop+1], sort_coords[itop]+1)

    def set_single_layer(self, layer_index=None, bonding=None):
        """
        对于已经预移动的晶胞，直接确定边界
        """
        from jamip.structure.bonding import Bonding
        if bonding is None:
            bonding = Bonding(self.structure, method='min', cutoff=3.5, offset=0, factor=1.2, constraint=None, pbc=[True,True,True])
        if layer_index is None:
            layer_index = 0
            
        parents = np.array(bonding.data.get_parents())
        unique, inverse = np.unique(parents, return_inverse=True)
        indices = np.where(inverse == layer_index)
        coords = self.structure.get_positions(type='direct')[indices]
        self._terminal = (coords[:, self._axis].min(), coords[:, self._axis].max())

    def set_terminal(self, thickness:int, loweratom=None, upperatom=None, tol=1, z0=0):
        """
        对于体相结构，需要根据原子类型及厚度设置终端
        """
        from jamip.utils.utils import is_periodic
        from jamip.structure.atomic_number import number
        from jamip.utils.convert import counter
        import json

        # get structure
        lattice_parameters=self.structure.lattice_parameters
        lattice = self.structure.lattice
        elements = self.structure.get_elements(type='symbol')
        positions = self.structure.get_positions(type='direct')
        coords = positions[:, self._axis]

        # 将结构移至晶胞中心，进行分层操作
        self.set_center(tol=tol)
        cmin, cmax = self._terminal
        shift_coords = np.where((coords-cmin)>-1e-8, coords, coords+1)
        sort_indices = np.argsort(shift_coords)

        atom_layers = []
        layer = [sort_indices[0]]
        for index in np.arange(1,len(coords)):
            i,j = sort_indices[index-1], sort_indices[index]
            d = (shift_coords[j] - shift_coords[i]) * lattice_parameters[self._axis]
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
        
        print(layer_species)
        print(layer_index)

        # find first layer
        i = 0
        if loweratom != None:
            for i,species in enumerate(layer_species):
                cmin = np.min(shift_coords[atom_layers[i]])
                print(species, cmin, z0)
                if loweratom in json.loads(species) and cmin - z0 > -1e-8:
                    break

        # find end layer
        nlayer = len(layer_species)
        for i in range(i, 20*nlayer):
            if i < mini_periodic: continue
            Z = i // normal_periodic 
            i = i % normal_periodic
            cmax = np.max(shift_coords[atom_layers[i]]) + Z
#            print(i, json.loads(layer_species[layer_index[i]]))
            if (cmax-cmin)*lattice_parameters[self._axis] > thickness:
                if upperatom == None or upperatom in json.loads(layer_species[i]):
                    print(upperatom, json.loads(layer_species[layer_index[i]]))
                    break

        shift = 0.01/lattice_parameters[self._axis]
        atomshift = self._terminal[0]
        cmin = np.around(cmin+atomshift-shift,4)
        cmax = np.around(cmax+atomshift+shift,4)
        print(cmin, cmax)
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
                for atom in list(self._structure.atomic_positions):
                    if atom.scale_coord[2] - self._terminal[0]/Z < -1e-8 or atom.scale_coord[2] - self._terminal[1]/Z > 1e-8:
                        unit_atoms.append(atom)
#                        print(atom.scale_coord)
#                print(unit_atoms)
                self.del_atoms(unit_atoms)

        return self._structure 

    def get_vacuum(self):
        coords = self.structure.get_positions(type='direct')[:, self._axis]
        cmin = coords.min()
        cmax = coords.max()
        c = np.linalg.norm(self._structure.lattice[2])
        vacuum = (1 - (cmax-cmin)) * c
        return vacuum

class InterfaceFactory(object):

    axis = {'x':0, 'y':1, 'z':2}
    
    def __init__(self, layers, **kwargs):
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
            self._layers.append(LayerFactory(layer, supercell_matrix=None))
        self._structure = None
        # Check that the lattice vectors are perpendicular to the direction 

    @property
    def raw_layers(self):
        """
        raw structures.
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
                    miller1 = np.round(np.dot(vector1[i], np.linalg.pinv(vec_base1)))
                    miller2 = np.round(np.dot(vector2[j], np.linalg.pinv(vec_base2)))
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


    def twister_match(self, angle, nmiller=20, fix_angle=None, axis_index=2, 
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
        

    def attach(self, weights=None, shift=None, spacing=None, axis_index:int=2, **kwargs):
        """
        direction: [x, y, z, dtype]
        """
        from jamip.structure import Structure, write
        from .check import formated_rotation_angle

        # n = 0 
        gammas = []
        unioncell = {'lattice':[], 'positions':[], 'numbers':[]}
        for layer in self._layers:
            lattice, positions, elements = layer.structure.to_cell()
            unioncell['lattice'].append(lattice)
            unioncell['positions'].append(positions)
            unioncell['numbers'].append(elements)
            gamma = vec_angle(lattice[0], lattice[1])
            gammas.append(gamma)
            print('gamma1', gamma / np.pi * 180)
            # write(layer.structure, '%d.vasp' %n, )
            # n += 1

        # print(gammas)
        if np.max(gammas) - np.min(gammas) > 5 / 180 * np.pi:
            raise ValueError(f'gamma mismatch in axis {axis_index} ! max is {np.max(gammas)}, min is {np.min(gammas)}')

        # check lattice parameters
        if axis_index not in (0,1,2):
            raise ValueError('Only axis directions are supported')

        # set wetght % 
        if weights is None:
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
            if 'angle' in kwargs:
                angle = formated_rotation_angle(kwargs['angle'])[0]
            else:
                angle = get_match_rotate_angle(lattice2[:2], lattice1[:2])
            if angle != 0:
                # angle=-angle
                R = np.array([[np.cos(angle),-np.sin(angle),0],
                              [np.sin(angle),np.cos(angle),0],
                              [0,0,1]])
                unioncell['lattice'][1] = lattice2 @ R
                print('rotate lattice1 by %f'%(angle/np.pi*180))
                print('lattice1:',unioncell['lattice'][0])
                print('lattice2:',unioncell['lattice'][1])
                
                gamma1 = vec_angle(lattice1[0], lattice1[1])
                print('gamma3', gamma1 / np.pi * 180)
                gamma1 = vec_angle(lattice2[0], lattice2[1])
                print('gamma3', gamma1 / np.pi * 180)
                gamma2 = vec_angle(unioncell['lattice'][0][0], unioncell['lattice'][1][0])        
                print('gamma3', gamma2 / np.pi * 180)
                if abs(gamma2 - np.pi) < 1e-3:
                    unioncell['lattice'][1] = lattice2 @ R.T
                # exit()

        # shift positions
        for i,vec in enumerate(shift):
            unioncell['positions'][i] += vec

        # set lattice
        lattice = np.eye(3)
        for i in range(3):
            # angle match %
            vectors = np.array([lattice[i] for lattice in unioncell['lattice']])
            # print(vectors)
            d = pdist(vectors, 'cosine')
            # print(d)
            #assert max(d) < 0.01, 'angle mismatch in axis %s ! max cosine is %.2f' % (i, max(d))

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
        
        s = Structure.from_cell((lattice, positions, numbers))
        return s

    def attach_with_sites(self, units, layersites, layershifts, weights=None):
        from jamip.structure.symmtery import EqualTools, get_lattice_transport
        from jamip.structure.dimension import Unit, normalized
        from jamip.structure.structure import Structure
        from collections import defaultdict

        def rescale(lattice, a,b,c=None):
            a1,b1,c1 = np.linalg.norm(lattice, axis=1)
            lattice[0] *= a/a1
            lattice[1] *= b/b1
            if c != None:
                lattice[2] *= c/c1
            return lattice

        # 此时晶格是需要重新生成的, 其中xy方向基于权重的缩放，z方向基于直接累加
        # 由于这里处理的都是四方相或六方向，a/b长度相图，不需要对a/b排序
        # 直接提取a1, a2, a' = w1a1 + w2a2, x = x / a1 * a' , y = y / a1 / a'
        # 重设lattice1和lattice2，生成结构layers
        if weights is None:
            weights = np.ones(len(layersites))/len(layersites)

        va = []
        vb = []
        vc = []
        for site in layersites:
            # 如果stdcell与原结构平移不等价怎么办?
            unit = units[site.index]
            stdcell = unit.stdcell
            lattice = stdcell.lattice            
            # get vc %
            va.append(np.linalg.norm(lattice[0]))
            vb.append(np.linalg.norm(lattice[1]))
            vc.append(unit.spacing_in_cartesian)

        #write(units[0].unitcell, 'u1.vasp')
        #write(units[1].unitcell, 'u2.vasp')

        for shift in layershifts:
            vc.append(shift.spacing_in_cartesian)

        mva = np.sum(np.array(va)*weights)
        mvb = np.sum(np.array(vb)*weights)
        mvc = np.sum(vc)
        #print(vc, mvc)

        std_lattice = rescale(units[0].stdcell.lattice, mva, mvb, mvc)
        equal = EqualTools.from_lattice(std_lattice)
        #print(std_lattice)
        #print(units[0].stdcell.lattice)
        #thickness = []

        units_before_shift = []
        for site in layersites:
            stdcell = units[site.index].stdcell
            cell = stdcell.to_cell()
            pbc_vectors = Unit.get_pbc_vectors(cell[1], axis_index=2)
            operation = np.linalg.inv(equal.get_operation(site.symop))
            lattice = rescale(cell[0], mva, mvb)
            matrix = get_lattice_transport(lattice, std_lattice)
            positions = (cell[1] - pbc_vectors) @ operation @ matrix
            positions[:,2] -= np.floor(np.min(positions[:,2]))
            #write(stdcell, 'i%d.vasp' %site.index)

            # 重建Unit类,以获得上下界面的原子
            unitcell = Structure.from_cell((std_lattice, positions, cell[2]))
            unit = Unit.from_structure(unitcell, np.arange(len(positions)))
            unit.valences = units[site.index].valences
            if matrix[2,2] < 1: 
                unit.reset_pbc_vectors()
            unit.get_stdcell()
            unit.get_atoms_data(axis_index=2, tol=0.1, pbc=False)
            unit.get_edge_indices(charge=True, tol=0.3)
            #unit.get_interface_atoms(axis_index=2, tol=0.0001, pbc=False, charge=True)
            #unit.get_interface_atoms(axis_index=2, tol=0.01, pbc=False, charge=False)
            units_before_shift.append(unit)

        # 建好结构后，计算层间的平移向量
        layer_shifts = []
        for i,layer in enumerate(layershifts):
            j = i
            k = 0 if i+1 == len(layershifts) else i+1
            # 初始化上下层的信息, 其中unit1为下层，unit2为上层
            unit1 = units_before_shift[j]
            unit2 = units_before_shift[k]
            dsites = unit1.get_positions_by_key('u1')
            usites = unit2.get_positions_by_key('d1')

            # 基于下层位点+层间成键，获取上层位点
            coord_shift = dsites[None,:,:] + layer.shift[:,None,:]
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

            # 基于上层位点，求解平移向量
            indices = EqualTools.istranslate(unique_shifts, usites)
            if indices == None:
                print(layer)
                print("===1===")
                print(coords)
                print("===2===")
                print(unique_shifts)# - unique_shifts[2])
                print("===3===")
                coords2 = coords2 # *np.array([-1,-1,1])
                print(coords2)# - coords2[2])
                print("get shifts failed.")
                raise RuntimeError("get shifts failed.")

            layer_shift = normalized(unique_shifts - usites[indices]).mean(axis=0)
#            print(layer_shift)
            layer_shift[2] += layer.spacing_in_cartesian / mvc 
#            print(layer.spacing_in_cartesian / mvc)
#            print(layer_shift)
            layer_shifts.append(layer_shift)

            # 验证一下成键距离
            vectors = normalized(dsites[:,None,:] - usites[None,:,:] - layer_shift)
            grid = np.mgrid[-1:2, -1:2, 0:1].reshape(3,-1).T # x,3
            grid_vectors = grid[:,None,None,:] + vectors[None,:,:,:]
            distances = np.min(np.linalg.norm(grid_vectors @ std_lattice, axis=3), axis=0)
            distance_in_layer = np.sqrt((np.linalg.norm(std_lattice[0]) * layer.distance_area_rate)**2 + layer.spacing_in_cartesian**2)
#            print('ds', distances, distance_in_layer)
            #if layer.distance - np.min(distances) > 0.1:
            #    # 原则上应该完全一样，但这里我们只修正距离近的情况
            #    print("distance set error! layer_distance: %.4f, min_distance: %.4f" %(layer.distance, np.min(distances)))
            #    # 如果希望修正，需要重设mvc, 还是交给m3gnet吧
            #print(layer_shifts)
            #print(j, coords1)

        # 累加每层的平移向量，一个周期的平移应为0
        #print(np.array(layer_shifts))
        layer_shifts_sum = np.roll(np.cumsum(layer_shifts, axis=0),1, axis=0)
        if np.sum(np.abs(normalized(layer_shifts_sum[0]))) > 3e-3:
            raise OSError("PBC condition failed")

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

        d1 = np.min(structure.get_all_distances())
        d2 = np.min([layer.spacing_in_cartesian for layer in layershifts])
        #print(units, d1, d2)
        #if (d1+1 < d2 and d1 < 2) or d1 < 1.5:
        #    print(d1,d2)
        #    write(structure, 'test.vasp')
        #    print(layershifts)
        #    exit()
        
        return structure

