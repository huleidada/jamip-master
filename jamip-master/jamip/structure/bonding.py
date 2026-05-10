import numpy as np
from collections import namedtuple, defaultdict
from jamip.structure.elementInfo import ElementData

covalent_radius = ElementData['covalent_radius'].to_dict()

Atom = namedtuple('Atom',['element','coord','unid'])
BaseBond = namedtuple('BaseBond',['a1','a2','distance','vector'])
VoroBond = namedtuple('VoroBond',['a1','a2','distance','vector','solid_angle','volume', 'area'])

def get_rdf(structure, shifts, r_cutoff):
    from scipy.spatial import KDTree

    # center cell
    lattice = structure.lattice
    positions = structure.get_positions(type='direct')
    positions -= np.floor(positions)
    num_atoms = len(structure)

    # neighbor cell
    neighbors = positions[None,:,:] + shifts[:,None,:]
    points = np.r_[positions, neighbors.reshape(-1,3)] @ lattice

    tree = KDTree(points)
    pairs = tree.query_pairs(r_cutoff)
    pairs = np.array(list(pairs))

    # update cutoff
    if pairs.size == 0:
        pairs = tree.query_pairs(r_cutoff*1.5)
        pairs = np.array(list(pairs))

    if pairs.size > 0:
        mask = (pairs[:, 0] < num_atoms) | (pairs[:, 1] < num_atoms)
        pairs = pairs[mask]
    vector = points[pairs[:,1]] - points[pairs[:,0]]  # 成键向量
    distances = np.linalg.norm(vector, axis=-1)       # 成键长度（向量的模）

    pair_indices = pairs % num_atoms 
    shift_indices = pairs[:,1] // num_atoms

    return distances, pair_indices.T, shift_indices

def solid_angle(center, coords):
    """
    Helper method to calculate the solid angle of a set of coords from the
    center.

    Args:
        center (3x1 array): Center to measure solid angle from.
        coords (Nx3 array): List of coords to determine solid angle.

    Returns:
        The solid angle.
    """
    r = np.array(coords) - center
    r_norm = np.linalg.norm(r, axis=1)

    # Compute the solid angle for each tetrahedron that makes up the facet
    #  Following: https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
    angle = 0
    for i in range(1, len(r) - 1):
        j = i + 1
        tp = np.abs(np.dot(r[0], np.cross(r[i], r[j])))
        de = (
            r_norm[0] * r_norm[i] * r_norm[j]
            + r_norm[j] * np.dot(r[0], r[i])
            + r_norm[i] * np.dot(r[0], r[j])
            + r_norm[0] * np.dot(r[i], r[j])
        )
        if de == 0:
            my_angle = 0.5 * np.pi if tp > 0 else -0.5 * np.pi
        else:
            my_angle = np.arctan(tp / de)
        angle += (my_angle if my_angle > 0 else my_angle + np.pi) * 2

    return angle


class BondCore:

    def __init__(self, distances, pair_indices, shift_indices, cell, shifts, radius_maps=None):
        self.distances = distances          # [nbond, ]
        self.pair_indices = pair_indices    # [nbond, 2]
        self.shift_indices = shift_indices  # [nbond, ]
        self.lattice = np.array(cell[0])    # [3, 3]
        self.positions = np.array(cell[1])  # [natom, 3]
        self.elements = np.array(cell[2])   # [natom, ]
        self.shifts = shifts
        if radius_maps is None:
            radius_maps = covalent_radius 
        self.covalent_radius = radius_maps
        self.graph = None

    def to_matrix(self, full=True):
        # [atom1, atom2, pair_id, mask]
        npair = len(self.distances)
        if full is False:
            atom12_indices = self.pair_indices.T
            pair_indices = np.arange(npair, dtype=int)
            mask =np.ones(npair, dtype=int)
        else:
            atom12_indices = np.c_[self.pair_indices, np.flip(self.pair_indices, axis=0)].T
            pair_indices = np.r_[np.arange(npair),np.arange(npair)]
            mask = np.r_[np.ones(npair), -1*np.ones(npair)].astype(int) 
        atom_mask = np.c_[atom12_indices, pair_indices, mask]
        return atom_mask

    def to_list(self):
        matrix = self.to_matrix()
        datas = []
        for row in matrix:
            distance = self.distances[row[2]]
            shift = self.shifts[self.shift_indices[row[2]]] * row[3]
            bb = BaseBond(row[0], row[1], distance, shift)
            datas.append(bb)
        return datas
   
    def get_parents(self):
        from scipy.cluster.hierarchy import DisjointSet

        graph = DisjointSet(np.arange(len(self.positions)))
        for i,j in self.pair_indices.T:
            graph.merge(i,j) 
        parents = [graph[i] for i in graph]
        print(parents)
        return parents

    def get_neighbors(self, atom_id:int, unique_atoms:bool=True):
        # [idx, distance, (shift), (vector)] 
        data = []
        positions = self.positions
        indices = np.where(self.pair_indices[0]==atom_id)
        indices_i = self.pair_indices[0][indices]
        indices_j = self.pair_indices[1][indices]
        if len(indices_j):
            dists = self.distances[indices]
            shifts = self.shifts[self.shift_indices[indices]]
            vectors = positions[indices_i] - positions[indices_j] + shifts @ self.lattice
            for i,d,s,v in zip(indices_j, dists, shifts, vectors):
                data.append([i,d,s,v])

        indices = np.where(self.pair_indices[1]==atom_id)
        indices_i = self.pair_indices[1][indices]
        indices_j = self.pair_indices[0][indices]
        if len(indices_j):
            dists = self.distances[indices]
            shifts = -self.shifts[self.shift_indices[indices]]
            vectors = positions[indices_i] - positions[indices_j] + shifts @ self.lattice
            for i,d,s,v in zip(indices_j, dists, shifts, vectors):
                data.append([i,d,s,v])

        return data

    def get_graph(self, return_bonds=False):
        from scipy.cluster.hierarchy import DisjointSet
        from .dimension import build_adjacency_list, traverse_component_graphs, merge_mutual_visits

        # boundary
        positions = self.positions
        vertex = np.where(positions==False)[0]
        boundary = {i:positions[i] for i in vertex}
        bonds = []    # 用于构建外键维度信息，形式为 [(a,b,shift), (a,b,shift), ...]
        active_axis = None

        graph = DisjointSet(np.arange(len(self.positions)))
        for a,b,p,s in self.to_matrix():
            if self.shift_indices[p]==0:    
                graph.merge(a,b)
            else:
                shift = self.shifts[self.shift_indices[p]] * s
                # a or b in boundary
                ainb = (a in boundary and sum(abs(boundary[a]*shift))<1e-8)
                binb = (b in boundary and sum(abs(boundary[b]*shift))<1e-8) 
                axis = shift[np.flatnonzero(shift)[0]]
                if active_axis == None and (ainb or binb):
                    active_axis = -axis if ainb else axis

                if (ainb and axis==active_axis) or (binb and axis == -active_axis):
                    graph.merge(a,b)
                else:
                    bonds.append((a,b,tuple(shift)))
 
        parents = [graph[i] for i in graph]
        adjacency = build_adjacency_list(parents, bonds)
        visited, ranks = traverse_component_graphs(adjacency)
        merged, visited, ranks = merge_mutual_visits(visited, ranks, graph)
        if return_bonds:
            return graph, merged, bonds, visited, ranks
        else:
            return graph, merged, visited, ranks

    def get_dimension(self):
        graph, merged, visited, ranks = self.get_graph()
        return max(ranks.values())

    def get_dimension_with_pbc(self):
        from .dimension import build_adjacency_list, traverse_component_graphs, merge_mutual_visits
        from jamip.structure.dimension import DimensionAnalysis, get_2d_axis
        from jamip.structure.structure import Structure
        graph, merged, bonds, visited, ranks = self.get_graph(return_bonds=True)
        maps = {'base': ranks}
        axes = {'base': get_main_axis(self.lattice, visited, ranks)}

        # group bonds by species
        data = defaultdict(list)
        for bond in bonds:
            i,j,vector = bond
            key = tuple(sorted([self.elements[i], self.elements[j]]))
            data[key].append(bond)

        parents = [graph[i] for i in graph]
        for key,value in data.items():
            adjacency = build_adjacency_list(parents, value)
            visited, ranks = traverse_component_graphs(adjacency)
            merged, visited, ranks = merge_mutual_visits(visited, ranks, graph)
            maps[key] = ranks
            axes[key] = get_main_axis(self.lattice, visited, ranks)
           
        return maps, axes 

    def get_digraph(self, matrix: np.ndarray = None, axis: np.ndarray = None, di=False):
        '''
        get directed graph
        Parameters:
            matrix: input bonding matrix
            axis: zaxis, np.ndarray (3,) or int (0-2)
        '''
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Please install networkx to use this method.")

        if matrix is None:
            matrix = self.to_matrix()
        if isinstance(axis, np.ndarray) and axis.shape == (3,):
            axis = axis.reshape(3,)
        elif isinstance(axis, int) and axis < 3:
            axis = np.array([0, 0, 0])
            axis[axis] = 1

        # 获取有向图
        if axis is None:
            dimatrix = [tuple(row[:2]) for row in matrix]
        else:
            vectors = np.sum(self.get_scaled_vectors(matrix) * axis, axis=1)
            matrix = matrix[np.where(vectors>1e-8)]
            if di == False:
                dimatrix = [tuple(row[:2]) if row[3]==1 else tuple(row[:2][::-1]) for row in matrix]
            elif di == True:
                dimatrix = []
                for row in matrix:
                    #dimatrix.append(tuple([row[0],row[1]]))
                    if row[3] > 0:
                        dimatrix.append(tuple([row[0],row[1],{'w':-int(row[3])}]))
                        dimatrix.append(tuple([row[1],row[0],{'w':int(row[3])}]))
                    else:
                        dimatrix.append(tuple([row[0],row[1],{'w':int(row[3])}]))
                        dimatrix.append(tuple([row[1],row[0],{'w':-int(row[3])}]))

        #vectors = np.sum(self.get_scaled_vectors(matrix)
        G = nx.DiGraph(dimatrix)

        positions = self.positions @ np.linalg.inv(self.lattice)
        for row in dimatrix:
            d = positions[row[0]] - positions[row[1]]
            d = d - np.round(d)
            print(row, d, G[row[0]][row[1]]['w'])
        #print(dimatrix)
        # print(dimatrix)
        # nx.weakly_connected_components(G)
        # connected_components = list(nx.weakly_connected_components(G))
        # print(connected_components)
        return G  

    def remove_bond_by_units(self, atom_indices):
        from scipy.cluster.hierarchy import DisjointSet

        graph = DisjointSet(np.arange(len(self.positions)))
        for atoms in atom_indices:
            if len(atoms) > 1:
                i = atoms[0]
                for j in atoms[1:]:
                    graph.merge(i,j) 
        parents = [graph[i] for i in graph]

        indices = []
        for idx,(i,j) in enumerate(self.pair_indices.T):
            if parents[i] == parents[j]:
                indices.append(idx)
            
        self.distances = self.distances[indices]
        self.pair_indices = self.pair_indices[:,indices]
        self.shift_indices = self.shift_indices[indices]

    def remove_bond_by_species(self, species:list):

        atom_indices = []
        for i,specie in enumerate(self.elements):
            if specie in species:
                atom_indices.append(i)

        indices = []
        for idx,(i,j) in enumerate(self.pair_indices.T):
            if i in atom_indices and j in atom_indices:
                indices.append(idx)
            elif i not in atom_indices and j not in atom_indices:
                indices.append(idx)

        self.distances = self.distances[indices]
        self.pair_indices = self.pair_indices[:,indices]
        self.shift_indices = self.shift_indices[indices]

    def get_coordination_matrix(self):
        # [natom, natom] 配位数矩阵
        natoms = len(self.elements)
        matrix = np.zeros((natoms,natoms), dtype=np.int8)
        indices = (self.pair_indices[0], self.pair_indices[1])
        np.add.at(matrix, indices, 1) 
        matrix = matrix + matrix.T 
        return matrix

    def get_covalent_matrix(self, gaussian_decay=False):
        # 共价矩阵
        radius = np.array([self.covalent_radius[i] for i in self.elements])
        covalent_distances = radius[self.pair_indices[0]] + radius[self.pair_indices[1]]
        if gaussian_decay:
            matrix = np.ones_like(self.distances)
            mask = np.where(self.distances-covalent_distances>0)
            matrix[mask] = np.exp(-1*((self.distances-covalent_distances)**2) / (2*(0.05**2)))[mask]
            return matrix
        else:
            return self.distances / covalent_distances

    def to_angle_matrix(self):
        # [atom1, atom2, atom3, pair_id1, pair_id2, mask1, maks2]
        atom_mask = self.to_matrix()
        atom_mask12 = atom_mask[:,0][:,None] - atom_mask[:,0][None,:]
        atom_mask12[np.tril_indices(len(atom_mask),0)] = -1
        pair1, pair2 = np.where(atom_mask12==0)
        pair_mask = np.c_[atom_mask[pair1],atom_mask[pair2]]
        pair_mask = pair_mask[:, [0,1,5,2,6,3,7]]
        return pair_mask

    def get_vectors(self, matrix, norm=False):
        # 提取输入矩阵对应的向量
        vectors = self.positions[matrix[:,0]] - self.positions[matrix[:,1]] + (self.shifts[self.shift_indices[matrix[:,2]]] * matrix[:,3][:,None])  @ self.lattice
        if norm: vectors /= self.distances[matrix[:,2]][:,None]
        return vectors

    def get_scaled_vectors(self, matrix):
        positions = self.positions @ np.linalg.inv(self.lattice)
        vectors = positions[matrix[:,0]] - positions[matrix[:,1]] + (self.shifts[self.shift_indices[matrix[:,2]]] * matrix[:,3][:,None])
        return vectors

    def section(self, indices):
        indices = np.sort(np.unique(indices))
        distances = self.distances[indices]
        pair_indices = self.pair_indices[:,indices]
        shift_indices = self.shift_indices[indices]
        cell = (self.lattice, self.positions, self.elements)
        shifts = self.shifts
        return BondCore(distances, pair_indices, shift_indices, cell, shifts)

    def classify(self, tol=0.2):
        from scipy.cluster.hierarchy import fcluster, single
        #from scipy.cluster.hierarchy import inconsistent, maxdists, cophenet
        matrix = self.to_matrix()
        pairs = np.sort(self.elements[matrix[:,:2]], axis=1)
        u, indices = np.unique(pairs, axis=0, return_inverse=True)
        groups = {}
        for i,pair in enumerate(u):
            ds = self.distances[matrix[np.where(indices==i),2]]
            Z = single(ds.T)
            dsum = 0
            num = 0
            dtol = tol
            for j,row in enumerate(Z): 
                dsum = Z[:,2][:j][-20:].sum()
                # dsum += row[2]
                if dsum > tol:
                    dtol = min(dtol, row[2]-1e-8)
                    break
                    
            fc = fcluster(Z, t=dtol, criterion='distance')
            u2, indices2 = np.unique(fc, return_inverse=True)
            for j,ftype in enumerate(u2):
                key = (pair[0], pair[1], j)
                groups[key] = matrix[np.where(indices==i)][np.where(indices2==j)]
            
        return groups
       
    def full_repr(self):
        matrix = self.to_matrix()
        # add distances and sort %
        distances = self.distances[matrix[:,2]]
        # [atom1, atom2, pair_id, mask]
        lines = defaultdict(list)
        for i in np.argsort(distances):
            atom1, atom2 = matrix[i,:2]
            lines[atom1].append('Atom %-2s: %.4f' %(self.elements[atom2], distances[i])) 
        string = ''
        for i,elm in enumerate(self.elements):
            string += 'ion %-2s:  ' %(elm) + ' '.join(lines[i]) + '\n'
        return string
    
    def get_non_adjacent_vectors(self, node1, node2):
        if self.graph is None:
            graph = defaultdict(dict)
            for i,(u,v) in enumerate(self.pair_indices.T):
                graph[u][v] = self.shifts[self.shift_indices[i]]
                graph[v][u] = self.shifts[self.shift_indices[i]] * -1
            self.graph = graph
        else:
            graph = self.graph

        paths = find_all_paths(graph, node1, node2)
        if not paths:
            return None

        return [
            (path, [sum(graph[path[i]][path[i+1]][j] for i in range(len(path)-1)) for j in range(3)])
            for path in paths
        ]

class Bonding(object):

    radius = covalent_radius

    def __init__(self, 
                 structure, 
                 method:str='min', 
                 cutoff:float=3.5, 
                 offset:float=0,
                 factor:float=1.2, 
                 constraint=None,
                 pbc=[True,True,True], 
                 **kwargs):
        """
        Parameters:
            method: min / radius / rdf 
            tolerance: 1.2
            tolerance_method: add / multiple
        """

        # load structure %
        self.structure = structure
        self.distances = structure.get_all_distances(pbc=True, diag=True)
        # bond search percision %
        if 'radius' in kwargs:
            self.radius = kwargs['radius'] 
        else:
            self.radius = covalent_radius

        # initialize shifts %
        lattice = np.array(structure.lattice)
        elements = structure.get_elements(type='symbol')
        positions = structure.get_positions(type='cartesian')
        if method == 'rdf':
            shifts = self.compute_shifts(lattice, pbc, cutoff)
            distances, pair_indices, shift_indices = get_rdf(structure, shifts, cutoff)
        else:
            if method == 'radius':
                cutoff = np.array([self.radius[i] for i in elements]) 
            elif method == 'min':
                cutoff = np.min(self.distances, axis=1)
            else:
                raise ValueError('Unknown radius type.')

            # add tolerance %
            cutoff = cutoff * factor + offset

            # constraint
            constraint = self.get_constraint(constraint)
         
            # get neighbors %
            shifts = self.compute_shifts(lattice, pbc)
            shifts_value = shifts @ lattice
            distances, pair_indices, shift_indices = self.get_neighbors(positions, shifts_value, cutoff, method, constraint)

        # mini radius %
        mini_radius = {specie:self.radius[specie] for specie in structure.species_of_elements}

        # save %
        all_shifts = np.r_[[[0,0,0]], shifts]
        cell = (lattice, positions, elements)
        self.data = BondCore(distances, pair_indices, shift_indices, cell, all_shifts, mini_radius)

    @classmethod
    def get_neighbors(self, coord, shifts, cutoff, method='rdf', constraint=True):
        num_atoms = coord.shape[0]
        # center cell 
        center_pair_indices = np.array(np.triu_indices(num_atoms, k=1))
        center_shifts_vectors = np.zeros((center_pair_indices.shape[-1], 3))
 
        # shifts cell
        coord_indices = np.arange(num_atoms)
        pair_indices = np.array(np.meshgrid(coord_indices, coord_indices)).reshape(2,-1)
        pbc_pair_indices = np.repeat(pair_indices, shifts.shape[0], axis=0).reshape(2,-1)
        pbc_shifts_vectors = np.repeat(shifts, num_atoms**2, axis=0)
        # [2,neighbor]
        all_pair_indices = np.c_[center_pair_indices, pbc_pair_indices]
        # [neighbor,3]
        all_shifts_vectors = np.r_[center_shifts_vectors, pbc_shifts_vectors]
        # [neighbor,]
        all_shifts_indices = np.r_[np.zeros(center_pair_indices.shape[-1]),
                                   np.repeat(np.arange(shifts.shape[0])+1,num_atoms**2, axis=0)].astype(int)
        
        # compute distances 
        selected_vectors = coord[all_pair_indices[0]] - coord[all_pair_indices[1]] + all_shifts_vectors 
        distances = np.linalg.norm(selected_vectors, axis=-1)
        if method == 'rdf':
            in_cutoff = (distances <= cutoff).nonzero()
        elif method == 'radius':
            cutoff = cutoff[all_pair_indices[0]] + cutoff[all_pair_indices[1]]
            in_cutoff = (distances <= cutoff).nonzero()
        elif method == 'min':
            cutoff = np.max(cutoff[all_pair_indices], axis=0)
            in_cutoff = (distances <= cutoff).nonzero()

        pair_indices = all_pair_indices[:,in_cutoff[0]]
        shift_indices = all_shifts_indices[in_cutoff]
        distances = distances[in_cutoff]

        # filter by constraint %
        if not np.all(constraint):
            filters = np.ones_like(distances, dtype=bool)
            for idx, (i,j) in enumerate(pair_indices.T):
                if not constraint[i,j]:
                    filters[idx] = False
            pair_indices = pair_indices[:,filters]
            shift_indices = shift_indices[filters]
            distances = distances[filters]

        return distances, pair_indices, shift_indices

    @classmethod
    def compute_shifts(self, lattice, pbc=[1,1,1], cutoff=None):
        if cutoff:
            length = np.linalg.norm(lattice, axis=1)
            grange = np.ceil(cutoff / length).astype(int)
        else:
            grange = np.ones(3, dtype=int)
        pbc = np.array(pbc, dtype=bool)
        grange = np.where(pbc, grange, np.zeros(3, dtype=int))

        o = np.zeros(1, dtype=int)
        r1 = np.arange(1, grange[0] + 1, dtype=int)
        r2 = np.arange(1, grange[1] + 1, dtype=int)
        r3 = np.arange(1, grange[2] + 1, dtype=int)
        r2_all = np.arange(-grange[1], grange[1] + 1, dtype=int)
        r3_all = np.arange(-grange[2], grange[2] + 1, dtype=int)

        shifts = np.c_[
            np.array(np.meshgrid(r1,r2_all,r3_all)).reshape(3,-1),
            np.array(np.meshgrid(o,r2,r3_all)).reshape(3,-1),
            np.array(np.meshgrid(o,o,r3)).reshape(3,-1)
        ].T
        return shifts

    def get_bond_by_atom(self,atoms):

        atom_indices = self.get_atom_indices(atoms)
        
        # get bonds step %
        matrix = self.data.to_matrix()
        selected_matrix = matrix[np.isin(matrix[:,0], atom_indices)]
        distances = self.data.distances[selected_matrix[:,2]]
        atom_matrix = np.c_[selected_matrix[:,:2], distances]
        return atom_matrix

    def get_angle_by_atom(self,atoms):

        atom_indices = self.get_atom_indices(atoms)
        
        # get bonds step %
        matrix = self.data.to_angle_matrix()
        selected_matrix = matrix[np.isin(matrix[:,0], atom_indices)]
        vector1 = self.data.get_vectors(selected_matrix[:,[0,1,3,5]], norm=True)
        vector2 = self.data.get_vectors(selected_matrix[:,[0,2,4,6]], norm=True)
        cosine = np.clip(np.sum(vector1*vector2, axis=1),-0.9999,0.9999)
        angle = np.arccos(cosine) # / np.pi * 180
        angle_matrix = np.c_[selected_matrix[:,:3], angle]
        return angle_matrix

    def get_bond_by_pair(self,pairs):

        pairs = np.array(pairs).reshape(-1,2)
        matrix = self.data.to_matrix()
        species12 = self.data.elements[matrix[:,:2]]
        pair_matrixs = []
        for pair in pairs:
            selected_matrix = matrix[(species12==pair).all(axis=1)]
            if pair[0] == pair[1]:
                selected_matrix = selected_matrix.reshape(2,-1,4)[0]
            pair_matrixs.append(selected_matrix)

        pair_matrix = np.concatenate(pair_matrixs, axis=0)
        distances = self.data.distances[pair_matrix[:,2]]
        pair_matrix = np.c_[pair_matrix[:,:2], distances]
        return pair_matrix

    def get_angle_by_pair(self,pairs):

        pairs = np.array(pairs).reshape(-1,3)
        matrix = self.data.to_angle_matrix()
        species123 = self.data.elements[matrix[:,:3]]
        pair_matrixs = []
        for pair in pairs:
            selected_matrix = matrix[(species123==pair).all(axis=1)]
            # TODO
            #if pair[0] == pair[1]:
            #    selected_matrix = selected_matrix.reshape(2,-1,4)[0]
            pair_matrixs.append(selected_matrix)

        pair_matrix = np.concatenate(pair_matrixs, axis=0)
        distances = self.data.distances[pair_matrix[:,2]]
        vector1 = self.data.get_vectors(pair_matrix[:,[0,1,3,5]], norm=True)
        vector2 = self.data.get_vectors(pair_matrix[:,[0,2,4,6]], norm=True)
        cosine = np.clip(np.sum(vector1*vector2, axis=1),-0.9999,0.9999)
        angle = np.arccos(cosine) # / np.pi * 180
        angle_matrix = np.c_[selected_matrix[:,:3], angle]
        return angle_matrix

    def get_bond_by_pair_and_orient(self,pairs,orient,tolerance=1e-3):

        # check orient %
        pairs = np.array(pairs).reshape(-1,2)
        norm_orient = np.array(orient).reshape(3,) / np.linalg.norm(orient)
        
        matrix = self.data.to_matrix()
        species12 = self.data.elements[matrix[:,:2]]
        pair_matrixs = []
        for pair in pairs:
            selected_matrix = matrix[(species12==pair).all(axis=1)]
            if pair[0] == pair[1]:
                selected_matrix = selected_matrix.reshape(2,-1,4)[0]
            pair_matrixs.append(selected_matrix)

        # compute orient
        pair_matrix = np.concatenate(pair_matrixs, axis=0)
        norm_vectors = self.data.get_vectors(pair_matrix, norm=True)
        cosine = 1 - abs(np.dot(norm_vectors, norm_orient)) 

        distances = self.data.distances[pair_matrix[:,2]]
        pair_matrix = np.c_[pair_matrix[:,:2], distances]
        in_pair_matrix = pair_matrix[np.where(cosine<tolerance)]
        out_pair_matrix = pair_matrix[np.where(cosine>tolerance)]

        return in_pair_matrix, out_pair_matrix

    def get_voronoi(self, atom_indices=None):
        """
        from pymatgen
        """
        from scipy.spatial import Voronoi

        vol_tetra = lambda vt1, vt2, vt3, vt4: np.abs(np.dot((vt1 - vt4), np.cross((vt2 - vt4), (vt3 - vt4)))) / 6

        core = self.data
        matrix = core.to_matrix()

        # get unique neighborhood atoms outside the cell %
        shift_indices = core.shift_indices[matrix[:,2]]
        shifts = core.shifts[shift_indices] * matrix[:,3][:,None]
        out_indices = np.where(shift_indices>0)[0]
        site_matrix = np.c_[matrix[:,0], shifts][out_indices]
        uniq_matrix = np.unique(site_matrix, axis=0)
        '''
        '''
        # get unique neighborhood atoms outside the cell %
        #grid = np.mgrid[-1:1, -1:1, -1:1].reshape(3,-1).T
        #shift_indices = np.repeat(np.arange(len(grid)), len(self.structure)) 

        # get atoms in cell %
        if atom_indices is None:
            atom_indices = np.arange(len(self.structure))

        natom = len(atom_indices)
        self_matrix = np.c_[atom_indices, np.zeros((natom,3), dtype=int)]
        all_matrix = np.r_[self_matrix, uniq_matrix]
        allsites = core.positions[all_matrix[:,0]] + all_matrix[:,1:] @ core.lattice

        # get voronoi
        data = []
        voro = Voronoi(allsites)
        # nn: idx of two atoms on either side of the voro plane
        # vind: idx of the vertices of the voro plane
        for idx1 in range(natom):
            for nn,vind in voro.ridge_dict.items():
                if idx1 in nn:
                    idx2 = nn[0] if nn[1] == idx1 else nn[1]
                    coord1 = allsites[idx1]
                    coord2 = allsites[idx2]
                    idx0 = all_matrix[idx2,0]
                    shift = all_matrix[idx2,1:]
                    if -1 in vind:
                        # -1 indices correspond to the Voronoi cell
                        #  missing a face
                        raise RuntimeError(
                            "This structure is pathological,"
                            " infinite vertex in the voronoi "
                            "construction"
                        )

                    # Calculate the solid angle
                    facets = [voro.vertices[i] for i in vind]
                    angle = solid_angle(coord1, facets)

                    # Calculate the tetrahedral volume
                    volume = 0
                    for j,k in zip(vind[1:], vind[2:]):
                        volume += vol_tetra(coord1,
                            voro.vertices[vind[0]],
                            voro.vertices[j],
                            voro.vertices[k],
                        )

                    # Compute the distance of the site to the face
                    distance = np.linalg.norm(coord1 - coord2) 

                    # Compute the area of the face 
                    area = 3 * volume / distance

                    # 'a1','a2','distance','vector','solid_angle','volume', 'area'
                    vb = VoroBond(idx1,idx0,distance,shift,angle,volume,area)
                    #if idx1 == 9:
                    #    print(idx1, idx2)
                    #    print(vb)
                    data.append(vb)
                    
        return data

    def get_bond_by_voronoi(self,atoms,tol=0.1):

        atom_indices = self.get_atom_indices(atoms)
        matrix = []

        # get voronoi
        data = self.get_voronoi(atom_indices)
        for idx in atom_indices:
            pdata = [row for row in data if row.a1 == idx]
            max_solid_angle = max([row.solid_angle for row in pdata])
            for row in pdata:
                if row.solid_angle / max_solid_angle > tol:
                    matrix.append([row.a1, row.a2, row.distance])

        return np.array(matrix)

    def get_adjacency_matrix(self, nu=1.5):
        # 邻接矩阵
        from sklearn.gaussian_process.kernels import Matern
        volume = self.structure.volume
        norm_distances = self.distances / np.power(volume, 1/3)
        np.fill_diagonal(norm_distances, 0)
        kernel = Matern(length_scale=0.5,nu=nu)
        return kernel(norm_distances)
        #return [get_matern(x) for x in norm_distances]

    def get_covalent_matrix(self):
        # 共价矩阵，高斯衰减
        radius = np.array([covalent_radius[i] for i in self.data.elements])
        covalent_distances = radius[None,:] + radius[:,None]
        matrix = np.ones_like(self.distances)
        mask = np.where(self.distances-covalent_distances>0)
        matrix[mask] = np.exp(-1*((self.distances-covalent_distances)**2) / (2*(0.05**2)))[mask]
        return np.around(matrix,8)

    def get_atom_indices(self, atoms):
        # elements to atom_indices 
        if isinstance(atoms,str): atoms=[atoms]
        elements = self.structure.get_elements(type='symbol')

        atom_indices = []
        for i in atoms:
            if isinstance(i, str):
                atom_indices.extend(np.where(elements==i)[0])
            else:
                atom_indices.append(i)
        atom_indices = np.sort(np.unique(atom_indices))
        assert atom_indices[-1] < len(elements), "atom index out of range"
        return atom_indices

    def get_constraint(self, constraint):

        cutoff = self.distances
        elements = self.structure.get_elements(type='symbol')

        if constraint is None:
            mask = np.ones_like(self.distances, dtype=bool)

        elif isinstance(constraint, str) and constraint.lower() == 'auto':
            mask = np.ones_like(self.distances, dtype=bool)
            # argsort to 2d indices
            flat_indices = np.argsort(self.distances, axis=None)  # axis=None 表示展平后排序
            row_indices, col_indices = np.unravel_index(flat_indices, cutoff.shape)

            records = set()
            for i,j in zip(row_indices, col_indices):
                if i in records and j in records: continue
                if mask[i,j] == False: continue

                indices = np.where(cutoff[i,:] > (cutoff[i,j]*1.5+0.3))
                mask[i,indices] = False
                indices = np.where(cutoff[:,j] > (cutoff[i,j]*1.5+0.3))
                mask[indices,j] = False

                if len(records) == len(cutoff): break
        else:

            # constraint：[[atoms1], [atoms2]]
            mask = np.zeros_like(self.distances, dtype=bool)
            for atoms in constraint:
                atom_indices = self.get_atom_indices(atoms)
                row_indices, col_indices = np.ix_(atom_indices, atom_indices)
                mask[row_indices, col_indices] = True

        return mask

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for neighbor in graph[start]:
        if neighbor not in path:
            new_paths = find_all_paths(graph, neighbor, end, path)
            for p in new_paths:
                paths.append(p)
    return paths

def get_matern(X, nu=1.5, length_scale=1):
    from scipy.special import gamma
    X = X / length_scale
    if nu == 0.5:
        k=np.exp(X)
    elif nu == 1.5:
        k=X * np.sqrt(3)
        k=(1+k) * np.exp(-k)
    elif nu == 2.5:
        k=X * np.sqrt(5)
        k=(1. + k + k ** 2./3.) * np.exp(-k)
    elif nu == np.inf:
        k=np.exp(-.5 * X**2)

    if nu != 'inf':
        tmp = (np.sqrt(2*nu) * X) ** nu
        Y = 2**(1-nu) / gamma(nu) * tmp * k
    else: # nu等于无穷
        Y = k

    return Y
