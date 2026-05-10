# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import itertools
        

class StructureAnalyser(object):
    
    def __init__(self, structure):
        """
        Arguments:
            structure: structure's object.
        """
        self.structure=structure

    
    def _calculate_RDF(self, formated_atom, symbol_of_element:list, max_r=10.0, min_r=1e-3, dr=0.1):
        """
        radius distribution function (RDF) for atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).
            symbol_of_element (default=structure.elements)
            
        Returns:
            {r:{symbol:positions}}
        """
        import pandas as pd
        
        atom0=self.structure.get_atom(formated_atom=formated_atom) # center
        if atom0 is None:
            raise ValueError('non-existed formated_atom')
        if dr > max_r:
            raise ValueError('beyond the upper boundary (< %.1f): dr' %max_r)
            
        lattice_parameters=self.structure.lattice_parameters
        grange = max_r / np.array(lattice_parameters[:3])
        left_grange = np.floor(atom0.position - grange)   # left of a in lattice for supercell
        right_grange = np.ceil(atom0.position + grange)   # right of a in lattice for supercell
        dim = np.c_[left_grange, right_grange].astype(int)
        grid = np.mgrid[left_grange[0]:right_grange[0]+1,
                        left_grange[1]:right_grange[1]+1,
                        left_grange[2]:right_grange[2]+1].reshape(3,-1).T
        
        # get position by symbol %
        positions = []
        elements = []
        atoms=list(self.structure.atoms)
        for atom1 in atoms:
            if atom1.element.symbol in symbol_of_element:
                positions.append(atom1.position)
                elements.append(atom1.element.symbol)
        positions = np.array(positions)
        elements = np.array(elements)

        # positions [Natom,3] -> supercell_positions [Ngrid, Natom, 3]
        all_positions = (positions[None,:,:] + grid[:,None,:]).reshape(-1,3)
        all_distances = np.linalg.norm( np.dot(all_positions - atom0.position, self.structure.lattice), axis=-1 )
        #formated_distances = np.around( all_distances / dr ) * dr
        df = pd.DataFrame({'distance': all_distances,
        #                   'position': list(np.around(all_positions,8)),
                           'element': np.tile(elements,len(grid))})
        df = df[(df['distance'] > min_r) & (df['distance'] <= max_r)]
        df['formated_distance'] = (np.around( df['distance'] / dr ) * dr)
        return df
    
    def get_RDF_of_atom(self, formated_atom, symbol_of_element=None, max_r=10.0, min_r=1e-3, dr=0.1):
        """
        radius distribution function (RDF) for atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).
            symbol_of_element (default=structure.elements)
            
        Returns:
            list-type [distance, number_of_atoms, [formated_atom1, formated_atom2,...],
                       ...
                      ]
        """
        if symbol_of_element == None:
            symbol_of_element = np.unique([atom.element.symbol for atom in self.structure.atoms])
            
        full_rdf=self._calculate_RDF(formated_atom=formated_atom, symbol_of_element=symbol_of_element, max_r=max_r, min_r=min_r, dr=dr)
        rdf=[] # [distance, number_of_atoms]}
        for name,group in full_rdf.groupby(['formated_distance','element']):
            distance, element = name
            rdf.append([distance, element, len(group)])
        return rdf
    
    def _calculate_RDF_of_all(self, max_r=10.0, min_r=1e-3, dr=0.1):
        """
        get the radial distribution function (RDF) of all atom.
        """
        """
        radius distribution function (RDF) for atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).

        Returns:
            {r:{symbol:positions}}
        """
        import pandas as pd

        # get all position %
        positions = []
        elements = []
        atoms=list(self.structure.atoms)
        for atom1 in atoms:
            positions.append(atom1.position)
            elements.append(atom1.element.symbol)
        positions = np.array(positions)
        elements = np.array(elements)

        # get range %
        grange = max_r / np.array(self.structure.lattice_parameters[:3])
        left_grange = np.floor(np.min(positions) - grange)   # left of a in lattice for supercell
        right_grange = np.ceil(np.max(positions) + grange)   # right of a in lattice for supercell
        dim = np.c_[left_grange, right_grange].astype(int)
        grid = np.mgrid[left_grange[0]:right_grange[0]+1,
                        left_grange[1]:right_grange[1]+1,
                        left_grange[2]:right_grange[2]+1].reshape(3,-1).T
        X,Y = np.meshgrid(elements,elements)
        element_tuples = np.c_[X.ravel(),Y.ravel()]
        all_element_tuples = np.tile(element_tuples,(len(grid),1))
        # positions [Natom,3] -> vectors [Natom, Natom, 3]
        vectors = (positions[None,:,:] + positions[:,None,:]).reshape(-1,3)
        # vectors -> supercell_vectors [Ngrid, Natom, Natom, 3]
        all_vectors = (vectors[None,:,:] + grid[:,None,:]).reshape(-1,3)
        all_distances = np.linalg.norm( np.dot(all_vectors, self.structure.lattice), axis=-1 )
        df = pd.DataFrame({'distance': all_distances,
        #                   'position': map(tuple,np.around(all_positions,8)),
                           'element': list(all_element_tuples)})
        df = df[(df['distance'] > min_r) & (df['distance'] <= max_r)]
        df['element'] = df['element'].apply(tuple)
        #list2str = lambda x: '-'.join(x)
        #df['element'] = df['element'].apply(list2str)
        df['formated_distance'] = (np.around( df['distance'] / dr ) * dr)
        return df
    
    def get_RDF_of_all(self, max_r=10.0, min_r=1e-3, dr=0.1):
        """
        radius distribution function (RDF) for all atom.
        
        Arguments:
            max_r (default=10.0): max radius (unit: Angstrom).
            min_r (default=1e-3): min radius (unit: Angstrom).
            dr (default=0.1): delta radius (unit: Angstrom).
            
        Returns:
            list-type [distance, number_of_atoms, [formated_atom1, formated_atom2,...],
                       ...
                      ]
        """
        full_rdf=self._calculate_RDF_of_all(max_r=max_r, min_r=min_r, dr=dr)
        rdf=[] # [distance, number_of_atoms]}
        for name,group in full_rdf.groupby(['formated_distance','element']):
        #for name,group in full_rdf.groupby(['formated_distance']):
            distance,element = name
            rdf.append([distance, element, len(group)])
        return rdf
    
    def get_tolerance_factor(self, r_a, r_b, r0):
        """
        get the Goldschmidt tolerance factor by given atomic radii.
        https://blog.csdn.net/weixin_44210796/article/details/106120423
        """
        return (r_a + r0) / (np.sqrt(2) * (r_b + r0))
    
    def get_XRD_pattern(self, target_of_anode_material=''):
        """
        calculate XRD pattern.
        
        Arguments:
            target_of_anode_material: target of anode material. 
                Anode   Cr   Fe   Co   Cu   Mo   Ag
                K_alpha 2.29 1.94 1.79 1.54 0.71 0.56
        
        Return:
            XRD pattern.
        """
        pass
    
class StrcutureMath(object):

    # 相较原版程序，外移了部分函数，从而简化了现有函数和功能


    """
    Class to match structures by similarity.

    Algorithm:

    1. Given two structures: s1 and s2
    2. Optional: Reduce to primitive cells.
    3. If the number of sites do not match, return False
    4. Reduce to s1 and s2 to Niggli Cells
    5. Optional: Scale s1 and s2 to same volume.
    6. Optional: Remove oxidation states associated with sites
    7. Find all possible lattice vectors for s2 within shell of ltol.
    8. For s1, translate an atom in the smallest set to the origin
    9. For s2: find all valid lattices from permutations of the list
       of lattice vectors (invalid if: det(Lattice Matrix) < half
       volume of original s2 lattice)
    10. For each valid lattice:

        a. If the lattice angles of are within tolerance of s1,
           basis change s2 into new lattice.
        b. For each atom in the smallest set of s2:

            i. Translate to origin and compare fractional sites in
            structure within a fractional tolerance.
            ii. If true:

                ia. Convert both lattices to Cartesian and place
                both structures on an average lattice
                ib. Compute and return the average and max rms
                displacement between the two structures normalized
                by the average free length per atom

                if fit function called:
                    if normalized max rms displacement is less than
                    stol. Return True

                if get_rms_dist function called:
                    if normalized average rms displacement is less
                    than the stored rms displacement, store and
                    continue. (This function will search all possible
                    lattices for the smallest average rms displacement
                    between the two structures)
    """

    def __init__(self, struct1, struct2,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
        primitive_cell: bool = True,
        scale: bool = True,
        attempt_supercell: bool = False,
        allow_subset: bool = False,
        niggli: bool = True,
    ) -> None:
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        self._primitive_cell = primitive_cell
        self._niggli = niggli
        self._scale = scale
        self._supercell = attempt_supercell
        # self._supercell_size = supercell_size
        self._subset = allow_subset
        self._struct1 = struct1
        self._struct2 = struct2


    def _get_lattices(self, target_lattice, s, supercell_size=1):
        """
        Yields lattices for s with lengths and angles close to the lattice of target_s. If
        supercell_size is specified, the returned lattice will have that number of primitive
        cells in it.

        Args:
            target_lattice (Lattice): target lattice.
            s (Structure): input structure.
            supercell_size (int): Number of primitive cells in returned lattice
        """
        from pymatgen.core.lattice import Lattice
        lattices = Lattice(s.lattice).find_all_mappings(
            Lattice(target_lattice),
            ltol=self.ltol,
            atol=self.angle_tol,
            skip_rotation_matrix=True,
        )
        for latt, _, scale_m in lattices:
            if abs(abs(np.linalg.det(scale_m)) - supercell_size) < 0.5:
                yield latt, scale_m

    def _get_supercells(self, s1, s2, fu, s1_supercell):
        """
        Computes all supercells of one structure close to the lattice of the
        other
        if s1_supercell is True, it makes the supercells of struct1, otherwise
        it makes them of s2.

        yields: s1, s2, supercell_matrix, average_lattice, supercell_matrix
        """
        from pymatgen.core.lattice import Lattice
        from pymatgen.util.coord import lattice_points_in_supercell
        # s1 = self._struct1
        # s2 = self._struct2

        def av_lat(l1, l2):
            # print(l1, l2)
            # l1 = Lattice(l1)
            l2 = Lattice(l2)
            params = (np.array(l1.parameters) + np.array(l2.parameters)) / 2
            return Lattice.from_parameters(*params)

        def sc_generator(s1, s2):
            s2_fc = s2.get_positions(type='direct')
            if fu == 1:
                cc = s1.get_positions(type='cartesian')                
                for latt, sc_m in self._get_lattices(s2.lattice, s1, fu):
                    fc = latt.get_fractional_coords(cc)
                    fc -= np.floor(fc)
                    yield fc, s2_fc, av_lat(latt, s2.lattice), sc_m
            else:
                fc_init = s1.get_positions(type='direct')
                for latt, sc_m in self._get_lattices(s2.lattice, s1, fu):
                    fc = np.dot(fc_init, np.linalg.inv(sc_m))
                    lp = lattice_points_in_supercell(sc_m)
                    fc = (fc[:, None, :] + lp[None, :, :]).reshape((-1, 3))
                    fc -= np.floor(fc)
                    yield fc, s2_fc, av_lat(latt, s2.lattice), sc_m

        if s1_supercell:
            for x in sc_generator(s1, s2):
                yield x
        else:
            for x in sc_generator(s2, s1):
                # reorder generator output so s1 is still first
                yield x[1], x[0], x[2], x[3]

    @classmethod
    def _cmp_fstruct(cls, s1, s2, frac_tol, mask):
        """
        Returns true if a matching exists between s2 and s2
        under frac_tol. s2 should be a subset of s1.
        """
        from pymatgen.util.coord_cython import is_coord_subset_pbc
        if len(s2) > len(s1):
            raise ValueError(f"{len(s1)} must be larger than {len(s2)}")
        if mask.shape != (len(s2), len(s1)):
            raise ValueError("mask has incorrect shape")

        return is_coord_subset_pbc(s2, s1, frac_tol, mask)

    @classmethod
    def _cart_dists(cls, s1, s2, avg_lattice, mask, normalization, lll_frac_tol=None):
        """
        Finds a matching in Cartesian space. Finds an additional
        fractional translation vector to minimize RMS distance.

        Args:
            s1: numpy array of fractional coordinates.
            s2: numpy array of fractional coordinates. len(s1) >= len(s2)
            avg_lattice: Lattice on which to calculate distances
            mask: numpy array of booleans. mask[i, j] = True indicates
                that s2[i] cannot be matched to s1[j]
            normalization (float): inverse normalization length
            lll_frac_tol (float): tolerance for Lenstra-Lenstra-Lovász lattice basis reduction algorithm

        Returns:
            Distances from s2 to s1, normalized by (V/atom) ^ 1/3
            Fractional translation vector to apply to s2.
            Mapping from s1 to s2, i.e. with numpy slicing, s1[mapping] => s2
        """
        from pymatgen.util.coord_cython import pbc_shortest_vectors
        from scipy.optimize import linear_sum_assignment 
        if len(s2) > len(s1):
            raise ValueError(f"{len(s1)} must be larger than {len(s2)}")
        if mask.shape != (len(s2), len(s1)):
            raise ValueError("mask has incorrect shape")

        # vectors are from s2 to s1
        vecs, d_2 = pbc_shortest_vectors(avg_lattice, s2, s1, mask, return_d2=True, lll_frac_tol=lll_frac_tol)
        # print(d_2)
        row_ind, col_ind = linear_sum_assignment(d_2)
        short_vecs = vecs[row_ind, col_ind]
        # print(row_ind, col_ind)
        # lin = LinearAssignment(d_2)
        # sol = lin.solution  # pylint: disable=E1101
        # short_vecs = vecs[np.arange(len(sol)), sol]
        translation = np.average(short_vecs, axis=0)
        f_translation = avg_lattice.get_fractional_coords(translation)
        new_d2 = np.sum((short_vecs - translation) ** 2, axis=-1)
        # print(new_d2)

        return new_d2**0.5 * normalization, f_translation, col_ind

    def _get_mask(self, struct1, struct2, fu, s1_supercell):
        """
        Returns mask for matching struct2 to struct1. If struct1 has sites
        a b c, and fu = 2, assumes supercells of struct2 will be ordered
        aabbcc (rather than abcabc).

        Returns:
        mask, struct1 translation indices, struct2 translation index
        """
        mask = np.zeros((len(struct2), len(struct1), fu), dtype=bool)

        inner = []
        for sp2, i in itertools.groupby(enumerate(struct2.species_of_elements), key=lambda x: x[1]):
            i = list(i)
            inner.append((sp2, slice(i[0][0], i[-1][0] + 1)))

        for sp1, j in itertools.groupby(enumerate(struct1.species_of_elements), key=lambda x: x[1]):
            j = list(j)
            j = slice(j[0][0], j[-1][0] + 1)
            for sp2, i in inner:
                mask[i, j, :] = not (sp1 == sp2)

        if s1_supercell:
            mask = mask.reshape((len(struct2), -1))
        else:
            # supercell is of struct2, roll fu axis back to preserve
            # correct ordering
            mask = np.rollaxis(mask, 2, 1)
            mask = mask.reshape((-1, len(struct1)))

        # find the best translation indices
        i = np.argmax(np.sum(mask, axis=-1))
        inds = np.where(np.invert(mask[i]))[0]
        if s1_supercell:
            # remove the symmetrically equivalent s1 indices
            inds = inds[::fu]
        return np.array(mask, dtype=int), inds, i

    def fit(self, symmetric: bool = False):
        """
        Fit two structures.

        Args:
            symmetric (bool): Defaults to False
                If True, check the equality both ways.
                This only impacts a small percentage of structures
            skip_structure_reduction (bool): Defaults to False
                If True, skip to get a primitive structure and perform Niggli reduction for struct1 and struct2

        Returns:
            bool: True if the structures are equivalent
        """
        s1 = self._struct1
        s2 = self._struct2
        
        # step1 : check if the composition is the same
        # if self._subset is True, get reduced formula
        f1 = s1.get_formula(reduced=self._subset, sort=True)
        f2 = s2.get_formula(reduced=self._subset, sort=True)
        if f1 != f2: 
            print('exit for elements')
            return False

        # step2: check if the lattice is the same
        struct1, struct2, fu, s1_supercell = self._preprocess()
        match1 = self._match(struct1, struct2, fu, s1_supercell, break_on_match=True)
        
        if not symmetric:
            if match1 is None:
                return False
            return match1[0] <= self.stol
        else:
            struct1, struct2, fu, s1_supercell = self._preprocess(inverse=True)
            match2 = self._match(struct1, struct2, fu, s1_supercell, break_on_match=True)

        if match1 is None or match2 is None:
            return False

        return max(match1[0], match2[0]) <= self.stol

    def get_rms_dist(self, struct1, struct2):
        """
        Calculate RMS displacement between two structures.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure

        Returns:
            rms displacement normalized by (Vol / nsites) ** (1/3)
            and maximum distance between paired sites. If no matching
            lattice is found None is returned.
        """
        struct1, struct2, fu, s1_supercell = self._preprocess()
        match = self._match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=False)

        if match is None:
            return None

        return match[0], max(match[1])

    def _get_supercell_size(self, s1, s2, supercell_size="num_atoms"):
        """        
        Returns the supercell size, and whether the supercell should be applied to s1.
        If fu == 1, s1_supercell is returned as true, to avoid ambiguity.
        """
        if supercell_size == "num_atoms":
            fu = len(s2) / len(s1)
        elif supercell_size == "volume":
            fu = s2.volume / s1.volume
        else:
            # supercell_size by part of elements
            if isinstance(supercell_size, str):
                supercell_size = [supercell_size]
            s1comp, s2comp = 0, 0
            for specie in supercell_size:
                s1comp += s1.composition.as_dict()[specie]
                s2comp += s2.composition.as_dict()[specie]
            if s1comp == 0 or s2comp == 0:
                raise ValueError("Supercell size not specified correctly")
            fu = s2comp / s1comp

        if fu < 2 / 3:
            return int(round(1 / fu)), False

        return int(round(fu)), True
    
    def _preprocess(self, skip_structure_reduction: bool = False, inverse=False):
        """
        Rescales, finds the reduced structures (primitive and niggli),
        and finds fu, the supercell size to make struct1 comparable to s2.
        If skip_structure_reduction is True, skip to get reduced structures (by primitive transformation and
        niggli reduction). This option is useful for fitting a set of structures several times.
        """
        from copy import deepcopy

        # print(len(self._struct1), len(self._struct2))

        if skip_structure_reduction:
            # Need to copy original structures to rescale lattices later
            struct1 = deepcopy(self._struct1)
            struct2 = deepcopy(self._struct2)
        else:
            struct1 = self._get_reduced_structure(self._struct1, self._primitive_cell, self._niggli)
            struct2 = self._get_reduced_structure(self._struct2, self._primitive_cell, self._niggli)
        
        if inverse:
            struct1, struct2 = struct2, struct1

        # print(len(struct1), len(struct2))

        if self._supercell:
            fu, s1_supercell = self._get_supercell_size(struct1, struct2)
        else:
            fu, s1_supercell = 1, True
        mult = fu if s1_supercell else 1 / fu

        # rescale lattice to same volume
        if self._scale:
            ratio = (struct2.volume / (struct1.volume * mult)) ** (1 / 6)
            struct1.lattice = struct1.lattice * ratio
            struct2.lattice = struct2.lattice / ratio

        return struct1, struct2, fu, s1_supercell

    def _match(self, struct1, struct2, fu, s1_supercell=True, use_rms=False, break_on_match=False):
        """Matches one struct onto the other."""
        ratio = fu if s1_supercell else 1 / fu
        if len(struct1) * ratio >= len(struct2):
            return self._strict_match(
                struct1,
                struct2,
                fu,
                s1_supercell=s1_supercell,
                break_on_match=break_on_match,
                use_rms=use_rms,
            )
        return self._strict_match(
            struct2,
            struct1,
            fu,
            s1_supercell=(not s1_supercell),
            break_on_match=break_on_match,
            use_rms=use_rms,
        )

    def _strict_match(self, struct1, struct2, fu, s1_supercell: bool = True,
        use_rms: bool = False,
        break_on_match: bool = False,
    ):
        """
        Matches struct2 onto struct1 (which should contain all sites in
        struct2).

        Args:
            struct1 (Structure): structure to match onto
            struct2 (Structure): structure to match
            fu (int): size of supercell to create
            s1_supercell (bool): whether to create the supercell of struct1 (vs struct2)
            use_rms (bool): whether to minimize the rms of the matching
            break_on_match (bool): whether to stop search at first match

        Returns:
            tuple[float, float, np.ndarray, float, Mapping]: (rms, max_dist, mask, cost, mapping)
                if a match is found, else None
        """
        from scipy.optimize import linear_sum_assignment 


        if fu < 1:
            raise ValueError("fu cannot be less than 1")

        mask, s1_t_inds, s2_t_ind = self._get_mask(struct1, struct2, fu, s1_supercell)

        if mask.shape[0] > mask.shape[1]:
            raise ValueError("after supercell creation, struct1 must have more sites than struct2")

        # check that a valid mapping exists
        if (not self._subset) and mask.shape[1] != mask.shape[0]:
            print("exit for subset")
            return None

        row_ind, col_ind = linear_sum_assignment(mask)
        min_cost = mask[row_ind, col_ind].sum()
        if min_cost > 0:  # pylint: disable=E1101
            print("exit for cost")
            return None

        best_match = None
        # loop over all lattices
        for s1fc, s2fc, avg_l, sc_m in self._get_supercells(struct1, struct2, fu, s1_supercell):
            # print(s1fc.shape, s2fc.shape)
            # compute fractional tolerance
            normalization = (len(s1fc) / avg_l.volume) ** (1 / 3)
            inv_abc = np.array(avg_l.reciprocal_lattice.abc)
            frac_tol = inv_abc * self.stol / (np.pi * normalization)
            # loop over all translations
            for s1i in s1_t_inds:
                t = s1fc[s1i] - s2fc[s2_t_ind]
                t_s2fc = s2fc + t
                if self._cmp_fstruct(s1fc, t_s2fc, frac_tol, mask):
                    inv_lll_abc = np.array(avg_l.get_lll_reduced_lattice().reciprocal_lattice.abc)
                    lll_frac_tol = inv_lll_abc * self.stol / (np.pi * normalization)
                    dist, t_adj, mapping = self._cart_dists(s1fc, t_s2fc, avg_l, mask, normalization, lll_frac_tol)
                    val = np.linalg.norm(dist) / len(dist) ** 0.5 if use_rms else max(dist)
                    # pylint: disable=E1136
                    if best_match is None or val < best_match[0]:
                        total_t = t + t_adj
                        total_t -= np.round(total_t)
                        best_match = val, dist, sc_m, total_t, mapping
                        if (break_on_match or val < 1e-5) and val < self.stol:
                            return best_match

        if best_match and best_match[0] < self.stol:
            return best_match

        return None

    def group_structures(self, s_list, anonymous=False):
        """
        Given a list of structures, use fit to group
        them by structural equality.

        Args:
            s_list ([Structure]): List of structures to be grouped
            anonymous (bool): Whether to use anonymous mode.

        Returns:
            A list of lists of matched structures
            Assumption: if s1 == s2 but s1 != s3, than s2 and s3 will be put
            in different groups without comparison.
        """
        if self._subset:
            raise ValueError("allow_subset cannot be used with group_structures")

        original_s_list = list(s_list)
        s_list = self._process_species(s_list)
        # Prepare reduced structures beforehand
        s_list = [self._get_reduced_structure(s, self._primitive_cell, niggli=True) for s in s_list]

        # Use structure hash to pre-group structures
        if anonymous:
            def c_hash(c):
                return c.anonymized_formula

        else:
            c_hash = self._comparator.get_hash

        def s_hash(s):
            return c_hash(s[1].composition)

        sorted_s_list = sorted(enumerate(s_list), key=s_hash)
        all_groups = []

        # For each pre-grouped list of structures, perform actual matching.
        for _, g in itertools.groupby(sorted_s_list, key=s_hash):
            unmatched = list(g)
            while len(unmatched) > 0:
                i, refs = unmatched.pop(0)
                matches = [i]
                if anonymous:
                    inds = filter(
                        lambda i: self.fit_anonymous(refs, unmatched[i][1], skip_structure_reduction=True),
                        list(range(len(unmatched))),
                    )
                else:
                    inds = filter(
                        lambda i: self.fit(refs, unmatched[i][1], skip_structure_reduction=True),
                        list(range(len(unmatched))),
                    )
                inds = list(inds)
                matches.extend([unmatched[i][0] for i in inds])
                unmatched = [unmatched[i] for i in range(len(unmatched)) if i not in inds]
                all_groups.append([original_s_list[i] for i in matches])

        return all_groups

    def _anonymous_match(self, struct1, struct2, fu: int, s1_supercell=True, use_rms=False, break_on_match=False, single_match=False):
        """
        Tries all permutations of matching struct1 to struct2.

        Args:
            struct1 (Structure): First structure
            struct2 (Structure): Second structure
            fu (int): Factor of unit cell of struct1 to match to struct2
            s1_supercell (bool): whether to create the supercell of struct1 (vs struct2)
            use_rms (bool): Whether to minimize the rms of the matching
            break_on_match (bool): Whether to break search on first match
            single_match (bool): Whether to return only the best match

        Returns:
            List of (mapping, match)
        """
        from jamip.structure.atom import Composition
        from copy import deepcopy

        # check that species lists are comparable
        species1 = struct1.species_of_elements
        species2 = struct2.species_of_elements
        if len(species1) != len(species2):
            return None

        ratio = fu if s1_supercell else 1 / fu
        swapped = len(struct1) * ratio < len(struct2)

        s1_comp = struct1.composition.as_dict()
        s2_formula = struct2.get_formula(sort=True, reduced=False)
        matches = []
        for perm in itertools.permutations(species2):
            sp_mapping = dict(zip(species1, perm))

            # do quick check that compositions are compatible
            mapped_dict = {sp_mapping[k]: v for k, v in s1_comp.items()}
            mapped_formula = Composition.from_dict(mapped_dict).get_formula(sort=True, reduced=False)
            if (not self._subset) and mapped_formula != s2_formula:
                continue

            mapped_struct = deepcopy(struct1)
            # replace species in struct1 with species from struct2
            for atom in mapped_struct.atomic_positions:
                for sp1, sp2 in sp_mapping.items():
                    if atom.specie == sp1:
                        atom.specie = sp2
                        break
            mapped_struct.update()
            # mapped_struct.replace_species(sp_mapping)
            if swapped:
                m = self._strict_match(struct2, mapped_struct, fu, (not s1_supercell), use_rms, break_on_match)
            else:
                m = self._strict_match(mapped_struct, struct2, fu, s1_supercell, use_rms, break_on_match)
            if m:
                matches.append((sp_mapping, m))
                if single_match:
                    break
        return matches

    @classmethod
    def _get_reduced_structure(cls, structure, primitive_cell: bool = True, niggli: bool = True, symprec=1e-3):
        """Helper method to find a reduced structure."""
        from .structureFactory import StructureFactory

        sf = StructureFactory(structure)
        if primitive_cell:
            sf.primitive(symprec=symprec)
        if niggli:
            sf.niggli_reduce()
        return sf.structure

    def get_rms_anonymous(self, struct1, struct2):
        """
        Performs an anonymous fitting, which allows distinct species in one
        structure to map to another. E.g., to compare if the Li2O and Na2O
        structures are similar.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure

        Returns:
            (min_rms, min_mapping)
            min_rms is the minimum rms distance, and min_mapping is the
            corresponding minimal species mapping that would map
            struct1 to struct2. (None, None) is returned if the minimax_rms
            exceeds the threshold.
        """
        struct1, struct2, fu, s1_supercell = self._preprocess()

        matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=False)
        if matches:
            best = sorted(matches, key=lambda x: x[1][0])[0]
            return best[1][0], best[0]

        return None, None

    def get_best_electronegativity_anonymous_mapping(self, struct1, struct2):
        """
        Performs an anonymous fitting, which allows distinct species in one
        structure to map to another. E.g., to compare if the Li2O and Na2O
        structures are similar. If multiple substitutions are within tolerance
        this will return the one which minimizes the difference in
        electronegativity between the matches species.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure

        Returns:
            min_mapping (dict): Mapping of struct1 species to struct2 species
        """
        struct1, struct2, fu, s1_supercell = self._preprocess()

        matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=True)

        if matches:
            min_X_diff = np.inf
            for match in matches:
                X_diff = 0
                for key, val in match[0].items():
                    X_diff += struct1.composition[key] * (key.X - val.X) ** 2
                if X_diff < min_X_diff:
                    min_X_diff = X_diff
                    best = match[0]
            return best

        return None

    def get_all_anonymous_mappings(self, struct1, struct2, niggli=True, include_dist=False):
        """
        Performs an anonymous fitting, which allows distinct species in one
        structure to map to another. Returns a dictionary of species
        substitutions that are within tolerance.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure
            niggli (bool): Find niggli cell in preprocessing
            include_dist (bool): Return the maximin distance with each mapping

        Returns:
            list of species mappings that map struct1 to struct2.
        """
        struct1, struct2, fu, s1_supercell = self._preprocess()

        matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, break_on_match=not include_dist)
        if matches:
            if include_dist:
                return [(m[0], m[1][0]) for m in matches]

            return [m[0] for m in matches]

        return None

    def fit_anonymous(self, niggli: bool = True):
        """
        Performs an anonymous fitting, which allows distinct species in one structure to map
        to another. E.g., to compare if the Li2O and Na2O structures are similar.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure
            niggli (bool): If true, perform Niggli reduction for struct1 and struct2
            skip_structure_reduction (bool): Defaults to False
                If True, skip to get a primitive structure and perform Niggli reduction for struct1 and struct2

        Returns:
            True/False: Whether a species mapping can map struct1 to stuct2
        """
        struct1, struct2, fu, s1_supercell = self._preprocess()

        matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, break_on_match=True, single_match=True)

        return bool(matches)

    def get_supercell_matrix(self, supercell, struct):
        """
        Returns the matrix for transforming struct to supercell. This
        can be used for very distorted 'supercells' where the primitive cell
        is impossible to find.
        """
        if self._primitive_cell:
            raise ValueError("get_supercell_matrix cannot be used with the primitive cell option")
        struct, supercell, fu, s1_supercell = self._preprocess()

        if not s1_supercell:
            raise ValueError("The non-supercell must be put onto the basis of the supercell, not the other way around")

        match = self._match(struct, supercell, fu, s1_supercell, use_rms=True, break_on_match=False)

        if match is None:
            return None

        return match[2]

    def get_transformation(self, struct1, struct2):
        """
        Returns the supercell transformation, fractional translation vector,
        and a mapping to transform struct2 to be similar to struct1.

        Args:
            struct1 (Structure): Reference structure
            struct2 (Structure): Structure to transform.

        Returns:
            supercell (numpy.ndarray(3, 3)): supercell matrix
            vector (numpy.ndarray(3)): fractional translation vector
            mapping (list(int or None)):
                The first len(struct1) items of the mapping vector are the
                indices of struct1's corresponding sites in struct2 (or None
                if there is no corresponding site), and the other items are
                the remaining site indices of struct2.
        """
        if self._primitive_cell:
            raise ValueError("get_transformation cannot be used with the primitive cell option")

        struct1, struct2 = self._process_species((struct1, struct2))

        s1, s2, fu, s1_supercell = self._preprocess()
        ratio = fu if s1_supercell else 1 / fu
        if s1_supercell and fu > 1:
            raise ValueError("Struct1 must be the supercell, not the other way around")

        if len(s1) * ratio >= len(s2):
            # s1 is superset
            match = self._strict_match(s1, s2, fu=fu, s1_supercell=False, use_rms=True, break_on_match=False)
            if match is None:
                return None
            # invert the mapping, since it needs to be from s1 to s2
            mapping = [list(match[4]).index(idx) if idx in match[4] else None for idx in range(len(s1))]
            return match[2], match[3], mapping
        # s2 is superset
        match = self._strict_match(s2, s1, fu=fu, s1_supercell=True, use_rms=True, break_on_match=False)
        if match is None:
            return None
        # add sites not included in the mapping
        not_included = list(range(len(s2) * fu))
        for i in match[4]:
            not_included.remove(i)
        mapping = list(match[4]) + not_included
        return match[2], -match[3], mapping

    def get_s2_like_s1(self, struct1, struct2, include_ignored_species=True):
        """
        Performs transformations on struct2 to put it in a basis similar to
        struct1 (without changing any of the inter-site distances).

        Args:
            struct1 (Structure): Reference structure
            struct2 (Structure): Structure to transform.
            include_ignored_species (bool): Defaults to True,
                the ignored_species is also transformed to the struct1
                lattice orientation, though obviously there is no direct
                matching to existing sites.

        Returns:
            A structure object similar to struct1, obtained by making a
            supercell, sorting, and translating struct2.
        """
        s1, s2 = self._process_species([struct1, struct2])
        trans = self.get_transformation(s1, s2)
        if trans is None:
            return None
        sc, t, mapping = trans
        sites = list(s2)
        # Append the ignored sites at the end.
        sites.extend([site for site in struct2 if site not in s2])
        temp = Structure.from_sites(sites)

        temp.make_supercell(sc)
        temp.translate_sites(list(range(len(temp))), t)
        # translate sites to correct unit cell
        for i, j in enumerate(mapping[: len(s1)]):
            if j is not None:
                vec = np.round(struct1[i].frac_coords - temp[j].frac_coords)
                temp.translate_sites(j, vec, to_unit_cell=False)

        sites = [temp.sites[i] for i in mapping if i is not None]

        if include_ignored_species:
            start = int(round(len(temp) / len(struct2) * len(s2)))
            sites.extend(temp.sites[start:])

        return Structure.from_sites(sites)

    def get_mapping(self, superset, subset):
        """
        Calculate the mapping from superset to subset.

        Args:
            superset (Structure): Structure containing at least the sites in
                subset (within the structure matching tolerance)
            subset (Structure): Structure containing some of the sites in
                superset (within the structure matching tolerance)

        Returns:
            numpy array such that superset.sites[mapping] is within matching
            tolerance of subset.sites or None if no such mapping is possible
        """
        if self._supercell:
            raise ValueError("cannot compute mapping to supercell")
        if self._primitive_cell:
            raise ValueError("cannot compute mapping with primitive cell option")
        if len(subset) > len(superset):
            raise ValueError("subset is larger than superset")

        superset, subset, _, _ = self._preprocess()
        match = self._strict_match(superset, subset, 1, break_on_match=False)

        if match is None or match[0] > self.stol:
            return None

        return match[4]
