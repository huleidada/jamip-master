# This code is adapted from elastic [https://github.com/jochym/Elastic] .
# The original code does not include the calculator part. Therefore, the 
# main purpose of this code is to provide a general calculation method 
# applicable to machine learning potentials.

import spglib
from jamip.structure import read,write,Structure
import numpy as np

lattice_types = ["Triclinic", "Monoclinic", "Orthorombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]
lattice_nums =  [          3,           16,            75,          143,        168,         195,     231]

def regular(u):
    '''
    Equation matrix generation for the regular (cubic) lattice.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{12}, C_{44}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
               [[uxx,   uyy + uzz,      0],
                [uyy,   uxx + uzz,      0],
                [uzz,   uxx + uyy,      0],
                [0,             0,      2*uyz],
                [0,             0,      2*uxz],
                [0,             0,      2*uxy]])


def tetragonal(u):
    '''
    Equation matrix generation for the tetragonal lattice.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{66}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
                [[uxx,   0,    uyy,  uzz,      0,      0],
                 [uyy,   0,    uxx,  uzz,      0,      0],
                 [0,     uzz,  0,    uxx+uyy,  0,      0],
                 [0,     0,    0,    0,        2*uxz,  0],
                 [0,     0,    0,    0,        2*uyz,  0],
                 [0,     0,    0,    0,        0,      2*uxy]])


def orthorombic(u):
    '''
    Equation matrix generation for the orthorombic lattice.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
                [[uxx,     0,    0,  uyy,  uzz,    0,     0,     0,     0],
                 [0,     uyy,    0,  uxx,    0,  uzz,     0,     0,     0],
                 [0,       0,  uzz,    0,  uxx,  uyy,     0,     0,     0],
                 [0,       0,    0,    0,    0,    0, 2*uyz,     0,     0],
                 [0,       0,    0,    0,    0,    0,     0, 2*uxz,     0],
                 [0,       0,    0,    0,    0,    0,     0,     0, 2*uxy]])


def trigonal(u):
    '''
    The matrix is constructed based on the approach from L&L
    using auxiliary coordinates: :math:`\\xi=x+iy`, :math:`\\eta=x-iy`.
    The components are calculated from free energy using formula
    introduced in :ref:`symmetry` with appropriate coordinate changes.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    # TODO: Not tested yet.
    # TODO: There is still some doubt about the :math:`C_{14}` constant.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
                [[   uxx,   0,    uyy,     uzz,     0,   2*uxz      ],
                 [   uyy,   0,    uxx,     uzz,     0,  -2*uxz      ],
                 [     0, uzz,      0, uxx+uyy,     0,   0          ],
                 [     0,   0,      0,       0, 2*uyz,  -4*uxy      ],
                 [     0,   0,      0,       0, 2*uxz,   2*(uxx-uyy)],
                 [ 2*uxy,   0, -2*uxy,       0,     0,  -4*uyz      ]])


def hexagonal(u):
    '''
    The matrix is constructed based on the approach from L&L
    using auxiliary coordinates: :math:`\\xi=x+iy`, :math:`\\eta=x-iy`.
    The components are calculated from free energy using formula
    introduced in :ref:`symmetry` with appropriate coordinate changes.
    The order of constants is as follows:

    .. math::
       C_{11}, C_{33}, C_{12}, C_{13}, C_{44}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    # TODO: Still needs good verification
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
                [[   uxx,   0,    uyy,     uzz,     0   ],
                 [   uyy,   0,    uxx,     uzz,     0   ],
                 [     0, uzz,      0, uxx+uyy,     0   ],
                 [     0,   0,      0,       0, 2*uyz   ],
                 [     0,   0,      0,       0, 2*uxz   ],
                 [   uxy,   0,   -uxy,       0,     0   ]])


def monoclinic(u):
    '''Monoclinic group,

    The ordering of constants is:

    .. math::
       C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
                [[uxx,  0,  0,uyy,uzz,  0,    0,    0,    0,uxy,  0,  0,  0],
                 [  0,uyy,  0,uxx,  0,uzz,    0,    0,    0,  0,uxy,  0,  0],
                 [  0,  0,uzz,  0,uxx,uyy,    0,    0,    0,  0,  0,uxy,  0],
                 [  0,  0,  0,  0,  0,  0,2*uyz,    0,    0,  0,  0,  0,uxz],
                 [  0,  0,  0,  0,  0,  0,    0,2*uxz,    0,  0,  0,  0,uyz],
                 [  0,  0,  0,  0,  0,  0,    0,    0,2*uxy,uxx,uyy,uzz,  0]])


def triclinic(u):
    '''Triclinic crystals.

    *Note*: This was never tested on the real case. Beware!

    The ordering of constants is:

    .. math::
       C_{11}, C_{22}, C_{33},
       C_{12}, C_{13}, C_{23},
       C_{44}, C_{55}, C_{66},
       C_{16}, C_{26}, C_{36}, C_{46}, C_{56},
       C_{14}, C_{15}, C_{25}, C_{45}

    :param u: vector of deformations:
        [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

    :returns: Symmetry defined stress-strain equation matrix
    '''

    # Based on the monoclinic matrix and not tested on real case.
    # If you have test cases for this symmetry send them to the author.
    uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
    return np.array(
    [[uxx,  0,  0,uyy,uzz,  0,    0,    0,    0,uxy,  0,  0,  0,  0,uyz,uxz,  0,  0],
     [  0,uyy,  0,uxx,  0,uzz,    0,    0,    0,  0,uxy,  0,  0,  0,  0,  0,uxz,  0],
     [  0,  0,uzz,  0,uxx,uyy,    0,    0,    0,  0,  0,uxy,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,2*uyz,    0,    0,  0,  0,  0,uxy,  0,uxx,  0,  0,uxz],
     [  0,  0,  0,  0,  0,  0,    0,2*uxz,    0,  0,  0,  0,  0,uxy,  0,uxx,uyy,uyz],
     [  0,  0,  0,  0,  0,  0,    0,    0,2*uxy,uxx,uyy,uzz,uyz,uxz,  0,  0,  0,  0]])


class Elastic:

    def __init__(self, structure, system:str=None, symprec=1e-3):
        self.structure = structure
        self.set_lattice_type(system, symprec)

    def set_lattice_type(self, system, symprec:float=1e-3):
        if system is None:
            dataset = spglib.get_symmetry_dataset(self.structure.to_cell(), symprec=symprec)
            try:
                lattype, bravais = self.get_lattice_type(dataset.number)
            except:
                lattype, bravais = self.get_lattice_type(dataset['number'])
        else:
            for i,j in enumerate(lattice_types):
                if system.lower()[:2] == j.lower()[:2]:
                    lattype, bravais = i,j
        self.lattype = lattype
        self.bravais = bravais

    @classmethod
    def get_lattice_type(cls, number):
    
        for i, l in enumerate(lattice_nums):
            if number < l:
                bravais = lattice_types[i]
                lattype = i+1
                break
    
        return lattype, bravais 


    @classmethod
    def from_file(cls, path, symprec=1e-3):
        s = read(path)
        return cls(s)

    def get_cart_deformed_cell(self, axis=0, size=1):
        '''Return the cell deformed along one of the cartesian directions
    
        Creates new deformed structure. The deformation is based on the
        base structure and is performed along single axis. The axis is
        specified as follows: 0,1,2 = x,y,z ; sheers: 3,4,5 = yz, xz, xy.
        The size of the deformation is in percent and degrees, respectively.
    
        :param base_cryst: structure to be deformed
        :param axis: direction of deformation
        :param size: size of the deformation
    
        :returns: new, deformed structure
        '''
        deform_i = [0,1,2,1,0,0]
        deform_j = [0,1,2,2,2,1]
        L = np.eye(3)
        L[deform_i[axis], deform_j[axis]] += size / 100

        lattice, positions, elements = self.structure.to_cell()
        lattice = np.dot(lattice, L)
        crystal = Structure.from_cell((lattice, positions, elements))
        return crystal

    def get_multi_cart_deformed_cell(self, axis:list, size=1):
        '''Return the cell deformed along one of the cartesian directions
    
        Creates new deformed structure. The deformation is based on the
        base structure and is performed along single axis. The axis is
        specified as follows: 0,1,2 = x,y,z ; sheers: 3,4,5 = yz, xz, xy.
        The size of the deformation is in percent and degrees, respectively.
    
        :param base_cryst: structure to be deformed
        :param axis: direction of deformation
        :param size: size of the deformation
    
        :returns: new, deformed structure
        '''
        deform_i = [0,1,2,1,0,0]
        deform_j = [0,1,2,2,2,1]
        L = np.eye(3)
        for i in axis:
            L[deform_i[i], deform_j[i]] += size / 100

        lattice, positions, elements = self.structure.to_cell()
        lattice = np.dot(lattice, L)
        crystal = Structure.from_cell((lattice, positions, elements))
        return crystal

    def get_elementary_deformations(self, n=5, d=2):
        '''Generate elementary deformations for elastic tensor calculation.
    
        The deformations are created based on the symmetry of the crystal and
        are limited to the non-equivalet axes of the crystal.
    
        :param cryst: Atoms object, basic structure
        :param n: integer, number of deformations per non-equivalent axis
        :param d: float, size of the maximum deformation in percent and degrees
    
        :returns: list of deformed structures
        '''
        # Deformation look-up table
        # Perhaps the number of deformations for trigonal
        # system could be reduced to [0,3] but better safe then sorry
        deform = {
            "Cubic": [[0, 3], regular],
            "Hexagonal": [[0, 2, 3, 5], hexagonal],
            "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
            "Tetragonal": [[0, 2, 3, 5], tetragonal],
            "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
            "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
            "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
        }
    
        axis, symm = deform[self.bravais]
    
        systems = []
        for a in axis:
            if a < 3:  # tetragonal deformation
                for dx in np.linspace(-d, d, n):
                    systems.append(
                            self.get_cart_deformed_cell(axis=a, size=dx))
            elif a < 6:  # sheer deformation (skip the zero angle)
                for dx in np.linspace(d/10.0, d, n):
                    systems.append(
                            self.get_cart_deformed_cell(axis=a, size=dx))
        return systems

    def get_multi_elementary_deformations(self, n=5, d=2):
        '''Generate elementary deformations for elastic tensor calculation.
    
        The deformations are created based on the symmetry of the crystal and
        are limited to the non-equivalet axes of the crystal.
    
        :param cryst: Atoms object, basic structure
        :param n: integer, number of deformations per non-equivalent axis
        :param d: float, size of the maximum deformation in percent and degrees
    
        :returns: list of deformed structures
        '''
        # Deformation look-up table
        # Perhaps the number of deformations for trigonal
        # system could be reduced to [0,3] but better safe then sorry
        deform = {
            "Cubic": [[0,1,2]],
            "Hexagonal": [[0,1], [0,2], [3,4]],
            "Trigonal": [[0,1], [0,2], [0,3], [0,4], [1,4]],
            "Tetragonal": [[0,1], [0,2], [1,2]],
            "Orthorombic": [[0,1], [0,2], [1,2]],
            "Monoclinic": [[0,1], [0,2], [1,2], [0,5], [1,5], [2,5], [3,4]],
            "Triclinic": [[0,1], [0,2], [1,2], 
                          [0,3], [0,4], [0,5],
                          [1,3], [1,4], [1,5],
                          [2,3], [2,4], [2,5],
                          [3,4], [3,5], [4,5]],
        }
    
        axis = deform[self.bravais]
    
        systems = []
        for a in axis:
            if a[-1] < 3:  # tetragonal deformation
                for dx in np.linspace(-d, d, n):
                    systems.append(
                            self.get_multi_cart_deformed_cell(axis=a, size=dx))
            elif a[-1] < 6:  # sheer deformation (skip the zero angle)
                for dx in np.linspace(d/10.0, d, n):
                    systems.append(
                            self.get_multi_cart_deformed_cell(axis=a, size=dx))
        return systems

    @classmethod
    def get_strain(self, lattice, ref_lattice):

        du = lattice-ref_lattice
        m = np.linalg.inv(ref_lattice)
        u = np.dot(m, du)
        u = (u+u.T)/2
        return np.array([u[0, 0], u[1, 1], u[2, 2], u[2, 1], u[2, 0], u[1, 0]])

    def get_elastic_tensor(self, systems):
        '''Calculate elastic tensor of the crystal.
    
        The elastic tensor is calculated from the stress-strain relation
        and derived by fitting this relation to the set of linear equations
        build from the symmetry of the crystal and strains and stresses
        of the set of elementary deformations of the unit cell.
    
        It is assumed that the crystal is converged and optimized
        under intended pressure/stress. The geometry and stress on the
        cryst is taken as the reference point. No additional optimization
        will be run. Structures in cryst and systems list must have calculated
        stresses. The function returns tuple of :math:`C_{ij}` elastic tensor,
        raw Birch coefficients :math:`B_{ij}` and fitting results: residuals,
        solution rank, singular values returned by numpy.linalg.lstsq.
    
        :param cryst: Atoms object, basic structure
        :param systems: list of Atoms object with calculated deformed structures
    
        :returns: tuple(:math:`C_{ij}` float vector,
                        tuple(:math:`B_{ij}` float vector, residuals, solution rank, singular values))
        '''
        import scipy
    
        # Deformation look-up table
        # Perhaps the number of deformations for trigonal
        # system could be reduced to [0,3] but better safe then sorry
        deform = {
            "Cubic": [[0, 3], regular],
            "Hexagonal": [[0, 2, 3, 5], hexagonal],
            "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
            "Tetragonal": [[0, 2, 3, 5], tetragonal],
            "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
            "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
            "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
        }
    
        axis, symm = deform[self.bravais]
        ref_lattice, ref_stress = systems[0]
        p = np.mean(ref_stress[:3])
        ref_stress = np.array([p, p, p, 0, 0, 0])
    
        ul = []
        sl = []
        for lattice, stress in systems[1:]:
            ul.append(self.get_strain(lattice, ref_lattice))
            # Remove the ambient pressure from the stress tensor
            sl.append(stress - ref_stress)
        #print(symm, ul)
        eqm = np.array([symm(u) for u in ul])
        # print(eqm)
        # print(eqm[0].shape, eqm.shape)
        eqm = np.reshape(eqm, (eqm.shape[0]*eqm.shape[1], eqm.shape[2]))
        # print(eqm)
        slm = np.array(sl).reshape(-1)
        #print(eqm.shape, slm.shape)
        # print(slm)
        Bij = scipy.linalg.lstsq(eqm, slm)
        # print(Bij[0] / units.GPa)
        # Calculate elastic constants from Birch coeff.
        # TODO: Check the sign of the pressure array in the B <=> C relation
        if (symm == orthorombic):
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p])
        elif (symm == tetragonal):
            Cij = Bij[0] - np.array([-p, -p, p, p, -p, -p])
        elif (symm == regular):
            Cij = Bij[0] - np.array([-p, p, -p])
        elif (symm == trigonal):
            Cij = Bij[0] - np.array([-p, -p, p, p, -p, p])
        elif (symm == hexagonal):
            Cij = Bij[0] - np.array([-p, -p, p, p, -p])
        elif (symm == monoclinic):
            # TODO: verify this pressure array
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p])
        elif (symm == triclinic):
            # TODO: verify this pressure array
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p,
                                  p, p, p, p, p, p, p, p, p])
        return Cij, Bij

    def get_elastic_tensor_by_energy(self, systems):
        '''Calculate elastic tensor of the crystal.
    
        The elastic tensor is calculated from the stress-strain relation
        and derived by fitting this relation to the set of linear equations
        build from the symmetry of the crystal and strains and stresses
        of the set of elementary deformations of the unit cell.
    
        It is assumed that the crystal is converged and optimized
        under intended pressure/stress. The geometry and stress on the
        cryst is taken as the reference point. No additional optimization
        will be run. Structures in cryst and systems list must have calculated
        stresses. The function returns tuple of :math:`C_{ij}` elastic tensor,
        raw Birch coefficients :math:`B_{ij}` and fitting results: residuals,
        solution rank, singular values returned by numpy.linalg.lstsq.
    
        :param cryst: Atoms object, basic structure
        :param systems: list of Atoms object with calculated deformed structures
    
        :returns: tuple(:math:`C_{ij}` float vector,
                        tuple(:math:`B_{ij}` float vector, residuals, solution rank, singular values))
        '''
        import scipy
        from collections import defaultdict
    
        # Deformation look-up table
        # Perhaps the number of deformations for trigonal
        # system could be reduced to [0,3] but better safe then sorry
        deform = {
            "Cubic": [[0, 3], regular],
            "Hexagonal": [[0, 2, 3, 5], hexagonal],
            "Trigonal": [[0, 1, 2, 3, 4, 5], trigonal],
            "Tetragonal": [[0, 2, 3, 5], tetragonal],
            "Orthorombic": [[0, 1, 2, 3, 4, 5], orthorombic],
            "Monoclinic": [[0, 1, 2, 3, 4, 5], monoclinic],
            "Triclinic": [[0, 1, 2, 3, 4, 5], triclinic]
        }
    
        axis, symm = deform[self.bravais]
        ref_lattice, ref_energy = systems[0]
        volume = abs(np.linalg.det(ref_lattice))
        #print(volume)
    
        data = defaultdict(list)
        for lattice, energy in systems[1:]:
            ul = self.get_strain(lattice, ref_lattice)
            axis = tuple(np.where(np.abs(ul)>1e-4, 1, 0))
            idx = np.argmax(np.abs(ul))
            data[axis].append((ul[idx], energy))
            print(ul, axis, energy)

        #print(data)

        results = []
        for i,rows in data.items():
            if sum(i) != 1: continue
            deltas = [0] + [v[0] for v in rows]
            energies = [ref_energy] + [v[1] for v in rows]
            #print(deltas, energies)
            coeffs = np.polyfit(deltas, energies, 2)
            if np.any(i[:3]):
                # ΔE/V = 1/2 C11
                results.append( 2 * coeffs[0] / volume * 160.21766208) # unit eV/Å³ > GPa
            else:
                # ΔE/V = 2 C44
                results.append( coeffs[0] / 2 / volume * 160.21766208) # unit eV/Å³ > GPa

        for i,rows in data.items():
            if sum(i) > 1:
                deltas = [0] + [v[0] for v in rows]
                energies = [ref_energy] + [v[1] for v in rows]
                coeffs = np.polyfit(deltas, energies, 2)
                if i[:3] == (1,1,1):
                    # ΔE/V = 3/2 (C11 + 2*C12)
                    result = (coeffs[0] / volume * 160.21766208 / 3 * 2 - results[0]) / 2
                    results.append(result)

        print(results)
        return results

        #print(symm, ul)
        eqm = np.array([symm(u) for u in ul])
        # print(eqm)
        # print(eqm[0].shape, eqm.shape)
        eqm = np.reshape(eqm, (eqm.shape[0]*eqm.shape[1], eqm.shape[2]))
        # print(eqm)
        slm = np.array(sl).reshape(-1)
        print(eqm.shape, slm.shape)
        # print(slm)
        Bij = scipy.linalg.lstsq(eqm, slm)
        # print(Bij[0] / units.GPa)
        # Calculate elastic constants from Birch coeff.
        # TODO: Check the sign of the pressure array in the B <=> C relation
        if (symm == orthorombic):
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p])
        elif (symm == tetragonal):
            Cij = Bij[0] - np.array([-p, -p, p, p, -p, -p])
        elif (symm == regular):
            Cij = Bij[0] - np.array([-p, p, -p])
        elif (symm == trigonal):
            Cij = Bij[0] - np.array([-p, -p, p, p, -p, p])
        elif (symm == hexagonal):
            Cij = Bij[0] - np.array([-p, -p, p, p, -p])
        elif (symm == monoclinic):
            # TODO: verify this pressure array
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p])
        elif (symm == triclinic):
            # TODO: verify this pressure array
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p,
                                  p, p, p, p, p, p, p, p, p])
        print(Bij, Cij, self.bravais)
        return Cij, Bij
    

    def get_cij_order(self):
        '''Give order of of elastic constants for the structure
    
        :param cryst: ASE Atoms object
    
        :returns: Order of elastic constants as a tuple of strings: C_ij
        '''
    
        orders = {
                1: ('C_11', 'C_22', 'C_33', 'C_12', 'C_13', 'C_23',
                    'C_44', 'C_55', 'C_66', 'C_16', 'C_26', 'C_36',
                    'C_46', 'C_56', 'C_14', 'C_15', 'C_25', 'C_45'),
                2: ('C_11', 'C_22', 'C_33', 'C_12', 'C_13', 'C_23',
                    'C_44', 'C_55', 'C_66', 'C_16', 'C_26', 'C_36', 'C_45'),
                3: ('C_11', 'C_22', 'C_33', 'C_12', 'C_13', 'C_23', 'C_44',
                    'C_55', 'C_66'),
                4: ('C_11', 'C_33', 'C_12', 'C_13', 'C_44', 'C_66'),
                5: ('C_11', 'C_33', 'C_12', 'C_13', 'C_44', 'C_14'),
                6: ('C_11', 'C_33', 'C_12', 'C_13', 'C_44'),
                7: ('C_11', 'C_12', 'C_44'),
                }
        return orders[self.lattype]
