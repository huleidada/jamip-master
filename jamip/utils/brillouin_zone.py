import numpy as np
import spglib
import math
import copy

class HighSymmetryKpath(object):

    def __init__(self):
        
        self.kpath = None
        self.pointgroup = None
        self.spacegroup = None
        self.transf_matrix_invQ = None
        self.transf_matrix_invP = None

    def rec_real_transf(self, structure_basis):

        after_transf = 2 * math.pi * np.linalg.inv(structure_basis)
        
        return (after_transf.T).tolist()
    
    def get_lattice_constant(self, structure_basis):

        va, vb, vc = np.array(structure_basis)
        a = np.linalg.norm(va)
        b = np.linalg.norm(vb)
        c = np.linalg.norm(vc)
        cosalpha = np.dot(vb, vc) / b / c
        cosbeta = np.dot(va, vc) / a / c
        cosgamma = np.dot(va, vb) / a / b

        return (a, b, c, cosalpha, cosbeta, cosgamma)

    def get_pgnum(self, pg_international):
   
        pg_dict = {u'C1': 1, u'C2': 3, u'C2h': 5, u'C2v': 7, \
                   u'C3': 16, u'C3h': 22, u'C3i': 17, u'C3v': 19, \
                   u'C4': 9, u'C4h': 11, u'C4v': 13, u'C6': 21, \
                   u'C6h': 23, u'C6v': 25, u'Ci': 2, u'Cs': 4, \
                   u'D2': 6, u'D2d': 14, u'D2h': 8, u'D3': 18, \
                   u'D3d': 20, u'D3h': 26, u'D4': 12, u'D4h': 15, \
                   u'D6': 24, u'D6h': 27, u'O': 30, u'Oh': 32, \
                   u'S4': 10, u'T': 28, u'Td': 31, u'Th': 29}

        return int(pg_dict[pg_international])

    def pg_inversion(self, pg_number):
   
        if pg_number in [2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32]:
            return True
        elif pg_number in [1, 3, 4, 6, 7, 9, 10, 12, 13, 14, 16, 
                           18, 19, 21, 22, 24, 25, 26, 28, 30, 31]:
            return False
        else:
            print("pg_number should be between 1 and 32")
        
    def cP(self, sgnum):

        self.transf_matrix_invP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transf_matrix_invQ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        kpoints = {'Gamma': [0, 0, 0], 
                   'R': [1./2, 1./2, 1./2], \
                   'M': [1./2, 1./2, 0], 
                   'X': [0, 1./2, 0], 
                   'X1': [1./2, 0, 0]}

        if sgnum in [195, 198, 200, 201, 205]:
            path = [['Gamma', 'X', 'M', 'Gamma', 'R', 'X'], ['R', 'M', 'X1']]
        else:
            path = [['Gamma', 'X', 'M', 'Gamma', 'R', 'X'], ['R', 'M']]
            
        return {'Kpoints': kpoints, 'Path': path}

    def cF(self, sgnum):
        
        self.transf_matrix_invP = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        self.transf_matrix_invQ = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        kpoints = {'Gamma': [0, 0, 0], 
                   'X': [1./2, 0, 1./2], \
                   'L': [1./2, 1./2, 1./2], \
                   'W': [1./2, 1./4, 3./4], 
                   'W2': [3./4, 1./4, 1./2], \
                   'K': [3./8, 3./8, 3./4], 
                   'U': [5./8, 1./4, 5./8]}

        if sgnum in [196, 202, 203]:      
            path = [['Gamma', 'X', 'U'], ['K', 'Gamma', 'L', 'W', 'X', 'W2']]
        else:
            path = [['Gamma', 'X', 'U'], ['K', 'Gamma', 'L', 'W', 'X']]

        return {'Kpoints': kpoints, 'Path': path}

    def cI(self):

        self.transf_matrix_invQ = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        self.transf_matrix_invP = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        kpoints = {'Gamma': [0, 0, 0], 'H': [1./2, -1./2, 1./2], \
                   'P': [1./4, 1./4, 1./4], 'N': [0, 0, 1./2]}

        path = [['Gamma', 'H', 'N', 'Gamma', 'P', 'H'], ['P', 'N']]

        return {'Kpoints': kpoints, 'Path': path}

    def cubic(self, sgnum):
        
        if sgnum in [195, 198, 200, 201, 205, 207, 208, \
                     212, 213, 215, 218, 221, 222, 223, 224]:
            self.kpath = self.cP(sgnum)

        elif sgnum in [196, 202, 203, 209, 210, 216, \
                       219, 225, 226, 227, 228]:
            self.kpath = self.cF(sgnum)

        elif sgnum in [197, 199, 204, 206, 211, 214, \
                       217, 220, 229, 230]:
            self.kpath = self.cI()

        else:
            print(f"Error! Unexpected value for space group number: {sgnum}")

    def tP(self):

        self.transf_matrix_invQ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transf_matrix_invP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        kpoints = {'Gamma': [0, 0, 0], 
                   'Z': [0, 0, 1./2], 
                   'M': [1./2, 1./2, 0], 
                   'A': [1./2, 1./2, 1./2], 
                   'R': [0, 1./2, 1./2], 
                   'X': [0, 1./2, 0]}
        path = [['Gamma', 'X', 'M', 'Gamma', 'Z', 'R', 'A', 'Z'], \
                ['X', 'R'], ['M', 'A']]

        return {'Kpoints': kpoints, 'Path': path}

    def tI(self, a, b, c):

        self.transf_matrix_invQ = np.array([[-1, 1, 0], [0, 0, 1], [1, 1, -1]])
        self.transf_matrix_invP = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        
        if c < a:
            eta = 1./4 * (1 + c**2/a**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'M': [-1./2, 1./2, 1./2], \
                       'X': [0, 0, 1./2], 
                       'P': [1./4, 1./4, 1./4], \
                       'Z': [eta, eta, -1 * eta], \
                       'Z0': [-1 * eta, 1 - eta, eta], 
                       'N': [0, 1./2, 0]}
            path = [['Gamma', 'X', 'M', 'Gamma', 'Z'], \
                    ['Z0', 'M'], ['X', 'P', 'N', 'Gamma']]
        elif c > a:
            eta = 1./4 * (1 + a**2/c**2)
            zeta = a**2/(2. * c**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'M': [1./2, 1./2, -1./2], \
                       'X': [0, 0, 1./2], 
                       'P': [1./4, 1./4, 1./4], \
                       'N': [0, 1./2, 0], 
                       'S0': [-1 * eta, eta, eta], \
                       'S': [eta, 1 - eta, -1 * eta], \
                       'R': [-1 * zeta, zeta, 1./2], 
                       'G': [1./2, 1./2, -1 * zeta]}
            path = [['Gamma', 'X', 'P', 'N', 'Gamma', 'M', 'S'], \
                    ['S0', 'Gamma'], ['X', 'R'], ['G', 'M']]
        else:
            print (f"Error! Unexpected value for Basis vectors: {a}, {b} and {c}")

        return {'Kpoints': kpoints, 'Path': path}

    def tetragonal(self, sgnum, a, b, c):

        if sgnum in [79, 82, 87, 88, 97, 98, 107, 108, \
                     109, 110, 119, 120, 121, 122, 139, 140, 141, 142]:
            self.kpath = self.tI(a, b, c)
        else:
            self.kpath = self.tP()

    def oP(self):

        self.transf_matrix_invP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transf_matrix_invQ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        kpoints = {'Gamma': [0, 0, 0], 
                   'X': [1./2, 0, 0], 
                   'Z': [0, 0, 1./2], \
                   'U': [1./2, 0, 1./2], 
                   'Y': [0, 1./2, 0], 
                   'S': [1./2, 1./2, 0], \
                   'T': [0, 1./2, 1./2], \
                   'R': [1./2, 1./2, 1./2]}
        path = [['Gamma', 'X', 'S', 'Y', 'Gamma', 'Z', 'U', 'R', 'T', 'Z'], \
                ['X', 'U'], ['Y', 'T'], ['S', 'R']]

        return {'Kpoints': kpoints, 'Path': path}

    def oF(self, a, b, c):

        self.transf_matrix_invQ = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        self.transf_matrix_invP = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        
        if a**-2 > b**-2 + c**-2:
            zeta = 1./4 * (1 + a**2/b**2 - a**2/c**2)
            eta = 1./4 * (1 + a**2/b**2 + a**2/c**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'T': [1, 1./2, 1./2], 
                       'Z': [1./2, 1./2, 0], \
                       'Y': [1./2, 0, 1./2], 
                       'Sigma0': [0, eta, eta], \
                       'U0': [1, 1 - eta, 1 - eta], 
                       'A0': [1./2, 1./2 + zeta, zeta], \
                       'C0': [1./2, 1./2 - zeta, 1 - zeta], \
                       'L': [1./2, 1./2, 1./2]}
            path = [['Gamma', 'Y', 'T', 'Z', 'Gamma', 'Sigma0'], ['U0', 'T'], \
                    ['Y', 'C0'], ['A0', 'Z'], ['Gamma', 'L']]

        elif c**-2 > a**-2 + b**-2:
            zeta = 1./4 * (1 + c**2/a**2 - c**2/b**2)
            eta = 1./4 * (1 + c**2/a**2 + c**2/b**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'T': [0, 1./2, 1./2], 
                       'Z': [1./2, 1./2, 1], \
                       'Y': [1./2, 0, 1./2], 
                       'Lambda0': [eta, eta, 0], \
                       'Q0': [1 - eta, 1 - eta, 1], 
                       'G0': [1./2 - zeta, 1 - zeta, 1./2], 
                       'H0': [1./2 + zeta, zeta, 1./2], 
                       'L': [1./2, 1./2, 1./2]}
            path = [['Gamma', 'T', 'Z', 'Y', 'Gamma', 'Lambda0'], ['Q0', 'Z'], \
                    ['T', 'G0'], ['H0', 'Y'], ['Gamma', 'L']]

        elif (a**-2 + b**-2 >= c**-2) or \
             (a**-2 + c**-2 >= b**-2) or \
             (b**-2 + c**-2 >= a**-2):
            eta = 1./4 * (1 + a**2/b**2 - a**2/c**2)
            delta = 1./4 * (1 + b**2/a**2 - b**2/c**2)
            phi = 1./4 * (1 + c**2/b**2 - c**2/a**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'T': [0, 1./2, 1./2], 
                       'Z': [1./2, 1./2, 0], 
                       'Y': [1./2, 0, 1./2], 
                       'A0': [1./2, 1./2 + eta, eta], 
                       'C0': [1./2, 1./2 - eta, 1 - eta], 
                       'B0': [1./2 + delta, 1./2, delta], 
                       'D0': [1./2 - delta, 1./2, 1 - delta], 
                       'G0': [phi, 1./2 + phi, 1./2], 
                       'H0': [1. - phi, 1./2 - phi, 1./2], 
                       'L': [1./2, 1./2, 1./2]}

            path = [['Gamma', 'Y', 'C0'], ['A0', 'Z', 'B0'], ['D0', 'T', 'G0'], \
                    ['H0', 'Y'], ['T', 'Gamma', 'Z'], ['Gamma', 'L']]
        else:
            print (f"Error! Unexpected value for Basis vectors: {a}, {b} and {c}")

        return {'Kpoints': kpoints, 'Path': path}

    def oI(self, a, b, c):

        self.transf_matrix_invQ = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        self.transf_matrix_invP = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        if max(a, b, c) == c:
            zeta = 1./4 * (1 + a**2/c**2)
            eta = 1./4 * (1 + b**2/c**2)
            delta = (b**2 - a**2)/(4. * c**2)
            mu = (a**2 + b**2)/(4. * c**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'X': [1./2, 1./2, -1./2], 
                       'S': [1./2, 0, 0], 
                       'R': [0, 1./2, 0], 
                       'T': [0, 0, 1./2], 
                       'W': [1./4, 1./4, 1./4], 
                       'Sigma0': [-1. * zeta, zeta, zeta], 
                       'F2': [zeta, 1. - zeta, -1. * zeta], 
                       'Y0': [eta, -1. * eta, eta], 
                       'U0': [1. - eta, eta, -1. * eta], 
                       'L0': [-1. * mu, mu, 1./2- delta], 
                       'M0': [mu, -1. * mu , 1./2 + delta], 
                       'J0': [1./2 - delta, 1./2 + delta, -1. * mu]}

            path = [['Gamma', 'X', 'F2'], ['Sigma0', 'Gamma', 'Y0'], \
                    ['U0', 'X'], ['Gamma', 'R', 'W', 'S', 'Gamma', 'T', 'W']]

        elif max(a, b, c) == a:
            zeta = 1./4 * (1 + b**2/a**2)
            eta = 1./4 * (1 + c**2/a**2)
            delta = (c**2 - b**2)/(4. * a**2)
            mu = (b**2 + c**2)/(4. * a**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'X': [-1./2, 1./2, 1./2], 
                       'S': [1./2, 0, 0], \
                       'R': [0, 1./2, 0], 
                       'T': [0, 0, 1./2],
                       'W': [1./4, 1./4, 1./4], \
                       'Y0': [zeta, -1. * zeta, zeta], 
                       'U2': [-1. * zeta, zeta, 1. - zeta], \
                       'Lambda0': [eta, eta, -1. * eta], 
                       'G2': [-1. * eta, 1. - eta, eta], 
                       'K': [1./2 - delta, -1. * mu, mu], 
                       'K2': [1./2 + delta, mu, -1. * mu], \
                       'K4': [-1. * mu, 1./2 - delta, 1./2 + delta]}

            path = [['Gamma', 'X', 'U2'], ['Y0', 'Gamma', 'Lambda0'], 
                    ['G2', 'X'], ['Gamma', 'R', 'W', 'S', 'Gamma', 'T', 'W']]

        elif max(a, b, c) == b:
            zeta = 1./4 * (1 + c**2/b**2)
            eta = 1./4 * (1 + a**2/b**2)
            delta = (a**2 - c**2)/(4. * b**2)
            mu = (a**2 + c**2)/(4. * b**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'X': [1./2, -1./2, 1./2], 
                       'S': [1./2, 0, 0], 
                       'R': [0, 1./2, 0], 
                       'T': [0, 0, 1./2], 
                       'W': [1./4, 1./4, 1./4], \
                       'Sigma0': [-1. * eta, eta, eta], 
                       'F0': [eta, -1. * eta, 1. - eta], \
                       'Lambda0': [zeta, zeta, -1. * zeta], 
                       'G0': [1. - zeta, -1. * zeta, zeta], \
                       'V0': [mu, 1./2 - delta, -1. * mu], 
                       'H0': [-1. * mu, 1./2 + delta, mu], \
                       'H2': [1./2 + delta, -1. * mu, 1./2 - delta]}

            path = [['Gamma', 'X', 'F0'], ['Sigma0', 'Gamma', 'Lambda0'], \
                    ['G0', 'X'], ['Gamma', 'R', 'W', 'S', 'Gamma', 'T', 'W']]
        else:
            print (f"Error! Unexpected value for Basis vectors: {a}, {b} and {c}")

        return {'Kpoints': kpoints, 'Path': path}

    def oS(self, sgnum, a, b, c):

        self.transf_matrix_invQ = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
        
        if sgnum in [20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 67, 68] and a < b:
            self.transf_matrix_invP = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])
            zeta = 1./4 * (1 + a**2/b**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'Y': [-1./2, 1./2, 0], 
                       'T': [-1./2, 1./2, 1./2], \
                       'Z': [0, 0, 1./2], 
                       'S': [0, 1./2, 0], 
                       'R': [0, 1./2, 1./2], \
                       'Sigma0': [zeta, zeta, 0], 
                       'C0': [-1. * zeta, 1. - zeta, 0], \
                       'A0': [zeta, zeta, 1./2], 
                       'E0': [-1. * zeta, 1. - zeta, 1./2]}

            path = [['Gamma', 'Y', 'C0'], ['Sigma0', 'Gamma', 'Z', 'A0'], \
                    ['E0', 'T', 'Y'], ['Gamma', 'S', 'R', 'Z', 'T']]

        elif sgnum in [38, 39, 40, 41] and b < c:
            self.transf_matrix_invP = np.array([[0, 1, -1], [0, 1, 1], [1, 0, 0]])
            zeta = 1./4 * (1 + b**2/c**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'Y': [-1./2, 1./2, 0], 
                       'T': [-1./2, 1./2, 1./2],
                       'Z': [0, 0, 1./2], 
                       'S': [0, 1./2, 0], 
                       'R': [0, 1./2, 1./2], 
                       'Sigma0': [zeta, zeta, 0], 
                       'C0': [-1. * zeta, 1. - zeta, 0], 
                       'A0': [zeta, zeta, 1./2], 
                       'E0': [-1. * zeta, 1. - zeta, 1./2]}
 
            path = [['Gamma', 'Y', 'C0'], ['Sigma0', 'Gamma', 'Z', 'A0'],\
                    ['E0', 'T', 'Y'], ['Gamma', 'S', 'R', 'Z', 'T']]

        elif sgnum in [20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 67, 68] and a > b:
            self.transf_matrix_invP = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])
            zeta = 1./4 * (1 + b**2/a**2)

            kpoints = {'Gamma': [0, 0, 0], 
                       'Y': [1./2, 1./2, 0], 
                       'T': [1./2, 1./2, 1./2], \
                       'T2': [1./2, 1./2, -1./2], 
                       'Z': [0, 0, 1./2], 
                       'Z2': [0, 0, -1./2], \
                       'S': [0, 1./2, 0], 
                       'R': [0, 1./2, 1./2], 
                       'R2': [0, 1./2, -1./2], \
                       'Delta0': [-1. * zeta, zeta, 0], 
                       'F0': [zeta, 1. - zeta, 0], \
                       'B0': [-1. * zeta, zeta, 1./2], 
                       'B2': [-1. * zeta, zeta, -1./2], \
                       'G0': [zeta, 1. - zeta, 1./2], 
                       'G2': [zeta, 1. - zeta, -1./2]}

            path = [['Gamma', 'Y', 'F0'], ['Delta0', 'Gamma', 'Z', 'B0'], \
                    ['G0', 'T', 'Y'], ['Gamma', 'S', 'R', 'Z', 'T']]

        elif sgnum in [38, 39, 40, 41] and b > c:
            self.transf_matrix_invP = np.array([[0, 1, -1], [0, 1, 1], [1, 0, 0]])
            zeta = 1./4 * (1 + c**2/b**2)
            kpoints = {'Gamma': [0, 0, 0], 
                       'Y': [1./2, 1./2, 0], 
                       'T': [1./2, 1./2, 1./2], \
                       'T2': [1./2, 1./2, -1./2], 
                       'Z': [0, 0, 1./2], 
                       'Z2': [0, 0, -1./2], \
                       'S': [0, 1./2, 0], 
                       'R': [0, 1./2, 1./2], 
                       'R2': [0, 1./2, -1./2], \
                       'Delta0': [-1. * zeta, zeta, 0], 
                       'F0': [zeta, 1. - zeta, 0], \
                       'B0': [-1. * zeta, zeta, 1./2], 
                       'B2': [-1. * zeta, zeta, -1./2], \
                       'G0': [zeta, 1. - zeta, 1./2], 
                       'G2': [zeta, 1. - zeta, -1./2]}

            path = [['Gamma', 'Y', 'F0'], ['Delta0', 'Gamma', 'Z', 'B0'], \
                    ['G0', 'T', 'Y'], ['Gamma', 'S', 'R', 'Z', 'T']]
        else:
            print(f"Error! Unexpected value for Basis vectors: {a}, {b}, {c}")

        return {'Kpoints': kpoints, 'Path': path}
            
    def orthorhombic(self, sgnum, a, b, c):

        if sgnum in [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, \
                     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]:
            self.kpath = self.oP()
        elif sgnum in [22, 42, 43, 69, 70]:
            self.kpath = self.oF(a, b, c)
        elif sgnum in [23, 24, 44, 45, 46, 71, 72, 73, 74]:
            self.kpath = self.oI(a, b, c)
        elif sgnum in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
            self.kpath = self.oS(sgnum, a, b, c)
        else:
            print(f"Error! Unexpected value for space group number: {sgnum}")

    def hP(self, sgnum):
        
        self.transf_matrix_invP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transf_matrix_invQ = np.array([[1, 0, 0], [-1, 1, 0], [0, 0, 1]])
        kpoints = {'Gamma': [0, 0, 0], 
                   'A': [0, 0, 1./2], 
                   'K': [1./3, 1./3, 0], \
                   'H': [1./3, 1./3, 1./2], 
                   'H2': [1./3, 1./3, -1./2], \
                   'M': [1./2, 0, 0], 
                   'L': [1./2, 0, 1./2]}
        
        if sgnum in [143, 144, 145, 147, 149, 151, 153, 157, 159, 162, 163]:
            path = [['Gamma', 'M', 'K', 'Gamma', 'A', 'L', 'H', 'A'], \
                    ['L', 'M'], ['H', 'K', 'H2']]
        else:
            path = [['Gamma', 'M', 'K', 'Gamma', 'A', 'L', 'H', 'A'], \
                    ['L', 'M'], ['H', 'K']]

        return {'Kpoints': kpoints, 'Path': path}

    def hexagonal(self, sgnum):

        self.kpath = self.hP(sgnum)
     
    def hR(self, a, b, c):

        self.transf_matrix_invQ = np.array([[1, -1, 0], [0, 1, -1], [1, 1, 1]])
        self.transf_matrix_invP = np.array([[1, 0, 1], [-1, 1, 1], [0, -1, 1]])
        
        if math.sqrt(3) * a < math.sqrt(2) * c:
            delta = a**2 / (4. * c**2)
            eta = 5./6 - 2. * delta
            nu = 1./3 + delta
            kpoints = {'Gamma': [0, 0, 0], 
                       'T': [1./2, 1./2, 1./2], 
                       'L': [1./2, 0, 0], \
                       'L2': [0, -1./2, 0], 
                       'L4': [0, 0, -1./2], 
                       'F': [1./2, 0, 1./2],\
                       'F2': [1./2, 1./2, 0], 
                       'S0': [nu, -1 * nu, 0], 
                       'S2': [1. - nu, 0, nu],\
                       'S4': [nu, 0, -1 * nu], 
                       'S6': [1. - nu, nu, 0], \
                       'H0': [1./2, -1. + eta, 1. - eta], 
                       'H2': [eta, 1. - eta, 1./2], \
                       'H4': [eta, 1./2, 1. - eta], 
                       'H6': [1./2, 1. - eta, -1. + eta], \
                       'M0': [nu, -1. + eta, nu], 
                       'M2': [1. - nu, 1. - eta, 1. - nu], \
                       'M4': [eta, nu, nu], 
                       'M6': [1. - nu, 1. - nu , 1. - eta], 
                       'M8': [nu, nu, -1. + eta]}

            path = [['Gamma', 'T', 'H2'], ['H0', 'L', 'Gamma', 'S0'], \
                    ['S2', 'F', 'Gamma']]

        elif math.sqrt(3) * a > math.sqrt(2) * c:
            zeta = 1./6 - c**2./(9. * a **2)
            eta = 1./2 - 2. * zeta
            nu = 1./2 + zeta
            kpoints = {'Gamma': [0, 0, 0], 
                       'T': [1./2, -1./2, 1./2], \
                       'P0': [eta, -1. + eta, eta], 
                       'P2': [eta, eta, eta], \
                       'R0': [1. - eta, -1. * eta, -1. * eta], \
                       'M': [1. - nu, -1. * nu, 1. - nu], \
                       'M2': [nu, -1. + nu, -1. + nu], \
                       'L': [1./2, 0, 0], 
                       'F': [1./2, -1./2, 0]}

            path = [['Gamma', 'L', 'T', 'P0'], ['P2', 'Gamma', 'F']]
        else:
            print(f"Error! Unexpected value for Basis vectors: {a}, {b}, {c}")
            
        return {'Kpoints': kpoints, 'Path': path}

    def trigonal(self, sgnum, a, b, c):

        if sgnum in [146, 148, 155, 160, 161, 166, 167]:
            self.kpath = self.hR(a, b, c)
        else: 
            self.kpath = self.hP(sgnum)
       
    def mP(self, a, c, cosbeta):
        
        self.transf_matrix_invP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.transf_matrix_invQ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        eta = (1. + (a/c) * cosbeta)/(2. * (1. - cosbeta**2))
        nu = 1./2 + eta * c * cosbeta / a
        kpoints = {'Gamma': [0, 0, 0], 'Z': [0, 1./2, 0], 'B': [0, 0, 1./2], \
                   'B2': [0, 0, -1./2], 'Y': [1./2, 0, 0], 'Y2': [-1./2, 0, 0],\
                    'C': [1./2, 1./2, 0], 'C2': [-1./2, 1./2, 0], 'D': [0, 1./2, 1./2], \
                   'D2': [0, 1./2, -1./2], 'A': [-1./2, 0, 1./2], 'E': [-1./2, 1./2, 1./2], \
                   'H': [-1. * eta, 0, 1. - nu], 'H2': [-1. + eta, 0, nu], \
                   'H4': [-1. * eta, 0, -1. * nu], 'M': [-1. * eta, 1./2, 1. - nu], \
                   'M2': [-1. + eta, 1./2, nu], 'M4': [-1. * eta, 1./2, -1. * nu]}

        path = [['Gamma', 'Z', 'D', 'B', 'Gamma', 'A', 'E', 'Z', 'C2', 'Y2', 'Gamma']]

        return {'Kpoints': kpoints, 'Path': path}

    def mC(self, a, b, c, cosbeta):

        self.transf_matrix_invQ = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])
        self.transf_matrix_invP = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
        
        if b < a * math.sqrt(1 - cosbeta**2):
            zeta = (2. + (a/c) * cosbeta) / (4. * (1 - cosbeta**2))
            eta = 1./2 - 2. * zeta * c * cosbeta / a
            psi = 3./4 - b**2 / (4. * a**2 * (1. - cosbeta**2))
            phi = psi - (3./4 - psi) * a * cosbeta / c
            kpoints = {'Gamma': [0, 0, 0], 'Y2': [-1./2, 1./2, 0], 'Y4': [1./2, -1./2, 0], \
                       'A': [0, 0, 1./2], 'M2': [-1./2, 1./2, 1./2], 'V': [1./2, 0, 0],\
                      'V2': [0, 1./2, 0], 'L2': [0, 1./2, 1./2], 'C': [1. - psi, 1. - psi, 0], \
                      'C2': [-1. + psi, psi, 0], 'C4': [psi, -1. + psi, 0], \
                       'D': [-1. + phi, phi, 1./2], 'D2': [1. - phi, 1. - phi, 1./2], \
                       'E': [-1. + zeta, 1. - zeta, 1. - eta], 'E2': [-1. * zeta, zeta, eta], \
                      'E4': [zeta, -1. * zeta, 1. - eta]}

            path = [['Gamma', 'C'], ['C2', 'Y2', 'Gamma', 'M2', 'D'], 
                    ['D2', 'A', 'Gamma'], ['L2', 'Gamma', 'V2']]

        elif b > a * math.sqrt(1. - cosbeta**2) and \
            -a * cosbeta / c + a**2 * (1. - cosbeta**2) / b**2 < 1:
            mu = 1./4 * (1. + a**2 / b**2)
            delta = -a * c * cosbeta / (2. * b**2)
            zeta = 1./4 * (a**2 / b**2 + (1. + (a/c) * cosbeta) / (1. - cosbeta**2))
            eta = 1./2 - 2. * zeta * c * cosbeta / a
            phi = 1. + zeta - 2. * mu
            psi = eta - 2. * delta
            kpoints = {'Gamma': [0, 0, 0], 'Y': [1./2, 1./2, 0], 'A': [0, 0, 1./2], \
                       'M': [1./2, 1./2, 1./2], 'V2': [0, 1./2, 0], 'L2': [0, 1./2, 1./2], \
                       'F': [-1. + phi, 1. - phi, 1. - psi], 'F2': [1. - phi, phi, psi], \
                      'F4': [phi, 1. - phi, 1. - psi], 'H': [-1. * zeta, zeta, eta], \
                      'H2': [zeta, 1. - zeta, 1. - eta], 'H4': [zeta, -1. * zeta, 1. - eta],\
                       'G': [-1. * mu, mu, delta], 'G2': [mu, 1. - mu, -1. * delta], \
                      'G4': [mu, -1. * mu, -1. * delta], 'G6': [1. - mu, mu, delta]}

            path = [['Gamma', 'Y', 'M', 'A', 'Gamma'], ['L2', 'Gamma', 'V2']]
        elif b > a * math.sqrt(1. - cosbeta**2) and \
            -1. * a * cosbeta / c + a**2 * (1. - cosbeta**2) / b**2 > 1:
            zeta = 1./4 * (a**2 / b**2 + (1. + (a/c) * cosbeta) / (1. - cosbeta**2))
            rho = 1. - zeta * b**2 / a**2
            eta = 1./2 - 2. * zeta * c * cosbeta / a
            mu = eta/2. + a**2 / (4. * b**2) + a * c * cosbeta / (2. * b**2)
            nu = 2. * mu - zeta
            omega = c / (2. * a * cosbeta) * (1. - 4. * nu + a**2 * (1. - cosbeta**2) / b**2)
            delta = -1./4 + omega/2. - zeta * c * cosbeta / a
            kpoints = {'Gamma': [0, 0, 0], 'Y': [1./2, 1./2, 0], 'A': [0, 0, 1./2], \
                       'M2': [-1./2, 1./2, 1./2], 'V': [1./2, 0, 0], 'V2': [0, 1./2, 0], \
                       'L2': [0, 1./2, 1./2], 'I': [-1. + rho, rho, 1./2], \
                       'I2': [1. - rho, 1. - rho, 1./2], 'K': [-1. * nu, nu, omega], \
                       'K2': [-1. +  nu, 1. - nu, 1. - omega], 'K4': [1. - nu, nu, omega], \
                        'H': [-1. * zeta, zeta, eta], 'H2': [zeta, 1. - zeta, 1. - eta], \
                       'H4': [zeta, -1. * zeta, 1. - eta], 'N': [-1. * mu, mu, delta], \
                       'N2': [mu, 1. - mu, -1. * delta], 'N4': [mu, -1. * mu, -1. * delta], \
                       'N6': [1. - mu, mu, delta]}

            path = [['Gamma', 'A', 'I2'], ['I', 'M2', 'Gamma', 'Y'], ['L2', 'Gamma', 'V2']]
        else:
            print(f"Error! Unexpected value for Basis vectors: {a}, {b}, {c}, {cosbeta}")

        return {'Kpoints': kpoints, 'Path': path}

    def monoclinic(self, sgnum, a, b, c, cosbeta):

        if sgnum in [3, 4, 6, 7, 10, 11, 13, 14]:
            self.kpath = self.mP(a, c, cosbeta)
        elif sgnum in [5, 8, 9, 12, 15]:
            self.kpath = self.mC(a, b, c, cosbeta)
        else:
            print("Error! Unexpected value for space group number: %d" % sgnum)

    def aP(self, coskalpha, coskbeta, coskgamma):

        if coskalpha < 0 and coskbeta < 0 and coskgamma < 0:
            kpoints = {'Gamma': [0, 0, 0], 'Z': [0, 0, 1./2], 'Y': [0, 1./2, 0], \
                       'X': [1./2, 0, 0], 'V': [1./2, 1./2, 0], 'U': [1./2, 0, 1./2], \
                       'T': [0, 1./2, 1./2], 'R': [1./2, 1./2, 1./2]}

            path = [['Gamma', 'X'], ['Y', 'Gamma', 'Z'], \
                    ['R', 'Gamma', 'T'], ['U', 'Gamma', 'V']]

        elif coskalpha > 0 and coskbeta > 0 and coskgamma > 0:
            kpoints = {'Gamma': [0, 0, 0], 'Z': [0, 0, 1./2], 'Y': [0, 1./2, 0], \
                       'Y2': [0, -1./2, 0], 'X': [1./2, 0, 0], 'V2': [1./2, -1./2, 0], \
                       'U2': [-1./2, 0, 1./2], 'T2': [0, -1./2, 1./2], \
                       'R2': [-1./2, -1./2, 1./2]}

            path = [['Gamma', 'X'], ['Y', 'Gamma', 'Z'], \
                    ['R2', 'Gamma', 'T2'], ['U2', 'Gamma', 'V2']]
        else:
            print(f"Error! Unexpected value for K-Basis vectors: {coskalpha}, {coskbeta}, {coskgamma}")

        return {'Kpoints': kpoints, 'Path': path}

    def triclinic(self, coskalpha, coskbeta, coskgamma):
        
        self.kpath = self.aP(coskalpha, coskbeta, coskgamma)

    def get_HSKP(self, structure, symprec=1e-3):
        from .version import get_spglib_version

        structure_dataset = spglib.get_symmetry_dataset(structure, symprec=symprec, angle_tolerance=-1.0)
        if get_spglib_version():
            # For spglib version >= 2.0.0
            self.pointgroup = structure_dataset.pointgroup
            self.spacegroup = structure_dataset.international
            structure_basis = structure_dataset.std_lattice
            self.sgnum = sgnum = structure_dataset.number
            hall_num = structure_dataset.hall_number
        else:
            self.pointgroup = structure_dataset['pointgroup']
            self.spacegroup = structure_dataset['international']
            structure_basis = structure_dataset['std_lattice']
            self.sgnum = sgnum = structure_dataset['number']
            hall_num = structure_dataset['hall_number']
        pg_international = spglib.get_spacegroup_type(hall_num)['pointgroup_schoenflies']
        inversion_symmetry = self.pg_inversion(self.get_pgnum(pg_international))
        a, b, c, cosalpha, cosbeta, cosgamma = self.get_lattice_constant(structure_basis)
        time_reversal = True
        
        if sgnum in range (195, 231):
            self.cubic(sgnum)
        elif sgnum in range (75, 143):
            self.tetragonal(sgnum, a, b, c)
        elif sgnum in range (16, 75):
            self.orthorhombic(sgnum, a, b, c)
        elif sgnum in range (168, 195):
            self.hexagonal(sgnum)
        elif sgnum in range (143, 168):
            self.trigonal(sgnum, a, b, c)
        elif sgnum in range (3, 16):
            self.monoclinic(sgnum, a, b, c, cosbeta)
        elif sgnum in range (1, 3):
            rec1 = self.rec_real_transf(structure_basis)
            rec2 = spglib.niggli_reduce(rec1)
            real2 = self.rec_real_transf(rec2)
            ka2, kb2, kc2, coskalpha2, coskbeta2, coskgamma2 = self.get_lattice_constant(rec2)
            conditions = np.array([abs(kb2 * kc2 * coskalpha2), \
                         abs(kc2 * ka2 * coskbeta2), \
                         abs(ka2 * kb2 * coskgamma2)])

            matrix_M2 = [np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), \
                         np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), \
                         np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]

            smallest_condition = np.argsort(conditions)[0]
            M2 = matrix_M2[smallest_condition]
            real3 = np.dot(np.array(real2).T, M2).T
            rec3 = self.rec_real_transf(real3)
            ka3, kb3, kc3, coskalpha3, coskbeta3, coskgamma3 = self.get_lattice_constant(rec3)
            
            if (coskalpha3 > 0. and coskbeta3 > 0 and coskgamma3 > 0) or \
               (coskalpha3 < 0 and coskbeta3 < 0 and coskgamma3 < 0):
                matrix_M3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            elif (coskalpha3 > 0 and coskbeta3 < 0 and coskgamma3 < 0) or \
                 (coskalpha3 < 0 and coskbeta3 > 0 and coskgamma3 > 0):
                matrix_M3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            elif (coskalpha3 < 0 and coskbeta3 > 0 and coskgamma3 < 0) or \
                 (coskalpha3 > 0 and coskbeta3 < 0 and coskgamma3 > 0):  
                matrix_M3 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

            elif (coskalpha3 < 0 and coskbeta3 < 0 and coskgamma3 > 0) or \
                 (coskalpha3 > 0 and coskbeta3 > 0 and coskgamma3 < 0):
                matrix_M3 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

            else:
                print('Error! Can not get M3 matrix for aP lattice')
                
            real4 = np.dot(real3.T, matrix_M3).T
            rec4 = self.rec_real_transf(real4)
            ka, kb, kc, coskalpha, coskbeta, coskgamma = self.get_lattice_constant(rec4)
            self.triclinic(coskalpha, coskbeta, coskgamma)
            
        if not inversion_symmetry and not time_reversal:
            augmented_path = True
        else:
            augmented_path = False

        if augmented_path:
            for pointname, coords in list(self.kpath['Kpoints'].items()):
                if pointname == 'Gamma':
                    continue
                self.kpath['Kpoints']["{}'".format(pointname)] = \
                                     [-coords[0], -coords[1], -coords[2]]
                
            for i in range(0, len(self.kpath['Path'])):
                path_list = []
                old_path = copy.deepcopy(self.kpath['Path'][i - 1])
                for path in old_path:
                    if path == 'Gamma':
                        new_path = path
                    else:
                        new_path = "{}'".format(path)
                    path_list.append(new_path)
                self.kpath['Path'].append(path_list)

        return self.kpath

class HighSymmetryKpath2d(object):

    def __init__(self):
        
        self.kpath = None
        self.pointgroup = None
        self.layergroup = None
        self.transf_matrix_invQ = None
        self.transf_matrix_invP = None

    def rec_real_transf(self, structure_basis):

        after_transf = 2 * math.pi * np.linalg.inv(structure_basis)
        
        return (after_transf.T).tolist()
    
    def get_lattice_constant(self, structure_basis):

        va, vb, vc = np.array(structure_basis)
        a = np.linalg.norm(va)
        b = np.linalg.norm(vb)
        c = np.linalg.norm(vc)
        cosalpha = np.dot(vb, vc) / b / c
        cosbeta = np.dot(va, vc) / a / c
        cosgamma = np.dot(va, vb) / a / b

        return (a, b, c, cosalpha, cosbeta, cosgamma)

    def get_pgnum(self, pg_international):
   
        pg_dict = {u'1': 1, u'-1': 2, u'2': 3, u'm': 4, \
                   u'2/m': 5, u'222': 6, u'mm2': 7, u'm2m': 8, \
                   u'mmm': 9, u'4': 10, u'-4': 11, u'4/m': 12, \
                   u'422': 13, u'4mm': 14, u'-42m': 15, u'-4m2': 15, u'4/mmm': 16, \
                   u'3': 17, u'-3': 18, u'32': 19, u'3m': 20, \
                   u'-3m': 21, u'6': 22, u'-6': 23, u'6/m': 24, \
                   u'622': 25, u'6mm': 26, u'-6m2': 27, u'6/mmm': 28}

        return int(pg_dict[pg_international])

    def pg_inversion(self, pg_number):
   
        if pg_number in [2, 5, 9, 12, 16, 18, 21, 24, 28]:
            return True
        elif 1 <= pg_number < 29:
            return False
        else:
            print("pg_number should be between 1 and 38")
        
    def tP(self):

        kpoints = {'Gamma': [0, 0, 0], 
                   'M': [1./2, 1./2, 0], 
                   'X': [0, 1./2, 0]}
        path = [['Gamma', 'M', 'X', 'Gamma']]

        return {'Kpoints': kpoints, 'Path': path}

    def tetragonal(self, sgnum):

        self.kpath = self.tP()

    def oP(self):

        kpoints = {'Gamma': [0, 0, 0], 
                   'X': [1./2, 0, 0], 
                   'Y': [0, 1./2, 0], 
                   'S': [1./2, 1./2, 0], }
        path = [['Gamma', 'X', 'S', 'Y', 'Gamma']]

        return {'Kpoints': kpoints, 'Path': path}

    def oC(self, sgnum, a, b):

        if a < b:
            zeta = 1./4 * (1 + a**2/b**2)
        elif a > b:
            zeta = 1./4 * (1 + b**2/a**2)

        kpoints = {'Gamma': [0, 0, 0],
                   'Y': [-1./2, 1./2, 0], 
                   'S': [0, 1./2, 0], 
                   'Sigma': [zeta, zeta, 0],
                   'Delta': [-1. * zeta, zeta, 0],}
        path = [['Gamma', 'Y', 'Delta'], ['Sigma', 'Gamma', 'S', 'Y']]

        return {'Kpoints': kpoints, 'Path': path}
            
    def orthorhombic(self, sgnum, a, b):

        if sgnum in [22, 26, 35, 36, 47, 48]:
            self.kpath = self.oC(sgnum, a, b)
        elif 19 <= sgnum < 49:
            self.kpath = self.oP()
        else:
            print("Error! Unexpected value for space group number: %d" % sgnum)

    def hP(self, sgnum):
        
        kpoints = {'Gamma': [0, 0, 0],
                   'M': [1./2, 0, 0], 
                   'K': [1./3, 1./3, 0],}
 
        path = [['Gamma', 'M', 'K', 'Gamma']]

        return {'Kpoints': kpoints, 'Path': path}

    def hexagonal(self, sgnum):

        self.kpath = self.hP(sgnum)
     
    def trigonal(self, sgnum, a, b):

        self.kpath = self.hP(sgnum)
       
    def mP(self):
        
        kpoints = {'Gamma': [0, 0, 0], 
                   'X': [0, 1./2, 0], 
                   'Y': [1./2, 0, 0], 
                   'S': [1./2, 1./2, 0], }

        path = [['Gamma', 'X', 'S', 'Y', 'Gamma']]

        return {'Kpoints': kpoints, 'Path': path}

    def mC(self, a, b):

        if a > b:
            psi = 3./4 - b**2 / (4. * a**2)
        elif b > a:
            psi = 1./4 * (1. + a**2 / b**2)

        kpoints = {'Gamma': [0, 0, 0], 
                   'Y': [-1./2, 1./2, 0],
                   'S': [1./2, 0, 0],
                   'C': [-1. + psi, psi, 0], 
                   'C2': [psi, -1. + psi, 0],}

        path = [['Gamma', 'C'], ['C2', 'Y', 'S', 'Gamma']]

        return {'Kpoints': kpoints, 'Path': path}

    def monoclinic(self, sgnum, a, b):

        if sgnum in [10, 13]:
            self.kpath = self.mC(a, b)
        if 3 <= sgnum < 19:
            self.kpath = self.mP()
        else:
            print("Error! Unexpected value for space group number: %d" % sgnum)

    def aP(self):

        kpoints = {'Gamma': [0, 0, 0], 
                   'Y': [0, 1./2, 0], 
                   'B': [1./2, 0, 0], 
                   'A': [1./2, -1./2, 0],}

        path = [['Gamma', 'Y'], ['B', 'Gamma', 'A']]

        return {'Kpoints': kpoints, 'Path': path}

    def triclinic(self):
        
        self.kpath = self.aP()

    def get_HSKP(self, structure, symprec=1e-3):
        from spglib import spglib
        from .version import get_spglib_version

        structure_dataset = spglib.get_symmetry_layerdataset(structure, symprec=symprec)
        if get_spglib_version():
            # For spglib version >= 2.0.0
            self.pointgroup = structure_dataset.pointgroup
            self.layergroup = structure_dataset.international
            structure_basis = structure_dataset.std_lattice
            self.lgnum = lgnum = structure_dataset.number
        else:
            self.pointgroup = structure_dataset['pointgroup']
            self.layergroup = structure_dataset['international']
            structure_basis = structure_dataset['std_lattice']
            self.lgnum = lgnum = structure_dataset['number']
        pgnum = self.get_pgnum(self.pointgroup)
        inversion_symmetry = self.pg_inversion(pgnum)
        a, b, c, cosalpha, cosbeta, cosgamma = self.get_lattice_constant(structure_basis)
        time_reversal = True
        
        if lgnum in range (49, 65):
            self.tetragonal(lgnum)
        elif lgnum in range (19, 49):
            self.orthorhombic(lgnum, a, b)
        elif lgnum in range (73, 81):
            self.hexagonal(lgnum)
        elif lgnum in range (65, 73):
            self.trigonal(lgnum, a, b)
        elif lgnum in range (3, 19):
            self.monoclinic(lgnum, a, b)
        elif lgnum in range (1, 3):
            #rec1 = self.rec_real_transf(structure_basis)
            #rec2 = spglib.niggli_reduce(rec1)
            #real2 = self.rec_real_transf(rec2)
            #ka, kb, kc, coskalpha, coskbeta, coskgamma = self.get_lattice_constant(rec4)
            #self.triclinic(coskalpha, coskbeta, coskgamma)
            self.triclinic()
            
        if not inversion_symmetry and not time_reversal:
            augmented_path = True
        else:
            augmented_path = False

        if augmented_path:
            for pointname, coords in list(self.kpath['Kpoints'].items()):
                if pointname == 'Gamma':
                    continue
                self.kpath['Kpoints']["{}'".format(pointname)] = \
                                     [-coords[0], -coords[1], -coords[2]]
                
            for i in range(0, len(self.kpath['Path'])):
                path_list = []
                old_path = copy.deepcopy(self.kpath['Path'][i - 1])
                for path in old_path:
                    if path == 'Gamma':
                        new_path = path
                    else:
                        new_path = "{}'".format(path)
                    path_list.append(new_path)
                self.kpath['Path'].append(path_list)

        return self.kpath
