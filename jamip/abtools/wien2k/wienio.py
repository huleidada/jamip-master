
__contributor__ = 'Kun Zhou'
# from ase.io.wien2k import write_structure
#================================================================
# baisc method for control the input and output 
#================================================================

import os
import numpy as np

#         symobl   z    mass        wfc     rho 
EspressoElements = {\
          'H'  :[  1,   1.007825,    60,   480],
          'He' :[  2,   4.002602,    50,   200],
          'Li' :[  3,   6.938,       40,   320],
          'Be' :[  4,   9.012182,    40,   320],
          'B'  :[  5,  10.806,       35,   280],
          'C'  :[  6,  12.0096,      45,   360],
          'N'  :[  7,  14.00643,     60,   480],
          'O'  :[  8,  15.99903,     50,   400],
          'F'  :[  9,  18.9984032,   45,   360],
          'Ne' :[ 10,  20.1797,      50,   200],
          'Na' :[ 11,  22.98976928,  40,   320],
          'Mg' :[ 12,  24.304,       30,   240],
          'Al' :[ 13,  26.9815386,   30,   240],
          'Si' :[ 14,  28.084,       30,   240],
          'P'  :[ 15,  30.973762,    30,   240],
          'S'  :[ 16,  32.059,       35,   280],
          'Cl' :[ 17,  35.446,       40,   320],
          'Ar' :[ 18,  39.948,       60,   240],
          'K'  :[ 19,  39.0983,      60,   480],
          'Ca' :[ 20,  40.078,       30,   240],
          'Sc' :[ 21,  44.955912,    40,   160],
          'Ti' :[ 22,  47.867,       35,   280],
          'V'  :[ 23,  50.9415,      35,   280],
          'Cr' :[ 24,  51.9961,      40,   320],
          'Mn' :[ 25,  54.938045,    65,   780],
          'Fe' :[ 26,  55.845,       90,  1080],
          'Co' :[ 27,  58.933195,    45,   360],
          'Ni' :[ 28,  58.6934,      45,   350],
          'Cu' :[ 29,  63.546,       55,   440],
          'Zn' :[ 30,  65.38,        40,   320],
          'Ga' :[ 31,  69.723,       70,   560],
          'Ge' :[ 32,  72.63,        40,   320],
          'As' :[ 33,  74.9216,      35,   280],
          'Se' :[ 34,  78.96,        30,   240],
          'Br' :[ 35,  79.901,       30,   240],
          'Kr' :[ 36,  83.798,       45,   180],
          'Rb' :[ 37,  85.4678,      30,   120],
          'Sr' :[ 38,  87.62,        30,   240],
          'Y'  :[ 39,  88.90585,     35,   280],
          'Zr' :[ 40,  91.224,       30,   240],
          'Nb' :[ 41,  92.90638,     40,   320],
          'Mo' :[ 42,  95.96,        35,   140],
          'Tc' :[ 43,  97.90721,     30,   120],
          'Ru' :[ 44, 101.07,        35,   140],
          'Rh' :[ 45, 102.9055,      35,   140],
          'Pd' :[ 46, 106.42,        45,   180],
          'Ag' :[ 47, 107.8682,      50,   200],
          'Cd' :[ 48, 112.41,        60,   480],
          'In' :[ 49, 114.818,       50,   400],
          'Sn' :[ 50, 118.71,        60,   480],
          'Sb' :[ 51, 121.76,        40,   320],
          'Te' :[ 52, 127.6,         30,   240],
          'I'  :[ 53, 126.90447,     35,   280],
          'Xe' :[ 54, 131.293,       60,   240],
          'Cs' :[ 55, 132.9054519,   30,   240],
          'Ba' :[ 56, 137.327,       30,   240],
          'La' :[ 57, 138.90547,     40,   320],
          'Ce' :[ 58, 140.116,       40,   320],
          'Pr' :[ 59, 140.90765,     40,   320],
          'Nd' :[ 60, 144.242,       40,   320],
          'Pm' :[ 61, 144.91276,     40,   320],
          'Sm' :[ 62, 150.36,        40,   320],
          'Eu' :[ 63, 151.964,       40,   320],
          'Gd' :[ 64, 157.25,        40,   320],
          'Tb' :[ 65, 158.92535,     40,   320],
          'Dy' :[ 66, 162.5,         40,   320],
          'Ho' :[ 67, 164.93032,     40,   320],
          'Er' :[ 68, 167.259,       40,   320],
          'Tm' :[ 69, 168.93421,     40,   320],
          'Yb' :[ 70, 173.054,       40,   320],
          'Lu' :[ 71, 174.9668,      45,   360],
          'Hf' :[ 72, 178.49,        50,   200],
          'Ta' :[ 73, 180.94788,     45,   360],
          'W'  :[ 74, 183.84,        30,   240],
          'Re' :[ 75, 186.207,       30,   240],
          'Os' :[ 76, 190.23,        40,   320],
          'Ir' :[ 77, 192.217,       55,   440],
          'Pt' :[ 78, 195.084,       35,   280],
          'Au' :[ 79, 196.966569,    45,   180],
          'Hg' :[ 80, 200.592,       50,   200],
          'Tl' :[ 81, 204.382,       50,   400],
          'Pb' :[ 82, 207.2,         40,   320],
          'Bi' :[ 83, 208.9804,      45,   360],
          'Po' :[ 84, 209.0,         75,   600],
          'At' :[ 85, 210.0,         50,   600],
          'Rn' :[ 86, 222.0,        120,   960],
}


class WienIO(object):

    @classmethod
    def write_struct(cls, structure, path:str, lattice='P', rmt=None, direct=True, **kwargs):
        from collections import defaultdict

        if rmt is None:
            rmt = [2.0] * len(structure)

        with open(path, 'w') as f:
            f.write('ASE generated\n')
            nat = len(structure)
            if rmt is None:
                rmt = [2.0] * nat
            f.write(f'{lattice}   LATTICE,NONEQUIV.ATOMS:%3i\n' % nat)
            f.write('MODE OF CALC=RELA unit=Bohr\n')
            lattice_parameters = structure.lattice_parameters
            lattice_parameters[:3] /= 0.5291772083  # convert Angstrom to Bohr
            f.write(('%10.6f' * 6 + '\n') % tuple(lattice_parameters))

            atom_index = defaultdict(int)
            for i, atom in enumerate(structure.atomic_positions):
                f.write('ATOM %3i: ' % (i + 1))
                # unit Ang to Bohr
                f.write('X=%10.8f Y=%10.8f Z=%10.8f\n' % tuple(atom.scale_coord))
                f.write('          MULT= 1          ISPLIT= 1\n')
                if atom.atomic_number > 71:
                    ro = 0.000005
                elif atom.atomic_number > 36:
                    ro = 0.00001
                elif atom.atomic_number > 18:
                    ro = 0.00005
                else:
                    ro = 0.0001
                atom_index[atom.specie] += 1
                f.write('%-2s%-8d NPT=%5i  R0=%9.8f RMT=%10.4f   Z:%10.5f\n' %
                        (atom.specie, atom_index[atom.specie], 781, ro, rmt[i], atom.atomic_number))
                # f.write(f'LOCAL ROT MATRIX:    {1.0:9.7f} {0.0:9.7f} {0.0:9.7f}\n')
                f.write(f'                     {1.0:9.7f} {0.0:9.7f} {0.0:9.7f}\n')
                f.write(f'                     {0.0:9.7f} {1.0:9.7f} {0.0:9.7f}\n')
                f.write(f'                     {0.0:9.7f} {0.0:9.7f} {1.0:9.7f}\n')
            f.write('   0\n')

    @classmethod
    def read_klist(cls, path:str):
        '''
        Read klist file and return kpoints in reciprocal space.

        example 1:
        #    index        kx        ky        kz      div weight  emin emax    total-kpoints mesh
                 1         1         1         1        14  2.0 -7.0  1.5       343 k, div: (  7  7  7)
        example 2:
        # label      kx   ky   kz  div weight emin emax   comment
        W            40   20    0   40  2.0-0.5 1.5       Template for fcc structure
        '''
        import re
        import pandas as pd
        
        with open(path, 'r') as f:
            lines = f.readlines()

        labels = {}
        points = []
        weights = []
        for i, line in enumerate(lines):
            if line.strip() == '':
                continue
            elif line.strip() == 'END':
                break
            if re.match(r'[A-z]+',line.split()[0]):
                labels[i] = line.split()[0]
                
            row = line[10:].split()
            points.append(row[:4])
            weights.append(re.findall(r'\d+\.\d+',row[4])[0])

        df = pd.DataFrame(np.array(points, dtype=int),columns=['x','y','z','div'])
        df['weight'] = np.array(weights, dtype=float)

        return df, labels

    @classmethod
    def write_klist(cls, kpoints, path:str, **kwargs):
        from fractions import Fraction

        with open(path, 'w') as f:
            # kpoints = kpoints.value
            if kpoints.model == 'Line Model':
                kpoints = kpoints.value
                for i,number in enumerate(kpoints.numbers):
                    if number == 1: continue
                    k1 = kpoints.sites[i].position
                    k2 = kpoints.sites[i+1].position
                    vector = k2 - k1
                    fractions = [Fraction(x).limit_denominator().denominator for x in k1] + [Fraction(x).limit_denominator().denominator for x in k2]
                    denominator = np.lcm.reduce(fractions) * number
                    for j in range(number):
                        site = k1 + j * vector / number
                        if j == 0 and (i == 0 or kpoints.numbers[i-1] == 1): # first point
                            f.write('%-10s %4d %4d %4d %4d  1.0\n' %
                                    (kpoints.sites[i].symbol, site[0]*denominator, site[1]*denominator, site[2]*denominator, denominator))

                        else:
                            f.write('           %4d %4d %4d %4d  1.0\n' %
                                    (site[0]*denominator, site[1]*denominator, site[2]*denominator, denominator))
                    # last point
                    site = kpoints.sites[i+1].position
                    f.write('%-10s %4d %4d %4d %4d  1.0\n' %
                            (kpoints.sites[i+1].symbol, site[0]*denominator, site[1]*denominator, site[2]*denominator, denominator))
                f.write('END\n')

            elif kpoints.model == 'Reciprocal':
                for i, kpoint in enumerate(kpoints.value):
                    fractions = [Fraction(x).limit_denominator().denominator for x in kpoint[:3]]
                    denominator = np.lcm.reduce(fractions)
                    print(kpoint[:3]*denominator, 'denominator:', denominator)
                    # print(fractions)
                    f.write('           %4d %4d %4d %4d %4.1f\n' % (
                        kpoint[0]*denominator, kpoint[1]*denominator, kpoint[2]*denominator, denominator, kpoint[3]))
                f.write('END\n')

    @classmethod
    def read_spaghetti(cls, path:str):
        '''
        Read spaghetti file, return kpoints and eigenvalues.
        > case.spaghetti
            bandindex 1
            kx ky kz x-axis eigenvalues
        Return:
            shape to (n_bands, n_kpoints, 5)
        '''
        with open(path, 'r') as f:
            indices = []
            data = []
            for line in f:
                row = line.split()
                if len(row) == 0:
                    continue
                elif row[0] == 'bandindex:':
                    indices.append(int(row[1]))
                elif len(row) == 5:
                    data.append(row)
        data = np.array(data, dtype=float).reshape(len(indices), -1, 5)  # reshape to (n_bands, n_kpoints, 5)
        return data
    
    @classmethod
    def read_absorp(cls, path:str):
        '''
        Read absorption file, return kpoints and absorption coefficients.
        > case.absorp
            kx ky kz x-axis absorption
        Return:
            shape to (n_kpoints, 5)
        '''
        with open(path, 'r') as f:
            data = []
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                row = line.split()
                if len(row) == 7:
                    data.append(row)
        data = np.array(data, dtype=float)  # reshape to (Energy, Re_sigma_xx, Re_sigma_yy, Re_sigma_zz, absorp_xx, absorp_yy, absorp_zz)
        return data