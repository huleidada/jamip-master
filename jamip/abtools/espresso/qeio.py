
__contributor__ = 'Kun Zhou'
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


class QEIO(object):

    @classmethod
    def write_structure(cls, structure, potentials:list, path:str, direct=True, **kwargs):

        with open(path, 'a+') as f:
            f.write("ATOMIC_SPECIES\n")
            for elm, pot in zip(structure.species_of_elements, potentials):
                mass = EspressoElements[elm][1]
                f.write("{0:6} {1:>8}  {2}\n".format(elm, mass, pot))
 
            f.write('CELL_PARAMETERS angstrom\n')
            if abs(structure.scale_factor-1.0) > 1e-8:
                lattice = structure.lattice * structure.scale_factor
            else:
                lattice = structure.lattice 
            for l in lattice:
                f.write(' '.join('{0:>16.8f}'.format(c) for c in l))
                f.write('\n')
 
            if direct is True:
                f.write('ATOMIC_POSITIONS crystal\n')
                for i, p in enumerate(structure.atomic_positions):
                    f.write('{0:>4}'.format(p.specie))
                    f.write(' '.join('{0:>13.8f}'.format(j) for j in p.scale_coord))
                    #if structure.select_dynamic:
                    #    fopen.write(' %5s %5s %5s' %p.freeze.xyz)
                    f.write('\n')
            elif direct is False:
                f.write('ATOMIC_POSITIONS angstrom\n')
                for i, p in enumerate(structure.atomic_positions):
                    f.write('{0:>4}'.format(p.specie))
                    f.write(' '.join('{0:>13.8f}'.format(j) for j in p.coord))
                    #if structure.select_dynamic:
                    #    fopen.write(' %5s %5s %5s' %p.freeze.xyz)
                    f.write('\n')

    @classmethod
    def write_kpoints(cls, incar, path:str, **kwargs):

        with open(path, 'a+') as f:
            kpoints = incar.kpoints.value
            if incar.kpoints.model == 'Line Model':
                f.write("K_POINTS crystal_b\n")
                f.write("%d\n" %len(kpoints.sites))
                f.write(kpoints.qeformat)
                
            elif incar.kpoints.model == 'Gamma':
                assert kpoints.size == 6
                f.write("K_POINTS automatic\n")
                f.write("{0[0]} {0[1]} {0[2]}".format(kpoints[0].astype(int)))
                f.write(" {0[0]} {0[1]} {0[2]}\n".format(kpoints[1]))
         
            elif incar.kpoints.model == 'Reciprocal':
                f.write("K_POINTS crystal\n")
                f.write(kpoints)

    @classmethod
    def write_input(cls, incar, path:str, **kwargs):

        from .utils import phonon_format

        with open(path,'w') as f:
            if incar.program == 'ph.x':
                f.write("phonon calculation.\n&inputph\n")
                incar = phonon_format(incar)
            elif incar.program in ('q2r.x','matdyn.x'):
                f.write("&input\n")
            else:
                f.write("&%s\n" %incar.name)

            for key in ['prefix','outdir']:
                f.write("  %-15s = '%-s'\n" %(key.lower(),incar.pop(key)))

            for key,value in incar.items():
                if 'fil' in key:
                    value = "'%s'" %value
                f.write("  %-15s = %-s\n" %(key.lower(),value))
            f.write("/\n")

            #if 'kpoints' in kwargs and kwargs['kpoints']: 
            #    f.write("%d\n" %len(incar.kpoints.kpath))
            #    f.write(incar.kpoints.qeformat)

    @classmethod
    def write_pwscf(cls, incar, path:str, **kwargs):
        
        from .utils import pwscf_format

        control, system, electrons, ions, cell = pwscf_format(incar)

        with open(path, 'w') as fopen:
            fopen.write("&CONTROL\n")
            for key,value in control.items():
                fopen.write("  %-15s = %-s\n" %(key.lower(),value))
            fopen.write("/\n")
         
            fopen.write("&SYSTEM\n")
            for key,value in system.items():
                fopen.write("  %-15s = %-s\n" %(key.lower(),value))
            fopen.write("/\n")
         
            fopen.write("&ELECTRONS\n")
            for key,value in electrons.items():
                fopen.write("  %-15s = %-s\n" %(key.lower(),value))
            fopen.write("/\n")
         
            fopen.write("&IONS\n")
            for key,value in ions.items():
                fopen.write("  %-15s = %-s\n" %(key.lower(),value))
            fopen.write("/\n")
         
            fopen.write("&CELL\n")
            for key,value in cell.items():
                fopen.write("  %-15s = %-s\n" %(key.lower(),value))
            fopen.write("/\n")
         
