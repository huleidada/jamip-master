__contributor__ = 'Kun Zhou'
#================================================================
# baisc method for control the input and output 
#================================================================

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


class CP2KIO(object):

    @classmethod
    def write_input(cls, incar, path:str, **kwargs):

        #from .utils import phonon_format
        from .utils import cp2k_format
 
        format_incar = cp2k_format(incar)

        with open(path,'w') as f:
            if incar.program.name == 'cp2k.popt':
                for key,value in format_incar.items():
                    format_dict(key, value, f, level=0)
            else:
                raise OSError(f"Unknown cp2k program: {incar.program}")

    @classmethod
    def read_input(cls, path):

        def load_dict(f, line):
            data = {}
            keys = line.split()
            key = keys[0][1:]
            if len(keys) == 2:
                value[key] = keys[1]

            for i,line in enumerate(f):
                if len(line.lstrip())==0 or line.lstrip()[0] == '#':
                    continue
                elif '&END' in line:
                    break
                elif '&' in line: 
                    k, v = load_dict(f, line)
                    data[k] = v
                else:
                    vs = line.split()
                    if len(vs) == 2:
                        data[vs[0]] = vs[1].strip('"')
                    else:
                        data['_%d' %i] = vs

            return key, data

        alldata = {}
        with open(path, 'r') as f:
            for line in f:
                if len(line.lstrip())==0 or line.lstrip()[0] == '#':
                    continue
                elif '&' in line:
                    key, value = load_dict(f, line)
                    alldata[key] = value
                elif len(line.strip()) == 0:
                    pass
                else:
                    raise ValueError(line)
                    
        return alldata 

    @classmethod
    def load_structure(cls, stdin):
        from jamip.structure import Structure
        from pathlib import Path

        stdin = Path(stdin)
        if stdin.is_file():
            data = cls.read_input(stdin)

        elif stdin.is_dir():
            data = cls.read_input(stdin/'cp2k.inp')
            project = data['GLOBAL']['PROJECT']
            path = stdin/f"{project}-1.restart"
            if path.exists(): 
                data = cls.read_input(path)
                stdin = path

        else:
            raise OSError('Input Path not exists!')

        print(f'Load structure from {stdin}')
        cell = [None, None, None] 
        for key,value in data['FORCE_EVAL']['SUBSYS']['CELL'].items():
            if key[0] == '_':
                if value[0] == 'A':
                    cell[0] = value[1:]
                elif value[0] == 'B':
                    cell[1] = value[1:]
                elif value[0] == 'C':
                    cell[2] = value[1:]
        cell = np.array(cell, dtype=float)
 
        direct = False
        elements = []
        positions = []
        for key,value in data['FORCE_EVAL']['SUBSYS']['COORD'].items():
            if key[0] == '_':
                elements.append(value[0])
                positions.append(value[1:])
            elif key.upper() == 'SCALED':
                if 'T' in value.upper():
                    direct = True
 
        positions = np.array(positions, dtype=float)
        return Structure.from_cell((cell, positions, elements), direct=direct)

def format_line(key, value, f, level):

    if key[0] == '_':
        text  = f'{value}\n'
    else:
        text  = f'{key.upper()} {value}\n'
    formatted_text = text.rjust(len(text) + level * 3) 
    f.write(formatted_text)

def format_dict(key, value, f, level):

    if key in value:
        text  = f'&{key.upper()} {value[key]}\n'
    else:
        text  = f'&{key.upper()}\n'
    formatted_text = text.rjust(len(text) + level * 3) 
    f.write(formatted_text)

    #print(key, value)
    for k,v in value.items():
        if k == key:
            pass
        elif isinstance(v, dict):
            format_dict(k, v, f, level+1)
        else:
            format_line(k, v, f, level+1)

    text  = f'&END {key.split()[0].upper()}\n'
    formatted_text = text.rjust(len(text) + level * 3) 
    f.write(formatted_text)
