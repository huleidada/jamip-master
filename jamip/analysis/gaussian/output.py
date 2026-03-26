import numpy as np
import os
import re
import pathlib

Record = [] 
MoreRecord = []
def AutoNext(Target):
    def NextTarget(*args):
        res = Target(*args)
        try: 
            next(res)
        except StopIteration:
            pass
        return res
    return NextTarget

@AutoNext
def OpenFile(Target):
    global Record,MoreRecord
    Record = []
    MoreRecord = []
    while True:
        F = yield
        with open(F,'r') as f:
            Target.send((f))

@AutoNext
def CatFile(Target):
    while True:
        f = yield
        for i in f :
            Target.send((i))

@AutoNext
def HeadFile(Target):
    while True:
        f = yield
        for i in f :            
            Target.send((i))
            if len(Record):
                break

@AutoNext
def MoreFile(Target):
    while True:
        f = yield
        for i in f :
            Target.send((i,f))

@AutoNext
def MoreLine(keyword,maxline,keyline=None):
    if keyline == None:
        keyline = range(maxline)
    elif isinstance(keyline,int):
        keyline = [keyline]
    elif not isinstance(keyline,list):
        raise TypeError
    while True:
        line,f = yield
        if (keyword in line):
            MoreRecord.append(line)
            for i in range(maxline):
                if i in keyline: 
                    Record.append(f.readline())
                else:
                    f.readline()

@AutoNext
def StartLine(keyword):
    while True:
        line = yield
        if line.startswith(keyword):
            Record.append(line)

@AutoNext
def RecordLine(keyword):
    while True:
        line = yield
        if (keyword in line):
            Record.append(line)

def run_with_ecc(func, path):
    try:
        func.send(path)
    except StopIteration:
        pass

class LogFinder(object):

    def __init__(self):
        pass

    def easysearch(self,path,keyword):   
        Gene = OpenFile(CatFile(RecordLine(keyword)))
        Gene.send(path)
        Record.reverse()
        for line in Record:
            result = re.findall(rf"{keyword}=\s*([-]?\d+)",line)
            if len(result):
                return int(result[0])

    def floatsearch(self,path,keyword,site='head'):
        Gene = OpenFile(CatFile(RecordLine(keyword)))
        Gene.send(path)
        if site == "head":
            result = re.findall(rf"{keyword}\s*=?\s*([-+|\.|\dEe]+)", Record[0])[0]
        elif site == "tail":              
            for line in Record[::-1]:
                if '=' in line:
                    result = re.findall(rf"{re.escape(keyword)}\s*=?\s*([-+|\.|\dEe]+)", line)[0]
                    break
        return float(result)

    def boolsearch(self,path,keyword):
        Gene = OpenFile(CatFile(RecordLine(keyword)))
        Gene.send(path)
        result = re.findall(rf"{keyword}:\s*(T|F)",Record[-1])[0]
        return True if result == 'T' else False

    # int %
    def natoms(self,path):
        return self.easysearch(path,'NAtoms')
  
    def nbands(self,path):
        return self.easysearch(path,'NBANDS')
  
    def nkpts(self,path):
        return self.easysearch(path,'NKPTS')

    def pstress(self,path):
        return self.easysearch(path,'PSTRESS')

    def num_electrons(self,path):
        return self.easysearch(path,'Number of electrons')
 
    def num_occupied_orbitals(self,path):
        return self.easysearch(path,'Number of occupied orbitals')

    def num_molecular_orbitals(self,path):
        return self.easysearch(path,'Number of molecular orbitals')

    def num_orbital_funcitons(self,path):
        return self.easysearch(path,'Number of orbital functions')

    # bool %
    def lwave(self,path):
        return self.boolsearch(path,'LWAVE')
  
    def lcharg(self,path):
        return self.boolsearch(path,'LCHARG')
  
    def lelf(self,path):
        return self.boolsearch(path,'LELF')
  
    # float % 
    def eps_scf(self,path):
        return self.floatsearch(path,'eps_scf')
  
    def step_size(self,path):
        return self.floatsearch(path,'step_size')

    def fermi_energy(self,path):
        return self.floatsearch(path,'Fermi energy')

    def emin(self,path):
        return self.floatsearch(path,'EMIN')

    def emax(self,path):
        return self.floatsearch(path,'EMAX')

    def E_RB3LYP(self,path):
        return self.floatsearch(path,'E(RB3LYP)','tail')

    def E_UB3LYP(self,path):
        return self.floatsearch(path,'E(UB3LYP)','tail')

    def E_RwB97XD(self,path):
        return self.floatsearch(path,'E(RwB97XD)','tail')

    # others %
    def starttime(self,path):
        Gene = OpenFile(HeadFile(RecordLine('STARTED')))
        Gene.send(path)
        result = ' '.join(Record[0].split()[:-2])
        return result

    def endtime(self,path):
        Gene = OpenFile(HeadFile(RecordLine('ENDED')))
        Gene.send(path)
        result = ' '.join(Record[0].split()[:-2])
        return result

    def energy(self,path):
        Gene = OpenFile(CatFile(RecordLine('SCF Done:  E')))
        run_with_ecc(Gene, path)
        result = float(Record[-1].split()[4])
        return result

    def polarizability(self,path):
        Gene = OpenFile(HeadFile(RecordLine('Isotropic polarizability')))
        run_with_ecc(Gene, path)
        result = float(Record[0].split()[5])
        return result

    def stoichiometry(self, path):
        Gene = OpenFile(HeadFile(RecordLine('Stoichiometry')))
        run_with_ecc(Gene, path)
        result = Record[0].split()[1]
        return result

    def degree_of_freedom(self, path):
        Gene = OpenFile(HeadFile(RecordLine('Deg. of freedom')))
        run_with_ecc(Gene, path)
        result = int(Record[0].split()[-1])
        return result
  
    def point_group(self,path):
        Gene = OpenFile(HeadFile(RecordLine('Full point group')))
        run_with_ecc(Gene, path)
        result = Record[-1].split()[3]
        return result

    def SCRF_radius(self,path):
        Gene = OpenFile(CatFile(RecordLine('Recommended a0 for SCRF calculation')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"Recommended a0 for SCRF calculation\s*=\s*(\d\.\d+)",Record[-1])[0]
        return float(result)

    def mol_volume(self, path):
        Gene = OpenFile(CatFile(RecordLine('Molar volume')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"Molar volume\s*=\s*(\d+\.\d+)",Record[-1])[0]
        return float(result)

    def dipole_moment(self, path):
        '''
         Dipole moment (field-independent basis, Debye):
            X=              3.8796    Y=             -0.6303    Z=              0.0000  Tot=              3.9305
        '''
        Gene = OpenFile(MoreFile(MoreLine(r'Dipole moment (field-independent basis, Debye)',1)))
        run_with_ecc(Gene, path)
        raw = Record[-1].split()
        # x, y, z, total
        results = raw[1::2]
        return np.array(results, dtype=float)

    def quadrupole_moment(self,path):
        '''
         Quadrupole moment (field-independent basis, Debye-Ang):
           XX=             -8.8494   YY=            -16.1262   ZZ=            -16.6195
           XY=             -1.4899   XZ=             -0.0000   YZ=             -0.0000
        '''
        Gene = OpenFile(MoreFile(MoreLine(r'Quadrupole moment (field-independent basis, Debye-Ang)',2)))
        run_with_ecc(Gene, path)
        results = []
        for line in Record[-2:]: 
            results.extend(line.split()[1::2])
        # xx, yy, zz, xy, xz, yz
        return np.array(results, dtype=float)
  
    def traceless_quadrupole_moment(self,path):
        '''
         Traceless Quadrupole moment (field-independent basis, Debye-Ang):
           XX=              5.0156   YY=             -2.2612   ZZ=             -2.7545
           XY=             -1.4899   XZ=             -0.0000   YZ=             -0.0000
        '''
        Gene = OpenFile(MoreFile(MoreLine(r'Traceless Quadrupole moment (field-independent basis, Debye-Ang)',2)))
        run_with_ecc(Gene, path)
        results = []
        for line in Record[-2:]: 
            results.extend(line.split()[1::2])
        # xx, yy, zz, xy, xz, yz
        return np.array(results, dtype=float)

    def octapole_moment(self,path):
        '''
         Octapole moment (field-independent basis, Debye-Ang**2):
          XXX=             17.1511  YYY=             -1.1764  ZZZ=              0.0000  XYY=              4.1211
          XXY=             -0.4844  XXZ=             -0.0000  XZZ=              4.3787  YZZ=             -2.0040
          YYZ=             -0.0000  XYZ=             -0.0000
        '''
        Gene = OpenFile(MoreFile(MoreLine(r'Octapole moment (field-independent basis, Debye-Ang**2)',3)))
        run_with_ecc(Gene, path)
        results = []
        for line in Record[-3:]: 
            results.extend(line.split()[1::2])
        # xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
        return np.array(results, dtype=float)

    def hexadecapole_moment(self,path):
        '''
         Hexadecapole moment (field-independent basis, Debye-Ang**3):
         XXXX=           -128.6014 YYYY=            -55.3146 ZZZZ=            -31.9150 XXXY=             -2.4092
         XXXZ=             -0.0000 YYYX=             -4.2983 YYYZ=              0.0000 ZZZX=              0.0000
         ZZZY=             -0.0000 XXYY=            -32.8370 XXZZ=            -30.8753 YYZZ=            -10.5771
         XXYZ=             -0.0000 YYXZ=             -0.0000 ZZXY=             -2.0506
        '''
        Gene = OpenFile(MoreFile(MoreLine(r'Hexadecapole moment (field-independent basis, Debye-Ang**3)',4)))
        run_with_ecc(Gene, path)
        results = []
        for line in Record[-4:]: 
            results.extend(line.split()[1::2])
        return np.array(results, dtype=float)

    def mulliken_charge(self,path):
        '''
         Mulliken charges:
                       1
             1  N   -0.688009
             2  C   -0.451781
             3  C   -0.735458
             4  H    0.459903
             5  H    0.463644
        '''
        natoms = self.natoms(path)
        Gene = OpenFile(MoreFile(MoreLine(r'Mulliken charges:',natoms+1)))
        run_with_ecc(Gene, path)
        datas = []
        for line in Record[-natoms:]:
            datas.append(line.split()[-1])
        return np.array(datas, dtype=float)

    def thermal_properties(self,path):
        """
                             E (Thermal)             CV                S
                              KCal/Mol        Cal/Mol-Kelvin    Cal/Mol-Kelvin
         Total                   70.732             14.609             65.699
         Electronic               0.000              0.000              0.000
         Translational            0.889              2.981             37.408
         Rotational               0.889              2.981             22.719
         Vibrational             68.955              8.647              5.572
        """
        Gene = OpenFile(MoreFile(MoreLine('E (Thermal)             CV                S',2)))
        run_with_ecc(Gene, path)
        # E (Thermal), CV, S 
        results = Record[-1].split()[1:]
        return np.array(results, dtype=float)
  
    def lattice_vectors(self,path,status='end'):        
        Gene = OpenFile(MoreFile(MoreLine('direct lattice vectors',4)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        if status == 'end':
            lines = Record[-4:]
        else:
            lines = Record[:4]

        datas = []
        for line in lines:
           data = re.findall(r"[-]?\d+\.\d+",line)
           if len(data) == 6:
               datas.append(data)
        return np.array(datas,dtype=float)
  
    def direct_lattice_vectors(self,path,status='end'):
        return self.lattice_vectors(path,status)[:,:3]

    def reciprocal_lattice_vectors(self,path,status='end'):
        return self.lattice_vectors(path,status)[:,3:]

    def length_of_vector(self,path,status='end'):
        Gene = OpenFile(MoreFile(MoreLine('length of vectors',1)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        if status == 'end':
            return np.array(re.findall(r"\d+\.\d+",Record[-1]),dtype=float)
        else:
            return np.array(re.findall(r"\d+\.\d+",Record[0]),dtype=float)

    def cell(self,path):
        import re
        natoms = self.natoms(path)
        with open(path) as f:
            for line in f:
                if 'Coordinates (Angstroms)' in line:
                    atomic_numbers = []
                    coordinates = []
                    f.readline()
                    f.readline()
                    for i in range(natoms):
                        result = f.readline().split()
                        atomic_numbers.append(result[1])
                        coordinates.append(result[-3:])
        return np.array(coordinates,dtype=float), np.array(atomic_numbers,dtype=int)
