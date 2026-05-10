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
def OpenFile(Target,filename="cp2k.log"):
    global Record,MoreRecord
    Record = []
    MoreRecord = []
    while True:
        F = yield
        with open(str(F)+os.sep+str(filename),'r') as f:
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

class LogFinder(object):

    def __init__(self):
        pass

    def easysearch(self,path,keyword):   
        Gene = OpenFile(CatFile(RecordLine(keyword)))
        Gene.send(path)
        Record.reverse()
        for line in Record:
            result = re.findall(rf"{keyword}:\s*([-]?\d+)",line)
            if len(result):
                return int(result[0])

    def floatsearch(self,path,keyword,site='head'):
        Gene = OpenFile(CatFile(RecordLine(keyword)))
        Gene.send(path)
        if site == "head":
            result = re.findall(rf"{keyword}:?\s*([-+|\.|\dEe]+)", Record[0])[0]
        elif site == "tail":              
            for line in Record[::-1]:
                if ':' in line:
                    result = re.findall(rf"{keyword}:?\s*([-+|\.|\dEe]+)", line)[0]
                    break
        return float(result)

    def boolsearch(self,path,keyword):
        Gene = OpenFile(CatFile(RecordLine(keyword)))
        Gene.send(path)
        result = re.findall(r"{0}:\s*(T|F)".format(keyword),Record[-1])[0]
        return True if result == 'T' else False

    # int %
    def max_scf(self,path):
        # vasp nsw
        return self.easysearch(path,'max_scf')
  
    def natoms(self,path):
        return self.easysearch(path,'Atoms')
  
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

    def total_energy(self,path):
        return self.floatsearch(path,'Total energy')

    def qs_energy(self,path):
        return self.floatsearch(path,'Total FORCE_EVAL ( QS ) energy [a.u.]')

    def fermi_energy(self,path):
        return self.floatsearch(path,'Fermi energy')

    def emin(self,path):
        return self.floatsearch(path,'EMIN')

    def emax(self,path):
        return self.floatsearch(path,'EMAX')

    def sigma(self,path):
        return self.floatsearch(path,'SIGMA')

    def free_energy(self,path):
        return self.floatsearch(path,'Total energy','tail')

    def energy_without_entropy(self,path):
        return self.floatsearch(path,'entropy','tail')

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

    def prec(self,path):
        Gene = OpenFile(HeadFile(RecordLine('PREC')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"PREC\s*=\s*(\w+)",Record[0])[0]
        return result

    def external_pressure(self, path):
        Gene = OpenFile(CatFile(RecordLine('external pressure')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"external pressure\s*=\s*(-?\d+\.\d+)\s*kB",Record[-1])[0]
        return float(result)
    
    def pullay_stress(self,path):
        '''
        unit kB (kbar)
        1 bar = 1.02 kgf / cm^2 ; 1 kgf / m^2 = 1/1.02 * 1e4 bar = 9.804e3 kbar
        '''
        Gene = OpenFile(CatFile(RecordLine('Pullay stress')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"Pullay stress\s*=\s*([-]?\d+\.\d+)\s*kB",Record[-1])[0]
        return float(result) 
    
    def eentro(self, path):
        Gene = OpenFile(CatFile(RecordLine('EENTRO')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"EENTRO\s*=\s*(-?\d+\.\d+)",Record[-1])[0]
        return float(result)
  
    def gga(self,path):
        Gene = OpenFile(CatFile(RecordLine('GGA')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"GGA\s*=\s*(\S+)",Record[-1])[0]
        if result== '--': return 'PE'
        return result

    def point_group(self,path):
        Gene = OpenFile(CatFile(RecordLine('The point group')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"([A-Z]_[a-z0-9]+)",Record[-1])[0]
        return result

    def volume(self,path):
        Gene = OpenFile(CatFile(RecordLine('volume of cell')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"volume of cell\s*:\s*(\S+)",Record[-1])[0]
        return float(result)

    def ionic_dipole_moment(self,path):
        Gene = OpenFile(CatFile(RecordLine('Ionic dipole moment')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"p\[ion\]=\(\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*\)",Record[-1])[0]
        return np.array(result, dtype=float)
  
    def electronic_dipole_moment(self,path):
        Gene = OpenFile(CatFile(RecordLine('Total electronic dipole moment')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"p\[elc\]=\(\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*\)",Record[-1])[0]
        return np.array(result, dtype=float)

    def force(self,path):
        nions = self.nions(path)
        Gene = OpenFile(MoreFile(MoreLine('TOTAL-FORCE',nions+1)))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for line in Record[1:]:
            datas.append(line.split()[3:])
        return np.array(datas, dtype=float)

    def cell_force(self,path):
        Gene = OpenFile(MoreFile(MoreLine('FORCE on cell',16)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        if len(Record):
            forces = []
            for line in Record:
                row = line.split()
                if row[0] == 'Total':
                    forces.append(row[1:])
            if len(forces):
                return np.array(forces[-1],dtype=float)
        return None
  
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

    def atominfo(self,path):
        with open(os.path.join(path,"POSCAR"),'r') as f:
            for i in range(5):
                f.readline()
            atom = f.readline().split()
            atom_num = f.readline().split()
        atom_num = map(int,atom_num) 
        return dict(zip(atom,atom_num)).items()

    def elements(self,path):
        with open(os.path.join(path,"POSCAR"),'r') as f:
            for i in range(5):
                f.readline()
            species = f.readline().split()
            numbers = f.readline().split()
        return np.repeat(species, numbers)

    def oszicar(self,path):
        import re
        energy = []
        with open(os.path.join(path,'OSZICAR')) as f:
            for line in f:
                if re.match(r'\s*\d+\s*F',line):
                    energy.append(re.findall(r"=\s*([-?+?|\.|\w]+)\s",line))
        energy = np.array(energy,dtype=float)
        return energy

