import numpy as np
import os
import re
import pathlib
#from ripgrepy import Ripgrepy

Record = [] 
MoreRecord = []
def AutoNext(Target):
    def NextTarget(*args):
        res = Target(*args)
        next(res)
        return res
    return NextTarget

@AutoNext
def InputGetPath(Target):
    InputPath = yield 
    PathGen = os.walk(InputPath)
    for i in PathGen:
        for j in i[-1]:
            FilePath = f"{i[0]}{os.sep}{j}"
            Target.send(FilePath)

@AutoNext
def OpenFile(Target):
    while True:
        F = yield
        with open(F) as f:
            Target.send((f))

@AutoNext
def OpenOutcar(Target):
    global Record,MoreRecord
    Record = []
    MoreRecord = []
    while True:
        F = yield
        with open(str(F)+os.sep+"OUTCAR",'r') as f:
            Target.send((f))

@AutoNext
def OpenProcar(Target):
    global Record,MoreRecord
    Record = []
    MoreRecord = []
    while True:
        F = yield
        with open(str(F)+os.sep+"PROCAR",'r') as f:
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

class GrepOutcar(object):

    def __init__(self):
        pass

    def easysearch(self,path,keyword):   
        Gene = OpenOutcar(CatFile(RecordLine(keyword)))
        try: 
            Gene.send(path)
            Record.reverse()
        except StopIteration:
            pass
        for line in Record:
            result = re.findall(r"{0}\s*=\s*([-]?\d+)".format(keyword),line)
            if len(result):
                return int(result[0])

    def floatsearch(self,path,keyword,site='head'):
        Gene = OpenOutcar(CatFile(RecordLine(keyword)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        if site == "head":
            result = re.findall(r"{0}\s*=\s*([-+|\.|\dEe]+)".format(keyword),Record[0])[0]
        elif site == "tail":
            result = re.findall(r"{0}\s*=\s*([-+|\.|\dEe]+)".format(keyword),Record[-1])[0]
        return float(result)

    def boolsearch(self,path,keyword):
        Gene = OpenOutcar(CatFile(RecordLine(keyword)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"{0}\s*=\s*(T|F)".format(keyword),Record[-1])[0]
        if result == 'T':
            return True
        else:
            return False


    # int %
    def nkdim(self,path):
        return self.easysearch(path,'NKDIM')
  
    def nedos(self,path):
        return self.easysearch(path,'NEDOS')
  
    def nions(self,path):
        return self.easysearch(path,'NIONS')
  
    def nbands(self,path):
        return self.easysearch(path,'NBANDS')
  
    def nkpts(self,path):
        return self.easysearch(path,'NKPTS')
  
    def istart(self,path):
        return self.easysearch(path,'ISTART')
  
    def icharg(self,path):
        return self.easysearch(path,'ICHARG')
  
    def ispin(self,path):
        return self.easysearch(path,'ISPIN')

    def nelm(self,path):
        return self.easysearch(path,'NELM')

    def nelmhf(self,path):
        return self.easysearch(path,'NELMHF')

    def nsw(self,path):
        return self.easysearch(path,'NSW')

    def lmaxmix(self,path):
        return self.easysearch(path,'LMAXMIX')

    def ibrion(self,path):
        return self.easysearch(path,'IBRION')

    def nfree(self,path):
        return self.easysearch(path,'NFREE')

    def isif(self,path):
        return self.easysearch(path,'ISIF')

    def isym(self,path):
        return self.easysearch(path,'ISYM')

    def pstress(self,path):
        return self.easysearch(path,'PSTRESS')

    def nelect(self,path):
        return self.easysearch(path,'NELECT')

    def ismear(self,path):
        return self.easysearch(path,'ISMEAR')

    def ialgo(self,path):
        return self.easysearch(path,'IALGO')

    def lorbit(self,path):
        return self.easysearch(path,'LORBIT')

    def dof(self,path):
        return self.easysearch(path,'DOF')
    
    # bool %
    def lsorbit(self,path):
        return self.boolsearch(path,'LSORBIT')
  
    def lwave(self,path):
        return self.boolsearch(path,'LWAVE')
  
    def lcharg(self,path):
        return self.boolsearch(path,'LCHARG')
  
    def lvtot(self,path):
        return self.boolsearch(path,'LVTOT')
  
    def lelf(self,path):
        return self.boolsearch(path,'LELF')
  
    # float % 
    def encut(self,path):
        return self.floatsearch(path,'ENCUT')
  
    def ediff(self,path):
        return self.floatsearch(path,'EDIFF')

    def ediffg(self,path):
        return self.floatsearch(path,'EDIFFG')

    def cshift(self,path):
        return self.floatsearch(path,'CSHIFT')

    def potim(self,path):
        return self.floatsearch(path,'POTIM')

    def emin(self,path):
        return self.floatsearch(path,'EMIN')

    def emax(self,path):
        return self.floatsearch(path,'EMAX')

    def sigma(self,path):
        return self.floatsearch(path,'SIGMA')

    def free_energy(self,path):
        return self.floatsearch(path,'TOTEN','tail')

    def energy_without_entropy(self,path):
        return self.floatsearch(path,'entropy','tail')

    # others %
    def date(self,path):
        Gene = OpenOutcar(HeadFile(RecordLine('date ')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"(\d{4}.\d{1,2}.\d{1,2})",Record[0])[0]
        return result

    def datetime(self,path):
        Gene = OpenOutcar(HeadFile(RecordLine('date ')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"(\d{1,2}:\d{1,2})",Record[0])[0]
        return result

    def cputime(self,path):
        Gene = OpenOutcar(HeadFile(RecordLine('Total CPU time')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"(\d+\.\d+)",Record[0])[0]
        return float(result)
  
    def vasp_version(self,path):
        Gene = OpenOutcar(HeadFile(RecordLine('vasp')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"vasp\.(\d+[\.\d+]+)\s",Record[0])[0]
        return result

    def prec(self,path):
        Gene = OpenOutcar(HeadFile(RecordLine('PREC')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"PREC\s*=\s*(\w+)",Record[0])[0]
        return result

    def fermi_energy(self,path):
        Gene = OpenOutcar(HeadFile(RecordLine('E-fermi')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"E-fermi\s*:\s*([-]?\d+\.\d+)",Record[0])[0]
        return float(result)    
        
    def external_pressure(self, path):
        Gene = OpenOutcar(CatFile(RecordLine('external pressure')))
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
        1 eV/Å³ ≈ 160.21766208 GPa = 1602.1766208 kbar
        '''
        Gene = OpenOutcar(CatFile(RecordLine('Pullay stress')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"Pullay stress\s*=\s*([-]?\d+\.\d+)\s*kB",Record[-1])[0]
        return float(result) 

    def virial_stress(self,path): 
        '''
        unit kB (kbar) 
        FORCE on cell =-STRESS
        Direction    XX          YY          ZZ          XY          YZ          ZX
        Total       0.00888     0.00888     0.00888     0.00000     0.00000    -0.00000   # unit eV (stress * volume)
        in kB      -0.12345     0.23456    -0.34567     0.45678    -0.56789     0.67890   # unit kB (stress)
        '''
        # TODO
        raise ValueError("not finish")
    
    def eentro(self, path):
        Gene = OpenOutcar(CatFile(RecordLine('EENTRO')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"EENTRO\s*=\s*(-?\d+\.\d+)",Record[-1])[0]
        return float(result)
  
    def gga(self,path):
        Gene = OpenOutcar(CatFile(RecordLine('GGA')))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"GGA\s*=\s*(\S+)",Record[-1])[0]
        if result== '--': return 'PE'
        return result

    def point_group(self,path):
        Gene = OpenOutcar(CatFile(RecordLine('The point group')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"([A-Z]_[a-z0-9]+)",Record[-1])[0]
        return result

    def volume(self,path):
        Gene = OpenOutcar(CatFile(RecordLine('volume of cell')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"volume of cell\s*:\s*(\S+)",Record[-1])[0]
        return float(result)

    def ionic_dipole_moment(self,path):
        Gene = OpenOutcar(CatFile(RecordLine('Ionic dipole moment')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"p\[ion\]=\(\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*\)",Record[-1])[0]
        return np.array(result, dtype=float)
  
    def electronic_dipole_moment(self,path):
        Gene = OpenOutcar(CatFile(RecordLine('Total electronic dipole moment')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = re.findall(r"p\[elc\]=\(\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*\)",Record[-1])[0]
        return np.array(result, dtype=float)

    def elastic(self, path):
        # unit (kBar) task: elastic
        Gene = OpenOutcar(MoreFile(MoreLine('TOTAL ELASTIC MODULI',9)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for line in Record:
           data = re.findall(r"[-]?\d+\.\d+",line)
           if len(data) == 6:
               datas.append(data)
        return np.array(datas,dtype=float)

    def force(self,path):
        nions = self.nions(path)
        Gene = OpenOutcar(MoreFile(MoreLine('TOTAL-FORCE',nions+1)))
        try:
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for line in Record[1:]:
            datas.append(line.split()[3:])
        return np.array(datas, dtype=float)

    def cell_force(self,path):
        Gene = OpenOutcar(MoreFile(MoreLine('FORCE on cell',16)))
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
        Gene = OpenOutcar(MoreFile(MoreLine('direct lattice vectors',4)))
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
        Gene = OpenOutcar(MoreFile(MoreLine('length of vectors',1)))
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

    def dynamical_matrix_eigenvalues(self,path):
        dof = self.dof(path)
        nions = self.nions(path)
        Gene = OpenOutcar(MoreFile(MoreLine('Eigenvectors and eigenvalues of the dynamical matrix',dof*(nions+3)+3)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for i in range(dof):
            line = Record[i*(nions+3) + 3]
            datas.append(line.split()[-2])
        return np.array(datas,dtype=float)

    def oszicar(self,path):
        import re
        energy = []
        with open(os.path.join(path,'OSZICAR')) as f:
            for line in f:
                if re.match(r'\s*\d+\s*F',line):
                    energy.append(re.findall(r"=\s*([-?+?|\.|\w]+)\s",line))
        energy = np.array(energy,dtype=float)
        return energy

    def locpot(self,path,axis=None):
        with open(os.path.join(path,'LOCPOT')) as f:
            text=f.readlines()
        skip=sum(np.array(text[6].split(),dtype=int))
        shape=np.array(text[9+skip].split(),dtype=int)
        data=[]
        for line in text[10+skip:] :
            data.extend(line.split())
        data=np.array(data,dtype=float).reshape(shape[::-1]).transpose(2,1,0)
        if axis == None:
            return data
        if axis.lower() == 'x':
            return np.mean(np.mean(data,axis=1),1)
        if axis.lower() == 'y':
            return np.mean(np.mean(data,axis=0),1)
        if axis.lower() == 'z':
            return np.mean(np.mean(data,axis=0),0)

    # Grepband %
    def grep_kpoint(self,path,nkpts=None):
        if nkpts==None: npkts=self.nkpts(path)
        Gene = OpenOutcar(MoreFile(MoreLine('k-points in reciprocal',nkpts)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        return Record[-nkpts:]

    # Grepband
    def grep_kpoint_weight(self,path,nkpts=None):
        if nkpts==None: npkts=self.nkpts(path)
        Gene = OpenOutcar(MoreFile(MoreLine('Following reciprocal coordinates',nkpts+1)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for line in Record[-nkpts:]:
            datas.append(line.split()[3])
        return np.array(datas,dtype=float)

    # Grepband %
    def grep_band(self,path,nbands=None):
        if nbands==None: nbands=self.nbands(path)
        Gene = OpenOutcar(MoreFile(MoreLine('occupation',nbands)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        return Record

    # Grepband %
    def gw_nelm(self,path):
        Gene = OpenOutcar(CatFile(RecordLine('QP shifts <psi_nk| G(iteration)W_0 |psi_nk>: iteration')))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        result = len(Record)
        return result

    def born(self,path):
        nions = self.nions(path)
        Gene = OpenOutcar(MoreFile(MoreLine('BORN EFFECTIVE CHARGES',4*nions+1)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for n in range(nions):
            data =[]
            for line in Record[4*n+2:4*n+5]:
                data.append(line.split()[1:4])
            datas.append(data)
        return np.array(datas,dtype=float)

    def frequency(self, path):
        # unit (cm-1) task: dielectric
        nions = self.nions(path)
        nline = nions*3*(nions+3)+2
        Gene = OpenOutcar(MoreFile(MoreLine('Eigenvectors and eigenvalues of the dynamical matrix', nline)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = Record[-nline:]
        freqs = []  # shape = nions * 3
        eigen = [] # shape = nions * 3, nions, 3
        for line in datas:
            c1 = re.compile(r'(\d+\.\d+) cm-1')
            c2 = re.compile(r'(\d+\.\d+)')
            if c1.search(line):
                freqs.extend(c1.findall(line))
            else:
                data = c2.findall(line)
                if len(data) == 6:
                   eigen.append(data)

        assert len(freqs) == 3*nions
        assert len(eigen) == 3*nions**2
        freqs = np.array(freqs,dtype=float)
        eigens = np.array(eigen,dtype=float).reshape(3*nions, nions, 6)

        return freqs, eigens

    def elastic_ionic(self, path):
        # unit (kBar) task: elastic
        Gene = OpenOutcar(MoreFile(MoreLine('ELASTIC MODULI IONIC CONTR',9)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for line in Record[-6:]:
           data = re.findall(r"[-]?\d+\.\d+",line)
           if len(data) == 6:
               datas.append(data)
        return np.array(datas,dtype=float)

    def piezoelectric(self, path):
        # unit (C/m^2) task: dielectric
        # Gene = OpenOutcar(MoreFile(MoreLine('PIEZOELECTRIC TENSOR  for field in x, y, z        (C/m^2)',6)))
        Gene = OpenOutcar(MoreFile(MoreLine('PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)',5)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        '''
        rg = Ripgrepy('PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)', path)
        Record = rg.I().after_context(5).run().as_string.rstrip().split('\n')
        '''
        datas = []
        for line in Record[-3:]:
           data = re.findall(r"[-]?\d+\.\d+",line)
           if len(data) == 6:
               datas.append(data)
        return np.array(datas,dtype=float)

    def piezoelectric_ionic(self, path):
        # unit (C/m^2) task: dielectric
        Gene = OpenOutcar(MoreFile(MoreLine('PIEZOELECTRIC TENSOR IONIC CONTR',5)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        datas = []
        for line in Record[-3:]:
           data = re.findall(r"[-]?\d+\.\d+",line)
           if len(data) == 6:
               datas.append(data)
        return np.array(datas,dtype=float)

    # Grepoptic %
    def dielectric_ionic(self,path):
        Gene = OpenOutcar(MoreFile(MoreLine('MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION',4)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        diel = []
        for line in Record[-3:]:
            diel.append(line.split())
        return np.array(diel,dtype=float)

    # Grepoptic %
    def dielectric(self,path):
        Gene = OpenOutcar(MoreFile(MoreLine('MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)',4)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        diel = []
        for line in Record[-3:]:
            diel.append(line.split())
        return np.array(diel,dtype=float)

    # Grepoptic %
    def dielectric_real(self,path,nedos=None):
        if nedos==None: nedos=self.nedos(path)
        Gene = OpenOutcar(MoreFile(MoreLine('REAL DIELECTRIC FUNCTION',nedos+2)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        diel = []
        for line in Record[2:nedos+2]:
            diel.append(line.split())
        return np.array(diel,dtype=float)

    # Grepoptic %
    def dielectric_imag(self,path,nedos=None):
        if nedos==None: nedos=self.nedos(path)
        Gene = OpenOutcar(MoreFile(MoreLine('IMAGINARY DIELECTRIC FUNCTION',nedos+2)))
        try: 
            Gene.send(path)
        except StopIteration:
            pass
        diel = []
        for line in Record[2:nedos+2]:
            diel.append(line.split())
        return np.array(diel,dtype=float)

    # others %
    def core_state(self,path):
        with open(os.path.join(path,'OUTCAR'), 'r') as f:
            for line in f:
                if 'the core state' in line:
                    core = []
                    for line in f:
                        state = {}
                        if re.match(r'\s+\d+-\s+', line):
                            for orbit,energy in re.findall(r'(\d[spdf])\s+(-?\d+\.\d+)', line):
                                state[orbit] = float(energy)
                            core.append(state)
                        elif re.match(r'\s+\d[spdf]\s+', line):
                            for orbit,energy in re.findall(r'(\d[spdf])\s+(-?\d+\.\d+)', line):
                                state[orbit] = float(energy)
                            core[-1].update(state)
                        else:
                            if len(line.strip()) == 0: continue
                            break
        return core
        
