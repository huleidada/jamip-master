from .outcar import GrepOutcar
import numpy as np
import pandas as pd
import re
import os
from numba import njit
from pathlib import Path
from jamip.utils.utils import lazy_property
import time
import spglib

@njit
def fmsd(data,nrepeat,ntotal):
    xout = np.zeros(nrepeat)
    for i in range(nrepeat,ntotal):
        xout += (data[i-nrepeat+1:i+1] - data[i])**2
    return np.flip(xout)/(ntotal-nrepeat)

@njit
def ffmsd(data,nrepeat,ntotal):
    xout = np.zeros(nrepeat)
    temp = np.zeros(nrepeat)
    for i in range(0,ntotal):
        if i < nrepeat:
            temp[i]=data[i]
        else:
            t0= (i+1) % nrepeat
            if t0==0 :t0=nrepeat
            temp[t0-1]=data[i]
            for j in range(1,nrepeat):
                t = t0-j
                if t <= 0 :t+=nrepeat
                xout[j]+=(temp[t0-1]-temp[t-1])**2
    return xout/(ntotal-nrepeat)

def dejump(coordinate0, coordinate1, coordinate_type, **kwargs):
    """
    remove boundary condition when atom moves across boundary.
    
    Args:
        
        coordinate0: the previous coordinate [3].
        coordinate1: the next coordinate [3]
        coordinate_type: type of atomic position ['Direct' or 'Cartesian'].
        
        kwargs:
            lattice: lattice parameter (if the coordinate's type is Cartesian, parameter 'lattice' need to be set) [3, 3].
                i.e. [[10., 0.5, 0.1],
                      [0.2, 10., 0.3],
                      [0.0, 0.0, 10.]]
                      
                For variable cell optimization, 'lattice' is the lattice parameter of the previous coordinate 
                (change very small between previous and next structure's cell).

    Returns:
        new coordinations1 [3].
    """
    coordinate0=np.array(coordinate0)
    coordinate1=np.array(coordinate1)
    
    if coordinate_type.lower().startswith('d'):
        distance=coordinate1-coordinate0
        for i in range(0, 3):
            if distance[i] > 0.5:
                coordinate1[i]=coordinate1[i]-1
            elif distance[i] < -0.5:
                coordinate1[i]=coordinate1[i]+1
    elif coordinate_type.lower().startswith('c'):
        if not('lattice' in kwargs):
            raise Exception('additional parameter: lattice')
        else:
            a=np.array(kwargs['lattice'])
            # check
            if a.shape != (3,3):
                raise AttributeError('unrecognized lattice')
        distance=coordinate1-coordinate0
        for i in range(0,a.shape[0]):
            length=np.linalg.norm(a[i]) # module of lattice. i.e. a b c
            proj=np.dot(distance,a[i]/length) # project length of 
            if proj > 0.5*length:
                coordinate1 -= a[i]
            elif proj < -0.5*length:
                coordinate1 += a[i]
    return coordinate1

def removeBoundaryCondition(structures, coordinate_type):
    """
    actually atomic moving trajectory (considering the period boundary condition).
    Args:
        structures: collection of structures in MD [steps, lattice+atoms, 3].
        coordinate_type: type of atomic coordinate in structure ['Direct' or 'CCartesian']

    Returns:
        new collection of structures with removing boundary condition [steps, lattice+atoms, 3].
    """
    for i in range(1, structures.shape[0]): # structures(nstructures, lattice+atoms, 3)
        for j in range(3, structures.shape[1]):
            tmp=[]
            if coordinate_type.lower().startswith('d'):
                tmp=dejump(structures[i-1][j], structures[i][j], coordinate_type)
            elif coordinate_type.lower().startswith('c'):
                tmp=dejump(structures[i-1][j], structures[i][j], coordinate_type, lattice=structures[i-1][:3])
            if len(tmp) != 0 :
                structures[i][j]=tmp
    return structures

def generate_lattice_lines(lattice_matrix):
    """生成晶格线框的线段"""
    from itertools import product
    # 定义单位晶格的8个顶点
    vertices = np.array(list(product([0,1], [0,1], [0,1])))
    
    # 将顶点转换为真实坐标
    real_vertices = np.dot(vertices, lattice_matrix)
    
    # 定义12条棱边（立方体的边）
    edges = [
        (0,1), (1,3), (3,2), (2,0),  # 底面
        (4,5), (5,7), (7,6), (6,4),  # 顶面
        (0,4), (1,5), (2,6), (3,7)   # 垂直边
    ]
    
    # 生成线段数据
    lines = []
    for start, end in edges:
        lines.append(real_vertices[start])
        lines.append(real_vertices[end])
        lines.append([None]*3)  # 添加None分隔线段
        
    return np.array(lines)




class Dynamics(GrepOutcar):
 
    def __init__(self, path:str):
        self.path = Path(path)
        self.ctype = 'nvt'
        for ctype in ['nve', 'nvt', 'npt']:
            if ctype in self.path.name.lower():
                self.ctype = ctype
                break

    @lazy_property
    def oszicar(self):
        '''
        get infomation from OSZICAR
        Return np.array([nstep, 5]), include [step, T, E, F, E0]
        '''
        results = []
        with open(self.path/'OSZICAR','r') as f:
            for line in f:
                if 'T=' in line:
                    results.append(re.findall(r'(\d+)\s*T=\s*(\d+\.\d*)\s*E=\s*(-?\d?\.\d+E[-+]?\d+)\s*F=\s*(-?\d?\.\d+E[-+]?\d+)\s*E0=\s*(-?\d?\.\d+E[-+]?\d+)*', line)[0])
        df = pd.DataFrame(results, columns=['step', 'T', 'E', 'F', 'E0'])
        return df

    @lazy_property
    def pressure(self):
        results = []
        with open(self.path/'OUTCAR','r') as f:
            for line in f:
                # if 'total pressure' in line:
                #     results.append(re.findall(r'total pressure\s*=\s*(-?\d+\.\d*)', line)[0])                
                if 'external pressure' in line:
                    pressure = float(line.split('=')[1].split('kB')[0])
                    results.append(pressure)
        return np.array(results, dtype=float)

    @lazy_property
    def forces(self):
        results = []
        with open(self.path/'OUTCAR','r') as f:
            for line in f:
                if 'TOTAL-FORCE (eV/Angst)' in line:                    
                    force = []
                    f.readline()
                    for line in f:
                        if len(line.split()) != 6: break
                        force.append(line.split()[3:])
                    results.append(force)
        return np.array(results, dtype=float)

    @lazy_property
    def xdatcar(self):
        results = []
        with open(self.path/'XDATCAR','r') as f:
            comment = f.readline()
            scale = float(f.readline().strip()) 
            lattice = []
            for i in range(3):
                lattice.append(f.readline().split())
            lattice = np.array(lattice, dtype=float) * scale
            element = np.array(f.readline().split())
            number = np.array(f.readline().split(), dtype=int)
            position = []
            for line in f:
                if line.startswith('Direct configuration='):
                    if len(position) == np.sum(number):
                        results.append(position)
                    position = []
                else:
                    position.append(line.split())
            # final         
            if len(position) == np.sum(number):
                results.append(position)
            results = np.array(results, dtype=float)
            return lattice, results

    @lazy_property
    def velocities(self, filename='v.dat'):

        with open(self.path/filename, 'r') as f:
            results = []
            for line in f:
                results.append(line.split())

        with open(self.path/'XDATCAR', 'r') as f:
            for i in range(6):
                f.readline()
            natoms = np.array(f.readline().split(), dtype=int).sum()

        results = np.array(results, dtype=float)[:len(results)//natoms*natoms]
        results = results.reshape(-1,natoms,3)

        return results

    def msd(self, data, nrepeat, **kwargs):
        """
        For 1d array: [nstep] -> msd [nstep]                 # single atom
        For 2d array: [nstep, natoms] -> msd [nstep]         # 1D materials
        For 3d array: [nstep, natoms, 3] -> msd [nstep, 3]   # 3D materials
        
        nrepeat: length for msd calculation.

        (A1) 对不同时间间隔的所有MSD求平均
        <r^2(n)> = 1/N Σ(i=1,N) [r(ni) - r(ni-n)]^2
        (A2) 对固定时间间隔下的所有原子求平均
        <r^2(n)> = 1/NA Σ(i=0,NA-1) [r(i+n) - r(i)]^2
        
        kwargs:
            natoms (default=natoms): if given number of atoms is less than natoms, it will random select given number of atoms.
        """
        #from .fmsd import msd as fmsd

        def random(data, natoms):
            from copy import deepcopy
            data=np.array(data)
            if natoms < data.shape[1]:
                atoms=np.random.choice(range(data.shape[1]), natoms)
                print('selected atoms:', atoms)
                deleted_atoms=np.setdiff1d(range(data.shape[1]), atoms)
                tmp=deepcopy(data)
                tmp=np.delete(tmp, deleted_atoms, axis=1)
                data=tmp
            return data

        msd = None
        data=np.array(data)
        if len(data.shape) == 1:
            msd = fmsd(data, nrepeat, len(data))

        elif len(data.shape) == 2:
            # check
            natoms=data.shape[1]
            if 'natoms' in kwargs:
                natoms=kwargs['natoms']
            data=random(data, natoms)
            
            for j in range(0, data.shape[1]): # natoms
                data0=data[:,j]
                tmp = ffmsd(data0, nrepeat, len(data0))
                if msd is None:
                    msd=tmp
                else:
                    msd += tmp
            
            msd /= data.shape[1]

        elif len(data.shape) == 3:
            # check, [nstep, nelm, 3]
            nstep, natoms, naxis = data.shape
            if 'natoms' in kwargs:
                natoms=kwargs['natoms']
            data=random(data, natoms)
            
            msd=np.zeros((data.shape[2],nrepeat))
            for k in range(0, data.shape[2]):         # directions
                for j in range(0, data.shape[1]):     # natoms
                    data0=data[:, j, k]
                    tmp=fmsd(data0, nrepeat, len(data0))
                    msd[k] += tmp
            
            msd = msd.T / data.shape[1]
        return msd

    def autocorrection(self, data):
        """
        For 1d array: [nstep] -> ac [nstep]
        For 2d array: [nstep, natoms] -> ac [nstep]
        For 3d array: [nstep, natoms, 3] -> ac [nstep, 3]
        """
        from scipy.signal import correlate

        ac=None
        data=np.array(data)
        if len(data.shape) == 1:
            data0=data
            result=correlate(data0, data0, mode='full')#, method='auto')
            ac=result[result.size//2:]
        elif len(data.shape) == 2:
            result=0
            for j in range(0, data.shape[1]): # natoms
                data0=data[:,j]
                tmp=correlate(data0, data0, mode='full')#, method='auto')
                tmp=tmp[tmp.size//2:]
                result += tmp
            result=np.array(result.tolist())
            result /= data.shape[1] # divde natoms
            ac=result
        elif len(data.shape) == 3:
            result=np.zeros((data.shape[2], data.shape[0]))
            for k in range(0, data.shape[2]): # directions
                for j in range(0, data.shape[1]): # natoms
                    data0=data[:,j,k]
                    tmp=correlate(data0, data0, mode='full')#, method='auto')
                    tmp=tmp[tmp.size//2:]
                    result[k] += tmp
            result = result.T / data.shape[1] # divde natoms
            ac=result
        return ac

    def temperature(self):
        return self.oszicar['T'].astype(float).values

    def entropy(self, filename='OUTCAR'):
        """abandon"""
        path= self.path / filename
        infile=os.popen(f'grep "energy  without entropy=" {path}')
        string=infile.readline()
        
        entropy=[]
        if "ML" in string :
            while string:
                entropy.append(float(string.split('=')[1].split( )[0]))
                string=infile.readline()
        else :
            while string:
                entropy.append(float(string.split('=')[1].split('energy')[0]))
                string=infile.readline()
        return entropy

    def structures(self, 
                   toCartesian=False, 
                   coordinate_type='Direct',
                   apart=1,
                   isRemoveBoundaryCondition=False,
                   filename='XDATCAR', **kwargs):
        """
        read structures from XDATCAR
        
        Args:
            toCartesian (default=False): whether to convert atomic coordinate from Direct to Cartesian.
            coordinate_type (default='Direct'): type of atomic coordinate for returned structures ['Direct' (default) or 'Cartesian'].
            apart (default=1): apart of tow adjacent time points in data sampling.
            isRemoveBoundaryCondition (default=False): whether to remove boundary condition. 
            filename (default='XDATCAR'): filename need to read.    
        
        Returns:
            collection of structures in MD [steps, lattice+atoms, 3].
        """
        ctype=self.ctype
        structures = []
        
        with open(self.path / filename, 'r') as f: 
            identifier = None
            counter=0
            for string in f:
                if identifier == None:
                    identifier = string.rstrip()

                if ctype.lower() == "npt":
                    if string.startswith(identifier):
                        scale=float(f.readline()) # scale of lattice parameter
                        # lattice parameter
                        a=[]
                        for i in range(0,3):
                            a.append(f.readline().split())
                            
                        a=np.array(a, dtype=float)*scale
                        
                        # type of element
                        element=np.array(f.readline().split())
                        elementNum=np.array(f.readline().split(), dtype=int)
                        # read atom coordinate
                        if f.readline().startswith('Direct configuration=') or f.readline().startswith(' '):
                            
                            atoms=self._atomCoordiante(f, a, elementNum, coordinate_type)
                            tmp=list(np.vstack([a, atoms]))
                            if np.mod(counter, apart) == 0:
                                structures.append(tmp)
                            counter += 1
         
                elif ctype.lower() == 'nvt':
                    if string.startswith(identifier):
                        scale=float(f.readline()) # scale of lattice parameter
                        # lattice parameter
                        a=[]
                        for i in range(0,3):
                            a.append(f.readline().split())
                        a=np.array(a, dtype=float)*scale
                        
                        # type of element
                        element=np.array(f.readline().split())
                        elementNum=np.array(f.readline().split(), dtype=int)
                        
                    # read atom coordinate
                    if string.startswith('Direct configuration=') or string.startswith(' '):
                        atoms=self._atomCoordiante(f, a, elementNum, coordinate_type)
                        
         
                        tmp=list(np.vstack((a, atoms))) # numpy to list
                        if np.mod(counter, apart) == 0:
                            structures.append(tmp)
                        counter += 1
         
                elif ctype.lower() == 'opt':
                    pass
                else:
                    raise ValueError("unrecognized ctype's value")
                                    
        # remove period boundary condition
        structures=np.array(structures)
        structures_ = np.empty([structures.shape[0],structures.shape[1],structures.shape[2]],dtype = float)

        for i in range(0,structures.shape[0]):
            structures_[i] = structures[0]
        difference_value = np.rint(structures-structures_)
        structures = structures-difference_value

        if toCartesian == True:
            atomic_coordinate = structures[:,3:,:]
            lattices =  structures[:,:3,:]
            structures[:,3:,:] = atomic_coordinate @ lattices
              
        if isRemoveBoundaryCondition:
            structures=removeBoundaryCondition(structures, coordinate_type)

        return structures

    def lattice_parameters(self, structure):
        """
        latttice parameters.
        
        Args:
            structure: given structure [lattice+atoms, 3].
            
        Returns:
            lattice parameters
        """
        lattice=np.array(structure[:3])
        
        a=np.linalg.norm(lattice[0])
        b=np.linalg.norm(lattice[1])
        c=np.linalg.norm(lattice[2])
        alpha=np.degrees(np.arccos(np.clip(np.dot(lattice[1]/b, lattice[2]/c), -1, 1)))
        beta=np.degrees(np.arccos(np.clip(np.dot(lattice[0]/a, lattice[2]/c), -1, 1)))
        gamma=np.degrees(np.arccos(np.clip(np.dot(lattice[0]/a, lattice[1]/b), -1, 1)))
        
        return np.array([a, b, c, alpha, beta, gamma])

    def velocities(self, filename='v.dat', **kwargs):
        """
        read atomic velocities of structures from v.dat file.
        
        Args:
            path: path of velocity file (v.dat).
            natoms: total atomic number in simulation box.
            
            kwargs:
                apart: apart of tow adjacent time points in data sampling.
            
        Returns:
            collection of atomic velocities [steps, atoms, 3].
        """
        
        natoms=self.structureInfo()['natoms']
        apart=kwargs.get('apart', 1)
            
        velocities=[]
        
        with open(self.path / filename,'r') as f:
            counter2atoms=0 # counter the atomic number in a loop
            vs2s=[] # velocities of a structure
            counter=0
            for string in f:
                if string != '':
                    if len(vs2s) < natoms:
                        vs2s.append(string.split())
                    else:
                        vs2s.append(string.split())
                        if np.mod(counter, apart) == 0:
                            velocities.append(vs2s)
                        vs2s=[]
                        counter += 1
            velocities=np.array(velocities, dtype=float)
            return velocities

    def _atomCoordiante(self, infile, lattice, elementNum, coordinate_type):
        """
        read atomic position of a structure from XDATCAR.
        
        Args:
            infile: read IO stream of XDATCAR.
            lattice: lattice of this structures [3, 3].
            elementNum: number of each element [elements]. 
            coordinate_type: type of atomic position needed to convert when reading position ['Direct' or 'Cartesian'].

        Returns:
           collection of atomic positions for a structure [atoms, 3].
        """
        atoms=[] # coordinate of atoms
        for i in range(0, np.sum(elementNum)):
            tmp=[]
            if coordinate_type.lower().startswith('d'):
                tmp=[float(s0) for s0 in infile.readline().split()]
            elif coordinate_type.lower().startswith('c'):
                tmp=np.dot(lattice, np.array([float(s0) for s0 in infile.readline().split()]))
            atoms.append(tmp)
        atoms=np.array(atoms)
        return atoms

    def averanged_structure(self, nstart=0, nend=None, nrepeat=1, **kwargs):
        from jamip.structure import Structure,read
        strucutres = self.structures()
        if self.ctype == 'npt' or self.ctype == 'nvt':
            sum = np.sum(strucutres[nstart:nend:nrepeat],axis=0)/(nend-nstart)
            lattice = sum[:3]
            positions = sum[3:]
            raw = read('POSCAR',ftype='poscar')
            Plattice,Ppositions,Pelements = raw.to_cell()
            raw1 = Structure.from_cell((lattice,positions,Pelements))

        return raw1

    def structureInfo(self, filename='POSCAR', isSortedByElement=True):
        """ get information of structure from POSCAR file.
        Args:
            filename (default='POSCAR'): filename need to read.
            isSortedByElement (default=True): whether to sort elements by element name.
        Returns:    
            info: a dict of information of structure, including:
                - composition: dict of element and number, e.g. {'C': 2, 'O': 1}
                - elements: list of elements, e.g. ['C', 'C', 'O']
                - natoms: total number of atoms, e.g. 3
                - formula: chemical formula, e.g. 'C2O1'
        """
        from collections import defaultdict
        
        with open(self.path / filename,'r') as f:
            lines = f.readlines()
            species = lines[5].split()
            num_species = np.array(lines[6].split(), dtype=int)

        if len(species) != len(num_species):
            raise ValueError(f'Error in elements of {filename}')

        composition = defaultdict(int)
        elements = []
        for specie, number in zip(species, num_species):
            elements += [specie]*number
            composition[specie] += number
                
        # formula
        species = list(composition.keys())
        numbers = list(composition.values())
        div = self.Z if np.gcd.reduce(numbers) is True else 1
        if isSortedByElement:
            # values,indices = np.unique(self.__species, return_index=True)
            indices = np.argsort(species)
        else:
            indices = range(len(species)) 

        formula = ''
        for i in indices:
            e = species[i]
            n = numbers[i]
            formula += '%s%d' %(e,n/div)

        info={'composition': composition,
              'elements': elements,
              'natoms': len(elements),
              'formula': formula,
        }
         
        return info

    def plotThermodynamcisInfo(self, nrepeat_for_MSD=None, path=None, skip=0, fname="info.png",**kwargs):
        """
        nrepeat_for_MSD:
        
        kwargs:
            For MSD:
                isMSD (default=True):
                natoms_for_MSD (default=natoms): if given number of atoms is less than natoms, it will random select given number of atoms.
                nstart_for_MSD (default=0):
                nend_for_MSD (default=nstep):
            For AC: 
                isVAC (default=False):
                nstart_for_AC (default=0):
                nend_for_AC (default=nstep):
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.font_manager import FontProperties

        start = time.time()

        fig=plt.figure(figsize=(20,15))
        outer_grid=gridspec.GridSpec(1, 1)
        inner_grid=gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer_grid[:, :], 
                                                    wspace=0.35, hspace=0.3)
        
        # -------------------- temperature, energy, pressure --------------------
        # Figure 1. Temperature %

        subplot=plt.Subplot(fig, inner_grid[0:1, 0:1])
        mddata = self.oszicar.iloc[skip:]
        temp=mddata['T'].astype(float).values
        t=np.arange(len(temp))/1000.0 # fs-ps
        start1 = time.time()
        print("temperatue cost:",start1-start)

        subplot.plot(t, temp, linestyle='-', lw=2)
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{T\ (K)}$", fontsize=22).set_fontweight("bold")
        subplot.ticklabel_format(useOffset=False)       
        fig.add_subplot(subplot)
        
        # Figure 2. Energy %
        subplot=plt.Subplot(fig, inner_grid[0:1, 1:2])
        H = mddata['E'].astype(float).values
        t=np.arange(len(H))/1000.0 # fs-ps
        start2 = time.time()
        print("energy cost:",start2-start1)

        subplot.plot(t, H, linestyle='-', lw=2)
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{H\ (eV)}$", fontsize=22).set_fontweight("bold")
        plt.gca().ticklabel_format(useOffset=False)
        fig.add_subplot(subplot)

        # Figure 3. Pressure %
        subplot=plt.Subplot(fig, inner_grid[0:1, 2:3])
        P=self.pressure[skip:]
        t=np.arange(len(P))/1000.0 # fs-ps        
        start3 = time.time()
        print("pressure cost:",start3-start2)

        subplot.plot(t, P, linestyle='-', lw=2)
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{P\ (kBar)}$", fontsize=22).set_fontweight("bold")
        subplot.ticklabel_format(useOffset=False) 
        fig.add_subplot(subplot)   
        
        # -------------------- trajectory --------------------        
        time3 = time.time()
        structures=self.structures(filename='XDATCAR',isRemoveBoundaryCondition=False)
        
        # lattice
        # lattice_a = None
        # lattice_b = None
        # lattice_c = None
        time1 = time.time()
        print('structure',time1-time3)
        '''
        for i in range(0, structures.shape[0]):
            structure0=structures[i]
            if lattice_a is None:
                lattice_a = structure0[0]
                lattice_b = structure0[1]
                lattice_c = structure0[2]
            else:
                lattice_a=np.vstack((lattice_a, structure0[0]))
                lattice_b=np.vstack((lattice_b, structure0[1]))
                lattice_c=np.vstack((lattice_c, structure0[2]))
        '''
        lattice_a = structures[:,0,:]
        lattice_b = structures[:,1,:]
        lattice_c = structures[:,2,:]
        # a, b, c
        a=np.linalg.norm(lattice_a,axis=1)
        b=np.linalg.norm(lattice_b,axis=1)
        c=np.linalg.norm(lattice_c,axis=1)
        # alpha, beta, gamma   
        alpha=np.degrees(np.arccos(np.divide(np.einsum('ij, ij->i', lattice_b, lattice_c),b*c)))
        beta=np.degrees(np.arccos(np.divide(np.einsum('ij, ij->i', lattice_c, lattice_a),c*a)))
        gamma=np.degrees(np.arccos(np.divide(np.einsum('ij, ij->i', lattice_a, lattice_b),a*b)))

        subplot=plt.Subplot(fig, inner_grid[1:2, 0:1])
        t=np.arange(structures.shape[0])/1000.0 # fs-ps
        l1, = subplot.plot(t, a, linestyle='-', lw=2, c='r', label='a')
        l2, = subplot.plot(t, b, linestyle='-', lw=2, c='g', label='b')
        l3, = subplot.plot(t, c, linestyle='-', lw=2, c='b', label='c')
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{Lattice\ parameters}$", fontsize=22).set_fontweight("bold")              
        fig.add_subplot(subplot)
        
        subplot0=plt.Subplot(fig, inner_grid[1:2, 0:1], sharex=subplot, frameon=False)        
        l4, = subplot0.plot(t, alpha, linestyle='-', lw=2, c='c', label=r'$\mathregular{\alpha}$')
        l5, = subplot0.plot(t, beta, linestyle='-', lw=2, c='m', label=r'$\mathregular{\beta}$')
        l6, = subplot0.plot(t, gamma, linestyle='-', lw=2, c='y', label=r'$\mathregular{\gamma}$')
        subplot0.set_ylim(0, 180)   
        subplot0.tick_params(labelsize=20)
        subplot0.yaxis.tick_right()
        subplot0.yaxis.set_label_position('right')
        handles = [l1, l2, l3, l4, l5, l6]
        labels = [line.get_label() for line in handles]

        # 在图形底部（或其他位置）添加统一图例
        fig.legend(handles, labels, 
                loc='lower center',   
                ncol=3,             
                bbox_to_anchor=(0.2, 0.07),  
                frameon=True,    
                prop=(FontProperties(weight="bold", size=18)))  
      
        fig.add_subplot(subplot0)
        
        #=========================================================
        # msd
        #==========================================================
        subplot=plt.Subplot(fig, inner_grid[1:2, 1:2])
        
        # isMSD=kwargs.get('isMSD', True)
        natoms_for_MSD=kwargs.get('natoms_for_MSD',structures.shape[1]-3)
        nstart_for_MSD=kwargs.get('nstart_for_MSD',0)
        nend_for_MSD=kwargs.get('nstart_for_MSD',structures.shape[0])
        nrepeat_for_MSD = 3000 if nend_for_MSD > 3000 else nend_for_MSD
        nrepeat_for_MSD=kwargs.get('nrepeat_for_MSD', 30)

        start4 = time.time()
        msd=self.msd(structures[nstart_for_MSD:nend_for_MSD,3:,:], nrepeat=nrepeat_for_MSD, natoms=natoms_for_MSD)
        t=np.arange(msd.shape[0])/1000.0 # fs-ps
        # x, y, z
        subplot.plot(t, msd[:,0], linestyle='-', lw=2, c='r', label='x')
        subplot.plot(t, msd[:,1], linestyle='-', lw=2, c='g', label='y')
        subplot.plot(t, msd[:,2], linestyle='-', lw=2, c='b', label='z')
        start5 = time.time()
        print("msd cost:",start5-start4)
        
        # subplot.set_xscale('log')
        # subplot.set_yscale('log')
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{MSD}$", fontsize=22).set_fontweight("bold")
        if np.max(msd) < 1e3:
            subplot.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            offset_text = subplot.yaxis.get_offset_text()
            offset_text.set_fontsize(16)
        # subplot.ticklabel_format(useOffset=False) 
        subplot.legend(loc=2, numpoints=2,
                       prop=(FontProperties(weight="bold", size=18)), frameon=True)              
        fig.add_subplot(subplot) 
        
        # AC
        isVAC = kwargs.get('isVAC',False)
        nstart_for_AC = kwargs.get('nstart_for_AC',0)
        nend_for_AC = kwargs.get('nend_for_AC',structures.shape[0])

        if isVAC:
            inner_grid2=gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=inner_grid[1:2, 2:3], 
                                                         wspace=0.1, hspace=0.05)
            # VAC
            velocities=self.velocities()
            vac=self.autocorrection(velocities[nstart_for_AC:nend_for_AC,:,:])
            t=np.arange(vac.shape[0])/1000.0 # fs-ps
        
            subplot=plt.Subplot(fig, inner_grid2[0:1, 0:1])

            # x, y, z
            subplot.plot(t, vac[:,0], linestyle='-', lw=2, c='r', label='x')
            subplot.plot(t, vac[:,1], linestyle='-', lw=2, c='g', label='y')
            subplot.plot(t, vac[:,2], linestyle='-', lw=2, c='b', label='z')
        
            subplot.tick_params(labelsize=20)
            subplot.set_xticks([])
#             subplot.set_xlabel("$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
            subplot.set_ylabel(r"$\mathregular{VAC}$", fontsize=22).set_fontweight("bold")
        
            subplot.legend(bbox_to_anchor=(1.0, 1.0), numpoints=2,
                           prop=(FontProperties(weight="bold", size=18)), frameon=True)
        
            fig.add_subplot(subplot)
        
            # FAC
            fac=self.autocorrection(self.forces[nstart_for_AC:nend_for_AC,:,:])
            t=np.arange(fac.shape[0])/1000.0 # fs-ps
        
            subplot=plt.Subplot(fig, inner_grid2[1:2, 0:1])

            # x, y, z
            subplot.plot(t, fac[:,0], linestyle='-', lw=2, c='r', label='x')
            subplot.plot(t, fac[:,1], linestyle='-', lw=2, c='g', label='y')
            subplot.plot(t, fac[:,2], linestyle='-', lw=2, c='b', label='z')
        
            subplot.tick_params(labelsize=20)
            subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
            subplot.set_ylabel(r"$\mathregular{FAC}$", fontsize=22).set_fontweight("bold")
        
            subplot.legend(loc=1, numpoints=2,
                           prop=(FontProperties(weight="bold", size=18)), frameon=True)
        
            fig.add_subplot(subplot)
        else:
            # FAC
            fac=self.autocorrection(self.forces[nstart_for_AC:nend_for_AC,:,:])
            t=np.arange(fac.shape[0])/1000.0 # fs-ps
        
            subplot=plt.Subplot(fig, inner_grid[1:2, 2:3])

            # x, y, z
            subplot.plot(t, fac[:,0], linestyle='-', lw=2, c='r', label='x')
            subplot.plot(t, fac[:,1], linestyle='-', lw=2, c='g', label='y')
            subplot.plot(t, fac[:,2], linestyle='-', lw=2, c='b', label='z')
        
            subplot.tick_params(labelsize=20)
            subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
            subplot.set_ylabel(r"$\mathregular{FAC}$", fontsize=22).set_fontweight("bold")

            subplot.legend(loc=1, numpoints=2,
                           prop=(FontProperties(weight="bold", size=18)), frameon=True)
        
            fig.add_subplot(subplot)

        start6 = time.time()
        print("FAC cost:",start6-start5)
              
        fig.tight_layout()
        fig.savefig(fname, dpi=600)  

    def plotAtomicTrajectories(self, projected_direction=None, atoms=None, colors=None, toCartesian=False, fname='Trajectories.png', **kwargs):
        """
        Arguments:
            structures: [step, lattice+atoms, 3].
            atoms: [atom0, atom1, atom2, ...].
            colors: {'Ca': 'r', 'N': 'b', ...}
            projected_direction: [0 ] or [0,1] or 0 | 0 -> x; 1 -> y; 2 -> z
            path: 
        
            kwargs:
                nstart (default=0):
                nend (default=nstep):
                nrepeat (default=1):
                isOutput (default=False): whether need to output data.
        
        Note that counting starts from 1 for atom1 and atom2. 
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.font_manager import FontProperties
              
        start= time.time()      
        # -------------------- trajectory --------------------
        structures=self.structures(toCartesian,filename='XDATCAR')
        structureInfo = self.structureInfo(isSortedByElement=False)
        end = time.time()
        print("structure cost:",end-start)

        nstart = kwargs.get('nstart',0)
        nend = kwargs.get('nend',structures.shape[0])
        nrepeat = kwargs.get('nrepeat',1)
        # isOutput = kwargs.get('isOutput',False)

        if not isinstance(atoms, list):
            atoms = range(0,structureInfo['natoms'])
        composition = structureInfo['composition']
        if not isinstance(colors, dict):
            color = ['r','g','b','y','c','m','k']
            colors = {}    
            for value, key in enumerate(composition):
                colors[key] = color[value % len(color)]

        elements = structureInfo['elements']            
        trajectories=structures[nstart:nend:nrepeat, 3:, :]
        start1 = time.time()    
        # lattice
        lattice_parameters=self.lattice_parameters(structures[0]) # [a, b, c, alpha, beta, gamma]
        
        if projected_direction == None:
            projected_direction = range(3)
        elif isinstance(projected_direction, int):
            projected_direction = [projected_direction]            

        for projected_direction in range(0,3):
            fig=plt.figure(figsize=(8,8))
            outer_grid=gridspec.GridSpec(1, 1)
            inner_grid=gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[:, :], 
                                                wspace=0.35, hspace=0.3)
            ploted_elements=[]
            subplot=plt.Subplot(fig, inner_grid[0:1, 0:1])
            xmax=None
            ymax=None
            
            if projected_direction == 0: # yz plane
                time_=time.time()
                xorder = np.argsort(structures[0,3:,0])
                for atom0 in xorder:
                    xs=trajectories[:, atom0, 1] # y
                    ys=trajectories[:, atom0, 2] # z
                    xmax=lattice_parameters[1]
                    ymax=lattice_parameters[2]
                    xlabel='Y'
                    ylabel='Z'
                    symbol0=elements[atom0]
                    if not(symbol0 in ploted_elements):
                        subplot.plot(xs, ys, linestyle='-', lw=2, c=colors[symbol0], label='%s' %(symbol0))
                        ploted_elements.append(symbol0)
                    else:
                        subplot.plot(xs, ys, linestyle='-', lw=2, c=colors[symbol0])
                time__=time.time()
                print('plot',time__-time_)

            elif projected_direction == 1: # xz plane
                yorder = np.argsort(structures[0,3:,1])
                for atom0 in yorder:
                    xs=trajectories[:, atom0, 0] # x
                    ys=trajectories[:, atom0, 2] # z
                    xmax=lattice_parameters[0]
                    ymax=lattice_parameters[2]
                    xlabel='X'
                    ylabel='Z'
                    symbol0=elements[atom0]
                    if not(symbol0 in ploted_elements):
                        subplot.plot(xs, ys, linestyle='-', lw=2, c=colors[symbol0], label='%s' %(symbol0))
                        ploted_elements.append(symbol0)
                    else:
                        subplot.plot(xs, ys, linestyle='-', lw=2, c=colors[symbol0])
            
            elif projected_direction == 2: # xy plane
                zorder = np.argsort(structures[0,3:,2])
                for atom0 in zorder:
                    xs=trajectories[:, atom0, 0] # x
                    ys=trajectories[:, atom0, 1] # y
                    xmax=lattice_parameters[0]
                    ymax=lattice_parameters[1]
                    xlabel='X'
                    ylabel='Y'
                    symbol0=elements[atom0]
                    if not(symbol0 in ploted_elements):
                        subplot.plot(xs, ys, linestyle='-', lw=2, c=colors[symbol0], label='%s' %(symbol0))
                        ploted_elements.append(symbol0)
                    else:
                        subplot.plot(xs, ys, linestyle='-', lw=2, c=colors[symbol0])

            end1 = time.time()
            print("plt cost",end1-start1)        
                
            subplot.tick_params(labelsize=20)
            subplot.set_xlabel(xlabel, fontsize=22).set_fontweight("bold")
            subplot.set_ylabel(ylabel, fontsize=22).set_fontweight("bold")
            
            if toCartesian==False:
                xmax = 1
                ymax = 1
            subplot.axis('scaled') # x axis = y axis
            subplot.axis([0,xmax,0,ymax])             
            subplot.legend(loc=1, numpoints=2,
                           prop=(FontProperties(weight="bold", size=18)), frameon=True)
            
            # fig.tight_layout()
            fig.add_subplot(subplot) 
            fname = Path(fname)
            subname = fname.parent / f'{fname.stem}_{projected_direction}.png'
            fig.savefig(subname, dpi=600) 

    def plotAtomicMSD(self, fname='atomic_msd.png', **kwargs):
        """
        plot atomic mean square displacement (MSD) of all atoms.
        """        
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.font_manager import FontProperties

        structures = self.structures()
        nrepeat = kwargs.get('nrepeat',1000)
        atoms = kwargs.get('atoms',0)
        order = kwargs.get('order',False)

        average_structure = (np.sum(structures,axis=0)/structures.shape[0]).reshape(1,structures.shape[1],structures.shape[2]).repeat(structures.shape[0],axis=0)
        DIF = ((structures-average_structure)@structures[0,:3,:])[:,3:,:] # difference value
        # structure0 = structures[0,:,:]
        # structures0=structure0.reshape(1,structure0.shape[0],structure0.shape[1]).repeat(structures.shape[0],axis=0)        
        # DIF = ((structures-structures0)@structures[0,:3,:])[:,3:,:] # difference value

        square = np.multiply(DIF,DIF)
        msd_atoms = np.sum(square,axis=0)/(structures.shape[0])

        if order:            
            # temp = np.sort(msd_atoms[27:108,:],axis=1)
            temp_1 = msd_atoms[0:27,:]
            temp1 = np.sum(temp_1,axis=0)/(temp_1.shape[0])
            print('temp1',temp1)

            temp_2 = msd_atoms[64:256,:]
            temp_2 = np.sort(msd_atoms[64:256,:])
            temp2 = np.sum(temp_2,axis=0)/(temp_2.shape[0])
            print('temp2',temp2)

        atoms_msd=np.array([None]*3)
        for i in range(3):
            if atoms == 0:
                atoms_msd[i] = fmsd(structures[:,3,i],nrepeat, structures.shape[0])
            else :
                atoms_msd[i] = fmsd(structures[:,atoms,i], nrepeat, structures.shape[0])
        atoms_msd=np.transpose(atoms_msd.tolist())

        fig = plt.figure(figsize=(12,8))
        outer_grid=gridspec.GridSpec(1,1)
        inner_grid=gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[:, :], 
                                                    wspace=0.35, hspace=0.3)
        
        subplot =plt.Subplot(fig, inner_grid[0:1, 0:1])
        t=np.arange(atoms_msd.shape[0])/1000.0 # fs-ps
        subplot.plot(t, atoms_msd[:,0], linestyle='-', lw=2, c='r', label='x')
        subplot.plot(t, atoms_msd[:,1], linestyle='-', lw=2, c='g', label='y')
        subplot.plot(t, atoms_msd[:,2], linestyle='-', lw=2, c='b', label='z')

        # subplot.set_xscale('log')
        # subplot.set_yscale('log')
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{MSD}$", fontsize=22).set_fontweight("bold")
        subplot.legend(loc=2, numpoints=2,
                       prop=(FontProperties(weight="bold", size=18)), frameon=True)
              
        fig.add_subplot(subplot) 

        subplot =plt.Subplot(fig, inner_grid[0:1, 1:2])
        t=np.arange(square.shape[0])/1000.0 # fs-ps
        subplot.plot(t, square[:,atoms,0], linestyle='-', lw=2, c='r', label='x')
        subplot.plot(t, square[:,atoms,1]-1, linestyle='-', lw=2, c='g', label='y')
        subplot.plot(t, square[:,atoms,2]-2, linestyle='-', lw=2, c='b', label='z')

        # subplot.set_xscale('log')
        # subplot.set_yscale('log')
        subplot.tick_params(labelsize=20)
        subplot.set_xlabel(r"$\mathregular{T\ (ps)}$", fontsize=22).set_fontweight("bold")
        subplot.set_ylabel(r"$\mathregular{MSD}$", fontsize=22).set_fontweight("bold")
        subplot.legend(loc=2, numpoints=2,
                       prop=(FontProperties(weight="bold", size=18)), frameon=True)
              
        fig.add_subplot(subplot) 

        fig.tight_layout()
        fig.savefig(fname, dpi=600)

        return

    def plotAtomicTrajectories3D(self, atoms=None, colors=None, delaunay=True, fname='atomic_trajecttories_3D.html', **kwargs):
        """
        Arguments:
            structures: [step, lattice+atoms, 3].
            atoms: [atom0, atom1, atom2, ...].
            colors: {'Ca': 'r', 'N': 'b', ...}
            projected_direction: [0 ] or [0,1] or 0 | 0 -> x; 1 -> y; 2 -> z
            path: 
        
            kwargs:
                nstart (default=0):
                nend (default=nstep):
                nrepeat (default=1):
                isOutput (default=False): whether need to output data.
        
        Note that counting starts from 1 for atom1 and atom2. 
        """        
        import plotly.graph_objects as go
        from collections import defaultdict

        # -------------------- trajectory --------------------
        structures = self.structures(toCartesian=True, filename='XDATCAR')
        structureInfo = self.structureInfo(isSortedByElement=False)

        nstart = kwargs.get('nstart', 0)
        nend = kwargs.get('nend', structures.shape[0])
        nrepeat = kwargs.get('nrepeat', 1)
        
        if not isinstance(atoms, list):
            atoms = range(0, structureInfo['natoms'])
        composition = structureInfo['composition']

        if not isinstance(colors, dict):
            color = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black']
            colors = {}    
            for value, key in enumerate(composition):
                colors[key] = color[value % len(color)]
        
        elements = structureInfo['elements']          
        trajectories = structures[nstart:nend:nrepeat, 3:, :]

        # Create figure
        fig = go.Figure()

        # 1. 先添加晶格线框
        lattice = structures[0, :3, :]  # Lattice vectors
        lattice_lines = generate_lattice_lines(lattice)
        fig.add_trace(go.Scatter3d(
            x=lattice_lines[:,0],
            y=lattice_lines[:,1],
            z=lattice_lines[:,2],
            mode='lines',
            line=dict(
                color='rgba(100,100,100,0.5)',  # 半透明灰色
                width=5
            ),
            name='Lattice',
            hoverinfo='none'
        ))

        # Process atom order
        order_list = [(ii, structures[0, 3:, 0][ii]) for ii in atoms]
        order_list = sorted(order_list, key=lambda x: x[1])
        atoms = [jj[0] for jj in order_list]

        # Group trajectories by element
        element_trajectories = defaultdict(list)
        for atom0 in atoms:
            symbol = elements[atom0]
            xs = trajectories[:, atom0, 0]
            ys = trajectories[:, atom0, 1]
            zs = trajectories[:, atom0, 2]
            element_trajectories[symbol].append((xs, ys, zs))

            # print(f"Processing atom {atom0} ({symbol}): "
            #       f"X range: {np.min(xs)} to {np.max(xs)}, "
            #       f"Y range: {np.min(ys)} to {np.max(ys)}, "
            #       f"Z range: {np.min(zs)} to {np.max(zs)}")

        # Add traces for each element
        for symbol in element_trajectories:
            # Combine all trajectories for this element
            all_x = np.concatenate([traj[0] for traj in element_trajectories[symbol]])
            all_y = np.concatenate([traj[1] for traj in element_trajectories[symbol]])
            all_z = np.concatenate([traj[2] for traj in element_trajectories[symbol]])

            fig.add_trace(go.Scatter3d(
                x=all_x,
                y=all_y,
                z=all_z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[symbol],
                    opacity=0.8
                ),
                name=symbol,
                showlegend=True
            ))

        # Set axis limits and labels
        xmin, xmax = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        ymin, ymax = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
        zmin, zmax = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', range=[xmin, xmax]),
                yaxis=dict(title='Y', range=[ymin, ymax]),
                zaxis=dict(title='Z', range=[zmin, zmax]),
                # aspectmode='manual',
                # aspectratio=dict(x=1, y=ymax/xmax, z=zmax/xmax)
                aspectmode='cube',
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            title="Atomic Trajectories"
        )

        # Save to HTML
        fig.write_html(fname, include_plotlyjs='cdn')
        return fig

    def plotAtomicTrajectories_imshow(self, nstart=0, nend=None, nrepeat=1, filename='XDATCAR', **kwargs):
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        structures=self.structures(toCartesian=True, filename=filename)
        if nend is None:
            nend = structures.shape[0]

        trajectories=structures[nstart:nend:nrepeat, 3:, :]  
        lattice_parameters=self.lattice_parameters(structures[0]) # [a, b, c, alpha, beta, gamma]
        fig,ax = plt.subplots(figsize=(8/2.54,8/2.54))
        x_num = 200
        y_num = 200
        density=np.zeros([x_num,y_num])

        x_len = lattice_parameters[0]
        y_len = lattice_parameters[1]

        for pp in range(x_num):
            temp1 = np.where((trajectories[:,:,0]>=(x_len*pp/x_num))&
                             (trajectories[:,:,0]<(x_len*(pp+1)/x_num)))
            for qq in range(y_num):
                temp2 = trajectories[:,:,1][temp1]
                temp3 = np.where((temp2>=(y_len*qq/y_num))&
                                 (temp2<(y_len*(qq+1)/y_num)))
                density[pp,qq]=temp3[0].shape[0]
        
        cmap = cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=100)
        x= np.arange(0,x_len,x_len/x_num)
        y= np.arange(0,y_len,y_len/y_num)
        # im = ax.imshow(density,cmap=cmap,norm=norm)
        # gap = math.floor(x_len/5)
        # x_ticks = np.arange(0,x_len,gap)
        # ax.set_xticks(x_ticks*x_num/x_len,x_ticks,fontsize=8)
        # y_ticks = np.arange(0,y_len,gap)
        # ax.set_yticks(y_ticks*y_num/y_len,y_ticks,fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel('X',fontsize=8)
        ax.set_ylabel('Y',fontsize=8)
        X , Y = np.meshgrid(x, y)
        im = ax.contourf(X,Y,density,levels=np.linspace(0,100,20),extend='max',cmap=cmap,norm=norm)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = fig.colorbar(im,cax=cax,shrink=0.5,extend='max')
        cbar.set_ticks([0,100])
        cbar.ax.tick_params(labelsize=8)
        ax.set_aspect('equal', 'box') 
        fig.tight_layout()
        fig.savefig('imshow.png', dpi=600) 
        return
    
    def plotAtomicTrajectories3Dx(self, atoms=None, colors=None, toCartesian=True, path=None, name=None, **kwargs):
        """
        Arguments:
            structures: [step, lattice+atoms, 3].
            atoms: [atom0, atom1, atom2, ...].
            colors: {'Ca': 'r', 'N': 'b', ...}
            projected_direction: [0 ] or [0,1] or 0 | 0 -> x; 1 -> y; 2 -> z
            path: 
        
            kwargs:
                nstart (default=0):
                nend (default=nstep):
                nrepeat (default=1):
                isOutput (default=False): whether need to output data.
        
        Note that counting starts from 1 for atom1 and atom2. 
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.font_manager import FontProperties

        # -------------------- trajectory --------------------
        structures=self.structures(toCartesian,filename='XDATCAR')
        structureInfo = self.structureInfo(isSortedByElement=False)

        nstart = kwargs.get('nstart',0)
        nend = kwargs.get('nend',structures.shape[0])
        nrepeat = kwargs.get('nrepeat',1)
        # isOutput = kwargs.get('isOutput',False)
        if isinstance(atoms,list):
            pass
        else:
            atoms = list(range(0,structureInfo['natoms']))
        composition = structureInfo['composition']

        if isinstance(colors,dict):
            pass
        else:
            color = ['r','g','b','y','c','m','k']
            colors = {}    
            for value, key in enumerate(composition):
                colors[key] = color[value]
        element=list(composition.keys())
        nums=composition.values()
        elements = []
        for mm,nn in enumerate(nums):
            [elements.append(element[mm]) for oo in range(0,nn)]
        
        trajectories=structures[nstart:nend:nrepeat, 3:, :]
          
        # lattice
        lattice_parameters=self.lattice_parameters(structures[0]) # [a, b, c, alpha, beta, gamma]

        fig=plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')

        ploted_elements=[]
        order_list = []
        list1 = structures[0,3:,0]
        for ii in atoms:
            order_list.append((ii,list1[ii]))
        order_list=sorted(order_list, key= lambda x:x[1])
        order = []
        [order.append(jj[0]) for jj in order_list]
        atoms = order
        for atom0 in atoms:
            xs=trajectories[:, atom0, 0] # x
            ys=trajectories[:, atom0, 1] # y
            zs=trajectories[:, atom0, 2] # z

            xmax=lattice_parameters[0]
            ymax=lattice_parameters[1]
            zmax=lattice_parameters[2]
            
            xlabel='x'
            ylabel='y'
            zlabel='Z'
            symbol0=elements[atom0]
            if not(symbol0 in ploted_elements):
                ax.scatter3D(xs, ys, zs, c=colors[symbol0], lw=2, label='%s' %(symbol0))
                ploted_elements.append(symbol0)
            else:
                ax.scatter3D(xs, ys, zs, c=colors[symbol0], lw=2)
        
        ax.set_xlim3d(0,xmax)
        ax.set_ylim3d(0,ymax)
        ax.set_zlim3d(0,zmax)

        ax.set_xlabel(xlabel, fontsize=22).set_fontweight("bold")
        ax.set_ylabel(ylabel, fontsize=22).set_fontweight("bold")
        ax.set_zlabel(zlabel, fontsize=22).set_fontweight("bold")
        
        ax.legend(loc=1, numpoints=2,
                               prop=(FontProperties(weight="bold", size=18)), frameon=True)

        if path is None:
                path=self.path
        
        ax.view_init(45,30)
        fig.savefig('%s/atomic_trajecttories_3D.png' %(path), dpi=600)    

class TDEP:#Parser:
    """rewrite base on tdep process_outcar_5.3.py"""
    
    def __init__(self, workdir=None, skip_steps=3000):
        self.inf = float('inf')
        self.skip_steps = skip_steps
        self.initialize_regex_patterns()
        self.reset_state()
        if workdir != None:
            self.workdir = Path(workdir)
        
    def initialize_regex_patterns(self):
        self.re_patterns = {
            'POTIM': re.compile(r"\s*POTIM\s*=\s*(\d+\.?\d+)\s+time-step for ionic-motion"),
            'NIONS': re.compile(r".*NIONS =\s+(\d+)"),
            'LATTICE': re.compile(r"\s+Lattice vectors\:\s+"),
            'VECTOR': re.compile(r"\s*A? = \(\s*(-?\d+\.\d+),\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\s*\)"),
            'LATTICE2': re.compile(r"\s+direct lattice vectors\s+reciprocal lattice vectors"),
            'VECTOR2': re.compile(r"\s+(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*.*"),
            'TEMP': re.compile(r"\s*nergy EKIN\s*=\s*(\d+\.\d+)\s*"),
            'TEMP2': re.compile(r"\s*lattice  EKIN_LAT=\s*(\d+\.\d+)\s* \(temperature\s*(\d*\.\d*) K\)"),
            'ETOT': re.compile(r"\s*total energy   ETOTAL =\s*(-?\d*\.\d*)\s*eV"),
            'EPOT': re.compile(r"%?\s*ion-electron   TOTEN  =\s*(-?\d*\.\d*)\s*.*"),
            'EPOT2': re.compile(r"%?\s*free  energy   TOTEN  =\s*(-?\d*\.\d*)\s*.*"),
            'STRESS': re.compile(r"\s*in kB (.*)"),
            'PRESSURE': re.compile(r"\s*external pressure =\s*(-?\d*\.\d*)\s*kB  Pullay stress =.*"),
            'POSITION': re.compile(r"\s*POSITION\s*TOTAL-FORCE.*"),
            'TEBEG': re.compile(r"\s*TEBEG\s*=\s*(\d+\.?\d+);\s*TEEND\s*=\s*(\d+\.\d+)\s*temperature during run"),
            'SIGMA': re.compile(r"\s*ISMEAR\s*=\s*(\+?-?\d*);\s*SIGMA\s*=\s*(\d*\.\d*)\s*broadening.*"),
            'IBRION': re.compile(r"\s*IBRION\s*=\s*(\+?-?\d*)\s*"),
            'MAG': re.compile(r"\s*magnetization \(x\)\s*")
        }
    
    def reset_state(self):
        """重置解析器状态"""
        self.potim = self.inf
        self.nions = self.inf
        self.tebeg = self.inf
        self.ismear = self.inf
        self.sigma = self.inf
        self.ibrion = None
        self.lattice = None
        self.reciprocal_lattice = None
        self.step = 0
        
    def read_poscar_lattice(self, poscar_file):
        """从POSCAR文件读取晶格信息"""
        try:
            f = open(poscar_file, 'r')
            
            f.readline()  # 注释行
            a = float(f.readline())  # 晶格常数
            
            if a < 0:
                raise ValueError("Volume scaling in POSCAR not supported")
            
            lattice = np.zeros((3, 3))
            for i in range(3):
                lattice[i] = list(map(float, f.readline().strip().split()))
            lattice = lattice * a
            
            f.close()
            return lattice
            
        except IOError as e:
            print(f"Could not read POSCAR file {poscar_file}: {e}")
            return None
    
    def load_info(self, infile="OUTCAR"):
        
        params_found = {
            'potim': False,
            'nions': False, 
            'tebeg': False,
            'sigma': False,
            'ismear': False,
            'lattice': False
        }

        with open(infile, 'r') as f:
            for line in f:
                # 解析POTIM
                if not params_found['potim']:
                    m = self.re_patterns['POTIM'].match(line)
                    if m:
                        self.potim = float(m.group(1))
                        params_found['potim'] = True
                
                # 解析NIONS
                if not params_found['nions']:
                    m = self.re_patterns['NIONS'].match(line)
                    if m:
                        self.nions = int(m.group(1))
                        params_found['nions'] = True
                
                # 解析TEBEG
                if not params_found['tebeg']:
                    m = self.re_patterns['TEBEG'].match(line)
                    if m:
                        self.tebeg = float(m.group(1))
                        params_found['tebeg'] = True
                
                # 解析SIGMA和ISMEAR
                if not params_found['sigma'] or not params_found['ismear']:
                    m = self.re_patterns['SIGMA'].match(line)
                    if m:
                        self.ismear = int(m.group(1))
                        self.sigma = float(m.group(2))
                        params_found['sigma'] = True
                        params_found['ismear'] = True
                
                # 解析IBRION
                if self.ibrion is None:
                    m = self.re_patterns['IBRION'].match(line)
                    if m:
                        self.ibrion = m.group(1)
                
                # 解析晶格（第一种格式）
                if not params_found['lattice'] and self.re_patterns['LATTICE'].match(line):
                    f.readline()  # 跳过空行
                    lattice = []
                    for i in range(3):
                        line = f.readline()
                        m = self.re_patterns['VECTOR'].search(line)
                        if m:
                            lattice.append([m.group(1), m.group(2), m.group(3)])
                    self.lattice = np.array(lattice, dtype=float)
                    self.reciprocal_lattice = np.linalg.inv(self.lattice)
                    params_found['lattice'] = True
                
                # 解析晶格（第二种格式）
                if not params_found['lattice'] and self.re_patterns['LATTICE2'].match(line):
                    lattice = []
                    for i in range(3):
                        line = f.readline()
                        m = self.re_patterns['VECTOR2'].search(line)
                        if m:
                            lattice.append([m.group(1), m.group(2), m.group(3)])
                    self.lattice = np.array(lattice, dtype=float)
                    self.reciprocal_lattice = np.linalg.inv(self.lattice)
                    params_found['lattice'] = True
                
                if all(params_found.values()):
                    break
        
        return all(params_found.values())
    
    def parse_timestep_data(self, infile):
        """解析时间步数据"""
        f = open(infile, 'r')
        
        filestep = 0
        current_stresses = None
        current_pressure = None
        mag_buffer = ""
        self.step = 0
        
        line = f.readline()
        while line:
            # 解析应力
            if self.re_patterns['STRESS'].match(line):
                m = self.re_patterns['STRESS'].search(line)
                current_stresses = list(map(float, m.group(1).split()))
                line = f.readline()
                m = self.re_patterns['PRESSURE'].search(line)
                current_pressure = float(m.group(1))
                filestep += 1
            
            # 解析位置和力
            elif self.re_patterns['POSITION'].match(line):
                f.readline()  # 跳过标题行
                positions = []
                forces = []
                for i in range(self.nions):
                    data = list(map(float, f.readline().strip().split()))

                    if self.reciprocal_lattice is not None:
                        frac_coords = self.reciprocal_lattice @ data[0:3]
                    else:
                        raise
                    positions.append(frac_coords)
                    forces.append(data[3:6])
                
                if filestep > self.skip_steps:
                    yield {
                        'type': 'positions_forces',
                        'step': filestep,
                        'positions': positions,
                        'forces': forces
                    }
            
            # 解析能量（标准格式）
            elif self.re_patterns['EPOT'].match(line):
                m = self.re_patterns['EPOT'].search(line)
                epot = float(m.group(1))
                
                line = f.readline()  # Ekin
                m = self.re_patterns['TEMP'].search(line)
                ekin = float(m.group(1))
                
                line = f.readline()  # kin. lattice ...
                m = self.re_patterns['TEMP2'].search(line)
                temp = float(m.group(2))
                
                # 跳过相关行
                for _ in range(3):
                    line = f.readline()
                
                line = f.readline()  # etotal
                m = self.re_patterns['ETOT'].search(line)
                etot = float(m.group(1))
                
                if filestep > self.skip_steps and current_stresses is not None:
                    self.step += 1

                    yield {
                        'type': 'energy',
                        'step': self.step,
                        'filestep': filestep,
                        'etot': etot,
                        'epot': epot,
                        'ekin': ekin,
                        'temp': temp,
                        'pressure': current_pressure,
                        'stresses': current_stresses,
                        'magnetic': mag_buffer if mag_buffer else None
                    }
                    mag_buffer = ""

            # 匹配能量块（对于IBRION != 0的情况，与原代码逻辑一致）
            elif self.re_patterns['EPOT2'].match(line) and self.ibrion != "0":
                m = self.re_patterns['EPOT2'].search(line)
                epot = float(m.group(1))
                etot = epot
                temp = 0
                ekin = 0

                # 只有在跳过足够步数后才处理
                if filestep > skip_steps:
                    self.step += 1

                    # 返回能量数据
                    yield {
                        'type': 'energy',
                        'step': self.step,
                        'filestep': filestep,
                        'etot': etot,
                        'epot': epot,
                        'ekin': ekin,
                        'temp': temp,
                        'pressure': current_pressure,
                        'stresses': current_stresses
                    }
            
            # 解析磁矩
            elif self.re_patterns['MAG'].match(line):
                f.readline()  # 跳过标题行
                f.readline()  # 跳过列标题
                f.readline()  # 跳过分隔线
                mag_buffer = ""
                for _ in range(self.nions):
                    mag_buffer += f.readline()
            
            line = f.readline()
        
        f.close()

    def get_workdir(self,workdir=None):
        if workdir is None:
            workdir = self.workdir
        else:
            workdir = Path(workdir)
        if not workdir.exists():
            workdir.mkdir(parents=True)
        return workdir

    def load_timestep(self, infile="OUTCAR", workdir=None, statfile="infile.stat", posfile="infile.positions", forcefile="infile.forces"):

        workdir = self.get_workdir(workdir)

        statfileobj = open(workdir/statfile, 'w')
        posfileobj = open(workdir/posfile, 'w')
        forcefileobj = open(workdir/forcefile, 'w')

        for data in self.parse_timestep_data(infile):
            if data['type'] == 'energy':
                # write stat file
                step = data['step']
                line = ("{0:<6d} {1:<8.2f}  {2: .6f}  {3: .6f}  {4: .6f}  {5: .2f}  {6: .3f} " +
                        "{7: .3f}  {8: .3f}  {9: .3f}  {10: .3f}  {11: .3f}  {12: .3f}\n").format(
                        step, (step - 1) * self.potim, data['etot'], data['epot'], 
                        data['ekin'], data['temp'], 0.1 * data['pressure'],
                        0.1 * data['stresses'][0], 0.1 * data['stresses'][1], 
                        0.1 * data['stresses'][2], 0.1 * data['stresses'][3], 
                        0.1 * data['stresses'][4], 0.1 * data['stresses'][5])
                statfileobj.write(line)

            elif data['type'] == 'positions_forces':
                # write position file
                for pos in data['positions']:
                    posfileobj.write(f"{pos[0]: .7f} {pos[1]: .7f} {pos[2]: .7f}\n")

                # write force file
                for force in data['forces']:
                    forcefileobj.write(f"{force[0]: .7f} {force[1]: .7f} {force[2]: .7f}\n")
    
    def write_meta(self, workdir=None, filename="infile.meta"):

        path = self.get_workdir(workdir) / filename

        with open(path, 'w') as f:
            f.write(f"{self.nions}\n")
            f.write(f"{self.step}\n")
            f.write(f"{self.potim}\n")
            f.write(f"{self.tebeg}\n")

    '''
    This script is used to convert the force constant in TDEP to the corresponded format of Phonopy and ShengBTE.
    Created on May 4, 2019
    
    @author: fu
    '''

    def getUCinfo(self, workdir=None, filename='infile.ucposcar'):
        """
        get atomic coordinate and symbol of unit cell.

        Arguments:
            filename (default='infile.ucposcar'): filename of unit cell structure.

        Return:
            atoms: array of atomic coordinates. [natoms, 3]
            symbols: array of elemental symbols. [natoms]
        """
        from jamip.structure import read

        path = self.get_workdir(workdir) / filename
        self.uc = read(path, ftype="vasp")

    def getSSinfo(self, workdir=None, filename='infile.ssposcar'):
        """
        get size of supercell.

        Arguments:
            filename (default='infile.ssposcar'): filename of supercell structure.

        Return:
            supercell: size of supercell. e.g. [3,3,3]
            natoms: total atomic numbers in supercell structure.
        """
        from jamip.structure import read
        path = self.get_workdir(workdir) / filename
        self.ss = read(path, ftype="vasp")
        self.natoms = len(self.ss)
        self.supercell = np.array(self.ss.comment_line.split()[0].split('x'), dtype=int)

    def read2dFC(self, workdir=None, filename='outfile.forceconstant'):
        """
        read second-order force constant of TDEP calculation.

        Return:
            ifc2: second-order force constant. Note that the matrix elements beyond given cutoff radius are set to zero. num_keys: natoms_uc x natoms
        """
        from collections import OrderedDict

        infile = self.get_workdir(workdir) / filename
        infile=open(infile, 'r')

        natoms_in_uc=int(infile.readline().split()[0])
        cutoff=float(infile.readline().split()[0])

        # atoms in supercell
        natoms=self.natoms

        ifc2={}
        for i in range(0, natoms_in_uc):
            neighbor=int(infile.readline().split()[0])
            for j in range(0, neighbor):
                tmp=infile.readline().split('In the unit cell, what is the index of neighbour')
                index_in_uc=int(tmp[0]) # index of mapped neighbor atom in unit cell
                ineigh=int(tmp[1].split()[0]) # index of neighbor
                iatom1=int(tmp[1].split()[-1])
                vector=[float(s0) for s0 in infile.readline().split()]
                iatom2=self.index(index_in_uc, vector, self.supercell)

                phi0=np.zeros((3,3)) # [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
                for k in range(0, 3):
                    phi0[k]=[float(s0) for s0 in infile.readline().split()]

                ifc2['%d-%d' %(iatom1, iatom2)]=phi0
            for k in range(1, natoms+1):
                if not('%d-%d' %(iatom1, k) in ifc2.keys()):
                    ifc2['%d-%d' %(iatom1, k)]=np.zeros((3,3))
        #print('before items',ifc2.items())
        ifc2=OrderedDict(sorted(ifc2.items(), key=lambda x : [int(s0) for s0 in x[0].split('-')]))
        #print('ifc2',ifc2)
        return ifc2

    def index(self, index_in_uc, vector, supercell):
        """
        get corresponding index in supercell for Phonopy.

        Arguments:
            index_in_uc: index of atom in unit cell. Note that counting is from 1.
            vector: the translation vector of subcell in supercell corresponding to unit cell. e.g. mirror cell of unit cell on the right is [1, 0, 0].

        Return:
            corresponded index of atom in supercell.
        """
        vector=self.removePBC(vector, supercell)
        index=(index_in_uc-1)*(supercell[0]*supercell[1]*supercell[2])+(1+vector[0]+vector[1]*supercell[0]+vector[2]*supercell[0]*supercell[1])
        return index

    def removePBC(self, vector, supercell):
        """
        remove the periodic boundary condition in TDEP's force constant. move atom in mirror cell to supercell. Likely, if a atom with coordinate of (0.5,0.0,0.0) is in subcell of [-1,0,0].
        If the supercell is [1,1,1]. moving vector is [1,0,0] to unit cell.
        If the supercell is [3,3,3]. moving vector is [3,0,0] to subcell of [2,0,0].

        Arguments:
            vector: vector of atom in TDEP's force constant.
            supercell: supercell size in MD calculation.

        Return:
            normalized vector.
        """
        for i in range(0, len(vector)):
            v=vector[i]
            #print('supercell',supercell)
            if supercell[i] == 1:
                if v < -1e-6:
                    v += 1
                elif v > 1e-6:
                    v -= 1
            elif supercell[i] > 1:
                if v < 0:
                    v += supercell[i]
                elif v > supercell[i]:
                    v -= supercell[i]

            vector[i]=v
        return vector

    def indexTable(self):
        """
        index table between index and vector in supercell. The countting of index is firstly along x, then y and last z.

        Return:
         directory of index table.
        """
        from collections import OrderedDict

        indexTable=OrderedDict()
        supercell=self.supercell
        for z in range(0, supercell[2]):
            for y in range(0, supercell[1]):
                for x in range(0, supercell[0]):
                    vector0=np.array([x,y,z])
                    for index_in_uc in range(1, len(self.uc)+1):
                        index=self.index(index_in_uc, vector0, supercell)
                        indexTable[index]={'index_in_uc':index_in_uc, 'vector':vector0}

        return indexTable

    def getmap(self, iatom1, iatom2):
        """
        get the index in supercell for the given pair (iatom1 and iatom2) from TDEP's force constant.

        Arguments:
            iatom1: index of atom1 in TDEP's force constant.
            iatom2: index of atom2 in TDEP's force constant.

        Return:
            corresponded index in supercell. [index1, index2]
        """
        indexTable=self.indexTable

        # atom1
        tmp1=indexTable[iatom1]
        index_in_uc1=tmp1['index_in_uc']
        vector1=tmp1['vector']
        # atom2
        tmp2=indexTable[iatom2]
        index_in_uc2=tmp2['index_in_uc']
        vector2=tmp2['vector']

        # method1 (moving vector is the vector1)
        vector3=vector2-vector1
        index3=self.index(index_in_uc2, vector3, self.supercell)

        # method2 (moving vector is the vector2)
        vector4=vector1-vector2
        index4=self.index(index_in_uc1, vector4, self.supercell)

        return [index_in_uc1, index3]

    def toPhonopy(self, workdir=None, filename='FORCE_CONSTANTS'):
        """
        convert TDEP's force constant to that of Phonopy.
        """
        from collections import OrderedDict

        path = self.get_workdir(workdir) / filename
        outfile=open(path, 'w')

        self.getUCinfo() # natoms: total atomic numbers in supercell structure.
        self.getSSinfo() # natoms: total atomic numbers in supercell structure.
        natoms=self.natoms
        supercell=self.supercell

        self.indexTable=self.indexTable()
        ifc2=self.read2dFC()
        outfile.write('%d\n' %self.natoms)

        new_ifc2=OrderedDict()
        for i in range(1, natoms+1): # atom1
            for j in range(1, natoms+1): # atom2
                map=self.getmap(i, j)
                new_ifc2['%d-%d' %(i, j)]=ifc2['%d-%d' %(map[0], map[1])]
                outfile.write('%d %d\n' %(i, j))
                for k in range(0, 3):
                    outfile.write('%.12f %.12f %.12f\n' %(new_ifc2['%d-%d' %(i, j)][k][0],
                                                          new_ifc2['%d-%d' %(i, j)][k][1],
                                                          new_ifc2['%d-%d' %(i, j)][k][2]))
        print ('finish conversion')


    def getUCLatticeVecotr(self, workdir=None, filename='infile.ucposcar'):
        """
        get lattice vector of unit cell.

        Arguments:
            filename (default='infile.ucposcar'): filename of unit cell structure.

        Return:
            lattvec: lattice vector. [3x3]
        """
        import linecache

        infile = self.get_workdir(workdir) / filename

        scale=float(linecache.getline(infile, 2))
        lattvec=[]
        for i in range(3, 6):
            lattvec.append([float(s0)*scale for s0 in linecache.getline(infile, i).split()])
        lattvec=np.array(lattvec)

        return lattvec

    def read3rdIFC(self, workdir=None, filename='outfile.forceconstant_thirdorder'):
        """
        read third-order force constant of TDEP's calculation.

        Return:
            ifc3: [[iatom1, iatom2, iatom3, vec1, vec2, vec3, phi0],
                   ...]
        """
        from collections import OrderedDict

        ifc3=[]
        infile = self.get_workdir(workdir) / filename
        infile=open(infile, 'r')

        natoms_in_uc=int(infile.readline().split()[0])
        cutoff=float(infile.readline().split()[0])

        index=0
        for i in range(0, natoms_in_uc): # atom0
            neighbor=int(infile.readline().split()[0])
            for j in range(0, neighbor):
                iatom1=int(infile.readline().split()[0])
                iatom2=int(infile.readline().split()[0])
                iatom3=int(infile.readline().split()[0])
                vec1=[float(s0) for s0 in infile.readline().split()]
                vec2=[float(s0) for s0 in infile.readline().split()]
                vec3=[float(s0) for s0 in infile.readline().split()]
                phi0=[]
                for k in range(0, 9):
                    phi0.append([float(s0) for s0 in infile.readline().split()])

                ifc3.append([iatom1, iatom2, iatom3, vec1, vec2, vec3, phi0])
        return ifc3

    def toShengBTE(self, workdir=None, filename='FORCE_CONSTANTS_3RD'):
        """
        convert TDEP's third-order force constant to that of ShengBTE.
        """
        path = self.get_workdir(workdir) / filename
        outfile=open(path, 'w')

        self.getUCinfo() # natoms: total atomic numbers in supercell structure.
        lattvec=self.uc.lattice
        ifc3=self.read3rdIFC()

        # matrix for phi
        matrix=[]
        for i in range(1,4):
            for j in range(1,4):
                for k in range(1,4):
                    matrix.append([i,j,k])

        outfile.write('%s\n' %len(ifc3))
        for i in range(0, len(ifc3)):
            iatom1=ifc3[i][0]
            iatom2=ifc3[i][1]
            iatom3=ifc3[i][2]
            vec1=np.array(ifc3[i][3])
            vec2=np.array(ifc3[i][4])
            vec3=np.array(ifc3[i][5])
            phi0=np.array(ifc3[i][6])
            if np.linalg.norm(vec1) > 1e-6:
                raise ValueError("vec1 doesn't equal to [0.0, 0.0, 0.0]")

            outfile.write('\n')
            outfile.write('    %d\n' %(i+1))
            d2=np.dot(lattvec, vec2)
            d3=np.dot(lattvec, vec3)
            outfile.write('%.10f %.10f %.10f\n' %(d2[0], d2[1], d2[2]))
            outfile.write('%.10f %.10f %.10f\n' %(d3[0], d3[1], d3[2]))
            outfile.write('    %d %d %d\n' %(iatom1, iatom2, iatom3))
            for j in range(0, 9):
                outfile.write(' %d %d %d\t %.10f\n' %(matrix[3*j][0], matrix[3*j][1], matrix[3*j][2], phi0[j][0]))
                outfile.write(' %d %d %d\t %.10f\n' %(matrix[3*j+1][0], matrix[3*j+1][1], matrix[3*j+1][2], phi0[j][1]))
                outfile.write(' %d %d %d\t %.10f\n' %(matrix[3*j+2][0], matrix[3*j+2][1], matrix[3*j+2][2], phi0[j][2]))
        print ('finish conversion')

#path=os.getcwd()
#TDEP2Phonopy(path=path).toPhonopy()
#TDEP2ShengBTE(path=path).toShengBTE()

