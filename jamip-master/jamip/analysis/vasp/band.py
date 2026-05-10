import re
import numpy as np
from jamip.analysis.base import Finder
from jamip.utils.utils import lazy_property
from .outcar import GrepOutcar
from .eigenval import Eigenval
from .procar import Procar
from .chgcar import Chgcar
from .xml import Xml
from collections import namedtuple
from typing import Union
import pathlib

Bandside = namedtuple('BandSide', ['energy','kpoints','ikpt','iband'])
Emass = namedtuple('Emass', ['mass','energy','kpoints'])
Kpath = namedtuple('Kpath', ['kpts','insert','dirs','bestkpts'])
Projection = namedtuple('projection', ['atoms', 'orbits','label'])

class Kpoints:

    def get_kpath_from_kpoints(self, path, filename='KPOINTS'):
        '''
        input :
            path: get kpath from KPOINTS.
        '''
        path = pathlib.Path(path)
        if path.is_dir():
            path = path / filename
            
        with open(path, 'r') as f:
            f.readline()   
            nkpt = int(f.readline())
            model = f.readline().strip()
            if model.startswith(('L','l')):
                kpath = []
                for i in f:
                    kpoint = re.findall(r"[-]?\d\.\d+", re.split(r"[A-z!#_]+",i)[0])
                    if len(kpoint) == 3 and re.search(r"[A-z]", i):
                        kpath.append(re.findall(r"\S*[A-z]+\S*", i)[0])
            else:
                raise OSError(f'KPOINTS model should be Line Model, not {model} !')

        assert (len(kpath) % 2) == 0, 'The number of Kpoints must be a multiple of 2!'
        kpath_total = []
        kpath_part = [kpath[0]]
        for i,k in enumerate(kpath[1::2],1):
            if len(kpath) == 2*i:
                kpath_part.append(k)
                kpath_total.append(kpath_part)
            elif k == kpath[2*i]:
                kpath_part.append(k)
            else:
                kpath_part.append(k)
                kpath_total.append(kpath_part)
                kpath_part = [kpath[2*i]]
        nkpts = [nkpt] * int(len(kpath)/2)

        return Kpath(kpts=kpath_total, insert=nkpts, dirs=[path.parent], bestkpts=None)

    def get_kpath_from_info(self, path, key='kpoints/band'):
        '''
        input :
            structure: seek kpath by structure symmetry.
            banddir: a reference for whether seek path was calculated.

        '''
        from jamip.utils.logger import load_hdf5

        path = pathlib.Path(path)
        if path.is_dir():
            path = path / 'info.hdf5'
        kpoints = load_hdf5(key, path)

        # banddirs = []
        # for i,kpts in enumerate(kpoints['path']):
        #     for j in range(len(kpts)-1):
        #         A_B = kpts[j].strip('\\')+'-'+kpts[j+1].strip('\\')
        #         banddirs.append(A_B)
                
        return kpoints

    def get_kpath_from_dirs(self, dirs:list):
        ''' 
        input :
            dirs: band paths in segments, list, for instance,  
                  ['Gamma-X','X-U','U-R','R-Gamma']

        return :
            self._kpath = [['Gamma','X','U','R','Gamma']]
        
        Warning! The final band path may not be consistent with 
        the initial setting!
        '''
        # search next %
        def head(k, dirs):
            knext=[]
            bandn=[]
            bandpop=[]
            for d in dirs:
                if re.search('^%s-'%k,d):
                    knext.append(d.split('-')[1])
                    bandpop.append(d)
                else:
                    bandn.append(d)
            if len(knext) == 1:
                return knext[0],bandn
            else:
                return None, dirs
        # search last %
        def tail(k, dirs):
            klast=[]
            bandn=[]
            for d in dirs:
                if re.search('-%s$'%k,d):
                    klast.append(d.split('-')[0])
                else:
                    bandn.append(d)
            if len(klast) == 1:
                return klast[0],bandn
            else:
                return None, dirs
        sort_kpath=[]
        while len(dirs):
            init = dirs.pop(-1)
            kpath = init.split('-')
            klen = len(dirs)
            while klen:
                # search next %
                knext, dirs = head(kpath[-1], dirs)
                if knext:
                    kpath.append(knext)
                # search last %
                klast, dirs = tail(kpath[0], dirs)
                if klast:
                    kpath.insert(0,klast)
                if len(dirs) == klen:
                    break
                else:
                    klen=len(dirs)
            sort_kpath.append(kpath)
        klen = [len(k) for k in sort_kpath] 
        kklen = len(klen)
        sorted_kpath=[]
        while kklen > 1:
            kid = np.argsort(klen)[-1]
            init = sort_kpath[kid]
            for k in np.argsort(klen)[::-1][1:]:
                if sort_kpath[k][0] == init[-1]:
                    sort_kpath[kid].extend(sort_kpath.pop(k)[1:])
                    break
                if sort_kpath[k][-1] == init[0]:
                    sort_kpath[k].extend(sort_kpath.pop(kid)[1:])
                    break
            klen = [len(k) for k in sort_kpath] 
            if len(klen) < kklen:
                kklen =len(klen)
            else:
                kid = np.argmax(klen)
                sorted_kpath.append(sort_kpath.pop(kid))
                klen.pop(kid)
                kklen = len(klen)
        sorted_kpath.append(sort_kpath[0]) 
        return sorted_kpath

    @classmethod
    def read_kpath(self,path):

        file = pathlib.Path(path)/'KPATH.in'
        if not file.exists():
            raise IOError('KPATH.in not exists!')
        with open(file,'r') as f:
            # skip title
            for i in range(3):
                f.readline()

            kpath = []
            kpaths = []
            insert = []
            # read kpath
            for line in f:
                if len(line.split()) == 6:
                    data = line.split()
                    if int(data[3]) == 0: continue
                    elif int(data[3]) == 1:
                        kpath.append(data[5])
                        insert.append(int(data[3]))
                        kpaths.append(kpath)
                        kpath = []
                    else:
                        kpath.append(data[5])
                        insert.append(int(data[3]))

            if len(kpath) > 0:
                kpaths.append(kpath)

        insert = np.array(insert,dtype=int)

        return kpaths,insert


class Emc:

    def __init__(self):
        pass

    def fd_effmass(self, energies, stepsize:float):
        # energies: eV -> H
        # stepsize: A0^-1 -> bohr^-1

        e = energies / 27.21138505
        h = stepsize * 0.52917721092
        ne = len(e)

        assert ne==19 or ne==49, "only st3/st5 valid ! nkpts = %d" %ne
        m = np.zeros((3,3))
        if ne == 19:
            m[0][0] = (e[1] - 2.0*e[0] + e[2])/h**2                         
            m[1][1] = (e[3] - 2.0*e[0] + e[4])/h**2                         
            m[2][2] = (e[5] - 2.0*e[0] + e[6])/h**2                         
                                                                            
            m[0][1] = (e[7] + e[8] - e[9] - e[10])/(4.0*h**2)               
            m[0][2] = (e[11] + e[12] - e[13] - e[14])/(4.0*h**2)            
            m[1][2] = (e[15] + e[16] - e[17] - e[18])/(4.0*h**2)            
                                                                            
            # symmetrize                                                        
            m[1][0] = m[0][1]                                               
            m[2][0] = m[0][2]                                               
            m[2][1] = m[1][2]   
        elif ne == 49:
            m[0][0] = (-(e[1]+e[4])  + 16.0*(e[2]+e[3])   - 30.0*e[0])/(12.0*h**2)
            m[1][1] = (-(e[5]+e[8])  + 16.0*(e[6]+e[7])   - 30.0*e[0])/(12.0*h**2)
            m[2][2] = (-(e[9]+e[12]) + 16.0*(e[10]+e[11]) - 30.0*e[0])/(12.0*h**2)
            #
            m[0][1] = (-63.0*(e[15]+e[20]+e[21]+e[26]) + 63.0*(e[14]+e[17]+e[27]+e[24]) \
                       +44.0*(e[16]+e[25]-e[13]-e[28]) + 74.0*(e[18]+e[23]-e[19]-e[22]))/(600.0*h**2) 
            m[0][2] = (-63.0*(e[31]+e[36]+e[37]+e[42]) + 63.0*(e[30]+e[33]+e[43]+e[40]) \
                       +44.0*(e[32]+e[41]-e[29]-e[44]) + 74.0*(e[34]+e[39]-e[35]-e[38]))/(600.0*h**2)
            m[1][2] = (-63.0*(e[47]+e[52]+e[53]+e[58]) + 63.0*(e[46]+e[49]+e[59]+e[56]) \
                       +44.0*(e[48]+e[57]-e[45]-e[60]) + 74.0*(e[50]+e[55]-e[51]-e[54]))/(600.0*h**2)
            #
            # symmetrize
            m[1][0] = m[0][1]
            m[2][0] = m[0][2]
            m[2][1] = m[1][2]

        return m

    def get_eff_masses(self, m):

        eigval, eigvec = np.linalg.eig(m)
        em = 1/eigval
        return em, eigvec

    def set_kpoints(self, k0, rec_lattice, mesh=0.01, type='st3'):
        if type == 'st3':
            a = [-1,1]
        elif type == 'st5':
            a = [-2,-1,1,2]
        else:
            raise ValueError("unknown emc type!")

        # gamma
        kpts = []
        kpts.append(np.zeros(3))

        # x/y/z axis
        for idx in range(3):
            for v in a:
                kpt = np.zeros(3)
                kpt[idx] = v
                kpts.append(kpt)

        # xy,xz,yz
        for idx,idy in [(0,1),(0,2),(1,2)]:
            for v1 in a:
                for v2 in a:
                    kpt = np.zeros(3)
                    kpt[idx] = v1
                    kpt[idy] = v2
                    kpts.append(kpt)

        kpts = np.array(kpts, dtype=float)
        cart_kpts = kpts*mesh + k0@rec_lattice
        frac_kpts = cart_kpts @ np.linalg.inv(rec_lattice)

        return frac_kpts

def get_boundary(structure, atom='C', axis='z', tol=0.1):
    '''
    支持最多识别材料A+材料B+真空层的结构
    其中AB至少包含两个组分周期
    '''
    from collections import defaultdict
    import pandas as pd

    axis = 'xyz'.index(axis)
    coords = structure.get_positions(type='direct')[:,axis]
    coords -= np.floor(coords)       # range to [0,1)
    elements = np.array(structure.get_elements(type='symbol'))
    c = np.linalg.norm(structure.lattice[axis])

    maps = defaultdict(list)
    # 拆分每种元素的z坐标，计算元素的区间和范围
    for i,j in enumerate(elements):
        maps[j].append(coords[i])

    data = []
    for key,value in maps.items():
        if len(value) < 3: continue
        sortcoords = np.sort(value)
        nextcoords = np.roll(sortcoords, -1)
        nextcoords[-1] += 1
        diff = nextcoords - sortcoords
        imax = np.argmax(diff)

        spacings = diff[np.where(diff*c>tol)]
        spacing = (np.sum(spacings) - diff[imax]) / ( len(spacings) - 1)
        data.append([key, nextcoords[imax]-1, sortcoords[imax], spacing])

    df = pd.DataFrame(data, columns=['specie', 'start', 'stop', 'spacing'])
    return df

def get_partial_locpot(locpot, start, stop, step, skip=5, m1='expand', m2=None):

    def conv(data, k):
        result = np.zeros_like(data)
        ks = np.arange(k//2*2+1) - k//2
        print(ks)
        for i in ks:
            if (k/2-abs(i)) < 1e-8:
                result += np.roll(data,i)/2
                print(k, i)
            else:
                result += np.roll(data,i)
        return result / k

    def expand(data, indices, k):
        result = np.zeros(len(indices))
        dim = len(data)
        ks = np.arange(k//2*2+1) - k//2
        for i in ks:
            x = indices+i
            if (k/2-abs(i)) < 1e-8:
                result += data[np.where(x>=dim, x-dim, x)]/2
            else:
                result += data[np.where(x>=dim, x-dim, x)]
        return result / k

    dim = len(locpot)
    b1 = round(start*dim) + skip
    b2 = round(stop*dim) - skip
    k = round(step*dim)

    x = np.arange(b1, b2+1)
    y = locpot[np.where(x>=dim, x-dim, x)]

    if m1 == 'expand':
        dy = expand(locpot, x, k)
    elif m1 == 'conv':
        dy = conv(y, k)

    b = np.median(dy)
    return x,y,b


class Proj:

    def __init__(self, elements:list, allorbits:tuple):
        self.elements = elements
        self.allorbits = allorbits

    def to_proj(self, atoms:tuple, orbits:tuple, label:str=''):
        # atoms2list
        result = []
        for i in atoms:
            if isinstance(i, str):
                rows = np.where(self.elements==i)[0].tolist()
                if len(rows) == 0: 
                    raise ValueError("No matching element %s" %i)
                result.extend(rows)
            else:
                result.append(i)

        # orbits2list
        index = []
        for i,orbit in enumerate(self.allorbits):
            if orbit in orbits:
                index.append(i)
        if len(self.allorbits) == 9:
            if 'p' in orbits: index += [1,2,3]
            if 'd' in orbits: index += [4,5,6,7,8]
        if len(self.allorbits) == 16:
            if 'p' in orbits: index += [1,2,3]
            if 'd' in orbits: index += [4,5,6,7,8]
            if 'f' in orbits: index += [9,10,11,12,13,14,15]

        return Projection(atoms=list(set(result)), orbits=tuple(set(index)), label=label)

class Outcar(GrepOutcar):

    def __init__(self, path:str, nkpts:int, nbands:int, nions:int, ispin:int=1):
        self.path = path
        self.nkpts = nkpts
        self.nbands = nbands
        self.nions = nions
        self.ispin = ispin
        self._bands = None
        self._kpoints = None

    @property
    def kpoints(self):
        if self._kpoints is None:
            self._kpoints = self._get_kpoint()
        return self._kpoints

    @property
    def bands(self):
        if self._bands is None:
            self._bands = self._get_band()
        return self._bands

    @classmethod
    def from_file(cls, path):
        p = pathlib.Path(path)
        if p.is_file():
            p = p.parent
        if not p.exists(): 
            raise IOError("File 'OUTCAR' not exists!") 
        path = str(p)

        ispin = GrepOutcar().ispin(path)
        nkpts = GrepOutcar().nkpts(path)
        nions = GrepOutcar().nions(path)
        nbands = GrepOutcar().nbands(path)
        
        return cls(path, nkpts, nbands, nions, ispin)

    def _get_kpoint(self, weight=False):
        kpoints = []
        kpointdat=self.grep_kpoint(self.path,self.nkpts)
        kpoints = []
        if weight:
            for dat in kpointdat:
                kpoints.append(dat.split()[:4])
        else:
            for dat in kpointdat:
                kpoints.append(dat.split()[:3])
        return np.array(kpoints,dtype=float)

    def _get_kpoint_weight(self, weight=False):
        return self.grep_kpoint_weight(self.path,self.nkpts)
            
    def _get_band(self):
        banddat = self.grep_band(self.path,self.nbands)
        assert len(banddat) == self.ispin * self.nkpts * self.nbands
        bands = []
        for dat in banddat:
            value = dat.split()
            if len(value)!= 3: continue
            bands.append(value[1:])
        bands = np.array(bands,dtype=float).reshape(self.ispin,self.nkpts,self.nbands,2)
        return bands

    def _get_gw_band(self,nelm,nstep=None):
        banddat = self.grep_band(self.path,self.nbands)
        assert len(banddat) == nelm * self.ispin * self.nkpts * (self.nbands+1)
        bands = []
 
        if nstep is None: nstep = nelm
        assert nstep <= nelm
        istart = nstep * self.ispin * self.nkpts * (self.nabnds + 1)
        iend = (nstep+1) * self.ispin * self.nkpts * (self.nabnds + 1)

        for i,dat in enumerate(banddat[istart:iend]):
            value = dat.split()
            if len(value)!= 7: 
                continue
            bands.append([value[2],value[-1]])
        bands = np.array(bands,dtype=float).reshape(self.ispin,self.nkpts,self.nbands,2)
        return bands


class BandFinder(Finder, Kpoints, GrepOutcar):

    class Result:

        def __init__(self, bands:np.ndarray, kpoints:np.ndarray, fermi:float, **kwargs):
            self.bands = bands
            self.kpoints = kpoints
            self.fermi = fermi
            self.metal = False
            if 'rec_vector' in kwargs: self.rec_vector = kwargs['rec_vector']
            if 'isorbit' in kwargs: self.isorbit = kwargs['isorbit']
            if 'wavecar' in kwargs: self.wavecar = kwargs['wavecar']
            # regroup path %
            if 'kpath' in kwargs and kwargs['kpath'] != None: 
                self.kpath = kwargs['kpath']
                if self.kpath.bestkpts != None:
                    self.regroup(self.kpath.bestkpts)
            if 'kpoint_weights' in kwargs:
                self.kpoints_weights = kwargs['kpoint_weights']

        def get_cbvb(self):
            """
            Extract valence band and conduction band indices. 
            Returns:
            list:[[vb,cb] for each spin]
            """
            def condition_not_vb(band):
                # If the maximum energy exceeds the Fermi level
                c1 = (max(band[:,0]) - self.fermi) > 2e-4 if self.fermi != None else False
                # If partially empty occupancy
                c2 = max(band[:,1]) < 0.001
                return (c1 or c2)

            def condition_metal(band, fermi):
                # If the minimum energy is less than the Fermi level
                c1 = (min(band[:,0]) - fermi) < -2e-4 if fermi != None else False
                # If partially full occupancy
                c2 = max(band[:,1]) > 0.999
                return (c1 or c2)

            cbvbs = []
            for spin in self.bands:
                for index in np.arange(spin.shape[1]):
                    if condition_not_vb(spin[:,index]):
                        cbvbs.append([index-1,index])
                        fermi = np.max(spin[:,index-1]) if self.fermi is None else self.fermi
                        if condition_metal(spin[:,index], fermi):
                            self.metal = True
                        break
                        
            return cbvbs

        def get_metal(self):
            """
            Extract valence band and conduction band indices. 
            Returns:
            list:[[vb,cb] for each spin]
            """
            cbvbs = self.get_metal_cbvb()
            metal = False
            for eb,fb in cbvbs:
                if eb-fb > 1:
                    metal = True
            return metal

        def get_metal_cbvb(self):
            """
            Extract valence band and conduction band indices. 
            Returns:
            list:[[vb,cb] for each spin]
            """
            def condition_not_full(band):
                # If the maximum energy exceeds the Fermi level
                c1 = (max(band[:,0]) - self.fermi) > 1e-4 if self.fermi != None else False
                # If partially empty occupancy
                c2 = min(band[:,1]) < 0.001
                return (c1 or c2)

            def condition_empty(band, fermi):
                # If the minimum energy is less than the Fermi level
                c1 = (min(band[:,0]) - fermi) > -0.005 if fermi != None else False
                # If complete empty occupancy
                c2 = max(band[:,1]) < 0.001
                return (c1 or c2)

            cbvbs = []
            for spin in self.bands:
                fb, eb = None, None
                for index in np.arange(spin.shape[1]):
                    if condition_not_full(spin[:,index]):
                        if fb is None:
                            fb = index-1
                            fermi = np.max(spin[:,fb]) if self.fermi is None else self.fermi
                        if condition_empty(spin[:,index], fermi):
                            eb = index
                            break
                cbvbs.append([fb,eb])

            return cbvbs
        
        def get_metal_cbmvbm(self):
            """
            Extract main information of CBM & VBM. 
            Returns:  Dict(namedtuple)
                {'vbm': ('iband':float, 'ikpt':float, 'energy': float, 'kpoint':(3,)),
                 'cbm': ('iband':float, 'ikpt':float, 'energy': float, 'kpoint':(3,))}
            """
            cbvbs = self.get_metal_cbvb()
            data = {}
            for ispin, (fb,eb) in enumerate(cbvbs):
                # get cbvb of full band %
                vbm, cbm = self._get_cbmvbm(ispin=ispin, vb=fb, cb=fb+1)
                data['full'] = {'vbm':vbm, 'cbm':cbm, 'gap':np.round(cbm.energy-vbm.energy,6)}
                # get cbvb of empty band %
                vbm, cbm = self._get_cbmvbm(ispin=ispin, vb=eb-1, cb=eb)
                data['empty'] = {'vbm':vbm, 'cbm':cbm, 'gap':np.round(cbm.energy-vbm.energy,6)}
                return data

        def _get_cbmvbm(self, ispin, vb, cb):
            """
            Extract main information of CBM & VBM. 
            Params:
                ispin: spin band index
                cb: conduction band index
                vb: valence band index

            Returns:  Bandside
            """
            # get data %
            bands = self.bands[ispin]
            kpoints = self.kpoints
            vbm = np.argmax(bands[:,vb,0])
            cbm = np.argmin(bands[:,cb,0])
            vbm = Bandside(iband=vb, ikpt=vbm, energy=bands[vbm,vb,0], kpoints=kpoints[vbm])
            cbm = Bandside(iband=cb, ikpt=cbm, energy=bands[cbm,cb,0], kpoints=kpoints[cbm])
            return vbm, cbm

        def get_cbmvbm(self):
            """
            Extract main information of CBM & VBM. 
            Params:
                allspin: whether to return two sets of dictionaries by spin

            Returns:  Dict(namedtuple)
                {'vbm': ('iband':float, 'ikpt':float, 'energy': float, 'kpoint':(3,)),
                 'cbm': ('iband':float, 'ikpt':float, 'energy': float, 'kpoint':(3,))}  (allspin is False)
            """
            cbvbs = self.get_cbvb()
            keys = ['-up','-down'] if len(cbvbs) > 1 else ['']
            data = {}
            for i,cbvb in enumerate(cbvbs):
                # get data %
                vbm, cbm = self._get_cbmvbm(ispin=i, vb=cbvb[0], cb=cbvb[1])
                # cache %
                data['vbm'+keys[i]] = vbm
                data['cbm'+keys[i]] = cbm

            # vbm_up/vbm_down -> vbm
            if len(cbvbs) > 1:
                if data['vbm-up'].energy > data['vbm-down'].energy:
                    data['vbm'] = data['vbm-up']
                else:
                    data['vbm'] = data['vbm-down']
                if data['cbm-up'].energy < data['cbm-down'].energy:
                    data['cbm'] = data['cbm-up']
                else:
                    data['cbm'] = data['cbm-down']

            data['gap'] = np.round(data['cbm'].energy-data['vbm'].energy,6) if self.metal == False else -1e-4
            return data

        def get_bandgap(self, allspin=False):
            """
            Extract bandgap. 
            Params:
                allspin: Returns two sets of bandgap values when ispin == 2
                
            Returns:
            dict: {'indirect': float, 'direct': float} (default)
            --- or ---
            list: [{'indirect':float, 'direct':float},
                   {'indirect':float, 'direct':float}] (ispin==2 and allspin==True)
            """
            cbvbs = self.get_cbvb()
            bands = self.bands
            
            if len(bands) > 1 and allspin:
                gaps = []
                for cbvb,spin in zip(cbvbs,bands):
                    vb = spin[:,cbvb[0],0]
                    cb = spin[:,cbvb[1],0]
                    gap = {'indirect': np.around(np.min(cb)-np.max(vb),4),
                           'direct'  : np.around(np.min(cb-vb),4)}
                    gaps.append(gap)
                return gaps
            else:
                vb = np.max([bands[i,:,j[0],0] for i,j in enumerate(cbvbs)], axis=0)
                cb = np.min([bands[i,:,j[1],0] for i,j in enumerate(cbvbs)], axis=0)
                gap = {'indirect': np.around(np.min(cb)-np.max(vb),4),
                       'direct'  : np.around(np.min(cb-vb),4)}
                return gap

        def get_emass(self, axis=None, bd=['cbm','vbm'], fit_range=3, method='base', ispin=1, **kwargs):

            import scipy.constants as sc

            def get_indices(center, fit_range, nkpt):
                indices = np.arange(center-fit_range, center+fit_range+1)
                if indices[-1] >= nkpt:
                    indices += (nkpt - 1 - indices[-1])
                elif indices[0] < 0:
                    indices += -indices[0]
                return indices

            cbvbs = self.get_metal_cbvb()
            xkpts = self.get_xkpt() * sc.pi * 2 / sc.angstrom
            nkpt = len(xkpts)
            assert nkpt >= 2*fit_range + 1
             
            if method in ['weighting', 'ploy']: 
                fermi = kwargs.get('fermi', self.fermi)
                T = kwargs.get('T', 300)

 
            tmps = []  
            for cbvb,spin in zip(cbvbs,self.bands): 
                vb = spin[:,cbvb[0],0]
                cb = spin[:,cbvb[1],0]
                emass = {}
                if 'cbm' in bd:
                    indices = get_indices(np.argmin(cb),fit_range, nkpt)
                    x = xkpts[indices] 
                    y = cb[indices]*sc.e
                    if method == 'base':
                        weights = np.ones(len(x))
                    else:
                        weights = (1 / (np.exp((cb[indices] - fermi) / (sc.Boltzmann * T / sc.e)) + 1))
                    p = np.polyfit(x,y,deg=2,w=weights)

                    if method == 'ploy':
                        function = np.poly1d(p)
                        #dedk = function.deriv(m=1)(x)
                        d2edk2 = function.deriv(m=2)(x)
                        # 倒数+调和平均
                        mass = 1 / np.average(d2edk2, weights=weights)
                        mass = np.around(mass*sc.hbar**2/sc.electron_mass, 6)

                    else:
                        mass = np.around(sc.hbar**2/(2*p[0])/sc.electron_mass, 6)

                    emass['cbm'] = Emass(mass=mass, energy=np.min(cb), kpoints=self.kpoints[np.argmin(cb)])

                if 'vbm' in bd:
                    indices = get_indices(np.argmin(vb),fit_range, nkpt)
                    x = xkpts[indices] 
                    y = vb[indices]*sc.e
                    if method == 'base':
                        weights = np.ones(len(x))
                    else:
                        weights = 1 - (1 / (np.exp((vb[indices] - fermi) / (sc.Boltzmann * T / sc.e)) + 1))
                    p = np.polyfit(x,y,deg=2,w=weights)

                    if method == 'ploy':
                        function = np.poly1d(p)
                        #dedk = function.deriv(m=1)(x)
                        d2edk2 = function.deriv(m=2)(x)
                        # 倒数+调和平均
                        mass = -1 / np.average(d2edk2, weights=weights)
                        mass = np.around(mass*sc.hbar**2/sc.electron_mass, 6)

                    else:
                        mass = np.around(sc.hbar**2/(-2*p[0])/sc.electron_mass, 6)

                    emass['vbm'] = Emass(mass=mass, energy=np.max(vb), kpoints=self.kpoints[np.argmax(vb)])                   
                tmps.append(emass)


            if len(self.bands) > 1 and ispin ==2:
                emass = {}
                for spin,emc in zip(['up', 'down'], tmps):
                    for key,value in emc.items():
                        nkey = '%s-%s' %(key, spin)
                        emass[key] = value
            else:
                emass = {}
                for key in tmps[0]:
                    spin1 = tmps[0][key]
                    spin2 = tmps[-1][key]
                    if ('cbm' in key) == ( spin1.energy <= spin2.energy):
                        emass[key] = spin1
                    else:
                        emass[key] = spin2
            return emass

        def get_all_transitions(self):
            """
            Get the energy difference between the whole valence band 
            and the conduction band at the same spin and kpt
            return nd.array [(ikpt,vb,cb),] for vb/cb in two spins might be different.
            """
            bands = self.bands
            cbvbs = self.get_cbvb()
            # all_transitions = np.zeros((nspin,nkpt,))
            # in case that bands of two spin have different band edge 
            # we have to use Dataframe to store the transitions
            dEs=[]
            for ispin in range(len(bands)):
                dE_spin = []
                for ikpt in range(len(self.kpoints)):
                    for vband in range(cbvbs[ispin,0]+1):
                        for cband in range(cbvbs[ispin,1],bands.shape[2]):
                            dE_spin.append(bands[ispin,ikpt,cband,0]-bands[ispin,ikpt,vband,0])
                dEs.append( np.array(dE_spin).reshape(len(self.kpoints),cbvbs[0,0]+1,-1) )
            return dEs

        def get_tdm(self,cbvbs=None):
            """
            Calculate the transition dipole between two bands
            kwargs:
                cbvb: [(vb,cb),],  contains the valence band and the conduction band index 
            """
            assert hasattr(self,'wavecar') == True, "Function is available only when self.wavecar is set !"
            if cbvbs == None:  cbvbs = self.get_cbvb()
            nspin = self.bands.shape[0]
            nkpts = self.bands.shape[1]
                   
            ovlaps = []
            tdms = []
            dEs = []
            for ispin in range(nspin): 
                tdm_spin = []
                for ikpt in range(nkpts):
                    vbk = [ispin,ikpt,cbvbs[ispin][0]]
                    cbk = [ispin,ikpt,cbvbs[ispin][1]]
                    e0, e1, dE, tdm = self.wavecar.tdm(vbk,cbk)
                    dEs.append(dE)
                    tdm_spin.append(np.abs(tdm)**2)
                tdms.append(tdm_spin)

            #self.ovlaps = np.array(ovlaps).reshape(nspin,nkpts,-1)
            self.tdms = np.array(tdms)
            return self


        def get_cpd(self,cbvbs=None):
            """
            Calculate the Circular Polarization between two bands
            kwargs:
                cbvb: [(vb,cb),],  contains the valence band and the conduction band index 
            """
            assert hasattr(self,'wavecar') == True, "Function is available only when self.wavecar is set !"
            if cbvbs == None:  cbvbs = self.get_cbvb()
            nspin = self.bands.shape[0]
            nkpts = self.bands.shape[1]
                   
            ovlaps = []
            cps = []
            for ispin in range(nspin): 
                cp_spin = []
                for ikpt in range(nkpts):
                    vbk = [ispin,ikpt,cbvbs[ispin][0]]
                    cbk = [ispin,ikpt,cbvbs[ispin][1]]
                    cp = self.wavecar.cpd(vbk,cbk)
                    cp_spin.append(cp)
                cps.append(cp_spin)

            self.cpds = np.array(cps)
            return self

        def get_xkpt(self,continuous:bool=True):
            """
            calaulate the x coords for bandstructure plot
            kwargs:
                xticks: bool, if True, return the xticks
                continuous: bool, if True, return the xticks for continuous kpoints
                            if False, return the xticks for discrete kpoints
            return:
                xkpt: np.array, x coords for kpoints
                xticks: np.array, x coords for xticks
            """
            # nkpt -> nkpt
            delta = np.linalg.norm((self.kpoints[1:,:3]-self.kpoints[:-1,:3]) @ self.rec_vector, axis=1)
            delta = np.insert(delta,0,0)
            xkpt = np.cumsum(delta)

            # 通过计算间隔判断该点是否可导，如果不可导点的间隔为1，即该处不连续，计算此两点的间隔无意义，重置为1e-8
            if continuous:
                disc = np.where(np.abs(np.diff(delta))>1e-4)[0]
                if len(disc) > 1:
                    for i in range(1,len(disc)):
                        if disc[i]-disc[i-1] == 1:
                            k0 = disc[i-1]
                            k1 = disc[i] 
                            xkpt[k1:] -= (xkpt[k1]-xkpt[k0]+1e-8) 

            return xkpt
        
        def get_xticks(self):
            """
            calculate the xticks for bandstructure plot
            return:
                xticks: np.array, x coords for xticks
                xlabels: np.array, x labels for xticks
            """
            from jamip.utils.convert import kpath2list
            xlabels = kpath2list(self.kpath.kpts)
            xkpt = self.get_xkpt()

            # set x axis % 
            xticks = [0] + np.cumsum(self.kpath.insert).tolist()
            xticks[-1] -= 1
            xticks = [xkpt[i] for i in xticks]
            return xticks, xlabels

        def regroup(self,new_kpath:Union[list,tuple]):
            assert hasattr(self,'kpath') == True, "Function is available only when self.kpath is set !"
            if type(new_kpath[0]) == str: new_kpath=[new_kpath]
            nkpt = self.kpath.insert
            def norm(kp1,kp2):
                kp1 = re.findall(r'\w+',kp1)[0]
                kp2 = re.findall(r'\w+',kp2)[0]
                return (kp1,kp2)            

            index = 0
            kmap = {}            
            for kp in self.kpath.kpts:
                for i in range(len(kp)-1):
                    key = norm(kp[i], kp[i+1])
                    kmap[key] = np.arange(index, index+nkpt[i], 1)
                    index += nkpt[i]

            indices = []
            multis = []
            for kp in new_kpath:
                for i,k in enumerate(kp[:-1]):
                    key = norm(kp[i], kp[i+1])
                    rkey = tuple(reversed(key))
                    if key in kmap:
                        indices.append(kmap[key])
                    elif rkey in kmap:
                        indices.append(kmap[rkey][::-1])
                    else:
                        raise ValueError("K-path %s-%s not exists" %key)
                    multis.append(key)

            insert = [len(i) for i in indices]
            self.kpath = Kpath(kpts=new_kpath, insert=insert, dirs=multis, bestkpts=None)
            indices = np.concatenate(indices, axis=0)
            self.kpoints = self.kpoints[indices]
            self.bands = self.bands[:,indices]
            if hasattr(self,'procar'):
                self.procar = self.procar[:,indices]

            return self
                
        def remove_duplicates(self):
            '''
            remove duplicates kpoints
            '''
            duplicates = np.where(np.diff(self.get_xkpt(continuous=False))==0)
            # reset insert 
            ikpt = np.cumsum(self.kpath.insert)
            for num in duplicates[0]:
                for i,j in enumerate(ikpt):
                    if j>num:
                        self.kpath.insert[i] -= 1
                        break 
            self.kpoints = np.delete(self.kpoints,duplicates,axis=0)
            self.bands = np.delete(self.bands,duplicates,axis=1)
            if hasattr(self,'procar'):
                self.procar = np.delete(self.procar,duplicates,axis=1)
            if hasattr(self,'ovlaps'):
                self.ovlaps = np.delete(self.ovlaps,duplicates,axis=1)
            if hasattr(self,'tdms'):
                self.tdms = np.delete(self.tdms,duplicates,axis=1)
            if hasattr(self,'cpds'):
                self.cpds = np.delete(self.cpds,duplicates,axis=1)
            return self

        def remove_grid(self, skip:int):
            '''
            remove grid kpoints for hse
            '''
            self.kpoints = self.kpoints[skip:]
            self.bands = self.bands[:,skip:]
            if hasattr(self,'procar'):
                self.procar = self.procar[:,skip:]
            if hasattr(self,'ovlaps'):
                self.ovlaps = self.ovlaps[:,skip:]
            if hasattr(self,'tdms'):
                self.tdms = self.tdms[:,skip:]
            if hasattr(self,'cpds'):
                self.cpds = self.cpds[:,skip:]
            return self

        def get_dos(self, nedos=3001, sigma=0.05, kpoint_weights=None, emin=None, emax=None, smear_type='gaussian'):
            from jamip.utils.convert import spectral_function

            ispin, nkpts, nbands, _ = self.bands.shape

            e0 = None
            if emin != None and emax != None:
                e0 = np.linspace(emin, emax, nedos)

            if kpoint_weights is None:
                kpoint_weights = np.ones(nkpts)
                if hasattr(self, 'kpoint_weights'):
                    kpoint_weights = self.kpoint_weights
                 
            # ispin, nodes, nkpts
            energy, total_dos = spectral_function(self.bands[..., 0], kpoint_weight=kpoint_weights, num=nedos, sigma=sigma, e0=e0)
            return energy, np.sum(total_dos, axis=2)

        # def to_hdf5(self, h5file, key='band', **kwargs):

        #     from jamip.utils.convert import kpath2list
        #     self.remove_duplicates()
        #     kpath=kpath2list(self.kpath.kpts)
        #     xkpt = self.get_xkpt()
            
        #     # set x axis % 
        #     xticks = [0] + np.cumsum(self.kpath.insert).tolist()
        #     xticks[-1] -= 1
        #     xticks = [xkpt[i] for i in xticks]
         
        #     # add bandgap %
        #     cv = self.get_cbmvbm()
        #     if self.metal:
        #         shift = self.fermi
        #     else:
        #         shift = cv['vbm'].energy
        #     #vacuum_level = vec
            
        #     data = {'xdata': xkpt,
        #             'xticks': xticks,
        #             'xlabel': kpath,
        #             'fermi_energy': bf.fermi,
        #             'vbm_energy': shift}
        #     data.update(kwargs)
        #     with h5py.File(str(h5file), "w") as hdf:
        #         datasets[key] = hdf.create_dataset(key, data=self.bands)
        #         for k,v in data.items():
        #             d.attrs[k] = v

        #     del data

    def __init__(self,stdin=None,multi=True):
        self.task = 'band'
        self.soft = 'vasp'
        self.multi = multi
        self.stdin = stdin

    @lazy_property
    def banddirs(self):
        '''
        band task directory
        '''
        file = self.file
        if self.file == 'jamip':
            if self.soft == 'vasp':
                path = self._stdin/'electric'/'band'
                file = self.seek_entry_type(path, task='band')
            else:
                raise ValueError(f'Soft {self.soft} not supported!')
        else:
            path = self._stdin

        subpaths = []
        if file != None:
            return [path]
        elif self.multi:
            for subpath in self.seek_multi_entries(path, task='band'):
                if '-' in subpath.name:
                    subpaths.append(subpath)
        else:
            raise OSError("DIR not exists!")
        return subpaths

    @lazy_property
    def emassdirs(self):
        '''
        effective mass directory
        '''
        file = self.file
        if self.file == 'jamip':
            if self.soft == 'vasp':
                path = self._stdin/'electric'/'emass'
                file = self.seek_entry_type(path, task='band')
            else:
                raise ValueError('Soft %s not supported!' % self.soft)
        else:
            path = self._stdin

        subpaths = []
        if file != None:
            return [path]
        elif self.multi:
            subpaths = self.seek_multi_entries(path, task='band')
        else:
            raise OSError("DIR not exists!")
        return subpaths

    @lazy_property
    def emcdirs(self):
        '''
        emc potential directory
        '''
        file = self.file
        if self.file == 'jamip':
            if self.soft == 'vasp':
                path = self._stdin/'electric'/'emc'
                file = self.seek_entry_type(path, task='band')
            else:
                raise ValueError('Soft %s not supported!' % self.soft)
        else:
            path = self._stdin

        subpaths = []
        if file != None:
            return [path]
        elif self.multi:
            subpaths = self.seek_multi_entries(path, task='band')
        else:
            raise OSError("DIR not exists!")

        return subpaths

    @lazy_property
    def dpdirs(self):
        '''
        deformation potential directory
        '''
        if self.file == 'jamip':
            if self.soft == 'vasp':
                #path = self._stdin/'electric'/'deformation'
                path = self._stdin/'electric'/'deform'
            else:
                raise ValueError('Soft %s not supported!' % self.soft)
        else:
            path = self._stdin

        subpaths = []
        for subpath in self.seek_multi_entries(path, task='band'):
            if (subpath.name == '1' or re.match(r'[xyz]-\d\.\d+',subpath.name)):
                subpaths.append(subpath)

        if len(subpaths) == 0:
            raise OSError("DIR not exists!")

        return subpaths

    def get_kpath(self, banddirs=None, source='KPOINTS'):

        if banddirs == None:
            banddirs = self.banddirs
        
        if source is None:
            source = 'KPOINTS'
        elif source.upper() == 'KPOINTS_OPT' or source.upper() == 'PROCAR_OPT':
            source = 'KPOINTS_OPT'
        else:
            source = 'KPOINTS'

        try:
            if len(banddirs) == 1:
                return self.get_kpath_from_kpoints(banddirs[0], source)
                # G-X-M insert=20 -> [20, 20] -> G19 X1 X19 M1
            elif self.file == 'jamip':
                kpoints = self.get_kpath_from_info(self._stdin)
                bestkpts = kpoints['paths']
                ikpts = kpoints['numbers']
            else:
                # get kpath from subdir
                bands = [dir.name for dir in banddirs]
                bestkpts = self.get_kpath_from_dirs(bands)

                kpath = []
                ikpts = []
                for path in banddirs:
                    k1,k2 = path.name.split('-')
                    kpath.append([k1,k2])
                    ikpts.append(GrepOutcar().nkpts(path))

            return Kpath(kpts=kpath, insert=ikpts, dirs=banddirs, bestkpts=bestkpts)
        except Exception as e:
            print("Get kpath failed:", e)
            return None

    def get_bands(self, banddirs:list=None, source='EIGENVAL'):
        """
        get band-structure data from OUTCAR

        Returns:
            ndarray: with shape (spin,kpt,band,2[energy, occupation] )
        """
        if banddirs == None:
            banddirs = self.banddirs

        if source.lower() == "xml":
            bands = []
            for dir in banddirs:
                data = Xml(dir)._get_band()
                bands.append(data)
            return np.concatenate(bands, axis=0)

        elif source.lower() == "outcar":
            func = Outcar
        elif source.lower() == "eigenval":
            func = Eigenval
        elif source.lower() == "procar":
            func = Procar
            func.filename = source.upper()
        elif source.lower() == "procar_opt":
            func = Procar
            func.filename = source.upper()
        elif source.lower() == "wavecar":
            from .wavecar import Wavecar
            func = Wavecar
        else:
            raise ValueError("unknown source type!")

        bands = []
        for dir in banddirs:
            data = func.from_file(dir) 
            if source.lower() == "wavecar":
                bands.append(data.get_bands())
            else:
                bands.append(data.bands)
        bands = np.concatenate(bands, axis=1) 
        return bands

    def _get_kpoints(self, banddirs:list=None, source='OUTCAR'):
        """
        get K-points in reciprocal lattice from OUTCAR

        Returns:
            ndarray: with shape (nkpt,3)
        """
        if banddirs == None:
            banddirs = self.banddirs

        if source.lower() == "xml":
            kpoints = []
            for dir in banddirs:
                data = Xml(dir)._get_kpoint()
                kpoints.append(data)
            return np.concatenate(kpoints, axis=0)
            
        elif source.lower() == "outcar":
            func = Outcar
        elif source.lower() == "eigenval":
            func = Eigenval
        elif source.lower() == "procar":
            func = Procar
            func.filename = source.upper()
        elif source.lower() == "procar_opt":
            func = Procar
            func.filename = source.upper()
        elif source.lower() == "wavecar":
            from .wavecar import Wavecar
            func = Wavecar
        else:
            raise ValueError("unknown source type!")

        kpoints = []
        for dir in banddirs:
            data = func.from_file(dir) 
            kpoints.append(data.kpoints)
        return np.concatenate(kpoints, axis=0)

    def get_kpoints(self, banddirs:list=None, source='OUTCAR'):
        return self._get_kpoints(banddirs, source)[:,:3]

    def get_kpoint_weights(self, banddirs:list=None, source='OUTCAR'):
        return self._get_kpoints(banddirs, source)[:,3]

    def get_data(self, banddirs=None, source='EIGENVAL', opt:bool=True, **kwargs):
        """
        get Result set with base func
        """
        if banddirs == None:
            banddirs = self.banddirs

        if opt and (pathlib.Path(banddirs[0])/'PROCAR_OPT').exists():
            source = 'PROCAR_OPT'
        default = source

        fermi = self.get_fermi(banddirs, source=source)
        bands = self.get_bands(banddirs, source=source)
        kpoints = self.get_kpoints(banddirs, source=source)
        kpath = self.get_kpath(banddirs, source=source)
        rec_vector = self.get_rec_vector(banddirs, source=kwargs.get('rec_vector', source))
        if bands.shape[1] != kpoints.shape[0]:
            print("bands shape:", bands.shape)
            print("kpoints shape:", kpoints.shape)
            raise ValueError("Data shapes do not match!")

        return self.Result(bands=bands, 
                           kpoints=kpoints, 
                           fermi=fermi,
                           rec_vector=rec_vector,
                           kpath=kpath)

    def get_data_from_wavecar(self, banddirs:list=None, lsorbit=None):
        from .wavecar import Wavecar
        if banddirs == None:
            banddirs = self.banddirs
        if len(banddirs) > 1:
            raise OSError("The program does not currently support this feature.")
        if lsorbit is None and not (banddirs[0]/'OUTCAR').exists():
            lsorbit = False
        wavecar = Wavecar.from_file(banddirs[0], lsorbit=lsorbit)
        kpath = self.get_kpath(banddirs)

        return self.Result(wavecar=wavecar,
                    bands=wavecar.get_bands(),
                    kpoints=wavecar.kpoints, 
                    fermi=self.get_fermi(),
                    rec_vector=wavecar.rec_cell/2/np.pi,
                    kpath=kpath)

    def get_fermi(self, banddirs=None, source='OUTCAR', func=max):
        """
        Get the fermi_energy from OUTCAR. If multiple energy values are read,
        the return value is determined according to the function

        Returns:
            float: (eV)
        """
        if banddirs == None:
            banddirs = self.banddirs

        efermi = []
        for path in banddirs:
            if source.lower() == 'xml':
                efermi.append(Xml(path).fermi_energy())
            else: #if source.lower() in ['ourcar','eigenval','procar']:
                if (path/"OUTCAR").exists():
                    efermi.append(GrepOutcar().fermi_energy(path))

        if len(efermi):
            return func(efermi)
        return None

    def get_elements(self, banddirs=None, source='OUTCAR'):
        if banddirs == None:
            banddirs = self.banddirs

        if source.lower() == 'xml':
            return Xml(banddirs[0]).elements()
        else:
            return self.elements(banddirs[0])

    def get_rec_vector(self, banddirs=None, source='OUTCAR'):
        if banddirs == None:
            banddirs = self.banddirs

        if source.lower() == 'xml':
            return Xml(banddirs[0]).reciprocal_lattice_vectors()
        elif source.lower() == 'chgcar':
            chg = Chgcar.from_file(banddirs[0]/"CHGCAR")
            return np.linalg.inv(chg.structure.lattice)
        else:
            return self.reciprocal_lattice_vectors(banddirs[0])

    def get_ldos(self, path=None, axis='z', energy=None, nedos=301, sigma=0.05, ethr=0.001, **kwargs):
        """
        Project eigenval in real-space with PARCAR.

        Params:
            path: input path
            axis: Direction of vacuum layer (x, y, z)
        """
        import re

        directions = {'x':(1,2), 'y':(0,2), 'z':(0,1)}
        def gaussian_smearing(e, e0):
            return (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-((e - e0)**2) / (2.0 * sigma**2))
        ethr_cutoff = ethr / (sigma * np.sqrt(2 * np.pi)) 


        bf = BandFinder(path) if path != None else self
        assert len(bf.banddirs) == 1, "It maybe work, but who care?"
        bfd = bf.get_data(**kwargs)
        bfd.kpoint_weights = bf.get_kpoint_weights(source='eigenval')
        parchgs = Chgcar.load_all_parchg(path=bf.banddirs[0]) 
        pattern = re.compile(r'^PARCHG\.(\d{4})\.(\d{4})$')

        # set volume
        for path,chg in parchgs.items():
            # unit Angtoa0 > Bohr 
            volume = chg.structure.volume * 1.8897261246257702**3
            break

        # set energy
        if energy is None:
            energys = []
            for path,chg in parchgs.items():
                match = pattern.match(path)
                if match:
                    ib, ik = match.groups()
                    ib, ik = int(ib)-1, int(ik)-1
                    energys.append(bfd.bands[0,ik,ib,0])
            emin = min(energys) - 5*sigma
            emax = max(energys) + 5*sigma
            energy = np.linspace(emin, emax, nedos)

        # shape: (ng3,nedos)
        ldos = []
        for path,chg in parchgs.items():
            match = pattern.match(path) 
            if match:
                ib, ik = match.groups()
                ib, ik = int(ib)-1, int(ik)-1
                # load charge = chg.chgcar
                charge_avg = chg.chgcar.mean(directions[axis])
                #smear = gaussian_smearing(energy, bfd.bands[0,ik,ib,0])
                smear = gaussian_smearing(energy, np.mean(bfd.bands[:,ik,ib,0]))
                
                # apply cutoff
                smear = np.where(smear > ethr_cutoff, smear, 0)
                ldos_avg = smear[:,None] * bfd.kpoint_weights[ik] * charge_avg / volume
                #ldos.append(smear*bfd.bands[0,ik,ib,1])
                ldos.append(ldos_avg)

        return energy, ldos

    def get_locpot(self, path=None, axis='z'):
        """
        Get the vacuum level from LOCPOT, applicable only to 2-d materials containing 
        a vacuum layer, the edge value is taken as the vacuum energy level  

        Params:
            path: input path
            axis: Direction of vacuum layer (x, y, z)

        Returns:
            float: (eV)
        """
        bf = BandFinder(path) if path != None else self
        if bf.file == 'jamip' and bf.scfdir != None:
            stdin = bf.scfdir
        elif isinstance(bf.file, str):
            stdin = bf.stdin
        else:
            stdin = bf.banddirs[0]
        locpot = self.locpot(stdin,axis)
        vacuum_level = max(locpot)
        return vacuum_level 

    def get_locpot_3d(self, path=None, axis='z', point='bottom'):
        """
        Get the vacuum energy level of the 3D material from LOCPOT,
        based on the average of the inflection points

        Params:
            path: input path
            axis: Direction of vacuum layer (x, y, z)

        Returns:
            float: (eV)
        """
        bf = BandFinder(path) if path != None else self
        if bf.file == 'jamip' and bf.scfdir != None:
            stdin = bf.scfdir
        elif isinstance(bf.file, str):
            stdin = bf.stdin
        else:
            stdin = bf.banddirs[0]
        locpot = self.locpot(stdin,axis)
        diff1 = locpot - np.roll(locpot, 1)
        diff2 = np.roll(locpot, -1) - locpot
        if point == 'bottom': 
            indices = np.where((diff1<=0) & (diff2>0))
        elif point == 'top': 
            indices = np.where((diff1>=0) & (diff2<0))

        vacuum_level = np.mean(locpot[indices])
        return vacuum_level 

    def get_locpot_with_element(self, path=None, axis='z'):
        """
        Get the vacuum energy level of the 3D material from LOCPOT,
        based on the average of the inflection points

        Params:
            path: input path
            axis: Direction of vacuum layer (x, y, z)

        Returns:
            float: (eV)
        """
        from jamip.structure import read

        bf = BandFinder(path) if path != None else self
        if bf.file == 'jamip' and bf.scfdir != None:
            stdin = bf.scfdir
        elif isinstance(bf.file, str):
            stdin = bf.stdin
        else:
            stdin = bf.banddirs[0]
        locpot = self.locpot(stdin,axis)
        s = read(stdin / "POSCAR")
        df = get_boundary(s)
 
        vacuum_level = {}
        for row in df.itertuples():
            x,y,b = get_partial_locpot(locpot, row.start, row.stop, row.spacing)
            vacuum_level[row.specie] = b

        return vacuum_level 

    def get_deformation_potential_data(self, path=None, reference='vacuum', core=None, vacuum='z'):
        """
        Deformation potential. 
        Params:
            path: dp calculation stdin. 
            core: Band alignment using core levels,  not vacuum levels
                  for example, core=('Si', '1s')

        Returns:
	    dict: {'dpv': {'index':(2,), 'energy': float, 'kpoint':(3,)},
	           'dpc': {'index':(2,), 'energy': float, 'kpoint':(3,)}}  (allspin is False)
        """
        import pandas as pd
        if path != None:
            dpdirs = BandFinder(path).dpdirs
        else:
            dpdirs = self.dpdirs
        if len(dpdirs) < 2:
            raise OSError('Invalid dp path!')
        
        axes = {'all': [True, True, True], 
                'x': [True, False, False], 
                'y': [False, True, False], 
                'z': [False, False, True]}
        vacuum_axis = vacuum

        datas = []
        for path in dpdirs:

            if re.match(r"[xyz]-\d\.\d+", path.name):
                axis = axes[path.name.split('-')[0]]
                scale = float(path.name.split('-')[1])
            else:
                axis = axes['all']
                scale = 1

            bf = BandFinder(path)
            cbmvbm = bf.get_data().get_metal_cbmvbm() # former:.get_cbmvbm
            Ecbm = cbmvbm['empty']['cbm'].energy #'cbm'
            Evbm = cbmvbm['empty']['vbm'].energy #'vbm'
            Efree = bf.get_info('free_energy')
            if reference == 'core_state':
                elements = bf.get_info('elements')
                corestate = bf.get_info('core_state')
                values = []
                if core != None:
                    for i,j in zip(elements,corestate):
                         if core[0] == i and core[1] in j:
                             values.append(j[core[1]])
                else:
                    for i,j in zip(elements,corestate):
                         if '1s' in j:
                             values.append(j['1s'])
                if len(values) == 0: 
                    raise ValueError("Find zero input core-state: %s" %core)
                vacuum = np.mean(values) 
            elif reference == 'vacuum':
                if isinstance(vacuum_axis, str):
                    vacuum = bf.get_locpot(path)
                else:
                    vacuum = bf.get_locpot_3d(path)
            else:
                raise ValueError("unknown reference.")

            data = [path.name, Ecbm, Evbm, Efree, vacuum, scale]
            data.extend(axis)
            datas.append(data)
        
        df = pd.DataFrame(datas, columns=['path','Ecbm','Evbm','Efree','vacuum','scale','x','y','z'])
        return df
         
    def get_deformation_potential(self, path=None, reference='vacuum', core=None, vacuum='z'):
        # DP unit(eV)
        from scipy.optimize import curve_fit

        df = self.get_deformation_potential_data(path, reference=reference, core=core, vacuum=vacuum)
        line = lambda x,a,b:a*x + b
        dps = {}
        for axis in 'xyz':
            if df[axis].sum() > 1:
                group = df[df[axis]]
                scale = group['scale']
                Ecbm = group['Ecbm'] - group['vacuum']
                Evbm = group['Evbm'] - group['vacuum']
                deform_c,b = curve_fit(line,scale,Ecbm)[0]
                deform_v,b = curve_fit(line,scale,Evbm)[0]
                dps['cbm-%s' %axis] = deform_c
                dps['vbm-%s' %axis] = deform_v

        return dps

    def get_emc_mass(self, path=None):
        """
        Extract effective mass by band fitting method. 
        Params:
            path: band calculation stdin. 

        Returns:
	    dict: {'vbm': float, 'cbm': float}, for single path, OR
	           'vbm-x': float, 'vbm-y': float, 'vbm-z': float, ...} for jamip path
        """
        if path != None:
            emcdirs = BandFinder(path).emcdirs
        else:
            emcdirs = self.emcdirs

        results = {}
        for path in emcdirs:
            bf = BandFinder(path).get_data()
            cbmvbm = bf.get_cbmvbm()
         
            cb = bf.bands[0,:,cbmvbm['cbm'].iband,0] 
            vb = bf.bands[0,:,cbmvbm['vbm'].iband,0] 
            kpts = (bf.kpoints[0] - bf.kpoints[1]) @ bf.rec_vector
            stepsize = max(abs(kpts))*2*np.pi
         
            m = Emc().fd_effmass(vb, stepsize)
            hmass,hvec = Emc().get_eff_masses(m)
            #print(np.around(hvec, 4))
         
            m = Emc().fd_effmass(cb, stepsize)
            emass,evec = Emc().get_eff_masses(m)
            #print(np.around(evec, 4))
         
            if re.match(r"[A-z](-[cv]bm)+", path.name):
                bd = path.name.split('-')[1:]
            else:
                bd = ['cbm','vbm']

            for i,j in enumerate('xyz'):
                if 'vbm' in bd:
                    results['vbm-'+j] = hmass[i] * -1
                if 'cbm' in bd:
                    results['cbm-'+j] = emass[i] 

        return results

    def get_emass(self, path=None, **kwargs):
        """
        Extract effective mass by band fitting method. 
        Params:
            path: band calculation stdin. 
            fit_range: The number of points used in fitting, (3 means three points on each side, total seven points. )  

        Returns:
	    dict: {'vbm': float, 'cbm': float}, for single path, OR
	           'vbm-x': float, 'vbm-y': float, 'vbm-z': float, ...} for jamip path
        """
        if path != None:
            emassdirs = BandFinder(path).emassdirs
        else:
            emassdirs = self.emassdirs
            
        if len(emassdirs) == 0:
            raise IOError("Emass calculation files not exists!")
        if len(emassdirs) == 1:
            bf = BandFinder(emassdirs[0]).get_data()
            emass = bf.get_emass(**kwargs)
            return emass
        else:        
            emass = {} 
            for path in emassdirs:
                if path.name == 'scf': continue
                if not (path/'OUTCAR').exists(): continue
                bf = BandFinder(path).get_data()                         
                # axis: x,y,z ; bd: [cbm,vbm]
                if re.match(r"[A-z](-[cv]bm)+", path.name):
                    axis = path.name.split('-')[0]
                    bd = path.name.split('-')[1:]
                else:
                    axis = path.name
                    bd = ['cbm','vbm']
                for key,value in bf.get_emass(axis,bd,**kwargs).items():
                    emass[key+'-'+axis] = value

            if len(emass) == 0:
                raise IOError("Emass calculation files not exists!")
            return emass

    def get_mobility_2d(self, emassdir=None, dpdir=None, elasticdir=None, axes=['x','y'], vacuum=None, core=None, T=300, 
                        masstype='fitting', elastictype='fitting', dptype='fitting', reference='vacuum'):

        """ DOI: 10.1007/978-3-642-25076-7 """
        from jamip.structure import read
        from scipy.optimize import curve_fit
        import scipy.constants as sc
        from scipy.stats.mstats import gmean
        import pandas as pd
    
        # Units %
        # Axis: x,y,z
        # T(temperature): K
        # dp(deformation potential): eV
        # m*(effective mass): me
        # mu(carrier mobility): cm^2/(V*s)
        # stretch : J/(1^2*m^2)
    
        line = lambda x,a,b:a*x + b
        mod = lambda x,a,b,c:a*x**2 + b*x + c

        assert len(axes) == 2, "2d mobility calcation."       
        if vacuum == None:
            vacuum = list(set('xyz') - set(['x','y']))[0]

        # data from dp-1 %
        if dpdir == None:
            dpdirs = self.dpdirs
        else:
            dpdirs = BandFinder(dpdir).dpdirs
        for path in dpdirs:
            if path.name == '1':
                scfdir = path

        # read structure 
        structure = read(scfdir / 'POSCAR')
        sectional_area = np.linalg.norm(np.cross(structure.lattice[0], structure.lattice[1]))
       
        # get emass data
        emass = None
        if masstype == 'fitting':
            emass = {}
            for key,value in self.get_emass(path=emassdir).items():
                emass[key] = value.mass
        elif masstype.lower() == 'boltztrap':
            from .boltztrap import Boltztrap
            if emassdir != None:
                emass = Boltztrap().get_effective_mass_dict(emassdir)
            elif self.file == 'jamip' and self.soft == 'vasp':
                path = self._stdin/'electric'/'boltztrap'
                if path.exists():
                    emass = Boltztrap().get_effective_mass_dict(path)
        if emass == None:
            raise ValueError("Get effective mass failed!")

        # 2d mean effective mass
        mc = [1/emass['cbm-'+i] for i in axes]
        mv = [1/emass['vbm-'+i] for i in axes]
        #md_c = gmean(mc)*sc.electron_mass
        #md_v = gmean(mv)*sc.electron_mass
        md_c = 1/np.mean(mc)*sc.electron_mass
        md_v = 1/np.mean(mv)*sc.electron_mass

        # data from deformation
        df = self.get_deformation_potential_data(path=dpdir, reference=reference, vacuum=vacuum, core=core)

        # get elastic moduli 
        elastic = {}
        if elastictype == 'fitting':
            for axis in axes:
                if df[axis].sum() > 1:
                    group = df[df[axis]]
                    scale = group['scale']
                    Efree = group['Efree']
                    stretch,b,c = curve_fit(mod,scale,Efree)[0]
                    elastic[axis] = stretch*2*sc.e/(sectional_area*sc.angstrom**2)
            # unit eV/Ang^2 -> J / m^2 
            # print(elastic)
        else:
            from jamip.analysis.vasp import GrepOutcar
            if elasticdir != None:
                data = GrepOutcar().elastic(elasticdir)
            elif self.file == 'jamip' and self.soft == 'vasp':
                path = self._stdin/'mechanic'/'elastic'
                if path.exists():
                    data = GrepOutcar().elastic(path)
            print(data)
            # unit kBar -> J/m^2
            a,b,c = structure.lattice_parameters[:3]
            elastic['x'] = data[0,0]*1e8 * 1e-10 * a
            elastic['y'] = data[1,1]*1e8 * 1e-10 * b
            elastic['z'] = data[2,2]*1e8 * 1e-10 * c
            # print(elastic)

        # get deformation potential 
        dp = {}
        if dptype == 'fitting':
            for axis in axes:
                if df[axis].sum() > 1:
                    group = df[df[axis]]
                    scale = group['scale']
                    Ecbm = group['Ecbm'] - group['vacuum']
                    Evbm = group['Evbm'] - group['vacuum']
                    deform_c,b = curve_fit(line,scale,Ecbm)[0]
                    deform_v,b = curve_fit(line,scale,Evbm)[0]
                    dp['cbm-%s' %axis] = deform_c
                    dp['vbm-%s' %axis] = deform_v
        else:
            from .amset import Amset
            amset = Amset(self.stdin)
            dpc, dpv = amset.get_deformation_potential(dpdir)

            for i,axis in enumerate('xyz'):
                if axis in axes:
                    dp['cbm-%s' %axis] = dpc[i,i]
                    dp['vbm-%s' %axis] = dpv[i,i]

        # get mobility
        datas = []
        for axis in axes:
            if df[axis].sum() > 1:
                stretch = elastic[axis]
                # unit ev -> J
                dpc = dp['cbm-%s' %axis]*sc.e
                dpv = dp['vbm-%s' %axis]*sc.e
                # unit m0 -> kg
                m_c = emass['cbm-%s' %axis]*sc.electron_mass
                m_v = emass['vbm-%s' %axis]*sc.electron_mass
                mu_c = sc.e*sc.hbar**3*stretch/(sc.k*T*m_c*md_c*dpc**2)
                mu_v = sc.e*sc.hbar**3*stretch/(sc.k*T*m_v*md_v*dpv**2)
                datas.append([axis, dp['cbm-%s' %axis], dp['vbm-%s' %axis], stretch,
                             emass['cbm-%s' %axis], emass['vbm-%s' %axis], mu_c*1e4, mu_v*1e4])
        
            else:
                raise OSError('axis %s not calculated!' %axis)
       
        df = pd.DataFrame(datas, columns=['axis','dp-cbm','dp-vbm','stretch',
                                          'm*-cbm','m*-vbm','mu-cbm','mu-vbm'])
        return df


    def get_mobility_3d(self, emassdir=None, dpdir=None, elasticdir=None, axes=['x','y','z'], vacuum=None, core=None, T=300, 
                        masstype='fitting', elastictype='fitting', dptype='fitting', reference='core_state'):

        """ DOI: 10.1007/978-3-642-25076-7 """
        from jamip.structure import read
        from scipy.optimize import curve_fit
        from scipy.stats.mstats import gmean
        import scipy.constants as sc
        import pandas as pd
    
        # Units %
        # Axis: x,y,z
        # T(temperature): K
        # dp(deformation potential): eV
        # m*(effective mass): me
        # mu(carrier mobility): cm^2/(V*s)
        # stretch : J/(1^2*m^2)
    
        line = lambda x,a,b:a*x + b
        mod = lambda x,a,b,c:a*x**2 + b*x + c

        # data from dp-1 %
        if dpdir == None:
            dpdirs = self.dpdirs
        else:
            dpdirs = BandFinder(dpdir).dpdirs
        for path in dpdirs:
            if path.name == '1':
                scfdir = path

        # read structure 
        volume = self.volume(scfdir) 
       
        # data from emass
        emass = None
        if masstype == 'fitting':
            emass = {}
            for key,value in self.get_emass(path=emassdir).items():
                emass[key] = value.mass
        elif masstype.lower() == 'boltztrap':
            from .boltztrap import Boltztrap
            if emassdir != None:
                emass = Boltztrap().get_effective_mass_dict(emassdir)
            elif self.file == 'jamip' and self.soft == 'vasp':
                path = self._stdin/'electric'/'boltztrap'
                if path.exists():
                    emass = Boltztrap().get_effective_mass_dict(path)
        if emass == None:
            raise ValueError("Get effective mass failed!")

        # 2d mean effective mass
        mc = [1/emass['cbm-'+i] for i in 'xyz']
        mv = [1/emass['vbm-'+i] for i in 'xyz']
        #md_c = gmean(mc)*sc.electron_mass
        #md_v = gmean(mv)*sc.electron_mass
        md_c = 1/np.mean(mc)*sc.electron_mass
        md_v = 1/np.mean(mv)*sc.electron_mass

        # data from deformation
        df = self.get_deformation_potential_data(path=dpdir, reference=reference, vacuum=vacuum, core=core)
       
        # get elastic moduli 
        elastic = {}
        if elastictype == 'fitting':
            for axis in axes:
                if df[axis].sum() > 1:
                    group = df[df[axis]]
                    scale = group['scale']
                    Efree = group['Efree']
                    stretch,b,c = curve_fit(mod,scale,Efree)[0]
                    elastic[axis] = stretch*2*sc.e/(volume*sc.angstrom**3)
                    #elastic[axis] = stretch*2*sc.e/(sectional_area*sc.angstrom**2)
        else:
            from jamip.analysis.vasp import GrepOutcar
            if elasticdir != None:
                data = GrepOutcar().elastic(elasticdir)
            elif self.file == 'jamip' and self.soft == 'vasp':
                elasticdir = self._stdin/'mechanic'/'elastic'
                if elasticdir.exists():
                    data = GrepOutcar().elastic(elasticdir)
                else:
                    raise OSError("Find elastic-dir failed. Try input it manually.")
            # unit kBar -> J/m^3
            elastic['x'] = data[0,0]*1e8
            elastic['y'] = data[1,1]*1e8
            elastic['z'] = data[2,2]*1e8

        # get deformation potential 
        dp = {}
        if dptype == 'fitting':
            for axis in axes:
                if df[axis].sum() > 1:
                    group = df[df[axis]]
                    scale = group['scale'].values
                    Ecbm = group['Ecbm'].values - group['vacuum'].values
                    Evbm = group['Evbm'].values - group['vacuum'].values
                    deform_c,b1 = curve_fit(line,scale,Ecbm)[0]
                    deform_v,b2 = curve_fit(line,scale,Evbm)[0]
                    # check %
                    diff = deform_c * scale + b1 - Ecbm
                    if np.mean(diff**2) > 1e-4:
                        print('Warning! dp-%s-cbm-rmse = %f' %(axis, np.max(np.abs(diff))))
                    diff = deform_v * scale + b2 - Evbm
                    if np.mean(diff**2) > 1e-4:
                        print('Warning! dp-%s-vbm-rmse = %f' %(axis, np.max(np.abs(diff))))
                    dp['cbm-%s' %axis] = deform_c
                    dp['vbm-%s' %axis] = deform_v
        else:
            from .amset import Amset
            amset = Amset(self.stdin)
            dpc, dpv = amset.get_deformation_potential(dpdir)
            #print(dpc, dpv)

            for i,axis in enumerate('xyz'):
                dp['cbm-%s' %axis] = dpc[i,i]
                dp['vbm-%s' %axis] = dpv[i,i]


        # get mobility
        datas = []
        for axis in axes:
            stretch = elastic[axis]
            # unit ev -> J
            dpc = dp['cbm-%s' %axis]*sc.e
            dpv = dp['vbm-%s' %axis]*sc.e
            # unit m0 -> kg
            m_c = emass['cbm-%s' %axis]*sc.electron_mass
            m_v = emass['vbm-%s' %axis]*sc.electron_mass
            mu_c = (2*np.sqrt(2*np.pi)*sc.e*sc.hbar**4*stretch) / (3*(sc.k*T)**(3/2)*(md_c)**(5/2)*dpc**2)
            mu_v = (2*np.sqrt(2*np.pi)*sc.e*sc.hbar**4*stretch) / (3*(sc.k*T)**(3/2)*(md_v)**(5/2)*dpv**2)
            datas.append([axis, dp['cbm-%s' %axis], dp['vbm-%s' %axis], stretch,
                         emass['cbm-%s' %axis], emass['vbm-%s' %axis], mu_c*1e4, mu_v*1e4])
       
        df = pd.DataFrame(datas, columns=['axis','dp-cbm','dp-vbm','stretch',
                                      'm*-cbm','m*-vbm','mu-cbm','mu-vbm'])
        return df
    
    def get_insert(self):
        """
        Get nkpts from KPOINTS or band_split
        """
        banddirs = self.banddirs
        # multi %
        if len(banddirs) > 1:
            func = getattr(GrepOutcar(),'nkpts')
            return func(banddirs[0])
        else:
            return self.get_kpath_from_kpoints(banddirs[0]).insert[0]

    def get_info(self,func):
        path = self.banddirs[0]
        func = getattr(GrepOutcar(),func)
        return func(path)

class ProFinder(BandFinder):

    class Result(BandFinder.Result):

        def __init__(self, bands:np.ndarray, kpoints:np.ndarray, fermi:float, 
                     procar:np.ndarray, elements:np.ndarray, **kwargs):
            self.procar = procar
            super().__init__(bands,kpoints,fermi,**kwargs)
            self.elements = elements
            assert procar.shape[4] in (10,17), 'procar.shape[4] must be 9(spd) or 16(spdf), now is %d' %(procar.shape[4]-1)

        @property
        def spin(self):
            return len(self.procar)
             
        def total(self):
            return self.procar[...,-1,-1]

        def get_proj(self, value):
            dorbits = ('s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2')
            forbits = ('s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2',
                       'fy3x2', 'fxyz', 'fyz2', 'fz3', 'fxz2', 'fzx2', 'fx3')
            allorbits = dorbits if self.procar.shape[4] == 10 else forbits

            if isinstance(value, str):
                projection = []
                for atom in np.unique(self.elements):
                    atom_indices = np.where(self.elements==atom)[0].tolist()
                    if value == 'emax':
                        projection.append(Projection(atoms=atom_indices, orbits=range(len(allorbits)), label=atom))
                    elif value == 'lmax':
                        projection.append(Projection(atoms=atom_indices, orbits=(0,), label='%s-s' %atom))
                        projection.append(Projection(atoms=atom_indices, orbits=(1,2,3), label='%s-p' %atom))
                        projection.append(Projection(atoms=atom_indices, orbits=(4,5,6,7,8), label='%s-d' %atom))
                        if len(allorbits) == 16:
                            projection.append(Projection(atoms=atom_indices, orbits=(9,10,11,12,13,14,15), label='%s-f' %atom))
                    elif value == 'mmax':
                        for i,orbit in enumerate(allorbits):
                            projection.append(Projection(atoms=atom_indices, orbits=(i,), label='%s-%s' %(atom,orbit)))
                    else:
                        raise ValueError("Unknown projection type.")
            else:
                projection = []
                proj = Proj(elements=self.elements, allorbits=allorbits)
                for val in value:
                    projection.append(proj.to_proj(*val))

            return projection

        def projection(self, proj):
            """
            proj: list of Projection or keywords (emax/lmax/mmax)
            # self.data [ispin, nkpt, nband, natom, norbit] -> [ispin, nkpt, nband, nproj]
            """
            procar = self.procar 
            data = []
            labels = []
            # info = np.zeros(procar.shape[:3]).astype(int)
            # value = np.zeros(procar.shape[:3])
            proj = self.get_proj(proj)
            for p in proj:
                # print(p.orbits, p.atoms)
                tmp = procar[...,p.orbits]
                data.append(np.sum(tmp[...,p.atoms,:],axis=(3,4)))
                labels.append(p.label)
            data = np.stack(data,axis=3)
            return data, labels

        def single_point(self, proj=None, ikpt:int=0, nedos:int=601, sigma:float=0.02):
            from jamip.utils.convert import spectral_function
            procar = self.procar[:,(ikpt,),...] #/self.total().max()
            energy = self.bands[:,(ikpt,),:,0]

            data = []
            labels = []
            proj = self.get_proj(proj)
            for p in proj:
                tmp = procar[...,p.orbits]
                dos = np.sum(tmp[...,p.atoms,:],axis=(-2,-1))                
                e0,sf = spectral_function(energy, dos, num=nedos, sigma=sigma)
                data.append(sf)
                labels.append(p.label)
            data = np.stack(data,axis=3)

            return e0,data,labels 

        def get_jdos(self, cutoff=6, sigma=0.02, nedos=3001, kpoint_weights=None, atoms=None, smear_type='gaussian'):
            # replace energy level by transition energies
            from jamip.utils.convert import spectral_function

            nspin, nkpts, nbands, _ = self.bands.shape
            if kpoint_weights is None:
                kpoint_weights = np.ones(nkpts)
                if hasattr(self, 'kpoint_weights'):
                    kpoint_weights = self.kpoint_weights

            transitions = []
            states = []
            for ispin in range(nspin):
                for ik in range(nkpts):
 
                    # split cb & vb by occupy
                    occupy = self.bands[ispin,ik,:,1]
                    vb_energies = self.bands[ispin,ik,:,0][occupy >= 0.5]
                    cb_energies = self.bands[ispin,ik,:,0][occupy < 0.5]
                    # bands, natom, norbit
                    vb_proj = self.procar[ispin, ik, :][occupy >= 0.5]
                    cb_proj = self.procar[ispin, ik, :][occupy < 0.5]
 
                    if len(vb_energies) > 0 and len(vb_energies) > 0:
 
                        transition_energies = cb_energies[:,None] - vb_energies[None,:]
 
                        if atoms is None:
                            vb_projections = vb_proj[:,-1,-1]
                            #vb_projections = np.sum(vb_proj[:,:-1,:-1],axis=(1,2))
                            cb_projections = cb_proj[:,-1,-1]
                            #if ispin == 0 and ik == 0:
                            #    raise ValueError(vb_projections)
                        else:
                            vb_projections = np.sum(vb_proj[:,atoms,-1],axis=1)
                            cb_projections = np.sum(cb_proj[:,atoms,-1],axis=1)
 
                        state_weights = cb_projections[:,None] * vb_projections[None,:]
 
                        # apply cutoff
                        indices = np.where(transition_energies<cutoff)
                        transitions.append(transition_energies[indices])
                        states.append(state_weights[indices])

                
                numbers = [len(data) for data in transitions]
                nmax = max(numbers)
                energies = np.zeros((nspin, nkpts, nmax))
                weights = np.zeros((nspin, nkpts, nmax))

                for i in range(nspin):
                    for j in range(nkpts):
                        idx = i*nkpts + j
                        energies[i,j,:numbers[idx]] = transitions[idx]
                        weights[i,j,:numbers[idx]] = states[idx]
                        

                e0 = np.linspace(0, cutoff, nedos)
                e0, total_dos = spectral_function(energies, weight=weights, num=nedos, sigma=sigma, e0=e0, kpoint_weight=kpoint_weights, smear_type=smear_type)

                #from scipy.integrate import cumtrapz
                #integrated_dos = cumtrapz(total_dos, energy_grid, initial=0)

                return e0, np.sum(total_dos, axis=2)

        def get_interpolation(self, procar=None,interpolation:float=1):
            from scipy.interpolate import interp1d
            xkpt = self.get_xkpt()
            # set x axis % 
            xticks = [0] + np.cumsum(self.kpath.insert).tolist()
            xticks[-1] -= 1
            min_step = (xkpt[-1] - xkpt[0]) / len(xkpt) / interpolation

            if procar is None:
                procar = self.procar

            xkpts = []
            bands = []
            procars = []
            for i in range(len(xticks)-1):
                l1,l2 = xticks[i],xticks[i+1]
                pxkpt = xkpt[l1:l2]
                num_step = int((pxkpt[-1] - pxkpt[0]) / min_step) + 1
                if num_step > l2 - l1:
                    f1 = interp1d(pxkpt, self.bands[:,l1:l2,:], kind='linear', axis=1)
                    f2 = interp1d(pxkpt, procar[:,l1:l2,:], kind='linear', axis=1)
                    nxkpt = np.linspace(pxkpt[0], pxkpt[-1], num_step)
                    xkpts.append(nxkpt)
                    bands.append(f1(nxkpt))
                    procars.append(f2(nxkpt))
                else:
                    xkpts.append(xkpt[l1:l2])
                    bands.append(self.bands[:,l1:l2,:])
                    procars.append(procar[:,l1:l2,:])

            nxkpt = np.concatenate(xkpts)
            bands = np.concatenate(bands, axis=1)
            procars = np.concatenate(procars, axis=1)
            return nxkpt, bands, procars

    def __init__(self, stdin=None, multi=True):
        self.task = 'band'
        self.soft = 'vasp'
        self.multi = multi
        self.stdin = stdin
        self.source = 'PROCAR'

    def get_procar(self, banddirs=None, source='PROCAR'):
        if banddirs == None:
            banddirs = self.banddirs

        func = Procar
        func.filename = source.upper()

        procars = []
        for dir in banddirs:
            data = func.from_file(dir)
            procars.append(data._get_procar())
        procars = np.concatenate(procars, axis=1)

        return procars

    def get_data(self, banddirs=None, source='PROCAR', opt:bool=True, **kwargs):
        """
        get Result set with base func
        """

        if opt:
            infopath = pathlib.Path(banddirs[0]) if banddirs != None else pathlib.Path(self.banddirs[0])
            if (infopath/'PROCAR_OPT').exists():
                source = 'PROCAR_OPT'

        fermi = self.get_fermi(banddirs, source=source)
        bands = self.get_bands(banddirs, source=source)
        procar = self.get_procar(banddirs, source=source)
        kpoints = self.get_kpoints(banddirs, source=source)
        elements = self.get_elements(banddirs, source=source)
        rec_vector = self.get_rec_vector(banddirs, source=source)
        kpath = self.get_kpath(banddirs, source=source)
        if bands.shape[1] != kpoints.shape[0]:
            print("bands shape:", bands.shape)
            print("parcar shape:", procar.shape)
            print("kpoints shape:", kpoints.shape)
            raise ValueError("Data shapes do not match!")

        return self.Result(bands=bands, 
                           kpoints=kpoints, 
                           procar=procar,
                           fermi=fermi,                      
                           elements=elements,
                           rec_vector=rec_vector,
                           kpath=kpath)

class GaussianBroadening:
    """Gaussian Broadening Mapper"""

    def __init__(self, sigma=0.1, cutoff=5.0, **kwargs):
        """
        Parameters:
        sigma: Standard deviation of Gaussian broadening
        cutoff: Cutoff distance (in units of sigma)
        """
        self.sigma = sigma
        self.cutoff = cutoff
        self.cutoff_width = cutoff * sigma

    def broaden_points(self, x_source, y_source, x_target):
        """
        Broaden source points (x_source, y_source) to target x_target

        Parameters:
        x_source: Source x coordinates [n,]
        y_source: Source y values [n,]
        x_target: Target x coordinates [m,]
        """
    
        x_source = np.asarray(x_source)
        y_source = np.asarray(y_source)
        x_target = np.asarray(x_target).reshape(-1, 1)
        
        # calculate distance matrix
        distances = x_target - x_source  # [m, n]
        
        # apply cutoff
        mask = np.abs(distances) <= self.cutoff_width
        
        # calculate gaussian weights
        weights = np.zeros_like(distances)
        weights[mask] = np.exp(-0.5 * (distances[mask] / self.sigma) ** 2)
        
        # calculate broaden
        y_broadened = y_source * weights
        
        return y_broadened.T        
