"""
from xyyang
"""
import pathlib 
import numpy as np
import pandas as pd
import scipy.constants as sc
from numba import njit  # type: ignore

Hartree = sc.physical_constants['Hartree energy in eV'][0]
Bohr = sc.physical_constants['Bohr radius'][0]
esu_pmV = 41.911148  # esu -> pm/V

@njit
def precess_bands(vmatrix, bands, occupy, w, kw, scissor, broadening, pgrid, cb, nbands, a, b, c):
    '''
    vmatrix: moment matrix 
    i: conduction band index
    j: valence band index
    '''
    chi_inter2 = np.zeros(pgrid, dtype=np.complex128)
    chi_intra2 = np.zeros(pgrid, dtype=np.complex128)
    chi_modul2 = np.zeros(pgrid, dtype=np.complex128)

    def clip(energy, e_min=1e-5):
        if abs(energy) < e_min:
            if abs(energy) < e_min**2:
                energy = 1
            else:
                energy = e_min**2/energy
        return energy

    for i in range(cb):               # valence bands
        for j in range(cb, nbands):   # conduction bands

            if scissor > 1e-3: 
                eji = bands[j]-bands[i] 
                vmatrix[:,j,i] = vmatrix[:,j,i] * (eji+scissor)/eji

            vji = vmatrix[:,j,i]
            dji = vmatrix[:,j,j] - vmatrix[:,i,i]
            eji = bands[j] - bands[i] + scissor
            
            inter = np.zeros(3, dtype=np.complex128)
            intra = np.zeros(3, dtype=np.complex128)
            modul = np.zeros(3, dtype=np.complex128)
            
            energy = eji**4
            intra[1] = np.conj(vji[a])*0.5*(dji[b]*vji[c]+vji[b]*dji[c])/(energy)*8.0*kw 
            modul[2] = vji[a]*0.5*(vji[b]*dji[c]+dji[b]*vji[c])/(energy)*0.5*kw 
            
            chi_intra2 += intra[1] / (eji-2*w+broadening)
            chi_modul2 -= modul[2] / (eji-w+broadening)
            
            for k in range(nbands):
            
                eki = bands[k] - bands[i]
                ekj = bands[k] - bands[j]

                if occupy[k] == 0.0:
                    eki = eki + scissor
                else:
                    ekj = ekj - scissor

                if k == i or k == j: continue
                if abs(eki) < 1e-8: continue
                if abs(ekj) < 1e-8: continue
                if abs(eki+ekj) < 1e-8: continue
                if abs(eki+eji) < 1e-8: continue
                if abs(ekj-eji) < 1e-8: continue
#                print(i,j,k,eki,ekj,eji)

                vik = vmatrix[:,i,k]
                vjk = vmatrix[:,j,k]
                vki = vmatrix[:,k,i]
                vkj = vmatrix[:,k,j]
            
                # interband
                energy = -eji * -ekj * eki * (eki+ekj)
                energy = clip(energy)
                inter[0] = np.conj(vji[a])*0.5*(np.conj(vkj[b])*np.conj(vik[c]) + np.conj(vik[b])*np.conj(vkj[c]))/energy
                
                energy = ekj * eji * -eki * -(eki+eji)
                energy = clip(energy)
                inter[1] = vkj[c]*0.5*(vji[a]*vik[b] + vik[a]*vji[b])/energy 
            
                energy = -eki * ekj * eji * (ekj-eji)
                energy = clip(energy)
                inter[2] = vik[b]*0.5*(vkj[c]*vji[a] + vji[c]*vkj[a])/energy 
            
                # intraband
                energy = -eki * ekj * eji**3
                energy = clip(energy)
                intra[0] = (eki*vik[b]*0.5*(vkj[c]*vji[a]+vji[c]*vkj[a]) 
                         + ekj*vkj[c]*0.5*(vji[a]*vik[b]+vik[a]*vji[b])) / energy
                
                energy = (eki * -ekj * eji**3) / (eki+ekj)
                energy = clip(energy)
                intra[2] = np.conj(vji[a])*0.5*(np.conj(vkj[b])*np.conj(vik[c]) + np.conj(vik[b])*np.conj(vkj[c]))/energy 
            
                # modulation
                energy = ekj*(-eki)*eji**3
                energy = clip(energy)
                modul[0] = (-eki)*vkj[a]*0.5*(vji[b]*vik[c] + vik[b]*vji[c])/energy
                
                energy = -eki*ekj*eji**3  
                energy = clip(energy)
                modul[1] = ekj*vik[a]*0.5*(vkj[b]*vji[c]+vji[b]*vkj[c])/energy
            
                inter[0] *= kw * 2.0
                inter[1] *= kw 
                inter[2] *= kw 
                intra[0] *= kw 
                intra[2] *= kw * 2.0
                modul[0] *= kw * 0.5
                modul[1] *= kw * 0.5
            
                chi_inter2 += inter[0]/(eji-2.0*w+broadening)
                chi_inter2 -= (inter[1]-inter[2])/(eji-w+broadening)
                chi_intra2 += intra[0]/(eji-w+broadening) + intra[2]/(eji-2.0*w+broadening)
                chi_modul2 += (modul[0]-modul[1])/(eji-w+broadening)

    return chi_inter2, chi_intra2, chi_modul2 

class Shg:

    def __init__(self):
        pass

    def set_input(self, path, direction, **kwargs): 
        from jamip.utils.logger import dump_yaml

        data = kwargs
        data['direction'] = direction
        if 'volume' not in data:
            data['volume'] = self.get_volume(path)
       
        '''
                'expgap':  1.6921,
                #'shgband': 1,
                'volume': 118.451602392,
                'pm': 1,
                'emini': 1e-5,
        '''
        print(data)
        dump_yaml(data, 'shg.yaml', default=True)

    def run(self, path, setting='shg.yaml', output=None):
        from jamip.utils.logger import load_yaml
        data = load_yaml(setting)
        
        # set bands
        gap = self.set_band(path)
        direct_gap = gap['direct']
        indirect_gap = gap['indirect']

        scissor = 0
        if 'expgap' in data:
           scissor = data['expgap'] - indirect_gap
        gap = direct_gap + scissor
        self.gap = gap
        self.scissor = scissor / Hartree

        # set nabij
        self.nabij = self.get_momentumatrix(path)

        # set weights
        volume = data['volume'] * sc.angstrom**3 / Bohr**3
        occupy = data.get('occupy', 2)
        self.weights = self.get_kpoint_weights(path) 
        self.Wk_omega = 1j*occupy/volume/np.sum(self.weights)
        
        pmax = data.get('pmax', 0)
        pgrid = data.get('pgrid', 0)
        if pmax == 0:
            pmax = np.clip(gap/2*3, 6, 12)
        if pgrid == 0:
            pgrid = int(np.ceil(pmax)*20)

        broadening_factor = data.get('broadening_factor', 0.1)
        self.broadening = broadening_factor / Hartree * 1j
        self.pmax = pmax / Hartree
        self.pgrid = pgrid

        self.e_min = data.get('e_min', 1e-5)
        self._lambda = data.get('lambda', 0)
        self.pm = data.get('pm', 1)

        # run
        direction = data['direction']
        if isinstance(direction, str):
            direction = np.array(direction.split(), dtype=int)
        a,b,c = direction
        df = self.shg_calculator(a,b,c)

        if output is None:
            output = pathlib.Path.cwd()
        else:
            output = pathlib.Path(output)

        if output.is_dir():
            filename = 'SHG_%d%d%d.csv' %(a,b,c)
            output = output/filename

        df.to_csv(output, index=0, float_format='%.6f')

    def set_band(self, path, source=None):
        '''
        获得直接带隙、间接带隙、修正带隙(剪切算符)
        '''
        from jamip.analysis.vasp.band import BandFinder

        path = pathlib.Path(path)
        if path.is_dir():
            source = 'EIGENVAL'
            path = path
        elif path.is_file():
            source = path.name
            path = path.parent

        bf = BandFinder(path).get_data(source=source)
        data = bf.get_bandgap()
        self.bands = bf.bands
        ispin,nkpts,nbands,_ = bf.bands.shape

        self.ispin = ispin
        self.nkpts = nkpts
        self.nbands = nbands

        return data
        
    def get_kpoint_weights(self, path):
        from jamip.analysis.vasp.band import Outcar
        path = pathlib.Path(path)
        if path.is_dir():
            path = path/"OUTCAR"
        weights = Outcar.from_file(path)._get_kpoint_weight()
        return weights 

    def get_volume(self, path, axis:int=None):
        from jamip.structure import read
        
        path = pathlib.Path(path)
        if path.is_dir():
            path = path/'POSCAR'

        s = read(path)
        volume = s.volume

        if axis != None:
            coord = s.get_positions()[:,axis]
            coord = coord - np.floor(coord)
            sorted_coord = np.sort(coord)
            diffs = np.diff(sorted_coord)
            diff0 = sorted_coord[0] - sorted_coord[-1] + 1
            thickness = max(max(diffs), diff0)
            volume = (1-thickness) * volume
            
        return float(round(volume,6))

    def get_volume_with_spacing(self, path, axis:int, spacing:float=None):
        from jamip.structure import read
        
        path = pathlib.Path(path)
        if path.is_dir():
            path = path/'POSCAR'

        s = read(path)
        volume = s.volume

        coord = s.get_positions()[:,axis]
        coord = coord - np.floor(coord)
        sorted_coord = np.sort(coord)
        diffs = np.diff(sorted_coord)
        diff0 = sorted_coord[0] - sorted_coord[-1] + 1
        diffs = np.sort(np.append(diffs,diff0))[::-1]
        thickness = diffs[0]
        if spacing is None:
            spacing = diffs[1]
        volume = (1-thickness+spacing) * volume
            
        return float(round(volume,6))

    def get_momentumatrix(self, path):

        path = pathlib.Path(path)
        if path.is_dir():
            path = path/'momentummatrix'

        with open(path, "rb") as fp:

            nabij = np.fromfile(fp, dtype=np.complex128)
            nabij = nabij.reshape(self.nkpts,3,self.nbands,self.nbands)
            nabij = nabij.transpose(0,1,3,2)
            return nabij * -1j

    def shg_calculator(self, a,b,c):
        from concurrent.futures import ProcessPoolExecutor

        w = np.arange(self.pgrid) * self.pmax / self.pgrid
        broadening = self.broadening
        scissor = self.scissor
        chi_inter2 = np.zeros(self.pgrid, dtype=np.complex128)
        chi_intra2 = np.zeros(self.pgrid, dtype=np.complex128)
        chi_modul2 = np.zeros(self.pgrid, dtype=np.complex128)

        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = []
            for ik in range(self.nkpts):
                vmatrix = self.nabij[ik] 
                kw = self.weights[ik] * self.Wk_omega
                bands = self.bands[0,ik,:,0] / Hartree
                occupy = self.bands[0,ik,:,1]
         
                cb = None
                for ib in range(self.nbands):
                     if occupy[ib] <= 1e-4:
                         cb = ib
                         break
         
                # loop over valence bands
                future = executor.submit(precess_bands, vmatrix, bands, occupy, w, kw, scissor, broadening, self.pgrid, cb, self.nbands, a, b, c)
                futures.append(future)

            data = []
            for i,f in enumerate(futures):
                print(i)
                data.append(f.result())

        data = np.array(data)
        # shape: nkpt, 3, nb
        chi = np.sum(data, axis=(0,1))

        #chi = chi_inter2+chi_intra2+chi_modul2
        chi0_imag = np.imag(chi[0])
        chi0_real = np.real(chi[0])
        shg_static = abs(chi0_real)*esu_pmV

        if self.pm==0:              # unit esu
            chi_real=np.real(chi)
            chi_imag=np.imag(chi)-chi0_imag
        elif self.pm==1:            # unit pm/V
            chi = chi*esu_pmV
            chi_real=np.real(chi)
            chi_imag=np.imag(chi)-chi0_imag*esu_pmV

        if self._lambda==1:

            gap = self.gap / Hartree 
            for i,j in enumerate(w):
                if j >= gap/2:
                    w_gap = i
                    break
            else:
                w_gap = self.pgrid
            self.w_gap = w_gap

            imax = np.ceil(self.w_gap**2)
            w = w[1:imax]
            chi_imag = chi_imag[1:imax],
            chi_real = chi_real[1:imax],

        data = {'energy': Hartree*w,
                'real': chi_real,
                'imag': chi_imag,
                'abs': np.sqrt(chi_real**2+chi_imag**2)
               }
        df = pd.DataFrame(data)
        return df
