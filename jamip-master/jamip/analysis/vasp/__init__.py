import os
# import re
import numpy as np
from jamip.analysis.base import Finder
from jamip.utils.utils import lazy_property
# from scipy.sparse import data
from .outcar import GrepOutcar
from .band import BandFinder, ProFinder
from .dos import DosFinder
from .xml import Xml
from .phonon import PhononFinder
from collections import namedtuple
import pathlib

Bandside = namedtuple('BandSide', ['energy','kpoints','ikpt','iband'])
Emass = namedtuple('Emass', ['mass','energy','kpoints'])
Kpath = namedtuple('Kpath', ['kpts','insert','dirs'])
Projection = namedtuple('projection', ['atoms', 'orbits','label'])

def reorder_eigenvalues(eigvecs1, eigvecs2, eigvals2):
    from scipy.optimize import linear_sum_assignment

    cost_matrix = -np.abs(eigvecs1.T @ eigvecs2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    eigvals2_sorted = eigvals2[col_ind]
    eigvecs2_sorted = eigvecs2[:, col_ind]

    return eigvals2_sorted, eigvecs2_sorted

def reorder_all_eigenvals(eigenvalues, eigenvectors):
    for idx in range(1, len(eigenvalues)):
        eigvecs_prev = eigenvectors[idx-1] 
        eigvals = eigenvalues[idx]
        eigvecs = eigenvectors[idx]
        eigvals_sorted, eigvecs_sorted = reorder_eigenvalues(eigvecs_prev, eigvecs, eigvals,)
        eigenvectors[idx] = eigvecs_sorted
        eigenvalues[idx] = eigvals_sorted
    return eigenvalues


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


class OpticsFinder(Finder, GrepOutcar):

    class Result:

        def __init__(self,energy:np.ndarray,data:np.ndarray,**kwargs):
            assert len(energy) == len(data)
            self.energy = energy
            self.data = data
            self.repair = True # False

        def absorb(self):
            """
            https://pubs.rsc.org/en/content/articlelanding/2015/tc/c5tc01622c
            E = hυ = hbar * ω  ; ω = E / hbar
            absorb = sqrt(2)*ω/c * (sqrt(ε1^2 + ε2^2) - ε1 )^(1/2)
            constant: ev -> cm-1 
            Return : cm-1
            """
            import scipy.constants as sc
            constant=np.sqrt(2)*sc.eV/sc.hbar/sc.c/100
            wavenumber = self.energy * constant
            real = self.real()
            imag = self.imag()
            return np.sqrt(np.sqrt(real**2+imag**2) - real) * wavenumber[:,None]

        def refract(self):
            '''
            refractive index n(ω) = sqrt(1/2) * (sqrt(ε1^2 + ε2^2) + ε1 )^(1/2)
            '''
            real = self.real()
            imag = self.imag()
            return np.sqrt(0.5) * np.sqrt(np.sqrt(real**2+imag**2) + real)

        def reflect(self):
            '''
            reflectivity R(ω) = |(np.sqrt(ε1 + iε2) - 1) / (np.sqrt(ε1 + iε2) + 1)|**2
            '''
            complex = self.real() + self.imag() * (0+1j)
            return np.abs((np.sqrt(complex)-1) / (np.sqrt(complex)+1))**2

        def extinction(self):
            '''
            extinction coefficient k(ω) = 1/sqrt(2) * ( sqrt(ε1^2 + ε2^2) - ε1 )^(1/2)
            '''
            real = self.real()
            imag = self.imag()
            return np.sqrt(0.5) * (np.sqrt(real**2+imag**2) - real)

        def energy_loss_spectrum(self):
            '''
            energy_loss_spectrum L(ω) = ε2 / (ε1^2 + i*ε2^2)
            '''
            real = self.real()
            imag = self.imag()
            return imag / (real**2+(0+1j)*imag**2)
            
        def optical_conductivity(self):
            '''
            E = hυ = hbar * ω  ; ω = E / hbar
            optical_conductivity = -i*ω/(4pi) * (ε1 + iε2 - 1)
            constant: ev -> fs-1 
            Return : fs-1         
            '''   
            import scipy.constants as sc
            constant= -1j/(4*np.pi)*sc.eV/sc.hbar*1e-15
            wavenumber = self.energy * constant
            complex = self.complex()
            return  wavenumber * (complex - 1)

        def real(self):
            eigenvalues = []
            eigenvectors = []
            for xx,yy,zz,xy,yz,xz in np.real(self.data):
                matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
                eigvals, eigvecs = np.linalg.eig(matrix)
                eigenvalues.append(eigvals)
                eigenvectors.append(eigvecs)

            if self.repair:
                eigenvalues = reorder_all_eigenvals(eigenvalues, eigenvectors)

            return np.array(eigenvalues)

        def imag(self):
            eigenvalues = []
            eigenvectors = []
            for xx,yy,zz,xy,yz,xz in np.imag(self.data):
                matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
                eigvals, eigvecs = np.linalg.eig(matrix)
                eigenvalues.append(eigvals)
                eigenvectors.append(eigvecs)

            if self.repair:
                eigenvalues = reorder_all_eigenvals(eigenvalues, eigenvectors)
                
            return np.array(eigenvalues)

        def complex(self):
            complex_dielectric = []
            for xx,yy,zz,xy,yz,zx in self.data:
                matrix=[[xx,xy,np.conj(zx)],[np.conj(xy),yy,yz],[zx,np.conj(yz),zz]]
                w=np.linalg.eigvals(matrix)
                # w=np.linalg.eigvalsh(matrix)
                complex_dielectric.append(w)
            return np.array(complex_dielectric)

        def fast_absorb(self):
            '''
            return np.ndarray (3,nedos)
            '''
            constant = 71618.96076
            absorption = []
            for i in range(3): # x,y,z
                real = np.real(self.data[:,i])
                imag = np.imag(self.data[:,i])
                absorb = self.energy * np.sqrt(np.sqrt(real**2+imag**2)-real) * constant
                absorption.append(absorb)
            return np.array(absorption)

        def dichroism(self, direction:str='x/y', energy:float=None, **kwargs):
            '''
            dichroism = max(ε1 / ε2, ε2 / ε1)
            '''
            def max_division(a, b):
                a_div_b = np.where(b != 0, a / b, 0)
                b_div_a = np.where(a != 0, b / a, 0)
                return np.maximum(a_div_b, b_div_a)

            absorption = self.fast_absorb()
            if energy is not None:
                absorption = absorption[:, np.argmin(np.abs(self.energy - energy))]

            if direction == 'x/y' or direction == 'y/x':
                return max_division(absorption[0], absorption[1])
            elif direction == 'x/z' or direction == 'z/x':
                return max_division(absorption[0], absorption[2])
            elif direction == 'y/z' or direction == 'z/y':
                return max_division(absorption[1], absorption[2])
            else:
                raise ValueError("Invalid input direction.")

        def to_pandas(self):
            import pandas as pd

            absorption = self.absorb()
            data = {'energy': self.energy,
                    'x': absorption[:,0], 
                    'y': absorption[:,1], 
                    'z': absorption[:,2], 
                    }
            df = pd.DataFrame(data)
            return df

        def _alpha_am(self, shift=0):
            # shift: hse_gap - pbe_gap (Unit eV)
            import scipy.constants as sc
            import jamip.analysis.vasp as jav
            import pandas as pd
            from scipy.interpolate import interp1d
            from importlib.resources import is_resource, open_text
            
            if is_resource(jav,'AM15.csv'):
                with open_text(jav,'AM15.csv') as df:
                    am15 = pd.read_csv(df).values
            else:
                raise OSError("Missing data file AM15.csv.")

            constant=np.sqrt(2)*sc.eV/sc.hbar/sc.c/100
            # Full-width at the half of the maximum
            # FWHM = 2 × sqrt(2 × ln2) × \sigma = 2.3548 
            sigma = 0.06/2.3548
            sqrt_2pi = np.sqrt(2.0*np.pi)

            deltaE = (self.energy[-1] - self.energy[0]) / len(self.energy)
            real_energy = self.energy + shift
            imag_energy = self.energy
            real = self.real()
            imag = self.imag()

            # cutoff 
            inds = np.where((real_energy>-1e-8) & (real_energy<5.0))[0] 
            data = []
            for i in inds:
         
                hv = real_energy[i]
                # cut imag by hv smear 
                val = ( (imag_energy-hv) / sigma )**2/2.0    
                indices = np.where(val<15.0) 
                wt = np.exp(-val[indices]) 
                imag_ = np.sum(wt[:,None] * imag[indices], axis=0) * deltaE / sqrt_2pi / sigma  
                real_ = real[i]
                alpha = np.sqrt(np.sqrt(real_**2+imag_**2) - real_) * hv * constant
                data.append(np.mean(alpha))

            # smear to am %
            f1 = interp1d(real_energy[inds], data, kind='linear')
            hv = am15[:,0]
            nhv = am15[:,1] 
            fa = f1(hv) 
            return np.c_[hv,nhv,fa]
         
        def _slme(self, data, L, dirgap:float, indirgap:float):
            """
            # Reference: L.Yu and A. Zunger, Phys. Rev. Lett. 108, 068701 (2012)

            alpha: calculate spectral weight to sloar spectral.
            L: film thickness, unit (cm)
            Eg: indirect band gap, unit (eV)
            dEg: ( indirect_band_gap - direct_band_gap ) unit (eV)

            Return: slme (Percent)
            """
            import scipy.constants as sc
            from scipy.integrate import simps
            from scipy.optimize import minimize
            # from scipy.interpolate import interp1d
            
            hv = data[:,0]
            nhv = data[:,1] 
            fa = data[:,2] 
            minEg = indirgap
            dEg = indirgap - dirgap

            T = 300 # K
            Vc = sc.k / sc.eV * T # K_eV * T = 8.61733E-5 ev/K * 300 K
            aE = 1.0 - np.exp(-2.0 * L * fa)
            fr = np.exp(dEg / Vc)
            isc = nhv * aE
            irb = hv**2 / (np.exp(hv/Vc)-1) * aE
           
            # 对太阳能光谱的入射能量进行积分
            Isc = simps(isc, hv)
            I0 = simps(irb, hv)
            I0 = I0 * 2.0*np.pi*sc.c/(sc.h*sc.c/sc.eV)**3 * fr
           
            def J(V):
                J = Isc - I0*(np.math.exp( V/Vc ) - 1.0)
                return J
            def power(V):
                p = J(V)*V
                return p
            def neg_power(V): # Minimizing the negative of power to optimize power
                return -power(V)
 
            bnds = ((0,minEg),)
            results = minimize(neg_power, x0 = [0.0001], bounds=bnds) 
            V_Pmax = float(results.x)
            P_m = power(V_Pmax)
            P_in = 1000 / sc.eV
            #P_in = simps(nhv, hv)
            efficiency = P_m / P_in * 100

            return efficiency

        def sl3me(self, thickness, dirgap:float, indirgap:float, shift=0, T:float=293.15):
            from scipy.integrate import simps
            from scipy.interpolate import interp1d
            from scipy.optimize import minimize
            import jamip.analysis.vasp as jav
            from importlib.resources import is_resource, open_text
            #from importlib.resources import files

            #amfile = files(jav).joinpath('am1.5G.dat')
            #if os.path.exists(amfile):
            if is_resource(jav,'am1.5G.dat'):
                with open_text(jav,'am1.5G.dat') as f:
                    solar_spectra_wavelength, solar_spectra_irradiance  = np.loadtxt(f, usecols = [0,1], unpack = True, skiprows=2)
                    solar_spectra_wavelength_meters = solar_spectra_wavelength * 10**-9
            else:
                raise OSError("Missing data file am1.5G.dat .")

            c = 299792458 # speed of light, m/s
            h = 6.62607004081E-34 # Planck's constant J*s (W)
            h_eV = 4.135667516E-15   # Planck's constant eV*s
            k = 1.3806485279E-23 # Boltzmann's constant J/K
            k_eV = 8.617330350E-5 # Boltzmann's constant eV/K
            e = 1.602176620898E-19  # Coulomb
            pi = np.pi

            dirgap += shift
            indirgap += shift
            minEg = indirgap
            delta = dirgap - indirgap

            fr = np.math.exp(-delta/(k_eV*T))
 
            ## need to convert solar irradiance from Power/m**2(nm) into photon#/s*m**2(nm)
            ## power is Watt, which is Joule / s
            ## E = hc/wavelength
            ## at each wavelength, Power * (wavelength(m)/(h(Js)*c(m/s))) = ph#/s
            solar_spectra_photon_flux = solar_spectra_irradiance * (solar_spectra_wavelength_meters/(h*c))

            ### Calculation of total solar power incoming
            P_in = simps(solar_spectra_irradiance, solar_spectra_wavelength)
 
            ### calculation of blackbody irradiance spectra
            ## units of W/(m**3), different than solar_spectra_irradiance!!! (This is intentional, it is for convenience)
            blackbody_irradiance = (2.0 * pi * h * c**2/(solar_spectra_wavelength_meters**5)) * ( 1.0/((np.exp(h*c/(solar_spectra_wavelength_meters*k*T)))-1.0))
 
            ## now to convert the irradiance from Power/m**2(m) into photon#/s*m**2(m)
            blackbody_photon_flux = blackbody_irradiance * (solar_spectra_wavelength_meters/(h*c))
 
            ## units of nm
            material_eV_for_absorbance_data = self.energy + shift
            material_wavelength_for_absorbance_data = ((c*h_eV)/(material_eV_for_absorbance_data+0.00000001))*10**9
 
            ### absorbance interpolation onto each solar spectrum wavelength
            material_absorbance_data = np.mean(self.absorb(), axis=1)
            ## creates cubic spline interpolating function, set up to use end values as the guesses if leaving the region where data exists
            material_absorbance_data_function = interp1d(material_wavelength_for_absorbance_data, material_absorbance_data,
                                                                                                                kind = 'cubic',
                                                                                                                fill_value = (material_absorbance_data[0], material_absorbance_data[-1]),
                                                                                                                bounds_error=False)
            material_interpolated_absorbance = np.zeros(len(solar_spectra_wavelength_meters))
            for i in range(0,len(solar_spectra_wavelength_meters)):
                ## Cutting off absorption data below the gap. This is done to deal with VASPs broadening of the calculated absorption data 
                if solar_spectra_wavelength[i] < ((c*h_eV)/dirgap)*10**9:
                    material_interpolated_absorbance[i] = material_absorbance_data_function(solar_spectra_wavelength[i])
 
            absorbed_by_wavelength = 1.0-np.exp(-2.0 * material_interpolated_absorbance * thickness)
 
            ###  Numerically integrating irradiance over wavelength array
            ## Note: elementary charge, not math e!  ## units of A/m**2   W/(V*m**2)  
            J_0_r = e * pi *simps( blackbody_photon_flux * absorbed_by_wavelength ,solar_spectra_wavelength_meters)
 
            J_0 = J_0_r / fr
 
            ###  Numerically integrating irradiance over wavelength array
            ## elementary charge, not math e!  ### units of A/m**2   W/(V*m**2)
            J_sc = e * simps( solar_spectra_photon_flux * absorbed_by_wavelength  , solar_spectra_wavelength)
 
            #    J[i] = J_sc - J_0*(1 - exp( e*V[i]/(k*T) ) )   #This formula from the original paper has a typo!!
            #    J[i] = J_sc - J_0*(exp( e*V[i]/(k*T) ) - 1)    #Bercx chapter and papers have the correct formula (well, the correction on one paper)
            def J(V):
                J = J_sc - J_0*(np.math.exp( e*V/(k*T) ) - 1.0)
                return J
            def power(V):
                p = J(V)*V
                return p
            def neg_power(V): # Minimizing the negative of power to optimize power
                return -power(V)
 
            results = minimize(neg_power, x0 = [0.0001] ) # passing a function as a variable, preconditioning at V~=0, since this is a smooth function
            # results.x are the inputs as an array that give the highest value of Power
            V_Pmax = float(results.x)
            P_m = power(V_Pmax)
 
            efficiency = P_m / P_in
 
            return  efficiency

    def __init__(self,stdin=None):
        self.task = 'optics'
        self.soft = 'vasp'
        self.stdin = stdin

    @property
    def opticsdir(self):
        return self._stdin/'optics'/'optics' if self.file == 'jamip' else self._stdin 
    
    @property
    def dielectricdir(self):
        return self._stdin/'optics'/'dielectric' if self.file == 'jamip' else self._stdin
    
    def get_data(self,source='xml'):
        if source == 'xml':
            imag,real = self.get_dielectric_func_from_xml()
        elif source == 'outcar':
            imag,real = self.get_dielectric_func_from_outcar()
        else:
            raise ValueError('Unknown optics data source.')

        return self.Result(energy=real[:,0],
                           data=real[:,1:]+imag[:,1:]*(0+1j))

    def get_slme(self, L, bandgap:dict, shift=0, method='sl3me'):
        '''
        L: thickness value or array, unit cm
        bandgap: dict, like {'dircet':0.7, 'indirect':0.62}
        shift: gap shift for corrected absorption spectrum
        '''
        dirgap = bandgap['direct']
        indirgap = bandgap['indirect']

        if method == 'sl3me':
            # calc with SL3ME %
            # 100 means percent %
            data = self.get_data(source='xml')
            if isinstance(L, (list,np.ndarray)):
                return [data.sl3me(l, dirgap, indirgap, shift)*100 for l in L]
            elif isinstance(L, float):
                return data.sl3me(L, dirgap, indirgap, shift)*100
            else:
                raise ValueError("Invalid input thickness")

        else:
            # calc with old version %
            data = self.get_data(source='outcar')
            alpha_am = data._alpha_am(shift=shift)
            if isinstance(L, (list,np.ndarray)):
                return [data._slme(alpha_am, l, dirgap, indirgap) for l in L]
            elif isinstance(L, float):
                return data._slme(alpha_am, L, dirgap, indirgap)
            else:
                raise ValueError("Invalid input thickness")

    def get_dielectric_func_from_outcar(self):
        nedos = np.ceil(self.nedos(self.opticsdir)/10).astype(int)
        imag = self.dielectric_imag(self.opticsdir,nedos)
        real = self.dielectric_real(self.opticsdir,nedos)
        return imag,real

    def get_dielectric_func_from_xml(self):
        import xml.etree.cElementTree as ET

        def xml2array(root):
            dim={}
            dim_order=[]
            field = []
            array = []
            shape = []
            for element in root:
                if element.tag == 'dimension':
                    dim[element.text] = 0
                    dim_order.append(element.text)
                if element.tag == 'field':
                    field.append(element.text)
                if element.tag == 'set':
                    for text in element.findall(".//"):
                        if text.tag == 'set':
                            dim[text.attrib['comment'].split()[0]]+=1
                        elif len(text.text.split()) == len(field):
                            array.append(text.text.split())
            product=1
            for i in reversed(dim_order[1:]):
                 shape.append(int(dim[i]/product))
                 product=dim[i]
            shape.extend([-1,len(field)])
            array=np.array(array,dtype=float).reshape(shape)
            return array
        
        clear = True
        imag = real = None
        xmlfile = self.opticsdir/'vasprun.xml'
        for event, elem in ET.iterparse(xmlfile,events=('start','end')):
            if event == 'start':
                if elem.tag == "dielectricfunction":
                    clear = False

            elif event == 'end':
                if elem.tag == "imag":
                    array = elem.findall('./array')[0]
                    imag = xml2array(array)
                    elem.clear()
                elif elem.tag == "real":
                    array = elem.findall('./array')[0]
                    real = xml2array(array)
                    elem.clear()
                elif elem.tag == "dielectricfunction":
                    break

            if clear:
                elem.clear()

        if imag is None or real is None:
            raise RuntimeError("Failed search keyword 'dielectricfunction'")
        return imag,real

    def get_dielectric_const_of_ionic(self,path=None):
        if path != None:
            path = OpticsFinder(path).dielectricdir
        else:
            path = self.dielectricdir

        if not path.exists():
            raise IOError('Dielectric path not exists!')
        diel = self.dielectric_ionic(path)
        return np.mean(np.linalg.eig(diel)[0])

    def get_dielectric_const(self,path=None):
        if path != None:
            path = OpticsFinder(path).dielectricdir
        else:
            path = self.dielectricdir
        if not path.exists():
            raise IOError('Dielectric path not exists!')
        diel = self.dielectric(path)
        return np.mean(np.linalg.eig(diel)[0])
    
    def get_dipoles_from_dpmatrix(self,path=None,directions=3):
        import pandas as pd

        if path != None:
            path = OpticsFinder(path).dielectricdir
        else:
            path = self.dielectricdir
        
        if not os.path.exists(path):
            raise IOError('Dipoles path not exists!')
        dpfile = os.path.join(path,'DPMATRIX')
        data = np.loadtxt(dpfile,skiprows=1)
        dipoles = pd.DataFrame({'ikpt':data[0::3,0], 
                                'kpt':data[0::3,1:4],
                                'weight':[kpt[3] for kpt in self.kpoints[data[0::3,0]]],
                                'ispin':data[0::3,4],
                                'vb':data[0::3,6],
                                'cb':data[0::3,7],
                                'dE':data[0::3,8],
                                'ampx':data[0::3,9],
                                'ampy':data[1::3,9],
                                'ampz':data[2::3,9],
                                'amp_avr':(data[0::3,9]+data[1::3,9]+data[2::3,9])/3})
        return dipoles
    
    #def get_dielectric_func_curve(self):
    #    from numpy.polynomial import Polynomial, Chebyshev
    #    imag, real = self.get_dielectric_func_from_xml()
    
    def get_dielectric_func_maximums(self,imag):
        """get maximums of imag part of epsilon

        Args:
            path (str, optional): workpath. Defaults to None.

        Returns:
            tuple: (array([energy_maximums]), array([directions of the maximums]))
            
            example: (array([0.7732, 2.8984, 3.567 , 3.567 , 4.4037, 4.4037, 5.2105, 6.2077,
                            6.2301, 6.2301, 8.3367, 8.6542, 8.6542, 9.6141, 9.7748]), 
                      array([2, 2, 0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2]))
        """        
        from scipy.signal import argrelextrema
        energy = imag[:,0]
        imag_maximums_indices = argrelextrema(imag[:,1:4],np.greater)
        imag_maximums_energies = (np.array([energy[indice] for indice in imag_maximums_indices[0]]), 
                                  imag_maximums_indices[1])
        return imag_maximums_energies
    
    def get_maximums_transitions(self,imag,diffE=0.05):
        maximums = self.get_dielectric_func_maximums(imag)
        return BandFinder(self._stdin).get_transitions_with_E(maximums[0],diffE)
    
    def get_maximums_dipoles(self,path=None):
        from .wavecar import Wavecar
        w = Wavecar.from_file(self._stdin)
        #w.wavecar
        
class OpticFinder(OpticsFinder):
    
    def __init__(self,stdin=None):
        super().__init__(stdin)

class VaspFinder(Finder):

    __task__ = ['band','dos','optics']

    def __new__(cls,task,stdin):

        if task == 'band':
            newcls = BandFinder
        else:
            raise KeyError("Unsupport task type %s" %task)

        Finder.seek(newcls,stdin)
        newcls.__task__ = task
        return super().__new__(newcls)

if __name__ == "__main__":
    vf = VaspFinder('band')

