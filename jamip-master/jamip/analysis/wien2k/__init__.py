import numpy as np
import pandas as pd
import pathlib


class DosFinder:

    class Result:

        def __init__(self,data:np.ndarray,**kwargs):
            self.data = data

        @property
        def energy_in_eV(self):
            """
            energy in eV
            """
            return self.data['ENERGY'].values
        
        @property
        def tdos(self):
            """
            total DOS
            """
            return self.data['total-DOS'].values
        
    def __init__(self,stdin=None,case:str=None):
        self.task = 'dos'
        self.soft = 'wien2k'
        self.stdin = pathlib.Path(stdin)
        if case is None:
            self.case = self.stdin.absolute().name
        else:
            self.case = case
        self.case = case
    
    def get_data(self,source='dos1ev'):
        if source == 'dos1ev':
            df = self.get_dos_from_dos1ev()
        else:
            raise ValueError('Unknown optics data source.')

        return self.Result(df)

    def get_dos_from_dos1ev(self):

        with open(self.stdin/f'{self.case}.dos1ev', 'r') as f:
            data  = []
            for line in f:
                if 'ENERGY' in line:
                    columns = line.split()[1:]
                if line.strip() and not line.startswith('#'):
                    values = [float(x) for x in line.split()]
                    data.append(values)

        data = np.array(data, dtype=float)
        df = pd.DataFrame(data, columns=columns)
        return df

class OpticsFinder:
    class Result:

        def __init__(self,data:np.ndarray,**kwargs):
            self.data = data
        
        def get_energy(self, unit='eV'):
            """
            energy in eV
            """
            energy = None
            for column in self.data.columns:
                if 'energy' in column.lower():
                    energy = self.data[column].values
                    break

            if energy is None:
                raise ValueError("Energy column not found in data.")
            if unit == 'eV':
                return energy if 'eV' in column else energy * 13.6057
            elif unit == 'Ry':
                return energy if 'Ry' in column else energy / 13.6057
            else:
                raise ValueError(f"Unknown energy unit: {unit}")

        @property
        def energy_in_eV(self):
            """
            energy in eV
            """
            return self.get_energy(unit='eV')
        
        @property
        def energy_in_Ry(self):
            """
            energy in Ry
            """
            return self.get_energy(unit='Ry')
        
        @property
        def real(self):
            """
            real part of dielectric function
            """
            res = []
            for column in self.data.columns:
                if 'Re' in column:
                    res.append(self.data[column].values)
            return np.array(res)
        
        @property
        def imag(self):
            """
            imaginary part of dielectric function
            """
            ims = []
            for column in self.data.columns:
                if 'Im' in column:
                    ims.append(self.data[column].values)
            return np.array(ims)
        
        @property
        def columns(self):
            """
            columns of the data
            """
            data = []
            for column in self.data.columns:
                if 'Re' in column:
                    data.append(column.split('_')[-1])
            return data

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
            wavenumber = self.energy_in_eV * constant
            real = self.real
            imag = self.imag
            return np.sqrt(np.sqrt(real**2+imag**2) - real) * wavenumber

        def refract(self):
            '''
            refractive index n(ω) = sqrt(1/2) * (sqrt(ε1^2 + ε2^2) + ε1 )^(1/2)
            '''
            real = self.real
            imag = self.imag
            return np.sqrt(0.5) * np.sqrt(np.sqrt(real**2+imag**2) + real)

        def reflect(self):
            '''
            reflectivity R(ω) = |(np.sqrt(ε1 + iε2) - 1) / (np.sqrt(ε1 + iε2) + 1)|**2
            '''
            complex = self.real + self.imag * (0+1j)
            return np.abs((np.sqrt(complex)-1) / (np.sqrt(complex)+1))**2

        def extinction(self):
            '''
            extinction coefficient k(ω) = 1/sqrt(2) * ( sqrt(ε1^2 + ε2^2) - ε1 )^(1/2)
            '''
            real = self.real
            imag = self.imag
            return np.sqrt(0.5) * (np.sqrt(real**2+imag**2) - real)

        def energy_loss_spectrum(self):
            '''
            energy_loss_spectrum L(ω) = ε2 / (ε1^2 + i*ε2^2)
            '''
            real = self.real
            imag = self.imag
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

        # def real(self):
        #     eigenvalues = []
        #     eigenvectors = []
        #     for xx,yy,zz,xy,yz,xz in np.real(self.data):
        #         matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        #         eigvals, eigvecs = np.linalg.eig(matrix)
        #         eigenvalues.append(eigvals)
        #         eigenvectors.append(eigvecs)
        #     return np.array(eigenvalues)

        # def imag(self):
        #     eigenvalues = []
        #     eigenvectors = []
        #     for xx,yy,zz,xy,yz,xz in np.imag(self.data):
        #         matrix = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
        #         eigvals, eigvecs = np.linalg.eig(matrix)
        #         eigenvalues.append(eigvals)
        #         eigenvectors.append(eigvecs)               
        #     return np.array(eigenvalues)

        # def complex(self):
        #     complex_dielectric = []
        #     for xx,yy,zz,xy,yz,zx in self.data:
        #         matrix=[[xx,xy,np.conj(zx)],[np.conj(xy),yy,yz],[zx,np.conj(yz),zz]]
        #         w=np.linalg.eigvals(matrix)
        #         # w=np.linalg.eigvalsh(matrix)
        #         complex_dielectric.append(w)
        #     return np.array(complex_dielectric)

        def dichroism(self, direction:str='x/y', energy:float=None, **kwargs):
            '''
            dichroism = max(ε1 / ε2, ε2 / ε1)
            NIR-1: 700-900 nm, 1.5-1.7 eV
            NIR-2: 1000-1700 nm, 0.7-1.2 eV
            '''
            def max_division(a, b):
                a_div_b = np.where(b != 0, a / b, 0)
                b_div_a = np.where(a != 0, b / a, 0)
                return np.maximum(a_div_b, b_div_a)

            absorption = self.absorb()
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

    def __init__(self,stdin=None,case:str=None):
        self.task = 'optics'
        self.soft = 'wien2k'
        self.stdin = pathlib.Path(stdin)
        if case is None:
            self.case = self.stdin.absolute().name
        else:
            self.case = case
    
    def get_data(self,source='epsilon'):
        if source == 'epsilon':
            df = self.get_dielectric_func_from_epsilon()
        else:
            raise ValueError('Unknown optics data source.')

        return self.Result(df)

    def get_dielectric_func_from_epsilon(self):

        with open(self.stdin/f'{self.case}.epsilon', 'r') as f:
            data  = []
            for line in f:
                if 'Energy [eV]' in line:
                    columns = [' '.join(line.split()[1:3])] + line.split()[3:]
                if line.strip() and not line.startswith('#'):
                    values = [float(x) for x in line.split()]
                    data.append(values)

        data = np.array(data, dtype=float)
        df = pd.DataFrame(data, columns=columns)
        return df
