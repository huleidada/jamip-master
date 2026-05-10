import os
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import scipy.constants as sc

stra = "Ef[eV]     T[K]           N          DOS(Ef)        Seebeck[uV/K]   sigma/tao         hall          thermal          powerfac[10^14]"
strb = "#       Ef[Ry] T [K]            N         DOS(Ef)           S             s/t               R_H        kappa0         c                 chi"
strc = '#   Ef[Ry]      T[K]                  N[e/uc]        DOS(ef)[1/(Ha*uc)]                    S[V/K]   sigma/tau0[1/(ohm*m*s)]                RH[m**3/C]    kappae/tau0[W/(m*K*s)]             cv[J/(mol*K)]             chi[m**3/mol]'
Ry = sc.physical_constants['Rydberg constant times hc in eV'][0]
Bohr_radius = sc.physical_constants['Bohr radius'][0] 

class BTP1:

    @classmethod
    def get_fermi(cls, path, filename='OUTCAR'):
        from jamip.analysis.vasp import BandFinder
        bf = BandFinder(path).get_data()
        return bf.get_cbmvbm()['vbm'].energy

    @classmethod
    def set_energy(cls, path, infile='EIGENVAL', output=None, soc=False, mode='BTP1'):
        """ 
        Transform energy for boltztrap in GENE model. (shift E-fermi before)

        INPUT: EIGENVAL & OUTCAR
        OUTPUT: $DIR.energy or $DIR.energyso (soc)
        """
        
        from jamip.analysis.vasp.eigenval import Eigenval
        from jamip.analysis.vasp.procar import Procar
        from jamip.analysis.vasp.outcar import GrepOutcar

        path = Path(path)
 
        # get info from OUTCAR 
        efermi = cls.get_fermi(path)
        lsorbit = GrepOutcar().lsorbit(path)
        # get bands
        if infile == 'EIGENVAL':
            bands = Eigenval.from_file(path/'EIGENVAL')
            if soc:
                bands = bands.soc_copy
        elif infile == "PROCAR_OPT":
            bands = Procar.from_file(path/"PROCAR_OPT")
        else:
            raise OSError("unsupport file type.")

        # filename
        if output is None:
            suffix = '.energyso' if lsorbit else '.energy'
            name = path.absolute().name if mode == 'BTP1' else 'SCF'
            filename = name + suffix
            output = path / filename

        bands.to_wien2k(output, efermi=efermi)

    @classmethod
    def set_struct(cls, path, output=None, soc=False, mode='BTP1', **kwargs):
        from jamip.structure import read
        from jamip.structure.io import write_boltztrap
 
        path = Path(path)
        if path.is_dir():
            path = path / "POSCAR"
        s = read(path)

        if output is None:
            name = path.absolute().parent.name if mode == 'BTP1' else 'SCF'
            filename = name + '.struct' 
            output = path.parent / filename

        if not soc:
            write_boltztrap(s, output, **kwargs)
        else:
            rotations = [np.diag([1,1,1]), np.diag([-1,-1,-1])]
            write_boltztrap(s, output, rotations=rotations)

    @classmethod
    def set_intrans(cls, path, infile='OUTCAR', output=None, mode='BTP1', lpfac=5, Tmax=300, Tgrid=300):
        from jamip.analysis.vasp.outcar import GrepOutcar
 
        # get info from OUTCAR 
        efermi = cls.get_fermi(path)
        nelect = GrepOutcar().nelect(path)

        path = Path(path)
        if output is None:
            name = path.absolute().name if mode == 'BTP1' else 'SCF'
            filename = name + '.intrans' 
            output = path/filename

        if mode == 'BTP1':
            with open(output, 'w') as f: 
                f.write("GENE              # use generic interface\n")
                f.write("0 1 0 0.0         # iskip (not presently used) idebug setgap shiftgap \n")
                f.write(" 0 0.0001 0.3 %6.1f     # Fermilevel (Ry), energygrid, energy span around Fermilevel, number of electrons\n"%nelect)
                f.write("CALC                      # CALC (calculate expansion coeff), NOCALC read from file\n")
                f.write("%d                         # lpfac, number of latt-points per k-point\n"%lpfac)
                f.write("BOLTZ                     # run mode (only BOLTZ is supported)\n")
                f.write(".60                       # (efcut) energy range of chemical potential\n")
                f.write("%d. %d.                 # Tmax, temperature grid\n"%(Tmax,Tgrid))
                f.write("0.                       # energyrange of bands given individual DOS output sig_xxx and dos_xxx (xxx is band number)\n")

        elif mode == 'BTPm':

            with open(output, 'w') as f: 
                f.write("VASP          # use vasp interface\n")
                f.write("0 1 0 0.0         # iskip (not presently used) idebug setgap shiftgap \n")
                f.write("%7.5 0.0005 0.4 %6.1f     # Fermilevel (eV), energygrid, energy span around Fermilevel, number of electrons\n"%(efermi/Ry, nelect))
                f.write("CALC                      # CALC (calculate expansion coeff), NOCALC read from file\n")
                f.write("%d                         # lpfac, number of latt-points per k-point\n"%lpfac)
                f.write("BOLTZ                     # run mode (only BOLTZ is supported)\n")
                f.write(".15                       # (efcut) energy range of chemical potential\n")
                f.write("%d. %d.                  # Tmax, temperature grid\n"%(Tmax,Tgrid))
                f.write("0.                     # energyrange of bands given individual DOS output sig_xxx and dos_xxx (xxx is band number)\n")
                f.write("HISTO\n")

class BTP2:

    @classmethod
    def save_calculation(cls, path:str, output:str):
        import BoltzTraP2
        from BoltzTraP2 import dft, sphere, fite, serialization, bandlib

        data = dft.DFTData(str(path))
        basis_tensors = BoltzTraP2.sphere.calc_tensor_basis(
            data.atoms, data.magmom
        )
        data.bandana(emin=data.fermi - 0.2, emax=data.fermi + 0.2)
        equivalences = sphere.get_equivalences(data.atoms, data.magmom,
                                       len(data.kpoints) * 5)
        
        coeffs = fite.fitde3D(data, equivalences)
        metadata = serialization.gen_bt2_metadata(
            data, derivatives=None,
        )
        # Save the result
        serialization.save_calculation(output, data, equivalences, coeffs, metadata)

    @classmethod
    def save_trace(cls, bt2file:str, tracefile:str, condtensfile:str, scattering_model="uniform_tau", T=300, **kwargs):
        import BoltzTraP2
        from BoltzTraP2 import sphere, fite, serialization, bandlib
        from BoltzTraP2.units import BOLTZMANN, Angstrom

        data, equivalences, coeffs, metadata = BoltzTraP2.serialization.load_calculation(bt2file)
        Tr = np.ravel(T).astype(float)
        lattvec = data.get_lattvec()
        eband, vvband, cband = fite.getBTPbands(
            equivalences, coeffs, lattvec, True, nworkers=8
        )
        epsilon, dos, vvdos, cdos = bandlib.BTPDOS(
            eband,
            vvband,
            cband,
            #npts=None,
            scattering_model=scattering_model,
            Tmin=Tr.min(),
        )
        mu0 = np.empty_like(Tr)
        for iT, T in enumerate(Tr):
            mu0[iT] = BoltzTraP2.bandlib.solve_for_mu(
                epsilon,
                dos,
                data.nelect,
                T,
                data.dosweight,
                refine=True,
                try_center=True,
            )
        # And at 0 K
        fermi = BoltzTraP2.bandlib.solve_for_mu(
            epsilon,
            dos,
            data.nelect,
            0.0,
            data.dosweight,
            refine=True,
            try_center=True,
        )
        margin = 9.0 * BOLTZMANN * Tr.max()
        mur_indices = np.logical_and(
            epsilon > epsilon.min() + margin, epsilon < epsilon.max() - margin
        )
        mur = epsilon[mur_indices]
        if mur.size == 0:
            raise ValueError("the energy window is too narrow")

        sdos = np.empty((Tr.size, epsilon.size))
        for iT, T in enumerate(Tr):
            sdos[iT, :] = BoltzTraP2.bandlib.smoothen_DOS(epsilon, dos, T)
        sdos = sdos[:, mur_indices]

        cv = BoltzTraP2.bandlib.calc_cv(
            epsilon, dos, mur, Tr, #dosweight=None
        )
        N, L0, L1, L2, Lm11 = BoltzTraP2.bandlib.fermiintegrals(
            epsilon,
            dos,
            vvdos,
            mur=mur,
            Tr=Tr,
            dosweight=data.dosweight,
            cdos=cdos,
        )

        # Rescale and combine the moments to get the Onsager transport coefficients
        vuc = data.atoms.get_volume() * Angstrom**3
        L11, seebeck, kappa, hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(
            L0, L1, L2, mur, Tr, vuc, Lm11
        )

        BoltzTraP2.io.save_trace(
            tracefile,
            data,
            Tr,
            mur,
            N,
            sdos,
            cv,
            L11,
            seebeck,
            kappa,
            hall,
            scattering_model=scattering_model,
        )
        BoltzTraP2.io.save_condtens(
            condtensfile,
            data,
            Tr,
            mur,
            N,
            L11,
            seebeck,
            kappa,
            scattering_model=scattering_model,
        )

class Boltztrap:

    def __init__(self,soft='vasp',mode='BTP1',prefix="SCF"):
        """
        soft: vasp & wien2k
        mode: BTP1, BTPm, BTP2
        """
        self.soft = soft
        self.mode = mode
        self.prefix = prefix

    class Result:
        def __init__(self, energy, N, dos, sigma_tao, seebeck, condtens, bandgap, volume, fermi):
            self.energy = energy # [eV]
            self.N = N           # [N]
            self.dos = dos       # [state/Volume]
            self.sigma_tao = sigma_tao
            self.seebeck = seebeck
            self.condtens = condtens
            self.bandgap = bandgap # [eV]
            self.volume = volume   # [ang^3]
            self.fermi = fermi     # [ev]

        def get_cbvb_by_carrier(self, concentration=1e18):
            '''get level by carrier concentration'''
            N = self.N / self.volume * 1e24
            ifermi = np.argmin(np.abs(N))
            ivb = np.argmin(np.abs(N - concentration))
            icb = np.argmin(np.abs(N + concentration))
            return ivb, icb

        def get_cbvb_by_gap(self):
            '''get level by carrier concentration'''
            ifermi = np.argmin(np.abs(self.energy-self.fermi))
            igap = np.argmin(np.abs(self.energy-self.fermi-self.bandgap))
            return ifermi, igap

        def get_cbvb_by_dos(self):
            '''get level by dos, dos[ivb] ~ dos[icb]'''
            ivb,icb = self.get_cbvb_by_gap()
            dos = self.dos

            if dos[ivb] < dos[icb]:
                for i in range(icb-ivb):
                    if dos[ivb-i] > dos[icb-i]:
                        break
                ivb = ivb-i
                icb = icb-i
            elif dos[ivb] > dos[icb]:
                for i in range(icb-ivb):
                    if dos[ivb+i] < dos[icb+i]:
                        break
                ivb = ivb+i
                icb = icb+i
            return ivb, icb

        def get_effective_mass(self, ivb, icb):
            hmass = self.N[ivb] / self.sigma_tao[ivb] *scipy.constants.e**2 / self.volume *1e30 / scipy.constants.m_e
            emass = -self.N[icb] / self.sigma_tao[icb] *scipy.constants.e**2 / self.volume *1e30 / scipy.constants.m_e
            return hmass, emass

        def get_carrier(self):
            #n = self.N / self.volume * 1e24 # cm-3
            return self.N / self.volume * 1e24

        def get_eff(self, ivb, icb):
            sigma_tao = self.condtens[:,[0,4,8]]
            seebecks = self.condtens[:,[9,13,17]]
            effx = seebecks[:,0]**2 * sigma_tao[:,0]
            effy = seebecks[:,1]**2 * sigma_tao[:,1]
            effz = seebecks[:,2]**2 * sigma_tao[:,2]
            sigtot = np.sum(sigma_tao, axis=1)
            a = effx*sigma_tao[:,0] + effy*sigma_tao[:,1] + effz*sigma_tao[:,2]
            effav = np.divide(a, sigtot, where=(sigtot!=0), out=np.zeros_like(a, dtype=float))

            factdv = self.volume * Ry*2 * scipy.constants.e * 1e-10**3
            #hn = self.N[:ivb] / self.volume * 1e24 # cm-3
            dvf = (self.dos[:ivb] / factdv)**(2/3)
            heff = np.divide(effav[:ivb], dvf , where=(dvf!=0), out=np.zeros_like(effav[:ivb], dtype=float))

            #en = self.N[icb:] / self.volume * 1e24 # cm-3
            dvf = (self.dos[icb:] / factdv)**(2/3)
            eeff = np.divide(effav[icb:], dvf , where=(dvf!=0), out=np.zeros_like(effav[icb:], dtype=float))
            
            return heff, eeff


        def get_effective_mass_with_direction(self, ivb, icb):
            #condtens = np.linalg.norm(self.condtens[ivb].reshape(3,3), axis=1)
            condtens = self.condtens[ivb,[0,4,8]]
            hmass =  self.N[ivb] / condtens * scipy.constants.e**2 / self.volume *1e30 / scipy.constants.m_e
            #condtens = np.linalg.norm(self.condtens[icb].reshape(3,3), axis=1)
            condtens = self.condtens[icb,[0,4,8]]
            emass = -self.N[icb] / condtens * scipy.constants.e**2 / self.volume *1e30 / scipy.constants.m_e
            return hmass, emass

    @classmethod
    def get_effective_mass_dict(self, path:str):
        '''H-mass e-mass
        only valid for BTPm
        '''
        path1 = Path(path) / 'boltztrap.dat'
        path2 = Path(path) / 'mass.dat'
        if path1.exists():
            with open(path1,'r') as f:
                lines = f.readlines()
            assert lines[5].split()[-1] == "e_mass"
            assert lines[6].split()[-1] == "H_mass"
            lines = lines[5:7]
        elif path2.exists():
            with open(path2,'r') as f:
                lines = f.readlines()
            assert lines[0].split()[-1] == "e_mass"
            assert lines[1].split()[-1] == "H_mass"
            lines = lines[0:2]
        else:
            return None

        # string convert dict
        data = {}
        t,x,y,z = lines[0].split()[:4]
        data['cbm'] = float(t)
        data['cbm-x'] = float(x)
        data['cbm-y'] = float(y)
        data['cbm-z'] = float(z)
        t,x,y,z = lines[1].split()[:4]
        data['vbm'] = float(t)
        data['vbm-x'] = float(x)
        data['vbm-y'] = float(y)
        data['vbm-z'] = float(z)
        return data

    @classmethod
    def get_effective_mass(self, path:str):
        '''H-mass e-mass'''
        data = self.get_effective_mass_dict(path)
        if data != None:
            return data['vbm'], data['cbm']
        return None
    
    def get_trace(self, path):

        path = Path(path)
        if path.is_dir():
            if self.mode in ['BTP1', 'BTP2']:
                path = path / (path.absolute().name + '.trace')
            else:
                path = path / 'SCF.trace'

        with open(path, 'r') as f:
            title = f.readline()
            if self.mode == 'BTPm':        # title == stra
                columns = ['Ef','T','N','DOS','S','sigma_tao','hall','thermal','powerfac'] 
                #units = ['eV','K','e/uc','e/uc','V/K','1/(ohm m)*1/s','m^3/C','W/mK*1/s','J/(mol K)']

            elif self.mode == 'BTP1':      # title == strb
                columns = ['Ef','T','N','DOS','S','sigma_tao','R_H','kappa0','c','chi']
                units = ['Ry','K','e/uc','e/uc','V/K','1/(ohm m)*1/s','m^3/C','W/mK*1/s','J/(mol K)']

            elif self.mode == 'BTP2':      # title == strc
                columns = ['Ef','T','N','DOS','S','sigma_tao','R_H','kappa0','c','chi']
                units = ['Ry','K','e/uc','1/(Ha*uc)','V/K','1/(ohm*m*s)','m**3/C','W/(m*K*s)','J/(mol*K)']
            else:
                raise ValueError("Unknown trace type.")

            data = []
            for line in f:
                data.append(line.split())

        data = np.array(data,dtype=float)
        return pd.DataFrame(data,columns=columns)

    def get_condtens(self, path):

        path = Path(path)
        if path.is_dir():
            if self.mode in ['BTP1', 'BTP2']:
                filename = path.absolute().name + '.condtens'
            else:
                filename = 'SCF.condtens'
            path = path / filename

        with open(path, 'r') as f:
            if self.mode in ['BTP1', 'BTP2']:   
                title = f.readline()
            columns = ['Ef','T','N','condxx','condxy','condxz',
                                    'condyx','condyy','condyz',
                                    'condzx','condzy','condzz',
                           'seebeckxx','seebeckxy','seebeckxz',
                           'seebeckyx','seebeckyy','seebeckyz',
                           'seebeckzx','seebeckzy','seebeckzz',]

            data = []
            for line in f:
                data.append(line.split()[:21])

        data = np.array(data,dtype=float)
        return pd.DataFrame(data,columns=columns)

    def get_data(self, path, T=300, with_condtens=True):
        volume = self.get_volume(path)
        bandgap = self.get_bandgap(path) 
        trace = self.get_trace(path)

        #if len(np.unique(trace['T'])) > 1:
        #    trace = trace[trace['T']==T]
        if self.mode in ['BTP1', 'BTPm']:
            fermi = 0
        else:
            fermi = BTP1.get_fermi(path)

        if self.mode == 'BTP1' or self.mode == 'BTP2':
            energy = trace['Ef'].values * Ry
            N = trace['N'].values
            dos = trace['DOS'].values
            sigma_tao = trace['sigma_tao'].values
            seebeck = trace['S'].values

        elif self.mode == 'BTPm':
            energy = trace['Ef'].values 
            N = trace['N'].values * (volume*scipy.constants.angstrom**6/Bohr_radius**3) * 1e6
            dos = trace['DOS'].values
            sigma_tao = trace['sigma_tao'].values
            seebeck = trace['S'].values

        condtens = None
        if with_condtens:
            condtens = self.get_condtens(path)
            condtens = condtens.values[:,3:]
            # condtens = condtens.values

        return self.Result(energy, N, dos, sigma_tao, seebeck, condtens, bandgap, volume, fermi)

    def get_bandgap(self, path, filename='OUTCAR'):
        from jamip.analysis.vasp import BandFinder
        bf = BandFinder(path).get_data()
        return bf.get_bandgap()['indirect']

    def get_volume(self, path):
        from jamip.analysis.vasp.outcar import GrepOutcar
        path = Path(path)
        if path.is_dir():
            file1 = path/"OUTCAR"
            name = path.absolute().name if self.mode != 'BTPm' else 'SCF'
            file2 = path/(name+'.outputtrans')
            if file1.exists():
                path = file1
            elif file2.exists():
                path = file2

        if path.name == "OUTCAR":
            # unit angstrom
            volume = GrepOutcar().volume(path.parent)
        elif path.suffix == '.outputtrans':
            volume = None 
            with open(path,'r') as f:
                for line in f:
                    if line.startswith("VOLUME"):
                        volume = float(line.split()[1])
                        break
            # unit Bohr radius -> angstrom
            volume = volume * Bohr_radius**3 * 1e10**3
        
        return volume

    @classmethod 
    def runBTP1(cls, path, mode='BTP1', opt=True, **kwargs):
        from jamip.analysis.vasp.outcar import GrepOutcar
        path = Path(path)

        def write_def(rows): 
            with open("BoltzTraP.def", 'w') as f:
                for row in rows:
                    f.write(f"{row[0]},'{row[1]}','{row[2]}','{row[3]}',{row[4]}\n")
            
        root = Path.cwd()
        if mode == 'BTP1':

            lsorbit = GrepOutcar().lsorbit(path)
            soc = True if lsorbit else False 

            if opt and (Path(path)/"PROCAR_OPT").exists():
                BTP1.set_energy(path, infile="PROCAR_OPT",mode=mode,soc=soc,**kwargs)
            else:
                BTP1.set_energy(path, mode=mode,soc=soc,**kwargs)
            BTP1.set_struct(path, mode=mode, soc=soc, **kwargs)
            BTP1.set_intrans(path, mode=mode, **kwargs)
 
            os.chdir(path)
            for line in os.popen("x_trans BoltzTraP").readlines():
                # print(line.rstrip())
                pass

            # Debug
            if not Path(f"{path.name}.trace").exists():
                with open("BoltzTraP.def", 'r') as f:
                    lines = f.readlines()

                rowdatas = []
                for line in lines:
                    row = line.split(',')
                    row = [i.strip().strip("'") for i in row]
                    infile = Path(row[1])
                    print(row)
                    if row[2] == 'old' and not infile.exists():
                        if infile.suffix == '.energy' and Path(f"{row[1]}so").exists():
                            row[1] = f"{row[1]}so"
                        elif row[0] == '-1':
                            pass
                        else:
                            raise OSError(f"Miss inputfile: {row[1]}")
                    rowdatas.append(row)
                write_def(rowdatas)

                for line in os.popen("BoltzTraP BoltzTraP.def").readlines():
                    print(line.rstrip())
                    pass


        elif mode == 'BTPm':
            from jamip.abtools.vasp.vaspio import VaspIO
            from jamip.structure import read

            if not (path/"SYMMETRY").exists():
                s = read(path/"POSCAR")
                VaspIO.write_symmetry(s, stdout=path, symprec=1e-4)

            os.chdir(path)
            for line in os.popen("./massall.x").readlines():
                # print(line.rstrip())
                pass

        os.chdir(root)

        return cls
      
    @classmethod 
    def runBTP2(cls, path, overwrite=False, **kwargs):

        root = Path(path).absolute() 
        bt2file = root / (root.name + '.bt2')
        tracefile = root / (root.name + '.trace') 
        condtensfile = root / (root.name + '.condtens') 
        #print(bt2file)
        if overwrite:
            BTP2.save_calculation(root, output=bt2file)
            BTP2.save_trace(bt2file, tracefile, condtensfile, **kwargs)
        else:
            if not bt2file.exists():
                BTP2.save_calculation(root, output=bt2file)
            if not tracefile.exists() or not condtensfile.exists():
                BTP2.save_trace(bt2file, tracefile, condtensfile, **kwargs)

        return cls
