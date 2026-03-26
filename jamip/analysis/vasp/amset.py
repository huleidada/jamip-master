import numpy as np
import pandas as pd
import pathlib
import warnings
import sys

warnings.simplefilter("ignore")

class Amset:

    def __init__(self, stdin, soft='vasp', **kwargs):
        """
        soft: vasp & wien2k
        """
        self.stdin = pathlib.Path(stdin)
        self.soft = soft
        self.energy_cutoff = kwargs.get('energy_cutoff', 1.5)
        self.zero_weighted_kpoints = kwargs.get('zero_weighted_kpoints', 'prefer')
        self.symprec = kwargs.pop("symprec", 1e-3)
        self.ibands = None

    def get_deform(self, deformdir=None, **kwargs):
        """
        Read deformation calculations and extract deformation potentials.
        Necessary for ADP
        """
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.util.string import unicodeify_spacegroup
        from amset.deformation.io import parse_calculation
        from amset.tools.deformation import check_calculation
        from amset.deformation.potentials import (
            calculate_deformation_potentials,
            get_strain_mapping,
            get_symmetrized_strain_mapping,
            strain_coverage_ok,
        )
        from amset.electronic_structure.symmetry import expand_bandstructure
 
        symprec = kwargs.pop("symprec", self.symprec)
        symprec_deformation = kwargs.pop("symprec_deformation", 1e-3)
        zwk = self.zero_weighted_kpoints

        if deformdir is None:
            deformdir = self.stdin / "electric" / "deform"

        bulk_folder = None
        deformation_folders = []
        for path in pathlib.Path(deformdir).iterdir():
            if path.name == 'scf':
                bulk_folder = path
            elif path.is_dir():
                deformation_folders.append(path)
 
        bulk_calculation = parse_calculation(bulk_folder, zero_weighted_kpoints=zwk)
        bulk_structure = bulk_calculation["bandstructure"].structure
        #sga = SpacegroupAnalyzer(bulk_structure, symprec=symprec)
        #spg_symbol = unicodeify_spacegroup(sga.get_space_group_symbol())
        #spg_number = sga.get_space_group_number()
 
        deformation_calculations = []
        for deformation_folder in deformation_folders:
            deformation_calculation = parse_calculation(deformation_folder, zero_weighted_kpoints=zwk)
            deformation_calculation = check_calculation(bulk_calculation, deformation_calculation)
            if deformation_calculation is not False:
                deformation_calculations.append(deformation_calculation)
 
        strain_mapping = get_strain_mapping(bulk_structure, deformation_calculations)
        bulk_calculation["bandstructure"] = expand_bandstructure(bulk_calculation["bandstructure"], symprec=symprec)
        strain_mapping = get_symmetrized_strain_mapping(
            bulk_structure,
            strain_mapping,
            symprec=symprec,
            symprec_deformation=symprec_deformation,
        )
 
        if not strain_coverage_ok(list(strain_mapping.keys())):
            sys.exit()
 
        deformation_potentials = calculate_deformation_potentials(bulk_calculation, strain_mapping)
        return bulk_calculation, deformation_potentials

    def get_deformation_potential(self, deformdir=None):

        # get data
        band, dp = self.get_deform(deformdir)
        kpoints = np.array([k.frac_coords for k in band['bandstructure'].kpoints])
        fermi = self.get_fermi(deformdir)

        # get cbvb
        cbm = None # ispin, iband, ikpt, energy
        vbm = None # ispin, iband, ikpt, energy
        for ispin,spin_band in band['bandstructure'].bands.items():
            for iband in np.arange(len(spin_band)):
                # If the maximum energy exceeds the Fermi level or partially empty occupancy
                if max(spin_band[iband]) > fermi:
                    ikpt = np.argmax(spin_band[iband-1])
                    energy = spin_band[iband-1, ikpt]
                    if vbm == None or vbm[3] < energy:
                        vbm = ispin, iband-1, ikpt, energy
                        
                    ikpt = np.argmin(spin_band[iband])
                    energy = spin_band[iband, ikpt]
                    if cbm == None or cbm[3] > energy:
                        cbm = ispin, iband, ikpt, energy
                    break
                    
        dp_cb = dp[cbm[0]][cbm[1], cbm[2]]
        dp_vb = dp[vbm[0]][vbm[1], vbm[2]]
        #print(cbm, kpoints[cbm[2]])
        #print(vbm, kpoints[vbm[2]])

        return dp_cb, dp_vb

    def set_deform(self, deformdir=None, output='deformation.h5', overwrite=False, **kwargs):
        """
        Read deformation calculations and extract deformation potentials.
        Necessary for ADP
        """
        from amset.deformation.io import write_deformation_potentials
        from amset.deformation.potentials import extract_bands
        from amset.electronic_structure.common import get_ibands
        from amset.electronic_structure.kpoints import get_kpoints_from_bandstructure
 
        if pathlib.Path(output).exists() and not overwrite:
            return

        bulk_calculation, deformation_potentials = self.get_deform(deformdir, **kwargs)
        bulk_structure = bulk_calculation["bandstructure"].structure
        ibands = self.ibands
        if ibands is None:
            ibands = get_ibands(self.energy_cutoff, bulk_calculation["bandstructure"])
            self.ibands = ibands
        print(self.ibands, ibands)

        deformation_potentials = extract_bands(deformation_potentials, ibands)
        kpoints = get_kpoints_from_bandstructure(bulk_calculation["bandstructure"])
        for k,v in deformation_potentials.items():
            print(self.ibands, k, v.shape)
        filename = write_deformation_potentials(
            deformation_potentials, kpoints, bulk_structure, filename=output
        )
        return deformation_potentials

    def set_wave(self, path=None, output='wavefunction.h5', **kwargs):
        """Extract wavefunction coefficients from a WAVECAR"""
        from pymatgen.io.vasp import BSVasprun
        from amset.electronic_structure.common import (
            get_band_structure,
            get_ibands,
            get_zero_weighted_kpoint_indices,
        )
        from amset.tools.wavefunction import _wavefunction_vasp
        from amset.wavefunction.io import write_coefficients

        if pathlib.Path(output).exists():
            return
        if path is None:
            path = self.stdin / "scf"
            #path = self.stdin / "optics" / "dielectric"
        else:
            path = pathlib.Path(path)
            if path.is_file():
                path = path.parent
 
        symprec_deformation = kwargs.pop("symprec_deformation", 1e-3)
        planewave_cutoff = kwargs.pop("planewave_cutoff", None)
        pawpyseed = kwargs.pop("pawpyseed", False)
 
        vr = BSVasprun(path/"vasprun.xml")
        bs = get_band_structure(vr, zero_weighted=self.zero_weighted_kpoints)
 
        ibands = self.ibands
        if ibands is None:
            ibands = get_ibands(self.energy_cutoff, bs)
            self.ibands = ibands
        print(self.ibands, ibands)
        ikpoints = get_zero_weighted_kpoint_indices(vr, self.zero_weighted_kpoints)
        coeffs, gpoints = _wavefunction_vasp(
            ibands, planewave_cutoff, ikpoints, 
            directory=path, wavecar="WAVECAR", vasp_type=None
        )
        kpoints = np.array([k.frac_coords for k in bs.kpoints])
        structure = vr.final_structure
        write_coefficients(coeffs, gpoints, kpoints, structure, filename=output)

    def set_input(self, scattering_type:list=["ADP","IMP","POP"], dfptdir=None, elasticdir=None, banddir=None, nworkers=10, 
            dopping="auto"):
        '''
        ADP (acoustic deformation potential scattering)
        IMP (ionized impurity scattering)
        PIE (piezoelectric scattering)
        POP (polar optical phonon scattering)
        CRT (constant relaxation time)
        MFP (mean free path scattering)
        '''
        from jamip.utils.logger import dump_yaml
        params = {"scattering_type": scattering_type,
                  #"doping": [-1e18,-5e18,-1e19,-5e19,-1e20,-5e20,-1e21,-5e21],
                  "doping": [-1e18,-1e19,1e18,1e19],
                  #"doping": [-1e15,-5e15,-1e16,-5e16,-1e17,-5e17,-1e18,-5e18],
                  "temperatures": [300,],
                  # electronic_structure settings
                  "interpolation_factor": 5,
                  # materials properties
                  "deformation_potential": "deformation.h5",
                  # performance settings
                  "nworkers": nworkers,
                  "cache_wavefunction": False,
                  "write_mesh": True,
                  "file_format":'txt'
                 }
        if dfptdir is None:
            dfptdir = self.stdin / "optics" / "dielectric"
            if not dfptdir.exists():
                dfptdir = self.stdin / "phonon" / "dfpt"
        if elasticdir is None:
            elasticdir = self.stdin / "mechanic" / "elastic"
        if banddir is None:
            banddir = self.stdin / "electric" / "band"

        # dfpt path
        path = pathlib.Path(dfptdir)
        if "POP" in scattering_type or "IMP" in scattering_type:
            params["static_dielectric"] = self.get_static_dielectric(path).tolist()
        if "POP" in scattering_type or "PIE" in scattering_type:
            params["high_frequency_dielectric"] = self.get_high_frequency_dielectric(path).tolist()
        if "POP" in scattering_type:
            params["pop_frequency"] = float(self.get_pop_frequency(path, path))
        if "PIE" in scattering_type:
            params["piezoelectric_constant"] = self.get_piezoelectric(path).tolist()
        if "IMP" in scattering_type:
            params["defect_charge"] = 1
            params["compensation_factor"] = 2

        params["bandgap"] = float(self.get_bandgap(banddir))
        if isinstance(dopping, str) and dopping == "auto":
            if params["bandgap"] < 1:
                params["dopping"] = [-1e15,-1e16,-1e17,-1e18,-1e19,-1e20,-1e21,1e15,1e16,1e17,1e18,1e19,1e20,1e21] 
            elif params["bandgap"] >3:
                params["dopping"] = [-1e13,-1e14,-1e15,-1e16,-1e17,-1e18,-1e19,1e13,1e14,1e15,1e16,1e17,1e18,1e19] 
            elif params["bandgap"] < 1:
                params["dopping"] = [-1e15,-1e16,-1e17,-1e18,-1e19,1e15,1e16,1e17,1e18,1e19]
        elif not (doppint is None):
            params["dopping"] = dopping

        if "ADP" in scattering_type or "PIE" in scattering_type:
            params["elastic_constant"] = self.get_elastic(elasticdir).tolist()
        dump_yaml(params, 'amset.yaml')     

    def get_elastic(self, path):
        from jamip.analysis.vasp import GrepOutcar
        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent
        path = str(path)
        # unit kBar -> GPa
        return GrepOutcar().elastic(path) / 10

    def get_piezoelectric(self, path, filename='OUTCAR'):
        from jamip.analysis.vasp import GrepOutcar
        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent
        path = str(path)
        piezo = GrepOutcar().piezoelectric(path)
        piezo_ionic = GrepOutcar().piezoelectric_ionic(path)
        return piezo + piezo_ionic

    def get_static_dielectric(self, path, filename='OUTCAR'):
        from jamip.analysis.vasp.outcar import GrepOutcar
        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent
        path = str(path)
        di = GrepOutcar().dielectric(path)
        di_ionic = GrepOutcar().dielectric_ionic(path)
        return di + di_ionic

    def get_high_frequency_dielectric(self, path, filename='OUTCAR'):
        from jamip.analysis.vasp.outcar import GrepOutcar
        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent
        path = str(path)
        di = GrepOutcar().dielectric(path)
        return di 

    def get_fermi(self, path, source='OUTCAR'):
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.vasp.xml import Xml
        if path is None:
            path = self.stdin / "electric" / "deform" / "scf"

        if source.lower() == 'xml':
            efermi = Xml(path).fermi_energy()
        else:
            efermi = GrepOutcar().fermi_energy(path)
        return efermi

    def get_bandgap(self, path=None):
        from jamip.analysis.vasp import BandFinder
        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent
        bf = BandFinder(path).get_data()
        return bf.get_bandgap()['indirect']

    def get_Cz(self, path):
        r"""
        $$ Cz = (2*Pi*e^2 / A) \sum {\frac {q*Z*e} {|q|\sqrt{2*m_l*wjq}}} $$
        """
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.structure.elementInfo import ElementDict
        from jamip.structure import read

        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent

        s = read(path/'POSCAR')

        path = str(path)
        # freq: natom,3
        # eigen: natom*3,natom,3
        # born: natom,3,3
        freq, eigen = GrepOutcar().frequency(path)
        # x,y,z,dx,dy,dz
        eigen = eigen[:,:,3:]
        born_charge = GrepOutcar().born(path)
        pi = 3.141592653
        electron = 1.6E-19
        proton = 1.673E-24 # in the unit of gram

        elements = s.get_elements(type='symbol')
        natoms = len(elements)
        area = np.linalg.norm(np.cross(s.lattice[0], s.lattice[1])) * 1e-16  # A^2 > cm^2
        coeff = 2.0 * pi * (electron ** 2) / area

        C_z = []
        for imode in range(3*natoms):
            si = 0.0
            for i in range(natoms):
                t1 = sum(born_charge[i][:2,:2] @ eigen[imode][i][:2])
                # t1 = born_charge[i][0][0] * eigen[imode][i][0] + \
                #      born_charge[i][0][1] * eigen[imode][i][1] + \
                #      born_charge[i][1][0] * eigen[imode][i][0] + \
                #      born_charge[i][1][1] * eigen[imode][i][1]
                t1 = t1 / np.math.sqrt(2.0 * freq[imode] * ElementDict[elements[i]]['mass'] * proton)
                si = si + t1
            C_z.append(abs(coeff*si*1.0E-7/electron/np.math.sqrt(2.0)))

        return np.mean(np.sort(C_z)[::-1][:2])

    def get_dielectric_thickness(self, path):
        """
        thickness + vdw_radius of surface_atom
        """
        from jamip.structure import read
        from mendeleev import element
        path = pathlib.Path(path)
        if path.is_file():
            path = path.parent
        s = read(path/'POSCAR')
        elements = s.get_elements(type='symbol')
        z = s.get_positions()[:,2]
        z = z - np.floor(z)
        zsort = np.sort(z)
        zdist = np.insert(zsort[1:]-zsort[:-1], 0, zsort[0]-zsort[-1]+1)
        idx = np.argmax(zdist)
        thickness_in_fraction = np.max(zdist)#np.max(zsort[1:]-zsort[:-1]), zsort[0]-zsort[-1]+1)

        species = elements[np.where(abs(z-zsort[idx-1])<1e-3)]
        r_vdw1 = max([element(i).vdw_radius for i in species]) / 100 # pm -> A
        species = elements[np.where(abs(z-zsort[idx])<1e-3)]
        r_vdw2 = max([element(i).vdw_radius for i in species]) / 100 # pm -> A
        #r_cov2 = max([element(i).covalent_radius_cordero for i in species]) / 100 # pm -> A

        thickness = (1-thickness_in_fraction) * np.linalg.norm(s.lattice[2]) + r_vdw1 + r_vdw2
        return thickness 

    def get_frohlich_epc_strength_2d(self, path):
        electron = 1.6E-19

        Cz = self.get_Cz(path)
        epsilon = self.get_static_dielectric(path)
        thickness = self.get_dielectric_thickness(path) 
        thickness = 6.0
        epsilon_in_plane = (epsilon[0,0] + epsilon[1,1])/2
        print(epsilon_in_plane)
        epsilon_in_plane = 15.5
        r_eff = epsilon_in_plane * thickness / 2
        print(epsilon_in_plane)
        print(r_eff)
        gFr = Cz / (1 + r_eff * electron) 
        return gFr

    def get_frohlich_epc_strength_3d(self, path):
        electron = 1.6E-19

        Cz = self.get_Cz(path)
        epsilon = self.get_static_dielectric(path)
        thickness = self.get_thickness(path) 
        epsilon_in_plane = (epsilon[0,0] + epsilon[1,1])/2
        r_eff = epsilon_in_plane * thickness / 2
        gFr = Cz / (1 + r_eff * electron) 
        return gFr

    def born(self,path):
        from jamip.analysis.vasp.outcar import GrepOutcar
        return GrepOutcar().born(path)

    #def get_polar_phonon_frequency(self, outcar, vasprun):
    def get_pop_frequency(self, outcar:str, vasprun:str):
        from pymatgen.io.vasp import Outcar, Vasprun
        from amset.tools.phonon_frequency import phonon_frequency, effective_phonon_frequency_from_vasp_files
 
        path = pathlib.Path(outcar)
        if path.is_dir():
            outcar = path/"OUTCAR"

        path = pathlib.Path(vasprun)
        if path.is_dir():
            vasprun = vasprun/"vasprun.xml"

        #effective_frequency = phonon_frequency(vasprun, outcar)
        effective_frequency, weights, freqs = effective_phonon_frequency_from_vasp_files(
            vasprun, outcar
        )
        try:
            effective_frequency = phonon_frequency(vasprun, outcar)
        except:
            outcar = Outcar(outcar)
            vasprun = Vasprun(vasprun)
         
            elements = vasprun.final_structure.composition.elements
            if len(set(elements)) == 1:
                raise Exception(
                    "This system only contains a single element and is therefore not polar.\n"
                    "There will no polar optical phonon scattering and you do not need to set "
                    "pop_frequency."
                )
         


        return effective_frequency

    def run(self, path=None, **kwargs):
        from amset.plot.rates import RatesPlotter
        from amset.plot.mobility import MobilityPlotter
        from jamip.utils.logger import load_yaml
        from amset.core.run import Runner

        if path is None: 
            vasprun = self.stdin/"scf"/"vasprun.xml"
        else:
            path = pathlib.Path(path)
            vasprun = path/"vasprun.xml" if path.is_dir() else path

        settings = load_yaml('amset.yaml')
        # get bandstructure & nelect & soc from vasprun.xml 
        runner = Runner.from_vasprun(vasprun, settings)
        amset_data = runner.run()
        amset_data.to_file(prefix="amset", file_format="yaml")
        
        plotter = RatesPlotter(amset_data)
        plt = plotter.get_plot()
        plt.savefig("Si_rates.png", bbox_inches="tight", dpi=400)
 
        plotter_2=MobilityPlotter(amset_data)
        plt_2=plotter_2.get_plot()
        plt_2.savefig("Si_mobility_negative.png", bbox_inches="tight", dpi=400)
        '''
        '''
