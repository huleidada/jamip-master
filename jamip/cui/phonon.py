import os
import numpy as np
from jamip.analysis.vasp import GrepOutcar
import pathlib



class PhononTools(GrepOutcar):

    __path = None
    __info = None

    def __init__(self, params, **kwargs):

        if params['phonon'] == 'help':
            self.write_help()
        else:
            self.path = params.get('pool', None)
            if params['phonon'] == 'fc2':                
                self.build_fc2()
            elif params['phonon'] == 'fc3':
                self.build_fc3()
            elif params['phonon'] == 'raman':
                self.build_raman()
            elif params['phonon'] == 'gruneisen':
                self.build_gruneisen()
            elif params['phonon'] == 'band':
                self.build_band()

    @property
    def path(self):
        return pathlib.Path(self.__path)

    @path.setter
    def path(self,value=None):

        if isinstance(value, list):
            value = value[0]

        if value == None: 
            value = pathlib.Path.cwd()
        else:
            value = pathlib.Path(value)
        if (value/'info.hdf5').exists():
            self.__info = True
        elif value.is_file() and value.name == 'info.hdf5':
            self.__info = True
            value = value.parent
        else:
            raise OSError("Found no jamip phonopy path")
        self.__path = value

    def build_fc2(self):
        from jamip.analysis.vasp import PhononFinder
        from phonopy.interface.phonopy_yaml import PhonopyYaml
        from phonopy.interface.vasp import get_born_OUTCAR

        pf = PhononFinder(self.path)
        if self.__info: 
            phonon = pf.get_phonon_from_info()
        else:
            phonon = pf.get_phonon_from_calculation()

        # get born %
        bornpath = self.path / 'optics' / 'born'
        if bornpath.exists():
            borns, epsilon, atom_indices = get_born_OUTCAR(bornpath/'POSCAR', bornpath/'OUTCAR', symprec=1e-5)
            with open('BORN','w') as f:
                text = "# epsilon and Z* of atoms "
                text += " ".join(["%d" % n for n in atom_indices + 1])
                f.write(text+'\n')
                f.write(("%13.8f " * 9 + '\n') % tuple(epsilon.flatten()))
                for z in borns:
                    f.write(("%13.8f " * 9 + '\n') % tuple(z.flatten()))

        # get force %
        forcepath = pf.fc2dir
        if forcepath.exists():
            phonon = pf.set_force_constants(phonon, stdin=forcepath, output=pathlib.Path.cwd())

        conf = {'dim': ' '.join(phonon.supercell_matrix.reshape(-1).astype(str)),
                'source': self.path,
                'cell_filename': "scf/POSCAR",
                'create_displacements': True,}
        # vasp calculator %
        units = {'force_constants_unit': 'eV/angstrom^2',
                 'length_unit': 'angstrom'}

        phpy_yaml = PhonopyYaml(settings={
                                    'force_sets': True,
                                    'born_effective_charge': False,
                                    'dielectric_constant': False,
                                    'displacements': True},
                                configuration=conf,
                                physical_units=units,
                               )
        phpy_yaml.set_phonon_info(phonon)
        with open('phonopy_disp.yaml', 'w') as w:
            w.write(str(phpy_yaml))

    def build_fc3(self):
        from jamip.analysis.vasp import PhononFinder
        from phono3py.interface.phono3py_yaml import Phono3pyYaml
        from phono3py.file_IO import write_FORCES_FC3, parse_FORCES_FC3

        pf = PhononFinder(self.path)
        if self.__info: 
            phonon = pf.get_phonon3_from_info()
        else:
            phonon = pf.get_phonon3_from_calculation()

        # get force %
        forcepath = pf.fc3dir
        if forcepath.exists():
            phonon.forces = pf.get_forces(forcepath)   # phonopy 3.0.3
            #dataset = phonon.displacement_dataset
            dataset = phonon.dataset
            write_FORCES_FC3(dataset,forces_fc3=phonon.fc3,filename='FORCES_FC3')

            # load force %
            parse_FORCES_FC3(dataset,filename='FORCES_FC3')
            #phonon.set_displacement_dataset(dataset)   # phonopy 3.0.3
            phonon.produce_fc3(is_compact_fc=False)

        conf = {'dim': ' '.join(phonon.supercell_matrix.reshape(-1).astype(str)),
                'source': self.path,
                'cell_filename': "scf/POSCAR",
                'create_displacements': True,}
        # vasp calculator %
        units = {'force_constants_unit': 'eV/angstrom^2',
                 'length_unit': 'angstrom'}

        phpy_yaml = Phono3pyYaml(settings={
                                    'force_sets': True,
                                    'born_effective_charge': False,
                                    'dielectric_constant': False,
                                    'displacements': True},
                                configuration=conf,
                                physical_units=units,
                               )
        phpy_yaml.set_phonon_info(phonon)
        with open('phono3py_disp.yaml', 'w') as w:
            w.write(str(phpy_yaml))

    def build_raman(self):
        from phonopy.interface.vasp import get_born_OUTCAR
        import h5py
        
        # get epsilon %
        ramanpath = self.path / 'phonon/raman'
        epsilons = []
        for dirname in np.sort(os.listdir(ramanpath)):
            poscar = ramanpath / dirname / 'POSCAR'
            outcar = ramanpath / dirname / 'OUTCAR'
            if outcar.exists():
                borns, epsilon, atom_indices = get_born_OUTCAR(poscar, outcar, symprec=1e-5)
                epsilons.append(epsilon)

        # get info %
        with h5py.File(self.path/"info.hdf5", "r") as h5:

            info = h5['raman']
            numstep = info.attrs['numstep']
            frequency = info.attrs['frequency']
            band_indices = info.attrs['band_indices']
            disps = info.attrs['displacement_step']
            maxdisps = info.attrs['max_cartesian_displacement']
            volume = info.attrs['volume']

        assert len(epsilons) == len(disps) == len(band_indices) * numstep, "Data mismatch"

        with open("Raman.yaml", 'w') as f:
            f.write("frequency_units: thz\n")
            f.write("step_units: sqrt(amu) * Ang\n")
            f.write("distance_units: Ang\n")
            f.write("volume_units: Ang ^ 3\n")
            f.write("cell_volume: %.4f\n\n" %volume)
            f.write("displacement_sets:\n")
            for i,index in enumerate(band_indices):
                f.write('- # %d\n' %(i+1))
                f.write('  band_index: %d\n' %(index+1))
                f.write('  frequency: %.10f\n' %frequency[0][index])
                f.write('  displacements:\n')
                for j in range(2):
                    f.write('   - # Step %d\n' %(j+1))
                    f.write('     displacement_step: %.8f\n' %disps[i*2+j])
                    f.write('     max_cartesian_displacement: %.8f\n' %maxdisps[i*2+j])
                    f.write('     epsilon_static:\n')
                    for k in range(3):
                        eps = epsilons[i*2+j][k]
                        f.write('     - [ %16.8f, %16.8f, %16.8f  ]\n' %(eps[0],eps[1],eps[2]))
        
        #info = {'mesh': [1,1,1], 'qpoint': [0,0,0], 'numstep':2, 'step': Ramanstep,
        #        'band_indices': band_indices, 'frequency': dataset['frequencies'],
        #        'disps': disps, 'main': 'phonon/raman'}

        

            
    def write_help(self):
        
        print("Rebuild phonopy disp.yaml and force_sets: jp --phonon fc2")
        print("Phonopy band structure: jp --phonon band")
        print("Thermal properties: phonopy -t -p mesh.conf")
        print("Band Structure: phonopy -p band.conf")
        print("PDOS: phonopy -p pdos.conf")
        print("")
        print("Rebuild phono3py disp.yaml and forces_fc3: jp --phonon fc3")
        print("Rename force_sets calculated by phonopy: mv FORCE_SETS FORCES_FC2")
        print("Create fc2.hdf5 and f3.hdf5: phono3py --sym-fc")
        print("Thermal conductivity calculation: phono3py --mesh='11 11 11' --br")
        print("")


    def build_gruneisen(self):
        from jamip.analysis.vasp import PhononFinder
        mesh = input("Mesh: ").split()
        mesh = [int(i) for i in mesh]
        pf = PhononFinder(self.path)
        gruneisen = pf.get_gruneisen_from_info()
        gruneisen.set_mesh(mesh)
        gruneisen.write_yaml_mesh()
        

    def build_band(self):
        # from jamip.utils.brillouin_zone import HighSymmetryKpath
        # from jamip.structure import read, Structure
        # from jamip.analysis.vasp import PhononFinder
        # from jamip.utils.logger import load_yaml, load_hdf5
        # from jamip.utils.convert import format_latex
        # from jamip.structure.atomic_number import number
        from phonopy.cui.load import load
        from phonopy.interface.phonopy_yaml import PhonopyYaml, PhonopyYamlLoader, load_yaml

        yamlfile = pathlib.Path('phonopy_disp.yaml')
        yaml_data = load_yaml(yamlfile)
        phyml_loader = PhonopyYamlLoader(yaml_data)
        phyml_loader.parse()
        symprec = phyml_loader._yaml['phonopy'].get('symmetry_tolerance', 1e-5)

        #phpy_yaml = read_phonopy_yaml(yamlfile)
        #phpy_yaml = PhonopyYaml()
        #phpy_yaml.read(yamlfile)
        #symprec = phpy_yaml._yaml['phonopy'].get('symmetry_tolerance', 1e-5)
        #symprec = phpy_yaml._data.data #symmetry #['phonopy'].get('symmetry_tolerance', 1e-5)

        if not yamlfile.exists():
            self.build()

        phonon = load(phonopy_yaml=yamlfile, symprec=symprec)
        #phonon.symmetrize_force_constants()
        phonon.auto_band_structure(write_yaml=True)

        '''
        # 先尝试从当前目录的phonon_disp.yaml读，如果没有在读info.hdf5
        if (self.path/'phonopy_disp.yaml').exists():
            phonon = load_yaml(self.path/'phonopy_disp.yaml')
            primcell = phonon['primitive_cell']
            lattice = primcell['lattice']
            elements = [number[i['symbol']] for i in primcell['points']]
            positions = [i['coordinates'] for i in primcell['points']]
            cell = (lattice, positions, elements)
            symprec = phonon['phonopy']['symmetry_tolerance']
        else:
            hdf5file = self.path/'info.hdf5'
            cell = load_hdf5('/structure/phonon', hdf5file)
            # structure = Structure.from_cell(cell)
            info = load_hdf5('phonon', hdf5file)
            symprec = info['symprec']

        bz = HighSymmetryKpath()
        kpoint = bz.get_HSKP(cell, symprec=symprec)
        labels = []
        coords = []
        # points to string %
        for points in kpoint['Path']:
            label = ' '.join([format_latex(p) for p in points])
            coord = np.around([kpoint['Kpoints'][p] for p in points], 4).reshape(-1)
            coord = ' '.join([str(i) for i in coord])
            labels.append(label)
            coords.append(coord)

        # write band.conf %
        with open('band.conf', 'w') as f:
            f.write('BAND = ' + ' , '.join(coords) + '\n')
            f.write('BAND_LABELS = ' + ' , '.join(labels) + '\n')
            f.write('BAND_POINTS = 51')
        '''

    def hdf5(self):
        import h5py

        if os.path.exists('info.hdf5'):
            with h5py.File("info.hdf5", 'r') as h5:
                #print(list(h5.keys()))
                h5.visititems(print)
