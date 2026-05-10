import os
import numpy as np
from jamip.analysis.base import Finder
from jamip.utils.logger import load_hdf5
from .outcar import GrepOutcar
from .xml import Xml
import pathlib
        

class PhononFinder(Finder):

    def __init__(self,stdin=None):
        self.task = 'phonon'
        self.soft = 'vasp'
        self.stdin = stdin

    @property
    def forcedir(self):
        """ Deprecated: Use `fc2dir` or `fc3dir` instead. """
        return self._stdin/'phonon'/'fc2' if self.file == 'jamip' else self._stdin

    @property
    def fc2dir(self):
        return self._stdin/'phonon'/'fc2' if self.file == 'jamip' else self._stdin
    
    @property
    def fc3dir(self):
        return self._stdin/'phonon'/'fc3' if self.file == 'jamip' else self._stdin

    @property
    def gruneisendir(self):
        return self._stdin/'phonon'/'gruneisen' if self.file == 'jamip' else self._stdin

    @property
    def softmodedir(self):
        return self._stdin/'phonon'/'softmode' if self.file == 'jamip' else self._stdin
    
    @property
    def h5(self):
        return self._stdin/'info.hdf5'

    def get_softmode_result(self):
        data = {}
        for path in self.softmodedir.iterdir():
            if not path.name.startswith('mode'): continue
            data[dir] = GrepOutcar().free_energy(path)
        return data

    def get_phonon_from_yaml(self):
        from phonopy.cui.load import load
        from phonopy.interface.phonopy_yaml import PhonopyYaml

        yamlfile = self.stdin/'phonopy_disp.yaml'
        if not yamlfile.exists():
            raise OSError("Yaml file not exists!")

        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(yamlfile)
        symprec = phpy_yaml._yaml['phonopy'].get('symmetry_tolerance', 1e-5)
        phonon = load(phonopy_yaml=yamlfile, symprec=symprec)
        return phonon

    def get_atoms_from_info(self, key:str):
        from phonopy.structure.atoms import PhonopyAtoms

        cell = load_hdf5(key, self.h5)
        if cell != None:
            lattice, positions, elements = cell
            unitcell = PhonopyAtoms(cell = lattice,
                        scaled_positions = positions,
                                 numbers = elements)
            return unitcell

    def get_phonon_from_info(self):
        from phonopy import Phonopy

        info = load_hdf5('fc2', self.h5)
        unitcell = self.get_atoms_from_info('/structure/fc2')
        phonon = Phonopy(unitcell, info['dim'], symprec=info['symprec'])
        phonon.generate_displacements()
        return phonon

    def get_phonon3_from_info(self):
        from phono3py import Phono3py

        info = load_hdf5('fc3', self.h5)
        unitcell = self.get_atoms_from_info('/structure/fc3')
        phonon3 = Phono3py(unitcell, info['dim'], symprec=info['symprec'])
        phonon3.generate_displacements()
        return phonon3
           
    def get_phonon_from_calculation(self, symprec=1e-3, mode='dfpt'):
        from phonopy import Phonopy
        from phonopy.interface.vasp import read_vasp
        unitcell = read_vasp(self.scfdir/'POSCAR')
        if (self.fc2dir/'dfpt'/'POSCAR').exists():
            supercell = read_vasp(self.fc2dir/'dfpt'/'POSCAR')
        elif (self.fc2dir/'0'/'POSCAR').exists():
            supercell = read_vasp(self.fc2dir/'0'/'POSCAR')
        elif (self.fc2dir/'00'/'POSCAR').exists():
            supercell = read_vasp(self.fc2dir/'00'/'POSCAR')
        elif (self.fc2dir/'000'/'POSCAR').exists():
            supercell = read_vasp(self.fc2dir/'000'/'POSCAR')
        else:
            raise OSError('Get supercell failed!')
        dim = [np.rint(np.linalg.norm(i)/np.linalg.norm(j)).astype(int) for i,j in zip(supercell.cell,unitcell.cell)]
        phonon = Phonopy(unitcell,dim,symprec=symprec)
        phonon.generate_displacements()
        return phonon

    def get_phonon(self):
        if self.file == 'jamip':
            try:
                phonon = self.get_phonon_from_info()
            except:
                phonon = self.get_phonon_from_calculation()
        else:
            phonon = self.get_phonon_from_yaml()
        return phonon

    def get_gruneisen_from_info(self):
        from phonopy import Phonopy
        from phonopy import PhonopyGruneisen
        from jamip.utils.logger import load_hdf5

        info = load_hdf5('gruneisen', self.h5)
        # 'dim': dim.tolist(), 'symprec': symprec, 'scale': [1-scale,1+scale]
        if info == None:
            raise KeyError('gruneisen not found')
        dim = info['dim']
        symprec = info['symprec']
        # scale = info['scale']

        phonons = []
        # hdf5 - orig %
        cell = self.get_atoms_from_info('/structure/gruneisen/orig')
        if cell == None:
            unitcell = self.get_atoms_from_info('/structure/fc2')
            stdin = self.fc2dir
        else:
            unitcell = cell
            stdin = self.gruneisendir/'orig'
        phonon = Phonopy(unitcell,dim,symprec=symprec)
        phonon = self.set_force_constants(phonon, stdin)
        phonons.append(phonon)

        # hdf5 - plus %
        unitcell = self.get_atoms_from_info('/structure/gruneisen/plus')
        stdin = self.gruneisendir/'plus'
        phonon = Phonopy(unitcell,dim,symprec=symprec)
        phonon = self.set_force_constants(phonon, stdin)
        phonons.append(phonon)

        # hdf5 - minus %
        unitcell = self.get_atoms_from_info('/structure/gruneisen/minus')
        stdin = self.gruneisendir/'minus'
        phonon = Phonopy(unitcell,dim,symprec=symprec)
        phonon = self.set_force_constants(phonon, stdin)
        phonons.append(phonon)

        gruneisen = PhonopyGruneisen(*phonons)
        
        return gruneisen

    @classmethod
    def write_forces(self, phonon, path:str, output:str=None):
        from phonopy.file_IO import write_FORCE_SETS
        forces = self.get_forces(path)
        phonon.set_forces(forces)
        dataset = phonon.displacement_dataset
        if output is None:
            output = pathlib.Path(path) / 'FORCE_SETS'
        else:
            output = pathlib.Path(output) 
        if output.is_dir():
            output = output / 'FORCE_SETS'
        write_FORCE_SETS(dataset, output)

    @classmethod
    def write_force_constants(self, phonon, path:str, output:str=None):
        from phonopy.file_IO import write_FORCE_CONSTANTS
        dataset = self.get_force_constants(path)
        if output is None:
            output = os.path.join(path, 'FORCE_CONSTANTS')
        if pathlib.Path(output).is_dir():
            output = os.path.join(output, 'FORCE_CONSTANTS')
        write_FORCE_CONSTANTS(dataset, output)

    def set_force_constants(self, phonon, stdin=None, output=None, mode=None):
        from phonopy.file_IO import parse_FORCE_SETS, parse_FORCE_CONSTANTS
        phonon.generate_displacements()
        stdin = self.fc2dir if stdin is None else pathlib.Path(stdin)
        output = stdin if output is None else output
        force_set = output / 'FORCE_SETS'
        force_constants = output / 'FORCE_CONSTANTS'
        # get mode
        if mode is None:
            mode = 'FORCE_SETS'
            if force_set.exists():
                mode = 'FORCE_SETS'
            elif force_constants.exists():
                mode = 'FORCE_CONSTANTS'
            elif (stdin/'vasprun.xml').exists():
                mode = 'FORCE_CONSTANTS'

        # write force
        if mode == 'FORCE_SETS':
            if not force_set.exists():
                self.write_forces(phonon, stdin, output)
        elif mode == 'FORCE_CONSTANTS':
            if not force_constants.exists():
                self.write_force_constants(phonon, stdin, output)
        else:
            raise ValueError("Only mode = FORCE_SETS or FORCE_CONSTANTS support now.")

        # update force %
        if mode == 'FORCE_SETS':
            force_sets=parse_FORCE_SETS(filename=force_set)
            phonon.set_displacement_dataset(force_sets)
            phonon.produce_force_constants(calculate_full_force_constants=False)
        else:
            fc = parse_FORCE_CONSTANTS(filename=force_constants)
            phonon.force_constants = fc
        return phonon

    @classmethod
    def get_forces(self, stdin:str):

        forces = []
        stdin = pathlib.Path(stdin)

        if (stdin/'vasprun.xml').exists():
            xml = Xml(stdin)
            forces.append(xml._get_forces())
        else:
            for dir in np.sort(os.listdir(stdin)):
                if dir == 'relax': continue
                xmlfile = stdin / dir / 'vasprun.xml'
                if xmlfile.exists():
                    xml = Xml(xmlfile)
                    forces.append(xml._get_forces())

        return np.array(forces, dtype=float)

    @classmethod
    def get_force_constants(self, stdin:str):
        stdin = pathlib.Path(stdin)
        if (stdin/'vasprun.xml').exists():
            xml = Xml(stdin)
            hessian = xml._get_hessian()
            mass = xml.mass()
            mass2 = -np.sqrt(mass[:,None] * mass[None,:])
            force_constants = hessian * mass2[:,:,None,None]
        else:
            raise OSError("file not exists!")

        return force_constants
