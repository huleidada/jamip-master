#from .monitor import Monitor
import os
import shutil
import logging
import numpy as np
from copy import deepcopy
from .check import CheckStatus
from collections import UserDict, defaultdict
from jamip.abtools.diyflow import get_diy_modules, import_diy_module
from jamip.abtools.base.flow import WorkFlow
from jamip.abtools.base.kpoints import Kpoints
import pathlib

_task_ = {'relax':('relax'),
          'scf':('scf'),
          'md':('md'),
          'nscf':('band'),
         }

class Task(UserDict):

    '''
    name:
    state:
    data:
    '''
    structure = None
    _kpoints_ = None
    xc_func = None
    bindir = None

    def __init__(self, data:dict, name=None):
        self.data = deepcopy(data)
        self.name = name
        self.path = None
        self.require = None
        self.spin = 1
        self.state = "W"
        self.group = defaultdict(list)
        self.xc_func = deepcopy(Task.xc_func)
        self.soft = None
        self.basis_set = {}

    def get_status(self, stdout:str):
        status = CheckStatus.get_status(stdout, self.name)
        if status['success'] and self.state != 'E':
            self.state = 'C'
        else:
            self.state = 'E'
        self.path = stdout / self.name
        return status

    def group_update(self, name:str, data:dict):
        self.data.update(data)
        self.group[name].extend(data.keys())

    @property
    def kpoints(self):
        return self._kpoints_

    @kpoints.setter
    def kpoints(self, value):
        if 'kspacing' in self.data:
            self.data.pop('kspacing')
        self._kpoints_ = value

    @property
    def program(self):
        softs = self.soft.split()
        if (self.bindir/softs[0]).exists():
            return self.bindir/self.soft 
        else:
            return self.soft

    def initialize(self, path):
        keys = path.split('/')
        
        node = self 
        for key in keys:
            if key not in node:
                node[key] = {}
            node = node[key]
        return node

class CP2KFlow(WorkFlow, CheckStatus):

    __exit = None

    def __init__(self, func, envdir, *args, **kwargs):

        from .setcp2k import SetCP2K
        from jamip.compute.cluster import Cluster

        # stdout directoty %
        self.rootdir = pathlib.Path.cwd()
        self.cluster = Cluster(envdir)

        # classify the tasks % 	
        if not isinstance(func, SetCP2K):
            raise TypeError('Invalid func. Make sure you input a SetVasp class!')

        # initialize %
        self.func = func
        self.raw_structure = func.structure
        self.initialize_tasks(func)

    def initialize_tasks(self, cp2k):

        from jamip.abtools.diyflow import import_diy_module
        from jamip.utils.logger import load_yaml

        configdir = pathlib.Path(self.cluster.root)
        incar_dict = load_yaml(configdir/'.incar')
        if incar_dict is None:
            raise OSError(".incar configure error !")

        # extra parameters %
        extra_dict = load_yaml(configdir/'.extra')
        outdir = os.path.relpath(self.rootdir, self.cluster.root)
        if extra_dict is None or outdir not in extra_dict:
            extra_dict = {}
        else:
            extra_dict = extra_dict[outdir]

        for key in ['hse','gw','soc']:
            if key in incar_dict:
                extra_dict[key] = incar_dict[key]
        self.extra = extra_dict

        # base parameters %
        base = incar_dict['base'] if 'base' in incar_dict else {}
        self.func.set_basis_set(**base)
        Task.kpoints = cp2k.kpoints
        Task.xc_func = cp2k.xc_func
        Task.bindir = pathlib.Path(cp2k.program)
        Task.structure = cp2k.structure

        tasks_dict = {}
        links_dict = load_yaml(configdir/'.link')
        for task in cp2k.tasks:
            if task in incar_dict:
                tasks_dict[task] = Task(base, name=task)
                tasks_dict[task].update(incar_dict[task])
                # seele %
                if task not in links_dict:
                    links_dict[task] = []

        # continue previous tasks %
        CheckStatus(self.rootdir).update_tasks(tasks_dict, cp2k.overwrite)

        # initialize links %
        self.links = CP2KFlow.Links(links_dict)
        if set(self.links.nodes) != set(tasks_dict.keys()):
            raise ValueError("Specified links don't match given FW")

        if len(self.links.nodes) == 0:
            raise ValueError("Workflow cannot be empty (must contain at least 1 FW)")

        self.tasks = tasks_dict
        for task in self.tasks:
            self.refresh(task)

        # std structure %
        if 'relax' in self.tasks and self.tasks['relax'].path != None:
            self.std_structure = self.tasks['relax'].path

    def cp2k_calculator(self):

        # submit tasks %
        while self.allstate and not self.__exit:
            self.launch()

    def run(self, task_id):
        '''
        Distributes tasks to specific execution functions
        '''
        logging.info("%s calculator start" %task_id)
        if task_id in _task_['relax']:
            self.relax_cell_ions(task_id)
        elif task_id in _task_['md']:
            self.molecular_dynamics(task_id)
        elif task_id in _task_['scf']:
            self.self_consistent_field(task_id)
        elif task_id in _task_['nscf']:
            self.nscf_property(task_id)
        elif task_id in _task_['optic']:
            self.optic_property(task_id)
        elif task_id in _task_['phonon']:
            self.phonon_property(task_id)

    def calculator(self, task_id, stdout:str, stdin:str=None, *args, **kwargs):

        incar = self.tasks[task_id]
        self.set_input(incar, stdout, stdin)
        self.mpirun(incar, stdout)

    #@Monitor
    def mpirun(self, incar, stdout):

        os.chdir(stdout)
        self.cluster.run(incar, infile='cp2k.inp',outfile='cp2k.log')
        os.chdir(self.rootdir)
  
#--------------------------------------------------------------#
#                  calculation functions                       #
#--------------------------------------------------------------#

    def relax_cell_ions(self, task_id='relax', steps=3, **kwargs):
        """
        function to relax the cell shape, internal inons and volume.
        """
        # set stdin & stdout %
        if len(self.links['relax']):
            stdin = self.get_stdin(task_id)
        else:
            stdin = None

        stdout = self.rootdir / 'relax'
        incar = self.tasks['relax']
        incar.soft = 'cp2k.popt'

        # task start %
        self.calculator('relax', stdout, stdin)

        # check status %
        status = self.tasks['relax'].get_status(stdout)
        self.write_status(status, stdout)
        # update status.yaml %
        if incar.state == 'C':
            self.std_structure = stdout
            incar.path = stdout

    def self_consistent_field(self, task_id='scf', **kwargs):

        stdout = self.rootdir / 'scf'
        incar = self.tasks['scf']
        incar.soft = 'cp2k.popt'
	
        # set stdin & stdout %
        if len(self.links['scf']):
            stdin = self.get_stdin(task_id)
            incar.structure = self.load_structure(stdin=stdin)
        else:
            stdin = None
            incar.structure = self.std_structure

        # task start %
        self.calculator('scf', stdout, stdin)

        # check status %
        status = self.tasks['scf'].get_status(stdout)
        self.write_status(status, stdout)

    def nscf_property(self, task_id, **kwargs):
        """
        select the case of tasks
        """
        from ..base.tasks import Incar
        from copy import deepcopy
        import numpy as np

        incar = self.tasks[task_id]
        incar.soft = 'cp2k.popt'
        stdout = self.rootdir / 'nscf' / task_id 

        # set stdin & stdout %
        if len(self.links[task_id]):
            stdin = self.get_stdin(task_id)
            incar.structure = self.load_structure(stdin=stdin)
        else:
            stdin = None
            incar.structure = self.std_structure

        if task_id == 'band':
            # get kpath - input.py or automatic generation % 
            incar.kpoints = self.get_kpath(task_id)
            # TODO: ADDED_MOS
            self.get_band_structure(incar)
            stdout = stdout 

            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'dos':

            stdout = stdout/'dos'
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

    def set_input(self, incar, stdout, stdin=None, **kwargs):
    
        """
        set task.in, task.save 
        """
        from .cp2kio import CP2KIO

        if not stdout.exists():
            os.makedirs(stdout)

        if incar.soft == 'cp2k.popt':
            self.set_kpoints(incar)
            self.get_subsys_structure(incar)
            self.get_subsys_potential(incar)
            self.get_xc_func(incar)

        # load wfn
        self.load_wfn(incar, stdout, stdin)

        CP2KIO.write_input(incar, stdout/'cp2k.inp')

        self.checkinput(incar.program, stdout)

    def checkinput(self, program, stdout):

        cmd = f'{program} -c {stdout}/cp2k.inp'
        print(cmd)
        lines = os.popen(cmd).readlines()
        if 'SUCCESS' in lines[0]:
            return True
        else:
            with open(f'{stdout}/cp2k.log', 'w') as f:
                for line in lines:
                    f.write(line.rstrip())
            return False

    def set_kpoints(self, incar, **kwargs):
        """
        """
        value = None
        if 'kspacing' in incar:
            value = float(incar.pop('kspacing'))
        elif incar.kpoints.model == 'kspacing':
            value = incar.kpoints.value

        if value != None:
            rec_lattice = np.linalg.inv(incar.structure.lattice)*2*np.pi
            kmesh = []
            for v in rec_lattice:
                kmesh.append(np.ceil(np.linalg.norm(v)/value))
            incar.kpoints = Kpoints('Gamma',kmesh)


    def set_vdw(self, elements=None, stdout=None):
        """
        """
        pass

    def get_nbands(self, incar):
	
        from jamip.analysis.cp2k.output import Output
        # TODO
        pass

    def get_kpath(self, task_id, insert=None, stdin=None, prec='suggest'):
        '''
        get hsym kpoints path base on structure.
        params:
            - prec: 
                  suggest: Continuous paths whose total length is greater than 5 segments.
                  all: All paths which time inversion are not considered
        '''
        from jamip.utils.brillouin_zone import HighSymmetryKpath
        from jamip.abtools.base.kpoints import BandPath

        if task_id in self.func.kpath:
            if isinstance(self.func.kpath[task_id], Kpoints):
                return self.func.kpath[task_id]
            elif isinstance(self.func.kpath[task_id], int):
                insert = self.func.kpath[task_id]

        if task_id == 'band':
            if insert == None: insert = 20

            bz = HighSymmetryKpath()
            structure = self.tasks[task_id].structure if task_id in self.tasks else self.std_structure
            kpoint = bz.get_HSKP(structure.to_cell(), symprec=1e-3)
         
            kpath = []
            for points in kpoint['Path']:
                # if add segments into suggest kpath % 
                if len(kpath) > 5 and prec != 'all':
                    num = 0
                    for x,y in zip(kpath[-1].position, kpoint['Kpoints'][points[0]]):
                        num += min(abs(x-y), abs((x+y)-int(x+y)))
                    if num > 0.001:
                        break
                kpath.append(points)
            bandpath = BandPath.from_symbols(kpath, kpoint['Kpoints'], insert) 

            return Kpoints(bandpath)

    @property
    def raw_structure(self):
        return self._raw_structure

    @raw_structure.setter
    def raw_structure(self, value):
        from jamip.structure import Structure

        if isinstance(value, Structure):
            self._raw_structure = deepcopy(value)
            self.sinfo('/structure/raw', value)

        else:
            raise ValueError("Invalid raw structure! ")

    @property
    def std_structure(self):

        return self._std_structure if self._std_structure != None else self._raw_structure

    @std_structure.setter
    def std_structure(self, stdin:str):

        s = self.load_structure(stdin=stdin)
        self._std_structure = s
        self.sinfo('/structure/std', s)
        logging.info('setting standard stducture from %s' %stdin)

    def get_stdin(self, task_id):

        return self.tasks[self.links[task_id][0]].path

    def get_band_structure(self, incar):

        node = incar.initialize('force_eval/dft/print/band_structure')
        kpoints = incar.kpoints.value
        if kpoints.model == 'Line Model':
            node['added_mos'] = len(incar.structure)
            node['file_name'] = 'cp2k.bs'
            kpts = {}
            kpts['units'] = 'b_vector'
            kpts['npoints'] = kpoints.get_insert()
            for i,kpt in enumerate(kpoints.sites):
                p = kpt.position
                s = kpt.symbol.strip("\\")
                kpts[f"_{i}"] = f'SPECIAL_POINT {p[0]:10.6f}{p[1]:10.6f}{p[2]:10.6f} #{s}'
            node['kpoint_set'] = kpts
        node = incar.initialize('force_eval/dft/scf')
        node['added_mos'] = len(incar.structure)
 
        return node
            
    def get_subsys_structure(self, incar):
        from jamip.structure import Structure, Molecule

        node = incar.initialize('force_eval/subsys')
        s = incar.structure
        if isinstance(s, Structure):
            cell = {}
            for i,axis in enumerate('ABC'):
                p = s.lattice[i]
                cell[f'_{i}'] = f'{axis:2}{p[0]:18.12f}{p[1]:18.12f}{p[2]:18.12f}'
            node['cell'] = cell

            coord = {}
            coord['SCALED'] = '.TRUE.'
            for i,atom in enumerate(s.atomic_positions):
                p = atom.scale_coord
                coord[f"_{i}"] = f'{atom.specie:2}{p[0]:18.12f}{p[1]:18.12f}{p[2]:18.12f}'
            node['coord'] = coord
        elif isinstance(s, Molecule):
            pass

    def get_subsys_potential(self, incar):

        node = incar.initialize('force_eval/subsys')
        potential = self.func.get_potential(incar)

        for key,value in potential.items():
            node[f"kind {key}"] = {'element': key, 'basis_set': self.func.basis_set, 'potential': value} 

    def get_xc_func(self, incar):

        node = incar.initialize('force_eval/dft/xc/xc_functional')
        if 'xc_functional' not in node:
            if len(self.func.xc_func) == 0:
                node['xc_functional'] = 'PBE'
            else:
                if 'pe' in self.func.xc_func:
                    node['xc_functional'] = 'PBE'

    def load_structure(self, stdin=None):
        from .cp2kio import CP2KIO
        from jamip.structure import Structure

        if stdin != None:
            # get project first %
            print(pathlib.Path(stdin)/'cp2k.inp')
            print(pathlib.Path.cwd())
            data = CP2KIO.read_input(pathlib.Path(stdin)/'cp2k.inp')
            project = data['GLOBAL']['PROJECT']

            path = pathlib.Path(stdin)/f"{project}-1.restart"
            if not path.exists():
                return self.std_structure

            data = CP2KIO.read_input(path)
            cell = [None, None, None] 
            for key,value in data['FORCE_EVAL']['SUBSYS']['CELL'].items():
                if key[0] == '_':
                    if value[0] == 'A':
                        cell[0] = value[1:]
                    elif value[0] == 'B':
                        cell[1] = value[1:]
                    elif value[0] == 'C':
                        cell[2] = value[1:]
            cell = np.array(cell, dtype=float)
 
            elements = []
            positions = []
            for key,value in data['FORCE_EVAL']['SUBSYS']['COORD'].items():
                if key[0] == '_':
                    elements.append(value[0])
                    positions.append(value[1:])
 
            positions = np.array(positions, dtype=float)
            return Structure.from_cell((cell, positions, elements))
        else:
            return self.std_structure

    def load_wfn(self, incar, stdout, stdin=None):
        from .cp2kio import CP2KIO
        from jamip.structure import Structure

        if stdin != None:
            # get project first %
            data = CP2KIO.read_input(pathlib.Path(stdin)/'cp2k.inp')
            project = data['GLOBAL']['PROJECT']
            path = pathlib.Path(stdin)/f"{project}-RESTART.wfn"
            if path.exists():
                #node = incar.initialize('force_eval/dft/scf')
                #node['scf_guess'] = 'RESTART'
                node = incar.initialize('force_eval/dft')
                node['wfn_restart_file_name'] = path.name
                shutil.copy(path, stdout/path.name)
                return True

        return False
