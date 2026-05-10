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
          'converge':('converge'),
          'nscf':('dos','band','projwfc'),
          'magnetic':('ferro','anti','ferri'),
          'optic':('optics','dielectric','gw','bse','shg','jdos'),
          'phonon':('phband','force','softmode','gruneisen'),
          'mechanic':('elastic','poisson'),
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
        if self.bindir/softs[0].exists():
            return self.bindir/self.soft 
        else:
            return self.soft


class QEFlow(WorkFlow, CheckStatus):

    __exit = None

    def __init__(self, func, envdir, *args, **kwargs):

        from .setqe import SetQE
        from jamip.compute.cluster import Cluster

        # stdout directoty %
        self.rootdir = pathlib.Path.cwd()
        self.savedir = self.rootdir/'qesave'
        self.cluster = Cluster(envdir)

        # classify the tasks % 	
        if not isinstance(func, SetQE):
            raise TypeError('Invalid func. Make sure you input a SetVasp class!')

        # initialize %
        self.func = func
        self.raw_structure = func.structure
        self.initialize_tasks(func)

    def qe_calculator(self):

        # submit tasks %
        while self.allstate and not self.__exit:
            self.launch()

    def calculator(self, task_id, stdout:str, *args, **kwargs):

        incar = self.tasks[task_id]
        self.set_input(incar, stdout)
        self.mpirun(incar, stdout)

    #@Monitor
    def mpirun(self, incar, stdout):

        os.chdir(stdout)
        self.cluster.run(incar, infile=incar.name+'.in',outfile=incar.name+'.out')
        os.chdir(self.rootdir)

    def initialize_tasks(self, qe):

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
        if qe.energy != None:
            base['etot_conv_thr'] = float(qe.energy)
        if qe.force != None:
            base['forc_conv_thr'] = float(qe.force)
        Task.kpoints = qe.kpoints
        Task.xc_func = qe.xc_func
        Task.bindir = pathlib.Path(qe.program)
        Task.structure = qe.structure

        tasks_dict = {}
        links_dict = load_yaml(configdir/'.link')
        for task in qe.tasks:
            if task in incar_dict:
                tasks_dict[task] = Task(base, name=task)
                tasks_dict[task].update(incar_dict[task])
                # seele %
                if task not in links_dict:
                    links_dict[task] = []

        # continue previous tasks %
        CheckStatus(self.rootdir).update_tasks(tasks_dict, qe.overwrite)

        # initialize links %
        self.links = QEFlow.Links(links_dict)
        if set(self.links.nodes) != set(tasks_dict.keys()):
            raise ValueError("Specified links don't match given FW")

        if len(self.links.nodes) == 0:
            raise ValueError("Workflow cannot be empty (must contain at least 1 FW)")

        self.tasks = tasks_dict
        for task in self.tasks:
            self.refresh(task)

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
        stdout = self.rootdir

        incar = self.tasks['relax']
        incar.soft = 'pw.x'
        incar['prefix'] = 'relax' 

        # task start %
        self.calculator('relax', stdout)

        # check status %
        status = self.tasks['relax'].get_status(stdout)
        self.write_status(status, stdout)

    def self_consistent_field(self, task_id='scf', **kwargs):
	
        # set stdin & stdout %
        if len(self.links['relax']):
            stdin = self.get_stdin(task_id)
        stdout = self.rootdir

        incar = self.tasks['scf']
        incar.soft = 'pw.x'
        incar['calculation'] = 'scf'
        incar['prefix'] = 'scf'

        # task start %
        self.calculator('scf', stdout)

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

        # get incar params%
        incar = self.tasks[task_id]
        if len(self.links[task_id]) == 0:
            raise ValueError("The calculation requires at least one dependency! ")
        incar.require = self.links[task_id][0]
        incar.soft = 'pw.x'
        stdout = self.rootdir

        if task_id == 'band':
            # get kpath - input.py or automatic generation % 
            incar.kpoints = self.get_kpath(task_id)

            incar['prefix'] = 'scf'
            incar['calculation'] = 'bands'
            if 'nbnd' not in incar:
                self.get_nbands(incar)

            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

            if status['success']:
                shutil.copy(self.rootdir/'save'/f"{incar['prefix']}.xml", self.rootdir/'save'/f"{task_id}.xml")

        elif task_id == 'dos':
            incar['prefix'] = 'scf'
            incar['calculation'] = 'nscf'

            # step 1 : nscf calculation %
            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            if incar.state != 'C':
                self.write_status(status, stdout)
                return

            if status['success']:
                shutil.copy(self.rootdir/'save'/f"{incar['prefix']}.xml", self.rootdir/'save'/f"{task_id}.xml")

            # step 2 : dos calculation %
            stdout = self.rootdir/'dos'
            incar.data = {'prefix': task_id,
                          'ngauss': 1, 
                          'deltae': 0.01, 
                          'degauss': 0.02,
                          'fildos': 'dos.dat'}
            incar.soft = 'dos.x'
            incar.soft = 'dos.x -pd .true.'
            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'projwfc':
            incar['prefix'] = 'scf'
            incar['calculation'] = 'nscf'

            # step 1 : nscf calculation %
            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            if incar.state != 'C':
                self.write_status(status, stdout)
                return

            if status['success']:
                shutil.copy(self.rootdir/'save'/f"{incar['prefix']}.xml", self.rootdir/'save'/f"{task_id}.xml")

            # step 2 : dos calculation %
            stdout = self.rootdir / 'dos'
            incar.soft = 'projwfc.x'
            incar.data = {'prefix': task_id,
                          'ngauss': 1, 
                          'deltae': 0.01, 
                          'degauss': 0.02,
                          'filpdos': 'dos',
                          'filproj': 'dos'}
            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)
	
    def phonon_property(self, task_id, **kwargs):
        """
        select the case of tasks
        """
        from copy import deepcopy

        # get incar params%
        incar = self.tasks[task_id]
        if len(self.links[task_id]) == 0:
            raise ValueError("The calculation requires at least one dependency! ")
        incar.require = self.links[task_id][0]
        stdout = self.rootdir/'phonon'
        
        if task_id == 'phband':
            # step 1
            incar.soft = 'ph.x'
            incar['prefix'] = 'scf'
            incar['fildyn'] = 'ph.dyn'

            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            if incar.state != 'C':
                self.write_status(status, stdout)
                return

            # step 2
            incar.soft = 'q2r.x'
            incar.data = {'fildyn': 'ph.dyn',
                          'filrc': 'ph.fc',
                          'zasr': 'simple'}

            self.calculator(task_id, stdout)#, filename='q2r')
            status = incar.get_status(stdout)
            if incar.state != 'C':
                self.write_status(status, stdout)
                return
            
            # step 3
            incar.soft = 'matdyn.x'
            incar.data = {'filrc': 'ph.fc',
                          'filrq': 'ph.freq',
                          'q_in_band_form': True,
                          'q_in_cryst_coord': True}
            incar.kpoints = self.get_kpath(task_id)
            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

    def set_input(self, incar, stdout, **kwargs):
    
        """
        set task.in, task.save 
        """
        from .qeio import QEIO

        if not stdout.exists():
            os.makedirs(stdout)
        # copy files %
        if incar.require != None:
            target_path = self.rootdir/'save'/f"{incar['prefix']}.save"
            source_path = self.rootdir/'save'/f'{incar.require}.save'

            if not target_path.exists():
                os.makedirs(target_path)
            if source_path.exists() and target_path != source_path:
                shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)

        filename = stdout / f'{incar.name}.in'
        incar['outdir'] = os.path.relpath(self.rootdir/'save', start=stdout)

        if incar.soft == 'pw.x':
            files = self.func.get_potential(incar)
            self.set_kpoints(incar)
         
            incar['ntyp'] = len(incar.structure.species_of_elements)
            incar['nat'] = sum(incar.structure.number_of_atoms)
            
            # INPUT: write input %
            QEIO.write_pwscf(incar, filename)
         
            # &ATOMIC_SPECIES - CELL_PARAMSTERS - ATOMIC_POSITIONS % 
            QEIO.write_structure(incar.structure, files, filename)
         
            # &K_POINTS: set kmesh % 
            QEIO.write_kpoints(incar, filename)

        else:
     	   
            QEIO.write_input(incar, filename)

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
	
        from jamip.analysis.qe.qexml import Xml

        # get nbands from scf calculation %
        xmlfile = self.rootdir/'save'/f'{incar.require}.save'/'data-file-schema.xml'
        npar = 4

        if self.func.nbands < 4:
            # nbands multiple %
            nbands = Xml().nbnd(xmlfile)
            incar['nbnd'] = int(np.ceil(nbands*self.func.nbands/npar)) * npar
        else:
            # nbands value %
            incar['nbnd'] = int(self.func.nbands)


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
    def std_structure(self, value):
        from jamip.structure import Structure

        if isinstance(value, Structure):
            self._std_structure = deepcopy(value)
            self.sinfo('/structure/std', value)

        else:
            raise ValueError("Invalid std structure! ")

    def get_stdin(self, task_id):

        return self.tasks[self.links[task_id][0]].path
