#from .monitor import Monitor
import os
import logging
import pathlib
import numpy as np
from copy import deepcopy
from .check import CheckStatus
from collections import UserDict, defaultdict
from jamip.abtools.base.flow import WorkFlow

_task_ = {'relax':('opt'),
          'scf':('sp'),
          'nscf':(#'frec',
                  'volume', 'scrf',
                  'irc','ircmax','scan','polar','admp','bomp','eet','force','stable',
                 ),
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
    label = 'jamip test'

    def __init__(self, data:dict, name=None):
        self.data = deepcopy(data)
        self.name = name
        self.path = None
        self.spin = 1
        self.state = "W"
        self.group = defaultdict(list)
        self.xc_func = deepcopy(Task.xc_func)
        self.task = None

    def get_status(self, stdout:str):
        status = CheckStatus.get_status(stdout, self.name)
        if status['success'] and self.state != 'E':
            self.state = 'C'
        else:
            self.state = 'E'
        self.path = os.path.join(stdout, self.name)
        return status

    def group_update(self, name:str, data:dict):
        self.data.update(data)
        self.group[name].extend(data.keys())


class GaussianFlow(WorkFlow, CheckStatus):

    __exit = None

    def __init__(self, func, envdir, *args, **kwargs):

        from .setgau import SetGaussian
        from jamip.compute.cluster import Cluster

        # stdout directoty %
        self.rootdir = pathlib.Path.cwd()
        self.cluster = Cluster(envdir)
        self.rundir = self.rootdir/'tmp'
        if not self.rundir.exists():
            self.rundir.mkdir()

        # classify the tasks % 	
        if not isinstance(func, SetGaussian):
            raise TypeError('Invalid func. Make sure you input a SetGaussian class!')

        # initialize %
        self.func = func
        self.raw_structure = func.structure
        self.initialize_tasks(func)

    def gaussian_calculator(self):

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
        if self.func.program == 'g16':
            cmd = '$g16root/g16/g16 < {0}.gjf > {0}.out'.format(incar.name)
        elif self.func.program == 'g09':
            cmd = '$g09root/g09/g09 < {0}.gjf > {0}.out'.format(incar.name)
        else: 
            raise
        os.popen(cmd).readline()
        os.chdir(self.rootdir)

    def initialize_tasks(self, gau):

        from os.path import join
        from jamip.utils.logger import load_yaml

        incar_dict = load_yaml(join(self.cluster.root,'.incar'))
        if incar_dict is None:
            raise OSError(".incar configure error !")

        # base parameters %
        base = incar_dict['base'] if 'base' in incar_dict else {}
        Task.xc_func = gau.xc_func
        Task.structure = gau.structure
        Task.charge = gau.charge
        Task.mspin  = gau.get_mspin()

        tasks_dict = {}
        links_dict = load_yaml(join(self.cluster.root,'.link'))
        for task in gau.tasks:
            if task in incar_dict:
                tasks_dict[task] = Task(base, name=task)
                tasks_dict[task].update(incar_dict[task])
                # seele %
                if task not in links_dict:
                    links_dict[task] = []

        # continue previous tasks %
        CheckStatus(self.rootdir).update_tasks(tasks_dict)

        # initialize links %
        self.links = GaussianFlow.Links(links_dict)
        if set(self.links.nodes) != set(tasks_dict.keys()):
            print(set(self.links.nodes), set(tasks_dict.keys()))
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
        elif task_id in _task_['scf']:
            self.self_consistent_field(task_id)
        elif task_id in _task_['nscf']:
            self.nscf_property(task_id)
  
#--------------------------------------------------------------#
#                  calculation functions                       #
#--------------------------------------------------------------#

    def relax_cell_ions(self, task_id='opt', steps=3, **kwargs):
        """
        function to relax the cell shape, internal inons and volume.
        """
        # set stdin & stdout %
        if len(self.links[task_id]):
            stdin = self.get_stdin(task_id)
        stdout = self.rootdir

        incar = self.tasks['opt']
        incar.label = 'opt test task'

        # task start %
        self.calculator('opt', stdout)

        # check status %
        status = incar.get_status(stdout)
        self.write_status(status, stdout)

    def self_consistent_field(self, task_id='sp', **kwargs):
	
        # set stdin & stdout %
        if len(self.links[task_id]):
            stdin = self.get_stdin(task_id)
        stdout = self.rootdir

        incar = self.tasks['sp']
        incar.label = 'sp test task'

        # task start %
        self.calculator('sp', stdout)

        # check status %
        status = incar.get_status(stdout)
        self.write_status(status, stdout)

    def nscf_property(self, task_id, **kwargs):
        """
        select the case of tasks
        """

        # get incar params%
        incar = self.tasks[task_id]
        #if len(self.links[task_id]) == 0:
        #    raise ValueError("The calculation requires at least one dependency! ")
        stdout = self.rootdir

        if task_id == 'freq':
            # get kpath - input.py or automatic generation % 
            incar.label = 'freq test task'

            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        else:
            incar.label = f'{task_id} test task'

            self.calculator(task_id, stdout)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)
	
    def set_input(self, incar, stdout, **kwargs):
    
        """
        set task.in, task.save 
        """
        from .gauio import GaussianIO

        stdout = pathlib.Path(stdout)
        stdout.mkdir(exist_ok=True)

        filename = stdout / f'{incar.name}.gjf'
        GaussianIO.write_input(incar, filename, print_level=self.func.print_level)

    @property
    def raw_structure(self):

        return self._raw_structure

    @raw_structure.setter
    def raw_structure(self, value):
        from jamip.structure import Structure, Molecule

        if isinstance(value, (Structure,Molecule)):
            self._raw_structure = deepcopy(value)
            self.minfo('/structure/raw', value)

        else:
            raise ValueError("Invalid raw structure! ")

    @property
    def std_structure(self):

        return self._std_structure if self._std_structure != None else self._raw_structure

    @std_structure.setter
    def std_structure(self, value):
        from jamip.structure import Structure, Molecule

        if isinstance(value, (Structure,Molecule)):
            self._std_structure = deepcopy(value)
            self.minfo('/structure/std', value)

        else:
            raise ValueError("Invalid std structure! ")

    def get_stdin(self, task_id):

        return self.tasks[self.links[task_id][0]].path
