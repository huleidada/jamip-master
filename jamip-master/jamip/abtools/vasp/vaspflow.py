import os
import pathlib
import logging
import numpy as np
from copy import deepcopy
from .check import CheckStatus
from .monitor import Monitor
from collections import UserDict, defaultdict
from jamip.utils.logger import dump_hdf5, load_hdf5
from jamip.abtools.base.kpoints import Kpoints
from jamip.abtools.diyflow import get_diy_modules

# initial hdf5 %
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

_task_ = {'relax':('relax',),
          'scf':('scf',),
          'md':('nve','nvt','npt','mlmd'),
          'converge':('converge',),
          'electric':('dos','band','partchg','emass','stm','deformation','zpe', 'emc','elf',
                      'hse_gap','hse_band','hse_emass','meta_gap','meta_band','chgdiff'),
          'magnetic':('ferro','anti','ferri'),
          'optics':('optics','dielectric','diag','gw','bse','shg','jdos','singlet','triplet','born'),
          'phonon':('fc2','fc3','softmode','gruneisen','raman','dfpt'),
          'mechanic':('elastic','poisson'),
          'diyflow': get_diy_modules(),
         }

def get_base_modules():
    base = []
    for key,value in _task_.items():
        if key != 'diyflow':
            base.extend(value)
    return base

class Task(UserDict):

    xc_func = None
    kpoints = None
    structure = None

    def __init__(self, data:dict, name:str=None):
        """Calculation Parameters for VASP task.

        Args:
            data (dict): Incar dict of task.
            name (str, optional): Task name. Defaults to None.

        Parameters:
            path (str): calculate path of task.
            spin (int): Spin polarization of task, 1 for non-spin-polarized, 2 for spin-polarized, 3 for spin-orbit coupling.
            nbds (int): Scale of number of bands for task.
            state (str): State of task, W for waiting, C for completed, H for hold, E for error.
            parallel (int): Number of parallel processes for task.
            group (dict): Group of task, key is group name, value is list of task names.
            xc_func (set): Exchange-correlation functional of task.
            kpoints (Kpoints): Kpoints object for task.
            structure (Structure): Structure object for task.
        """        
        self.data = deepcopy(data)
        self.name = name
        self.path = None
        self.spin = 1
        self.nbds = None
        self.state = "W"
        self.parallel = 1
        self.group = defaultdict(list)
        self.xc_func = deepcopy(Task.xc_func)
        self.kpoints_opt = None
    
    def get_status(self, stdout:str):
        """Get status of task.

        Args:
            stdout (str): Calculation directory.

        Returns:
            dict: Status of task.
        """        
        status = CheckStatus.get_status(stdout, self.name)
        if status['success'] and self.state != 'E':
            self.state = 'C'
        else:
            self.state = 'E'
        self.path = stdout
        return status

    def group_update(self, name:str, data:dict):
        """Update group of task.

        Args:
            name (str): Group name.
            data (dict): Group data.
        """
        self.data.update(data)
        self.group[name].extend(data.keys())

    def __setitem__(self, key, item): 
        
        self.data[key] = item
        if key == 'kspacing':
            self.kpoints = Kpoints(item)

class WorkFlow:

    class Links(UserDict):
        """Links of workflow."""
        
        @property
        def nodes(self):  
            """Get all nodes of workflow.

            Returns:
                set: All nodes of workflow.
            """                    
            rev = []
            for root,children in self.items():
                rev.append(root)  
                rev.extend(children)
            return set(rev)

        @property
        def reverse(self):
            """Get reverse links of workflow.

            Returns:
                dict: Reverse links of workflow.
            """
            rev = defaultdict(list)
            for root,children in self.items():
                for child in children:
                    rev[child].append(root)
            return rev

    def __init__(self):
        self._raw_structure = None

    def refresh(self, taskname):
        """Refresh state of task.

        Args:
            taskname (str): Task name.

        Returns:
            list: Update list.
        """        
        # W&H 
        if self.tasks[taskname].state in ["W", "H"]:
            tmp = []
            for parent in self.links[taskname]:
                tmp.append(self.tasks[parent].state) 
            if "W" in tmp or "H" in tmp or "E" in tmp:
                self.tasks[taskname].state = "H" 
            else:
                self.tasks[taskname].state = "W" 

        # E&C
        update = []
        if self.tasks[taskname].state in ["E", "C", "D"]:
            update = self.links.reverse[taskname]
        return update

    def launch(self):

        task_id = None
        for i,task in self.tasks.items():
            if task.state == "W":
                task_id = i
                break

        if task_id != None:
            self.run(task_id)
            queue = [task_id]
            for i in queue:
                r = self.refresh(i)
                queue.extend(r)
        else: 
            logging.info("JOB end.")

    def run(self):        
        raise TypeError("MetaClass")

    @property
    def state(self):
        '''
        > W: waiting
        > C: completed
        > H: hold
        > E: error
        > R: running
        > S: sleep
        '''
        return {task:value.state for task, value in self.tasks.items()}

    @property
    def allstate(self):
        state = [value.state for value in self.tasks.values()]
        return True if "W" in state else False

class VaspFlow(WorkFlow, CheckStatus):

    _raw_structure = None
    _std_structure = None
    __exit = None

    def __init__(self, func, envdir, *args, **kwargs):
        from .setvasp import SetVasp
        from jamip.compute.cluster import Cluster 

        # rootdir
        self.rootdir = pathlib.Path.cwd()
        self.cluster = Cluster(envdir)

        # classify the tasks %  
        if not isinstance(func, SetVasp):
            raise TypeError('Invalid func. Make sure you input a SetVasp class!')

        # initialize %
        self.func = func
        self.raw_structure = func.structure
        self.initialize_tasks(func)

    def vasp_calculator(self):
        """Launch next VASP calculator."""

        # submit tasks %
        while self.allstate and not self.__exit:
            self.launch()

    def calculator(self, task_id, stdout:str, stdin=None, **kwargs):
        """Run VASP calculator.

        Args:
            task_id (str): Task ID.
            stdout (str): Standard output directory.
            stdin (str): Standard input directory.
            **kwargs: Additional keyword arguments.
        """

        incar = self.tasks[task_id] if isinstance(task_id, str) else task_id
        if stdin != None:
            incar.structure = self.load_structure(stdin)
        
        self.set_input(incar, stdout, stdin)
        self.mpirun(incar, stdout)

    @Monitor
    def mpirun(self, incar, stdout):
        """Run VASP with mpirun.
        Args:
            incar (Task): Task object containing VASP parameters.
            stdout (str): Standard output directory.
        """
        os.chdir(self.rootdir/stdout)
        self.cluster.run(incar, infile=None, outfile='vasp.log')
        os.chdir(self.rootdir)

    def load_structure(self, stdin:str, stdout=None):
        from jamip.structure import read
        import shutil

        path = None
        stdin_path = pathlib.Path(stdin)
        structure = self.func.structure
        # read structure from stdin %
        if stdin_path.exists():
            try:
                structure = read(stdin_path/'CONTCAR')
                path = stdin_path/'CONTCAR'
            except:
                try:
                    structure = read(stdin_path/'POSCAR')
                    path = stdin_path/'POSCAR'
                except:
                    structure = self.func.structure

        if stdout != None and path is not None:
            stdout.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(path,stdout/'POSCAR')

        return structure

    def load_wavchg(self, incar, stdout:str, stdin:str):
        import shutil

        stdin = pathlib.Path(stdin)
        stdout = pathlib.Path(stdout)

        # restart from previous calculation % 
        if stdin.exists() and stdin != stdout:

           if not stdout.exists(): stdout.mkdir()

           # update wavecar %
           wavein = stdin / 'WAVECAR'
           waveout = stdout / 'WAVECAR'
           if wavein.exists() and wavein.stat().st_size > 0:
               if 'istart' not in incar:
                   incar['istart'] = 1
           elif 'istart' not in incar:
               incar['istart'] = 0

           # remove link anyway %
           if waveout.is_symlink(): waveout.unlink()

           # copy wavecar %
           if wavein.exists() and incar['istart'] == 1:
               if incar.get('lwave') == False:
                   if waveout.exists(): waveout.unlink()
                   relpath = os.path.relpath(wavein, start=stdout)
                   waveout.symlink_to(relpath)
               elif not waveout.exists() or not waveout.samefile(wavein):
                   shutil.copyfile(wavein,waveout)

           # update chgcar %
           chgin = stdin / 'CHGCAR'
           chgout = stdout / 'CHGCAR'
           if chgin.exists() and chgin.stat().st_size > 0:
               if 'icharg' not in incar:
                   incar['icharg'] = 1
           elif 'icharg' not in incar:
               incar['icharg'] = 0 if incar.get('istart',0) != 0 else 2
           elif incar['icharg'] == 11:
               raise IOError('CHGCAR not exists!')

           # remove link anyway %
           if chgout.is_symlink(): chgout.unlink()

           # copy chgcar %
           if chgin.exists() and incar['icharg'] in (1,11):
               if incar.get('lcharg') == False:
                   if chgout.exists(): chgout.unlink()
                   relpath = os.path.relpath(chgin, start=stdout)
                   chgout.symlink_to(relpath)
               elif not chgout.exists() or not chgout.samefile(chgin):
                   shutil.copyfile(chgin,chgout)
        
    def initialize_tasks(self, vasp):
        """Initialize tasks for VASP workflow.
        Args:
            vasp (SetVasp): SetVasp object containing VASP parameters.
        """
        from jamip.utils.logger import load_yaml

        incar_dict = load_yaml(self.cluster.root/'.incar')
        if incar_dict is None: 
            raise OSError(".incar configure error !")

        # set output key %
        outdir = os.path.relpath(self.rootdir, self.cluster.root)

        # initialize extra parameters %
        extra_dict = load_yaml(self.cluster.root/'.extra')
        if extra_dict is None or outdir not in extra_dict:
            extra_dict = {}
        else:
            extra_dict = extra_dict[outdir]

        for key in ['hse','gw','soc','ldau']:
            if key in incar_dict:
                extra_dict[key] = incar_dict[key]
        # ldau %
        if 'ldau' in incar_dict:
            for key in ['ldaul','ldauu','ldauj']:
                if key in extra_dict and extra_dict[key] != None:
                    extra_dict['ldau'][key] = extra_dict.pop(key)
        self.extra = extra_dict

        # initialize vasp potential dict %
        potential_dict = load_yaml(self.cluster.root / '.potential', strict=False)
        if potential_dict is None or outdir not in potential_dict:
            self.potential = vasp.get_potential(vasp.structure)
        else:
            self.potential = vasp.get_potential(vasp.structure, potential_dict[outdir])

        # base parameters %
        base = incar_dict['base'] if 'base' in incar_dict else {}
        if vasp.energy != None:
            base['ediff'] = float(vasp.energy)
        if vasp.force != None:
            base['ediffg'] = -float(vasp.force)
        Task.kpoints = vasp.kpoints
        Task.xc_func = vasp.xc_func
        Task.structure = vasp.structure

        # charged calculation %
        if 'nelect' in self.extra:
            base['nelect'] = self.potential.nelect + self.extra['nelect']
            
        # create tasks parameters %
        tasks_dict = {}
        links_dict = load_yaml(self.cluster.root/'.link')
        for task in vasp.get_all_tasks():
            if task in vasp.tasks:
                if task in incar_dict:
                    tasks_dict[task] = Task(base, name=task)
                    tasks_dict[task].update(incar_dict[task])
                    # seele %
                    if task not in links_dict:
                        links_dict[task] = []
            elif task not in ['hse','gw','soc','ldau']:
                if task in incar_dict:
                    tasks_dict[task] = Task(base, name=task)
                    tasks_dict[task].update(incar_dict[task])
                    tasks_dict[task].state = 'S'
                    # seele %
                    if task not in links_dict:
                        links_dict[task] = []

        # update tasks parameters %
        for task in tasks_dict:
            # update xc_func %
            if 'xc_func' in tasks_dict[task]:
                tasks_dict[task].xc_func = set(tasks_dict[task].pop('xc_func', []))

            # update mdtype %
            if task in _task_['md']:
                tasks_dict[task].group_update('md', incar_dict.get('md', {}))
                tasks_dict[task].group_update(task, incar_dict.get(task, {}))

            # update kpoints %
            if task in self.func.kpath:
                kpoints = self.func.kpath[task]
                # TODO debug!
                if isinstance(kpoints, Kpoints) and kpoints.model != 'Line Model':
                    tasks_dict[task].kpoints = kpoints
                # elif isinstance(kpoints, int):
                #     tasks_dict[task].kpoints = kpoints
                # else:
                #     raise TypeError("kpath must be int or Kpoints object!")

            # update parallel %
            if 'parallel' in tasks_dict[task]:
                tasks_dict[task].parallel = int(tasks_dict[task].pop('parallel'))

        # group parameters %
        for task, params in tasks_dict.items():
            for key in list(params.keys()):
                if key[0] == '+':
                    xc, value = key[1:], params.pop(key)
                    if xc == "kpoints":
                        if isinstance(value, float): value = [value]
                        tasks_dict[task].kpoints_opt = Kpoints(*value)
                    elif value == True and self.func.get_atuo_groups(xc):
                        tasks_dict[task].xc_func.add(xc)
        
        #for task in tasks_dict:
        #    print(task, tasks_dict[task].xc_func, tasks_dict[task].kpoints_opt)

        # continue previous tasks %
        CheckStatus(self.rootdir).update_tasks(tasks_dict, vasp.overwrite)

        # initialize links %
        self.links = VaspFlow.Links(links_dict)
        if set(self.links.nodes) != set(tasks_dict.keys()):
            link_tasks = ' '.join(list(set(self.links.nodes)))
            task_tasks = ' '.join(list(tasks_dict.keys()))
            raise ValueError('''
            WORKFLOW LACKS DEPENDENT TASKS:
                LINKS -> %s
                TASKS -> %s
            ''' %(link_tasks, task_tasks))

        if len(self.links.nodes) == 0:
            raise ValueError("Workflow cannot be empty (must contain at least 1 Task)")

        self.tasks = tasks_dict
        for task in self.tasks:
            self.refresh(task)

        # std structure %
        if 'scf' in self.tasks and self.tasks['scf'].path != None:
            self.std_structure = self.tasks['scf'].path
        elif 'relax' in self.tasks and self.tasks['relax'].path != None:
            self.std_structure = self.tasks['relax'].path
        
    def run(self, task_id):
        '''
        Distributes tasks to specific execution functions
        '''
        from jamip.abtools.diyflow import import_diy_module

        logging.info("%s calculator start" %task_id)
        if task_id in _task_['relax']:
            self.relax_cell_ions(task_id)
        elif task_id in _task_['md']:
            self.molecular_dynamics(task_id)
        elif task_id in _task_['scf']:
            self.self_consistent_field(task_id)
        elif task_id in _task_['converge']:
            self.converge_test(task_id)
        elif task_id in _task_['electric']:
            self.electric_property(task_id)
        elif task_id in _task_['optics']:
            self.optic_property(task_id)
        elif task_id in _task_['phonon']:
            self.phonon_property(task_id)
        elif task_id in _task_['magnetic']:
            self.magnetic_property(task_id)
        elif task_id in _task_['mechanic']:
            self.mechanic_property(task_id)
        elif task_id in _task_['diyflow']:
            module = import_diy_module(task_id)
            module(self).diy_calculator()
        else:
            raise SystemExit("Unknown task_id")

#--------------------------------------------------------------#
#                  calculation functions                       #
#--------------------------------------------------------------#

    def converge_test(self, task_id='converge', **kwargs):
        """
        function to compare the calculation results of different parameters
        """
        import shutil
        from jamip.analysis.vasp.outcar import GrepOutcar

        # get test params %
        incar = self.tasks['converge']
        try:
            key = incar.pop('key')
            value = incar.pop('value')
            if isinstance(value,str):
                value = list(set(value.split()))
        except:
            raise ValueError("Missing key & value in incar")

        # set stdin & stdout %
        stdin = None
        stdout = self.rootdir / 'converge' / key
        if len(self.links['converge']):
            stdin = self.tasks[self.links['converge'][0]].path
        if stdout.exists():
            shutil.rmtree(stdout)
        stdout.mkdir()

        # start calculation %
        subtasks = []

        for v in value:
            incar[key] = v
            output = stdout/str(v)
            self.set_input(incar, output, stdin)
            subtasks.append(output)
          
        self.batch_calculator(task_id, subtasks)

        outdir = []
        energy = []
        go = GrepOutcar()
        stdin = pathlib.Path(self.tasks[self.links[task_id][0]].path)
        for path in stdin.iterdir():
            if (path/'OUTCAR').exists(): 
                outdir.append(path)
                energy.append(go.free_energy(path))
        incar.path = outdir[np.argmin(energy)] if len(outdir) else None

    def relax_cell_ions(self, task_id='relax', steps=3, **kwargs):
        """
        function to relax the cell shape, ions and volume.
        """
        import shutil

        incar = self.tasks['relax']
        stdout = self.rootdir / 'relax'

        # set stdin & stdout %
        stdin = None
        if len(self.links['relax']):
            stdin = self.get_stdin(task_id)
        elif self.std_structure != None:
            incar.structure = self.std_structure

        self.backup_calculations(incar.name, stdout, stdin)

        # accelerate relax calculation % 
        if isinstance(self.func.accelerate,list):
            stdout = self.rootdir / 'relax' / 'S0'
            incar_acc = deepcopy(incar)

            # run vasp object % 
            for i,acc in enumerate(self.func.accelerate):
                incar_acc.update(acc)
                self.calculator(incar_acc, stdout, stdin)
                stdin = stdout
                # backup %
                if (stdout/'vasp.log').exists():
                    shutil.copy(stdout/'vasp.log', stdout/(f'step{i}.log')) 
                if (stdout/'OUTCAR').exists():
                    shutil.copy(stdout/'OUTCAR',stdout/(f'OUTCAR_{i}'))
                if (stdout/'POSCAR').exists():
                    shutil.copy(stdout/'POSCAR',stdout/(f'POSCAR_{i}'))

            # check accelerate status %         
            incar_acc.state = 'W'
            status = incar_acc.get_status(stdin)
            if status['finish']:
                if status['success']:
                    logging.info("Accelerate relax calculation success, try next relax calculation !")
                else:
                    logging.warning("Accelerate relax calculation failed, try next relax calculation !")
                status['success'] = False
                self.write_status(status, stdin)

        # standard relax calculation %
        n = 1
        last_status = None
        incar.state = 'E'
        while (n <= steps) and incar.state != 'C':
            stdout = self.rootdir / 'relax' / f'S{n}'
            self.calculator('relax', stdout, stdin)
            stdin = stdout

            # check current status %
            incar.state = 'W'
            status = incar.get_status(stdin)
            logging.info(f"Relaxation step {n} finished, incar.state={incar.state}, success={status['success']}, finish={status['finish']}")
            n += 1

            if status['finish']:
                self.write_status(status, stdin)
                last_status = status
                last_stdout = stdin

        # update status.yaml %
        if incar.state == 'C':
            self.std_structure = stdin
        elif last_status != None:
            incar.path = last_stdout
        else:
            incar.path = None

    def molecular_dynamics(self, task_id='md', **kwargs):
        """
        function to run molecular dynamics simulation.
        """

        incar = self.tasks[task_id]
        stdout = self.rootdir / 'md' / task_id

        def reshape(dim:str):
            dim = np.array(dim.split(), dtype=int)
            if dim.size == 3:
                dim = np.diag(dim)
            elif dim.size == 9:
                dim = dim.reshape(3,3)
            else:
                raise ValueError('Dim params is incorrectly set in the %s' %task_id)
            return dim

        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.get_stdin(task_id)
        elif incar.path != None:
            stdin = incar.path

        self.backup_calculations('md', stdout, stdin)

        # update structure if necessary %
        if 'dim' in incar:
            from jamip.structure import read,write
            from jamip.structure.convert import phonopy2jamip, jamip2phonopy
            from phonopy import Phonopy

            dim = incar.pop('dim')
            if dim.strip().lower() == 'auto':
                lattice_parameters = incar.structure.lattice_parameters[:3]
                dim = np.diag(np.ceil(10 / lattice_parameters).astype(int))
            else:
                dim = reshape(dim)

            # save unitcell & supercell
            stdout.mkdir(exist_ok=True, parents=True)
            write(incar.structure, stdout/"infile.ucposcar", ftype='vasp')
            unitcell = jamip2phonopy(incar.structure)
            phonon = Phonopy(unitcell, dim)
            phonon.generate_displacements()
            supercell = incar.structure = phonopy2jamip(phonon.supercell)
            if np.abs(np.diag(dim)).sum() == np.abs(dim).sum():
                l1,l2,l3 = np.diag(dim)
                suprecell.comment_line = f"{l1}x{l2}x{l3}"

            write(supercell, stdout/"infile.ssposcar", ftype='vasp')

        # task start %
        incar.kpoints = Kpoints('Gamma',[1,1,1]) 
        self.calculator(incar, stdout, stdin)

        # check status %
        status = incar.get_status(stdout)
        self.write_status(status, stdout)

    def self_consistent_field(self, task_id='scf', **kwargs):

        # set stdin & stdout %
        if len(self.links['scf']):
            stdin = self.get_stdin(task_id)
        elif self.tasks['scf'].path != None:
            stdin = self.tasks['scf'].path
        elif 'relax' in self.tasks and self.tasks['relax'].path != None:
            stdin = self.tasks['relax'].path
        else:
            stdin = None

        stdout = self.rootdir / 'scf'

        # task start %
        self.calculator('scf', stdout, stdin)
 
        # check status %
        status = self.tasks['scf'].get_status(stdout)
        self.write_status(status, stdout)

        # std_structure %
        if status['success'] is True:
            self.std_structure = stdout
                
    def electric_property(self, task_id, **kwargs):
        from .vaspio import VaspIO

        incar = self.tasks[task_id]
        stdout = self.rootdir / 'electric' / task_id
        if len(self.links[task_id]) == 0:
            raise ValueError("The calculation requires at least one dependency! ")
        stdin = self.get_stdin(task_id)
        incar.structure = self.load_structure(stdin)
        self.clear_status(task_id)
            
        if task_id == 'band':

            # Add parameters %
            incar.nbds = self.func.nbands
            kpoints = self.get_kpath(task_id)
            dump_hdf5('kpoints/band', kpoints.value)

            # band calculator %
            if getattr(self.func, 'band_split', False):
                # for parallel
                subtasks = []                
                for k, kpts in kpoints.value.split().items():
                    output = stdout / k
                    incar.kpoints = Kpoints(kpts)
                    # check status
                    status = incar.get_status(output)
                    if status['success']: continue
                    self.set_input(incar, output, stdin)
                    subtasks.append(output)
                    # for single
                    #self.calculator(task_id, output, stdin)

                # finally %
                self.batch_calculator(task_id, subtasks)

            else:
                incar.kpoints_opt = kpoints
                self.calculator(task_id, stdout, stdin)
                status = incar.get_status(stdout)
                self.write_status(status, stdout)

        elif task_id == 'emass':

            # Add parameters %
            bandin = self.tasks[self.links[task_id][-1]].path
            incar.nbds = self.func.nbands
            kpath = self.get_kpath(task_id, stdin=bandin)
        
            # emass calculator %
            for dir,kpts in kpath.items():
                
                output = stdout / dir
                if output.exists():
                    status = incar.get_status(output)
                    if status['success']: continue
                incar.kpoints_opt = kpts
                self.calculator(task_id, output, stdin)
                status = incar.get_status(output)

            # final update status
            if incar.state == 'C': 
                self.write_status(status, stdout)
                incar.path = stdout

        elif task_id == 'emc':

            # Add parameters %
            bandin = self.tasks[self.links[task_id][-1]].path
            incar.nbds = self.func.nbands
            kpath = self.get_kpath(task_id, stdin=bandin)
        
            # emass calculator %
            for dir,kpts in kpath.items():
                incar.kpoints = kpts
                output = stdout / dir
                self.calculator(task_id, output, stdin)
                status = incar.get_status(output)

            # final update status
            if incar.state == 'C': 
                self.write_status(status, stdout)
                incar.path = stdout

        elif task_id == 'stm':

            # Add parameters %
            bandin = self.tasks[self.links[task_id][-1]].path

            if incar['nbmod'] == -2 and 'eint' not in incar:
                edge = self.get_band_edge(bandin)
                e_cb_shift = incar.pop('EINT_CB', 0.3)
                e_vb_shift = incar.pop('EINT_VB', -0.3)

                output = stdout / 'cbm'
                incar['eint'] = '%.4f %.4f' %(edge['cbm'].energy, edge['cbm'].energy+e_cb_shift)
                self.calculator(task_id, output, stdin)
                status = incar.get_status(output)
                
                output = stdout / 'vbm'
                incar['eint'] = '%.4f %.4f' %(edge['vbm'].energy+e_vb_shift, edge['vbm'].energy)
                self.calculator(task_id, output, stdin)
                status = incar.get_status(output)

                if incar.state == 'C': 
                    self.write_status(status, stdout)
                    incar.path = stdout

            else:
                self.calculator(incar, stdout, stdin)
                status = incar.get_status(stdout)
                self.write_status(status, stdout)

        elif task_id == 'partchg':
            from jamip.utils.utils import relink
            from jamip.analysis.vasp.band import Outcar

            # Add parameters %
            bandin = self.tasks[self.links[task_id][-1]].path
            edge = self.get_band_edge(bandin)
            if bandin == 'scf':
                if not stdout.exists():
                    stdout.mkdir()
                relink(self.rootdir, stdout, 'scf')
                # check status %
                output = stdout / 'scf'
                status = incar.get_status(output)
                if incar.state != 'C': 
                    return False

                # get ibzkpt
                ibzkpt = Outcar.from_file(output)._get_kpoint(weight=True)
                kpoints = Kpoints("Reciprocal",ibzkpt)
                cb_ikpt = edge['cbm'].ikpt
                vb_ikpt = edge['vbm'].ikpt

            else:
                # ibzkpt = incar.kpoints.get_reciprocal_kpoints(cell=incar.structure.to_cell(), isym=incar.get('isym',1)).value
                # cbmkpt = [[*edge['cbm'].kpoints,0.01]]
                # vbmkpt = [[*edge['vbm'].kpoints,0.01]]
                # kpoints = np.r_[ibzkpt, cbmkpt, vbmkpt]
                cbmkpt = [[*edge['cbm'].kpoints,1]]
                vbmkpt = [[*edge['vbm'].kpoints,1]]
                kpoints = np.r_[cbmkpt, vbmkpt]
                kpoints = Kpoints("Reciprocal",kpoints)
                cb_ikpt = len(kpoints.value) - 1
                vb_ikpt = len(kpoints.value) 
             
                # scf calculator %
                std_id = self.links[task_id][0]
                incar_scf = deepcopy(self.tasks[std_id])
                incar_scf.kpoints = kpoints
                output = stdout / 'scf'
                self.calculator(incar_scf, output, stdin)
             
                # check status %
                status = incar.get_status(output)
                stdin = output
                if incar.state != 'C': 
                    return False

            # cbm & vbm calculator % 
            edge = self.get_band_edge(stdin)
            incar.kpoints = kpoints
            ncb = incar.pop('IBAND_CB', 1)
            nvb = incar.pop('IBAND_VB', 1)

            output = stdout / 'cbm'
            incar['kpuse'] = cb_ikpt
            ibands = np.array(edge['cbm'].iband+np.arange(ncb)+1, dtype=str)
            incar['iband'] = ' '.join(ibands)
            self.calculator(incar, output, stdin)
            status = incar.get_status(output)
            
            output = stdout / 'vbm'
            incar['kpuse'] = vb_ikpt
            ibands = np.array(edge['vbm'].iband-np.arange(nvb)+1, dtype=str)
            incar['iband'] = ' '.join(ibands)
            self.calculator(incar, output, stdin)
            status = incar.get_status(output)

            # final update status
            if incar.state == 'C': 
                self.write_status(status, stdout)
                incar.path = stdout
                 
        elif task_id == 'meta_gap' or task_id == 'hse_gap':

            if task_id == 'hse_gap' and 'hse' not in incar.xc_func:
                incar.xc_func.add('hse')

            # Add parameters %
            bandin = self.tasks[self.links[task_id][-1]].path
            edge = self.get_band_edge(bandin)
            cbmkpt = [[*edge['cbm'].kpoints,0]]
            vbmkpt = [[*edge['vbm'].kpoints,0]]
            if self.func.kpoints_opt:
                kpoints = np.r_[cbmkpt, vbmkpt]
                incar.kpoints_opt = Kpoints("Reciprocal", kpoints)
            else:
                ibzkpt = incar.kpoints.get_reciprocal_kpoints(cell=incar.structure.to_cell(), isym=incar.get('isym',1)).value
                kpoints = np.r_[ibzkpt, cbmkpt, vbmkpt]
                incar.kpoints = Kpoints("Reciprocal", kpoints)
            
            # hse-gap calculator %
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'meta_band' or task_id == 'hse_band':
            from .vaspio import VaspIO

            if task_id == 'hse_band' and 'hse' not in incar.xc_func:
                incar.xc_func.add('hse')

            # Add parameters %
            kpath = self.get_kpath(task_id)

            if self.func.kpoints_opt:
                dump_hdf5(f'kpoints/{task_id}', kpath.value)
                incar.kpoints_opt = kpath
            else:            
                # gamma kpoints -> ibzkpt %     
                isym = incar.get('isym', 1)
                cell = incar.structure.to_cell()
                mesh = incar.pop('mesh', 0.02)
                ibzkpt = incar.kpoints.get_reciprocal_kpoints(cell=cell, isym=isym)
                # kpath -> meshkpath %
                kpath.value.set_mesh(cell[0], mesh=mesh)
                VaspIO.write_kpoints(kpath, stdout, name='KPATH.in')
                dump_hdf5(f'kpoints/{task_id}', kpath.value)
                # merge ibzkpt & bandkpoints %
                bandkpt = kpath.get_reciprocal_kpoints()
                bandkpt.value[:,-1] = 0
                incar.kpoints = Kpoints("Reciprocal", np.r_[ibzkpt.value, bandkpt.value])
            
            # hse-gap calculator %
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'hse_emass':

            # Add parameters %
            bandin = self.tasks[self.links[task_id][-1]].path
            incar.nbds = self.func.nbands
            kpath = self.get_kpath(task_id, stdin=bandin)
        
            if not self.func.kpoints_opt:
                print("Too expensive. skip task")
                return
                
            incar.kpoints_opt = kpath
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'chgdiff':

            structure = self.load_structure(stdin)
            atom_indices = incar.pop('indices', '')
            if len(atom_indices) == 0:
                raise ValueError('indices is empty!')
            else:
                indices = []
                for index in atom_indices.split():
                    if '*' in index:
                        value, number = index.split('*')                     
                        indices.extend([int(value)]*int(number))
                    else:
                        indices.append(int(index))

            subtasks = []
            for idx in np.unique(indices):
                indice = np.where(indices == idx)[0]
                elements = []
                positions = []
                for i, atom in enumerate(structure.atomic_positions):
                    if i in indice:
                        elements.append(atom.specie)
                        positions.append(atom.scale_coord)

                atoms = structure.from_cell((structure.lattice, positions, elements))
                incar.structure = atoms
                self.set_input(incar, stdout/f'{idx}')
                subtasks.append(stdout/f'{idx}')

            # finally %
            self.batch_calculator(task_id, subtasks)

        elif task_id == 'deformation':

            # get bandedge & kpoints
            bandin = self.tasks[self.links[task_id][-1]].path
            isym = incar.get('isym',1)
            ibzkpt = incar.kpoints.get_reciprocal_kpoints(cell=incar.structure.to_cell(), isym=isym).value
            edge = self.get_band_edge(bandin)
            cbmkpt = [[*edge['cbm'].kpoints,0]]
            vbmkpt = [[*edge['vbm'].kpoints,0]]
            kpoints = np.r_[ibzkpt, cbmkpt, vbmkpt]
            incar.kpoints = Kpoints("Reciprocal",kpoints)
            incar.nbds = self.func.nbands

            # initialize scale
            scale = incar.pop('scale', '0.99 0.995 1.00 1.005 1.01').split()
            scale = np.array(scale, dtype=float)

            # initialize axis
            axis = incar.pop('axis', 'x y z').split()
            axis_map = {'x':0, 'y':1, 'z':2}
            axis = np.array([axis_map[i] for i in axis])

            subtasks = []
            # cal base structure %
            for s in scale:
                if abs(s-1) < 1e-8: 
                    output = stdout / '1'
                    self.set_input(incar, output)
                    subtasks.append(output)

            # cal deform structures %
            scf_cell = deepcopy(incar.structure)
            for i in axis:
                for s in scale:
                    if abs(s-1) < 1e-8: continue
                    # reshape lattice
                    cell = deepcopy(scf_cell)
                    lattice = cell.lattice
                    lattice[i] = lattice[i]*s
                    cell.lattice = np.array(lattice)
                    incar.structure = cell
                    # scf-gap
                    output = stdout / ('%s-%s'%('xyz'[i],s))
                    self.set_input(incar, output)
                    subtasks.append(output)
            
            # finally %
            self.batch_calculator(task_id, subtasks)

            
        else:   # dos and others
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

    def optic_property(self, task_id, **kwargs):

        from jamip.utils.utils import relink
        import shutil
        import re

        incar = self.tasks[task_id]
        stdout = self.rootdir / 'optics' / task_id
        if len(self.links[task_id]) == 0:
            raise ValueError("The calculation requires at least one dependency! ")
        stdin = self.get_stdin(task_id)
        self.clear_status(task_id)
            
        if task_id == 'optics':

            # Add parameters %
            incar.nbds = self.func.optics_nbands
            if 'cores' in incar:
                incar.cores = incar.pop('cores')
            # calculator %
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'diag':

            # Add parameters %
            incar.nbds = self.func.optics_nbands
            if 'lpead' in incar and incar['lpead'] == True:
                incar['npar'] = None
                incar['ncore'] = 1
            # calculator %
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'gw':
       
            # Add parameters %
            incar.nbds = self.func.optics_nbands
            #incar.nbandsgw = self.func.nbands
            # TODO: same in soc?
            if 'nbandsgw' not in incar:
                incar['nbandsgw'] = self.get_nbands(incar)
            if 'lpead' in incar and incar['lpead'] == True:
                incar['npar'] = None
                incar['ncore'] = 1

            # copy %
            gwin = pathlib.Path(self.tasks[self.links[task_id][-1]].path)
            if not (gwin/'WAVEDER').exists():
                raise OSError('WAVEDER not found in GW input directory.') 
            if not stdout.exists(): stdout.mkdir()
            shutil.copy(gwin/"WAVEDER", stdout/"WAVEDER")

            self.calculator(task_id, stdout, gwin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'bse':
            from jamip.analysis.vasp import BandFinder
       
            # Add parameters %
            incar.nbds = self.func.optics_nbands
            if 'nbandso' not in incar and 'nbandsv' not in incar:
  
                eint = float(incar.pop('eint', 3))
                bfd = BandFinder(stdin).get_data()
                cbvbs = bfd.get_cbvb()
                bands = bfd.bands
                # assert ispin = 1
                ivb,icb = cbvbs[0]
                Evbm = max(bands[:,:,ivb,0])
                Ecbm = min(bands[:,:,icb,0])
                # get nbandso
                obands = max(bands[:,:,:ivb,0])
                incar['nbandso'] = (obands-Evbm > -eint).sum()
                vbands = min(bands[:,:,:icb,0])
                incar['nbandsv'] = (vbands-Ecbm < eint).sum()

            # copy %
            gwin = pathlib.Path(self.tasks[self.links[task_id][-1]].path)
            # WAVEDER from diag
            if not (gwin/'WAVEDER').exists():
                raise OSError('WAVEDER not found in BSE input directory.')
            if not stdout.exists(): stdout.mkdir()
            shutil.copy(gwin/"WAVEDER", stdout/"WAVEDER")
            # Wxxxx.tmp from gw
            num = 0
            for filename in os.listdir(stdin):
                if re.match(r'WFULL\d+.tmp', filename):
                    relink(stdin, stdout, filename)
                    num += 1
            if num == 0:
                raise OSError('Wxxxx.tmp not found in BSE input directory.') 

            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'jdos':

            opticsin = pathlib.Path(self.tasks[self.links[task_id][-1]].path)
            if not (opticsin/'OPTIC').exists():
                raise OSError('OPTIC not found in JDOS input directory.')
            # copy files %
            if not stdout.exists(): stdout.mkdir()
            shutil.copyfile(opticsin/'CONTCAR', stdout/'POSCAR')
            shutil.copyfile(opticsin/'OPTIC', stdout/'OPTIC')
            if (opticsin,'IBZKPT').exists():
                shutil.copyfile(opticsin/'IBZKPT', stdout/'KPOINTS')
            else:
                shutil.copyfile(opticsin/'KPOINTS', stdout/'KPOINTS')

            # write input %
            with open(stdout/'OPTCTR','w') as f:
                for key,value in incar.items():
                    f.write('{0:12} = {1}\n'.format(key.upper(),value))

            # run optics
            os.chdir(stdout)
            if 'lsearch' in incar and incar['lsearch']:
                os.popen("optics > MATRIX").readline()
            else:
                os.popen("optics > optics.log").readline()
            os.chdir(self.rootdir)

            # check
            if (stdout/'OPTCTR').exists(): # JDOS+EPS
                status = {'task':'jdos','finish':True,'success':True}
                self.write_status(status, stdout)
                incar.state = 'C'
            else:
                incar.state = 'E'

        elif task_id == 'shg':

            stdin = pathlib.Path(stdin)
            bandin = self.tasks[self.links[task_id][-1]].path
            # copy %
            if not stdout.exists(): stdout.mkdir()
            for filename in ['EIGENVAL', 'IBZKPT', 'POSCAR']:
                shutil.copyfile(stdin / filename, stdout / filename)
            relink(stdin, stdout, 'momentummatrix')
            # bandgap %
            edge = self.get_band_edge(bandin)
            incar['expgap'] = edge['cbm']['energy'] - edge['vbm']['energy']

            # input %
            with open(stdout/'input','w') as f:
                for key in incar:
                    if key in ['direction', 'volume', 'expgap', 'broadening', 'pm', 'dim', 'pmax', 'pgrid']:
                        f.write('{} : {}\n'.format(key, incar[key]))

            incar.program = 'shg'
            if 'cores' in incar:
                incar.cores = incar.pop('cores')
            self.mpirun(incar, stdout)

            # check %
            incar.state = 'E'
            for file in os.listdir(stdout):
                if file.startswith('SHG'):
                    status = {'task':'shg','finish':True,'success':True}
                    self.write_status(status, stdout)
                    incar.state = 'C'
                    break
        
        elif task_id == 'born' or task_id == 'dielectric':
            if 'lepsilon' in incar and incar['lepsilon'] == True:
                incar['npar'] = None
                incar['ncore'] = 1
            incar.kpoints = incar.kpoints.get_gamma_kpoints(
                    cell=incar.structure.lattice,
                    model=self.func.force_create_kpoints
                    )

            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'singlet' or task_id == 'triplet':
            if task_id == 'singlet' and 'ferwe' not in incar:
                nelect = self.potential.nelect
                cb = nelect / 2
                if 'nbands' in incar:
                     nbands = incar['nbands']
                else:
                     npar = self.get_npar(incar)
                     nions = sum(incar.structure.number_of_atoms)
                     nbands = int(max(nelect/2+nions, nelect*0.6)/npar) * npar
                     incar['nbands'] = nbands
                # ncb = nbands - nelect
                incar['ferwe'] = '%d*1 0 1 %d*0 ' %(cb-1, nbands-cb-1)
                incar['ferdo'] = '%d*1 %d*0 ' %(cb, nbands-cb)

            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

    def mechanic_property(self, task_id, **kwargs):

        incar = self.tasks[task_id]
        stdout = self.rootdir / 'mechanic' / task_id
        stdin = None
        if len(self.links[task_id]):
            stdin = self.get_stdin(task_id)
            incar.structure = self.load_structure(stdin)
        elif self.std_structure != None:
            incar.structure = self.std_structure
        self.clear_status(task_id)
        
        if task_id == 'elastic':
            incar['npar'] = self.cluster.cores
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)

        elif task_id == 'poisson':
            # initialize scale
            scale = incar.pop('scale', '0.98 0.99 1.00 1.01 1.02').split()
            scale = np.array(scale, dtype=float)
            optcell_bk = self.func.optcell
            # initialize axis
            axis = incar.pop('axis', 'x y z').split()
            axis_map = {'x':0, 'y':1, 'z':2}
            index = np.array([axis_map[i] for i in axis])
            if len(axis) == 3:
                optcell = np.array([1,1,1])
            elif len(axis) == 2:
                optcell = np.array([0,0,0])
                optcell[index] = 1
            else:
                raise ValueError("Calculating Poisson's ratio requires choosing at least two directions")
            subtasks = []
            origin_lattice = deepcopy(incar.structure.lattice)
             
            for i,ax in zip(index,axis):
                ioptcell = deepcopy(optcell)
                ioptcell[i] = 0
                self.func.optcell = ioptcell

                for s in scale:
                    #if abs(scale -1) < 1e-4: continue
                    # resize lattice %
                    lattice = deepcopy(origin_lattice)
                    lattice[i] = lattice[i]*s
                    incar.structure.lattice = lattice
                    incar.structure.comment_line = 'POISSON-%s-%s' %(ax,s)
                    # calaulation %
                    output = stdout / ('%s-%s' %(ax, s))
                    self.set_input(incar, output, stdin)
                    subtasks.append(output)

            # finally %
            self.func._optcell = optcell_bk
            self.batch_calculator(task_id, subtasks)

    def phonon_property(self, task_id, **kwargs):

        from jamip.structure.convert import phonopy2jamip, jamip2phonopy
        from phonopy import Phonopy

        # set stdin & stdout %
        phonon = self.initialize_phonon(task_id)
        stdout = self.rootdir / 'phonon' / task_id
        stdin = None
        incar = self.tasks[task_id]
        subtasks = []

        if task_id == 'fc2':

            if incar.get('ibrion') == 8: # dfpt
                incar.structure = phonopy2jamip(phonon.supercell)
                incar['npar'] = None
                incar['ncore'] = 1
                self.calculator(task_id, stdout, stdin)
                status = incar.get_status(stdout)
                self.write_status(status, stdout)
            else:
                digit = len(str(len(phonon.supercells_with_displacements)))
                for i,supercell in enumerate(phonon.supercells_with_displacements):
                    output = stdout / str(i).zfill(digit)
                    incar.structure = phonopy2jamip(supercell)
                    self.set_input(incar, output, stdin)
                    subtasks.append(output)

        elif task_id == 'dfpt':

            incar.structure = phonopy2jamip(phonon.supercell)
            incar['npar'] = None
            incar['ncore'] = 1
            self.calculator(task_id, stdout, stdin)
            status = incar.get_status(stdout)
            self.write_status(status, stdout)
               
        elif task_id == 'fc3':

            digit = len(str(len(phonon.supercells_with_displacements)))
            for i,supercell in enumerate(phonon.supercells_with_displacements):
                output = stdout / str(i).zfill(digit)
                incar.structure = phonopy2jamip(supercell)
                self.set_input(incar, output, stdin)
                subtasks.append(output)

        elif task_id == 'raman':

            digit = len(str(len(phonon.supercells_with_displacements)))
            for i,cell in enumerate(phonon.supercells_with_displacements):
                output = stdout / str(i).zfill(digit)
                incar.structure = cell
                self.set_input(incar, output, stdin)
                subtasks.append(output)

        elif task_id == 'softmode':

            digit = len(str(len(phonon.get_modulated_supercells())))
            for i,supercell in enumerate(phonon.get_modulated_supercells()):
                output = stdout / str(i).zfill(digit)
                incar.structure = phonopy2jamip(supercell)
                self.set_input(incar, output, stdin)
                subtasks.append(output)
        
        elif task_id == 'gruneisen':

            structures = {}
            # relax %
            for scale, structure in phonon.items():
                incar.structure = structure
                output = stdout / scale / 'relax'
                self.calculator(task_id, output, stdin)
                status = incar.get_status(output)
                structures[scale] = self.load_structure(output)

            # check and update%
            if incar.state == 'E':
                raise RuntimeError('Gruneisen relax calculation failed.') 
            # update fc params %
            if len(self.links[task_id]):
                fc_id = self.links[task_id][0]
                fc_incar = deepcopy(self.tasks[fc_id])
                fc_incar.pop('dim',0)
                fc_incar.pop('symprec',0)
            else:
                fc_incar = deepcopy(incar)
                fc_incar.pop('isif', 4) 
                fc_incar.pop('nsw', 0) 

            # fc %
            for scale, structure in structures.items():

                unitcell = jamip2phonopy(structure)
                phonon = Phonopy(unitcell,incar.dim,symprec=incar.symprec)
                phonon.generate_displacements()
                dump_hdf5('structure/gruneisen/%s' %scale, structure)

                digit = len(str(len(phonon.supercells_with_displacements)))
                for i,supercell in enumerate(phonon.supercells_with_displacements):
                    fc_incar.structure = phonopy2jamip(supercell)
                    output = stdout / scale / str(i).zfill(digit)
                    self.set_input(fc_incar, output, stdin)
                    subtasks.append(output)

        self.batch_calculator(task_id, subtasks)

    def initialize_phonon(self, task_id):
        from jamip.structure.convert import jamip2phonopy
        from jamip.analysis.vasp import PhononFinder
        from phonopy.file_IO import parse_FORCE_SETS
        from phonopy import Phonopy
        from os.path import join, exists, relpath
        from jamip.structure import Structure

        def reshape(dim:str):
            dim = np.array(dim.split(), dtype=int)
            if dim.size == 3:
                dim = np.diag(dim)
            elif dim.size == 9:
                dim = dim.reshape(3,3)
            else: 
                raise ValueError('Dim params is incorrectly set in the %s' %task_id)
            return dim

        # set structure %
        structure = self.std_structure
        if task_id in ['fc2', 'fc3']:
            if len(self.links[task_id]):
                stdin = self.get_stdin(task_id)
                structure = self.load_structure(stdin)  
        elif task_id in ['raman', 'gruneisen', 'softmode']:
            if len(self.links[task_id]):
                if self.links[task_id][0] == 'fc2':
                    cell = load_hdf5('structure/fc2')
                    structure = Structure.from_cell(cell)
                elif self.links[task_id][0] == 'fc3':
                    cell = load_hdf5('structure/fc3')
                    structure = Structure.from_cell(cell)
                else:
                    stdin = self.get_stdin(task_id)
                    structure = self.load_structure(stdin)

        incar = self.tasks[task_id]
        symprec = 1e-4 
        if 'dim' in incar:
            dim = incar.pop('dim')
            if dim.strip().lower() == 'auto':
                lattice_parameters = structure.lattice_parameters[:3] 
                dim = np.diag(np.ceil(10 / lattice_parameters).astype(int))
            else:
                dim = reshape(dim)
        elif task_id == 'dfpt':
            dim = incar.pop('dim', '1 1 1')
            dim = reshape(dim)
        elif len(self.links[task_id]) and self.links[task_id][0] == 'fc2':
            info = load_hdf5('phonon')
            dim = info['dim']
            symprec = info['symprec']
            logging.info(f'initialize {task_id} supercell dimension from fc2')
        else:
            raise ValueError('Dim params is incorrectly set in {task_id}')

        if 'SYMMETRY_TOLERANCE' in incar:
            symprec = float(incar.pop('SYMMETRY_TOLERANCE'))

        if task_id == 'fc2' or task_id == 'dfpt':

            # init phonopy %
            unitcell = jamip2phonopy(structure)
            phonon = Phonopy(unitcell, dim, symprec=symprec)
            phonon.generate_displacements()

            # info params %
            info = {'dim': dim.tolist(), 'symprec': symprec, 'main': 'phonon/fc2'}
            dump_hdf5('fc2', info)
            dump_hdf5('structure/fc2', structure)

        elif task_id == 'fc3':
            from phono3py import Phono3py

            # init phonopy %
            unitcell = jamip2phonopy(structure)
            phonon = Phono3py(unitcell, dim, symprec=symprec)
            phonon.generate_displacements()

            # info params %
            info = {'dim': dim.tolist(), 'symprec': symprec, 'main': 'phonon/fc3'}
            dump_hdf5('fc3', info)
            dump_hdf5('structure/fc3', structure)

        elif task_id == 'raman':
            from jamip.structure.symmetry import RamanData
            ramanmaps = RamanData.set_index('pointgroup', drop=False).to_dict('index')

            # init structure % 
            lattice = structure.lattice
            positions = structure.get_positions(type='cartesian')
            elements = structure.get_elements()

            # init phonopy %
            pf = PhononFinder(self.rootdir)
            phonon = pf.get_phonon_from_info()
            # to compare with mesh.yaml
            for i,values in enumerate(phonon.dataset['first_atoms']):
                phonon.dataset['first_atoms'][i]['displacement'] = np.around(values['displacement'],16)

            # get fc2 class %
            fc2path = self.rootdir / 'phonon' / 'fc2'
            if exists(fc2path):
                forces = pf.get_forces(fc2path)
                phonon.set_forces(forces)
                phonon.produce_force_constants(calculate_full_force_constants=False)
                phonon.symmetrize_force_constants()
            else:
                raise OSError("phonopy fc2 calculation failed!")

            # run mesh %
            phonon.run_mesh(mesh=[1,1,1], with_eigenvectors=True)
            #phonon.mesh.write_yaml()
            dataset = phonon.get_mesh_dict() 
            eigenvectors = np.real(dataset['eigenvectors']) # shape: (nqpoint, nbands, nbands)
            eigenvectors = eigenvectors.reshape(2,3,6).transpose(2,0,1) # shape: (nqpoint, nbands, nbands)
            eigendisplacements = eigenvectors / np.sqrt(phonon.masses)[None,:,None]

            # run irreps % 
            band_indices = []
            phonon.set_irreps(q=[0,0,0])
            # TODO -4m2 or -42m?
            pgs = phonon.symmetry.pointgroup_symbol
            raman_active_modes = ramanmaps[pgs]['raman'].split()

            for i,label in enumerate(phonon.irreps._ir_labels):
                #print(label, phonon.irreps._degenerate_sets[i])
                if label in raman_active_modes:
                    band_indices.extend(phonon.irreps._degenerate_sets[i])
                    #print(pgs, label)
            #print(band_indices)
 
            # create displacements %
            disp_structures = []
            disps = []
            maxdisps = []
            Ramanstep = 0.01
            RamanStepType = 'norm_x'
            for i in band_indices:
                for scale in (-1,1):
                    dispstep = scale * Ramanstep / np.linalg.norm(eigendisplacements[i])
                    coords = positions + eigendisplacements[i] * dispstep
                    disp_structure = Structure.from_cell((lattice,coords,elements), direct=False)
                    disps.append(dispstep)
                    maxdisps.append( np.max(eigendisplacements[i]*dispstep)*scale )
                    disp_structures.append(disp_structure)
            phonon._supercells_with_displacements = disp_structures

            # info params %
            info = {'mesh': [1,1,1], 'qpoint': [0,0,0], 'numstep':2, 'step': Ramanstep,
                    'band_indices': band_indices, 'frequency': dataset['frequencies'],
                    'max_cartesian_displacement': maxdisps, 'displacement_step': disps, 
                    'volume': structure.volume, 'main': 'phonon/raman'}
            dump_hdf5('raman', info)

        elif task_id == 'softmode':
            from phonopy.file_IO import parse_FORCE_SETS

            # set stdin %
            if len(self.links[task_id]):
                stdin = self.get_stdin(task_id)
            else:
                raise ValueError("The calculation requires at least one dependency! ")

            # get Phonopy with force %
            pf = PhononFinder(self.rootdir)
            phonon = pf.get_phonon_from_info()
            if not exists(join(stdin, 'FORCE_SETS')):
                pf.write_forces(phonon, stdin)
            force_sets=parse_FORCE_SETS(filename=join(stdin, 'FORCE_SETS'))
            phonon.set_displacement_dataset(force_sets)
            phonon.produce_force_constants(calculate_full_force_constants=False)

            # Add parameters %
            try:
                amplitude = np.array(incar.pop('amplitude').split(),dtype = float)      
                if len(amplitude) == 3 and amplitude[2] < amplitude[1]:
                    amplitude = np.arange(amplitude[0],amplitude[1],amplitude[2])
                q = np.array(incar.pop('q').split(),dtype = float)                      
                argument = incar.pop('argument')                                        
                band_index = incar.pop('band_index')      
            except:
                raise ValueError('Missing necessary parameters in %s' %task_id)

            softmode = []
            for scale in amplitude:
                softmode.append([q, band_index, scale, argument])
            phonon.set_modulations(dim,softmode)

            # info params
            info = {'dim': dim.tolist(), 'symprec': symprec, 'q': q, 'band_index': band_index,
                    'scale' :amplitude, 'argument': argument, 'main': '/phonon/softmode'}
            dump_hdf5('softmode', info)

        elif task_id == 'gruneisen':
            from jamip.structure import Structure

            # initialize structure %
            scale = float(incar.pop('scale', 0.003))
            # minus %
            minus_structure = deepcopy(structure)
            minus_structure.scale_factor = 1-scale
            # plus %
            plus_structure = deepcopy(structure)
            plus_structure.scale_factor = 1+scale

            # skip force or not %
            if len(self.links[task_id]) and self.links[task_id][-1] == 'fc2':
                logging.info('gruneisen calculation skip stardand structure')
                cells = {'minus':minus_structure, 'plus':plus_structure}
                stdout = self.rootdir/'phonon'/'gruneisen'
                if not stdout.exists(): stdout.mkdir()
                if not (stdout/'orig').exists():
                    os.symlink(relpath(self.tasks['fc2'].path, start=stdout), stdout/'orig')
            else:
                cells = {'orig':structure, 'minus':minus_structure, 'plus':plus_structure}
                 
            incar.symprec = symprec
            incar.dim = dim
            phonon = cells
  
            # info params
            info = {'dim': dim.tolist(), 'symprec': symprec, 'scale': [1-scale,1+scale], 'main': '/phonon/gruneisen'}
            dump_hdf5('gruneisen', info)
                 
        return phonon
            
    def set_input(self, incar, stdout:str, stdin=None, **kwargs):
        from .vaspio import VaspIO

        stdout = pathlib.Path(stdout)
        stdout.mkdir(parents=True, exist_ok=True)
 
        # UPDATE GROUP KWARGS
        for xc in incar.xc_func:
            if xc in ['hse','soc','ldau']:
                incar.group_update(xc, self.extra[xc])

        # VDW %
        files = self.func.get_vdW(incar) 
        VaspIO.write_files(files, stdout)

        # debug metagga %
        if 'metagga' in incar:
            if 'gga' in incar:
                incar.pop('gga')
            if 'GGA' in incar:
                incar.pop('GGA')

        # WAVECAR & CHGCAR
        if stdin != None:
            self.load_wavchg(incar, stdout, stdin)

        # POTCAR %
        if incar.structure.get_formula() != self.potential.formula: 
            self.potential = self.func.get_potential(incar.structure)
        VaspIO.write_potcar(self.potential.files, stdout, name="POTCAR")
        if 'encut' not in incar:
            if self.func.cutoff > self.potential.enmin:
                incar['encut'] = self.func.cutoff
            else:
                incar['encut'] = min(round(self.func.cutoff*self.potential.enmax,2) ,600)

        # POSCAR %
        if stdin != None and self.func.force_copy_poscar:
            self.load_structure(stdin, stdout)
        else:
            VaspIO.write_poscar(incar.structure, stdout, name='POSCAR')

        # remove old KPOINTS %
        if (stdout / 'KPOINTS').exists():
            os.remove(stdout / 'KPOINTS')
        if (stdout / 'KPOINTS_OPT').exists():
            os.remove(stdout / 'KPOINTS_OPT')

        # KPOINTS_OPT
        if incar.kpoints_opt != None:
            if self.func.kpoints_opt:
                incar.kpoints_opt = incar.kpoints_opt.get_gamma_kpoints(
                        cell=incar.structure.lattice, 
                        model=self.func.force_create_kpoints
                        )
                VaspIO.write_kpoints(incar.kpoints_opt, stdout, name='KPOINTS_OPT')
            else:
                incar.kpoints = incar.kpoints_opt

        # KPOINTS %
        if self.func.force_create_kpoints:
            incar.kpoints = incar.kpoints.get_gamma_kpoints(
                    cell=incar.structure.lattice,
                    model=self.func.force_create_kpoints
                    )

        if incar.kpoints.model != 'kspacing':
            VaspIO.write_kpoints(incar.kpoints, stdout)
        elif 'kspacing' not in incar:
            incar['kspacing'] = incar.kpoints.value

        # LDAU %
        self.func.get_ldau(incar)

        # other files %
        VaspIO.write_files(self.func.external_files, stdout)
        if self.func.optcell != None:
            VaspIO.write_optcell(self.func.optcell, stdout)

        # get spin %
        if 'lsorbit' in incar and incar['lsorbit'] == True:
            incar.spin = 3
        elif 'ispin' in incar and incar['ispin'] == 2:
            incar.spin = 2

        # set nband %
        if 'nbands' not in incar and incar.nbds is not None:
            incar['nbands'] = self.get_nbands(incar)

        # initialize magmon %
        if incar.spin >= 2 and 'magmom' not in incar:
            incar['magmom'] = self.get_magmom(incar)

        # INCAR %
        VaspIO.write_incar(incar, stdout)

        # GET program
        if isinstance(self.func.program, str):
            incar.program = self.func.program

        elif incar.spin == 3 and 'ncl' in self.func.program:
            incar.program = self.func.program['ncl']

        elif 'gam' in self.func.program and incar.kpoints.model in ('Monkhorst-pack','Gamma') and not np.any(incar.kpoints.value[0]-1):
            incar.program = self.func.program['gam']

        elif 'std' in self.func.program:
            incar.program = self.func.program['std']

        else:
            logging.error("The current VASP program is unavailable")

#------------------------------------------------------------#
#                 get calculation parameters                 #
#------------------------------------------------------------#

    def get_magmom(self, incar):

        if 'magmom' in self.extra and self.extra['magmom'] != None:
            return self.extra['magmom']

        else:
            return self.func.get_magmom(incar)

    def get_npar(self, incar):

        if 'npar' in incar and incar['npar'] != None:
            return int(incar['npar'])
        elif 'ncore' in incar:
            return int(np.floor(self.cluster.cores/incar['ncore']))
        else:
            return 4

    def get_nbands(self, incar): 
        """
        job: optics, 
        """
        npar = self.get_npar(incar)
        natom = sum(incar.structure.number_of_atoms)
        if incar.spin == 3:
            nocc = self.potential.nelect 
        else:
            nocc = self.potential.nelect / 2
        nbands = incar.nbds

        if nbands < max(nocc,4):
            # nbands multiple %
            nelect = max(nocc+natom/2, nocc*1.2)
            nbands = int((nelect * nbands // npar + 1) * npar)
        else:
            # nbands value %
            nbands = int(nbands)

        return nbands

    def get_kpath(self, task_id, insert=None, stdin=None, prec='suggest', **kwargs):
        '''
        kwargs:
            task_id: get kpath type
            insert: kpath insert number
            prec: suggest or all, to decide calculate how many kpath 
            structure: use input structure rather than incar structure
        '''
        from jamip.utils.brillouin_zone import HighSymmetryKpath, HighSymmetryKpath2d
        from jamip.abtools.base.kpoints import BandPath

        kptin = None
        kpath = self.func.kpath
        if task_id in kpath:
            if isinstance(kpath[task_id], Kpoints):
                kptin = kpath[task_id]
            elif isinstance(kpath[task_id], int) and insert == None:
                insert = kpath[task_id]

        if 'structure' in kwargs:
            structure = kwargs['structure']
        elif task_id in self.tasks:
            structure = self.tasks[task_id].structure
        else:
            structure = self.std_structure

        if 'prec'in kwargs:
            prec = kwargs['prec']
        else:
            prec = 'suggest'
            if task_id in self.tasks and 'kpath' in self.tasks[task_id]:
                prec = self.tasks[task_id]['kpath']
 
        if task_id == 'band':
            if kptin != None: return kptin
            if insert == None: insert = 20

            if self.func.dimension == 2:
                bz = HighSymmetryKpath2d()
            else:
                bz = HighSymmetryKpath()
            kpoint = bz.get_HSKP(structure.to_cell(), symprec=1e-3)
            print(kpoint)
            # suggest path %
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

        elif task_id == 'emass' or task_id == "hse_emass":
            # from structure %
            rec_lattice = np.linalg.inv(structure.lattice)
            v1 = rec_lattice[0] 
            v2 = np.cross(v1, np.cross(v1, rec_lattice[1]))
            v3 = np.cross(v1, v2)
            # from incar %
            insert = int(self.tasks[task_id].pop('insert', 7))
            mesh = float(self.tasks[task_id].pop('mesh', 0.01))
            scale = np.arange(insert) - int(insert/2) 
            # from stdin %
            edge = self.get_band_edge(stdin)
            cbm = edge['cbm'].kpoints
            vbm = edge['vbm'].kpoints

            if self.func.dimension == 2:
                axes = self.tasks[task_id].pop('axis', 'xy')
            else:
                axes = self.tasks[task_id].pop('axis', 'xyz')

            if task_id == 'emass':

                kpaths = {}
                for i,vector in enumerate((v1, v2, v3)):
                    axis = 'xyz'[i]
                    if axis not in axes: continue
                    frac_vector = np.dot(vector, structure.lattice)
                    std_vector = np.round(frac_vector / np.linalg.norm(frac_vector),6) * mesh
             
                    if np.sum(np.abs(cbm - vbm)) < 0.001:
                        kpoints = [vbm + std_vector*d for d in scale]
                        kpaths[f'{axis}-cbm-vbm'] = Kpoints('r', kpoints)
                    else:
                        # cbm %
                        kpoints = [cbm + std_vector*d for d in scale]
                        kpaths[f'{axis}-cbm'] = Kpoints('r', kpoints)
                        # vbm %
                        kpoints = [vbm + std_vector*d for d in scale]
                        kpaths[f'{axis}-vbm'] = Kpoints('r', kpoints)
             
                return kpaths

            elif task_id == 'hse_emass':

                kpath = []
                kpoints = {}
                for i,vector in enumerate((v1, v2, v3)):
                    axis = 'xyz'[i]
                    if axis not in axes: continue
                    frac_vector = np.dot(vector, structure.lattice)
                    std_vector = np.round(frac_vector / np.linalg.norm(frac_vector),6) * mesh
             
                    if np.sum(np.abs(cbm - vbm)) < 0.001:
                        klabel1 = f'{axis}-cbm-vbm-1'
                        klabel2 = f'{axis}-cbm-vbm-2'
                        kpath.append([klabel1, klabel2])
                        kpoints[klabel1] = vbm + std_vector*min(scale) # P1
                        kpoints[klabel2] = vbm + std_vector*max(scale) # P2
                    else:
                        # cbm %
                        klabel1 = f'{axis}-cbm-1'
                        klabel2 = f'{axis}-cbm-2'
                        kpath.append([klabel1, klabel2])
                        kpoints[klabel1] = cbm + std_vector*min(scale) # P1
                        kpoints[klabel2] = cbm + std_vector*max(scale) # P2
                        # vbm %
                        klabel1 = f'{axis}-vbm-1'
                        klabel2 = f'{axis}-vbm-2'
                        kpath.append([klabel1, klabel2])
                        kpoints[klabel1] = vbm + std_vector*min(scale) # P1
                        kpoints[klabel2] = vbm + std_vector*max(scale) # P2
             
                bandpath = BandPath.from_symbols(kpath, kpoints, insert) 
                return Kpoints(bandpath)

        elif task_id == 'emc':
            from jamip.analysis.vasp.band import Emc
            # from structure %
            rec_lattice = np.linalg.inv(structure.lattice)
            # from incar %
            emctype = str(self.tasks[task_id].pop('emc', 'st3'))
            mesh = float(self.tasks[task_id].pop('mesh', 0.01))
            # from stdin %
            edge = self.get_band_edge(stdin)
            cbm = edge['cbm'].kpoints
            vbm = edge['vbm'].kpoints
            kpaths = {}

            if np.sum(np.abs(cbm - vbm)) < 0.001:
                kpoints = Emc().set_kpoints(cbm,rec_lattice,mesh,emctype)
                kpaths['t-cbm-vbm'] = Kpoints('r', kpoints)
            else:
                # cbm %
                kpoints = Emc().set_kpoints(cbm,rec_lattice,mesh,emctype)
                kpaths['t-cbm'] = Kpoints('r', kpoints)
                # vbm %
                kpoints = Emc().set_kpoints(vbm,rec_lattice,mesh,emctype)
                kpaths['t-vbm'] = Kpoints('r', kpoints)

            return kpaths
        
        elif 'band' in task_id:
            return self.get_kpath('band', insert=insert, prec=prec, structure=structure) if kptin == None else kptin

        else:
            raise ValueError(f'Invalid task_id: {task_id}')    


    def get_band_edge(self, stdin:str):
        '''
        '''
        from jamip.analysis.vasp import BandFinder
        bf = BandFinder(stdin).get_data()
        if hasattr(self.func, 'metal') and self.func.metal is True:
            cbvbs = bf.get_metal_cbmvbm()            
            gap_full = cbvbs['full']['gap']
            gap_empty = cbvbs['empty']['gap']
            if gap_full >= gap_empty:
                edge = cbvbs['full']
                logging.error("This calculation set bandgap between full and others")
            else:
                edge = cbvbs['empty'] 
                logging.error("This calculation set bandgap between empty and others")
        else:
            edge = bf.get_cbmvbm()

        gap = edge.pop('gap')
        if gap < 0:
            logging.error("This calculation does not apply to a zero band gap system")
            raise ValueError("This calculation does not apply to a zero band gap system. current gap is %f" %gap)

        return edge

    @property
    def raw_structure(self):
        return self._raw_structure

    @raw_structure.setter
    def raw_structure(self, value):
        from jamip.structure import Structure

        if isinstance(value, Structure):
            self._raw_structure = deepcopy(value)
            dump_hdf5('structure/raw', value)

        else:
            raise ValueError("Invalid raw structure! ")

    @property
    def std_structure(self):        
        return self._std_structure if self._std_structure != None else self._raw_structure

    @std_structure.setter
    def std_structure(self, stdin:str):

        s = self.load_structure(stdin=stdin)
        self._std_structure = s
        dump_hdf5('structure/std', s)
        logging.info('setting standard stducture from %s' %stdin)

    def get_stdin(self, task_id):
        return self.tasks[self.links[task_id][0]].path

    def batch_calculator(self, task_id, paths):
        from jamip.compute.pool import Pool
        from jamip.utils.utils import get_stdout
        from socket import gethostname
        import time

        incar = self.tasks[task_id]
        maximum = min(incar.parallel, len(paths))
        if maximum == 1:
            states = []
            for output in paths:
                status = CheckStatus.get_status(output, incar.name)
                if not status['success']:
                    self.mpirun(incar, output)
                    status = incar.get_status(output)
                    states.append(incar.state)
                else:
                    status = {'task':task_id, 'finish': True, 'success': True}
                    states.append('C')
            condition = all(s == 'C' for s in states)
            stdout = get_stdout(paths)
            if condition:
                incar.state = 'C'
                self.write_status(status, stdout)

        elif maximum > 1:
            basestatus = {'status': 'W', 'prior': 3, 'host': f'{gethostname()} '}
            poolfile = self.rootdir / task_id
            with Pool.open(poolfile, 'n') as pool:
                for path in paths:
                    outdir = os.path.relpath(path, self.rootdir)
                    pool[outdir] = basestatus

            outdir = f'{self.rootdir} {task_id}'
            self.cluster.script = 'batch.sh'
            self.cluster.write_script(incar.program, outdir, 'jamip.compute.batch')

            # submit 
            for i in range(maximum):
                os.popen(f'{self.cluster.cmd} batch.sh').readline()
                time.sleep(1)

            # exit current flow %
            self.__exit = True

if __name__ == "__main__":

    tasks = "relax scf band emass"
    links = {'scf':['relax'], 'band':['scf'], 'emass':['scf','band']}

    vasp = VaspFlow(tasks, links)
    print(vasp.state)
            
