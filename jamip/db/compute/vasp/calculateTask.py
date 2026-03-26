# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import shutil
from compute.task import Task, TaskBuilder
from utils.variables import state_of_calculation
from compute.vasp.incar import INCAR
from compute.vasp.poscar import POSCAR
from compute.vasp.potcar import POTCAR
from compute.vasp.kpoints import KPOINTS
from compute.vasp.optcell import OPTCELL
from compute.vasp.submitScript import SubmitScript


class CalculateTask(Task):
    """
    normal calculation's task. i.g. opt scf band ...
    """
    
    def __init__(self, name, path, structure):
        """
        Arguments:
            name: name of calculate task.
            path: store path of calculate task data.
            structure: structrue's object.
        """
        super().__init__(name=name, 
                         path=path)
# =============================================================================
#         super(CalculateTask, self).__init__(name=name, 
#                                             path=path)
# =============================================================================
        
        self._structure=structure
        self._state=state_of_calculation.unknown
        
        self._INCAR=None
        self._POSCAR=None
        self._POTCAR=None
        self._submitScript=None
        # optional
        self._KPOINTS=None
        self._OPTCELL=None
    
    def __str__(self):
        return '{0} | {1} | {2}'.format(self.path, self.name, self.state)
    
    @property
    def structure(self):
        return self._structure
    
    @property
    def state(self):
        """
        state of task.
        """
        return self.check()
    
    @property
    def INCAR(self):
        return self._INCAR

    @property
    def POSCAR(self):
        return self._POSCAR
    
    @property
    def POTCAR(self):
        return self._POTCAR
    
    @property
    def submitScript(self):
        return self._submitScript
    
    @property
    def KPOINTS(self):
        return self._KPOINTS
    
    @property
    def OPTCELL(self):
        return self._OPTCELL
    
    def check(self, **kwargs):
        """
        check the state of calculation. Note theis method will synchronously update the value of 'self.state'.
        
        Retrurn:
            Enum-type state of calculation.
        """
        
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
        else:
            path=self.path
            
        state=None
        if os.path.exists('{0}/{1}'.format(path, state_of_calculation.prepare.value)):
            state=state_of_calculation.prepare
        elif os.path.exists('{0}/{1}'.format(path, state_of_calculation.calculating.value)):
            state=state_of_calculation.calculating
        elif os.path.exists('{0}/{1}'.format(path, state_of_calculation.finished.value)):
            state=state_of_calculation.finished
        elif os.path.exists('{0}/{1}'.format(path, state_of_calculation.error.value)):
            state=state_of_calculation.error
        else:
            state=state_of_calculation.unknown
            
        # udpate
        self._state=state
        
        return state
        
    def run(self):
        """
        run this task.
        """
        from time import sleep
        
        self.set_state(state_of_calculation=state_of_calculation.calculating)
        self.notify_listener('calculate-task-starting\n')
        sleep(3)
        self.set_state(state_of_calculation=state_of_calculation.finished)
        self.notify_listener('calculate-task-finished\n')

    # flag file
    def set_state(self, state_of_calculation, **kwargs):
        """
        set state of calculate task, and create a corresponding flag file with blank.
        
        Arguments:
            state_of_calculation: state of calculation (Enum-type).
            
            kwargs:
                path: path of OPTCELL file.
        """
        # remove old state
        self.remove_state(**kwargs)
        
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.path
        
        filename=None
        if state_of_calculation == state_of_calculation.prepare:
            filename=state_of_calculation.prepare.value
        elif state_of_calculation == state_of_calculation.calculating:
            filename=state_of_calculation.calculating.value
        elif state_of_calculation == state_of_calculation.finished:
            filename=state_of_calculation.finished.value
        elif state_of_calculation == state_of_calculation.error:
            filename=state_of_calculation.error.value
        else:
            raise ValueError('unknown state_of_calculation')
        
        # write new state
        with open('{}/{}'.format(path, filename), 'w') as f:
            f.write('')
            
        # update
        self.check()
        
        return self
    
    def remove_state(self, **kwargs):
        """
        remove state of calculate task
        
        Arguments:
            kwargs:
                path: path of OPTCELL file.
        """
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.path
        
        # prepare
        path0='{}/{}'.format(path, state_of_calculation.prepare.value)
        if os.path.exists(path0):
            os.remove(path0)
        # calculating
        path0='{}/{}'.format(path, state_of_calculation.calculating.value)
        if os.path.exists(path0):
            os.remove(path0)
        # finished
        path0='{}/{}'.format(path, state_of_calculation.finished.value)
        if os.path.exists(path0):
            os.remove(path0)
        # error
        path0='{}/{}'.format(path, state_of_calculation.error.value)
        if os.path.exists(path0):
            os.remove(path0)
            
        # update
        self.check()
        
        return self
    

class CalculateTaskBuilder(TaskBuilder):
    
    def __init__(self, name, path, structure, **kwargs):
        """
        build a calculate task.
        
        Arguments:
            name: name of calculate task.
            path: store path of calculate task data.
            structure: structrue's object.
            
            kwargs:
                isClear: if true, clear old data in calculate task's dictionary.
        """
        self.task=CalculateTask(name=name, 
                                path=path, 
                                structure=structure)
        self.task._builder=self
                    
                    
        isClear=False
        if 'isClear' in kwargs:
            isClear=kwargs['isClear']                
            
        if os.path.exists(path):
            if isClear:
                shutil.rmtree(path)
                os.makedirs(path)
        else:
            os.makedirs(path)
    
    def set_INCAR(self, **kwargs):
        """
        generate INCAR file
        
        Arguments:
            kwargs:
                path: path of INCAR file.
                filename: filename of INCAR file.
                parameters: customized parameters (dictionary-type). i.e. {'EDIFF':1e-6, 'EDIFFG':-1e-4, ...}
                
        Return:
            builder's ojbect.
        """
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.task.path
        filename=None
        if 'filename' in kwargs:
            filename=kwargs['filename']
            kwargs.pop('filename')
        if filename is None:
            self.task._INCAR=INCAR(path=path, **kwargs).output()
        else:
            self.task._INCAR=INCAR(path=path,filename=filename, **kwargs).output()
        
        return self
    
    def set_POSCAR(self, **kwargs):
        """
        generate POSCAR file
        
        Arguments:
            kwargs:
                path: path of POSCAR file.
                filename: filename of POSCAR file.
        Return:
            builder's ojbect.
        """
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.task.path
        filename=None
        if 'filename' in kwargs:
            filename=kwargs['filename']
            kwargs.pop('filename')
        structure=self.task.structure
        if 'structure' in kwargs:
            structure=kwargs['structure']
            kwargs.pop('structure')
# =============================================================================
#         if filename is None:
#             self.task._POSCAR=POSCAR(path=path, **kwargs).output()
#         else:
#             self.task._POSCAR=POSCAR(path=path, filename=filename, **kwargs).output()
# =============================================================================
        if filename is None:
            self.task._POSCAR=POSCAR(path=path, structure=structure).output()
        else:
            self.task._POSCAR=POSCAR(path=path, filename=filename, structure=structure).output()
            
        return self
    
    def set_POTCAR(self, **kwargs):
        """
        generate POTCAR file
        
        Arguments:
            kwargs:
                dtype (default='PBE'): type of exchange-correlation functional. i.e. 'LDA', 'PBE', 'PW91'
                path: path of POTCAR file.
                filename: filename of POTCAR file.
        Return:
            builder's ojbect.
        """
        dtype=None
        if 'dtype' in kwargs:
            dtype=kwargs['dtype']
            kwargs.pop('dtype')
        else:
            dtype='PBE'
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.task.path
        filename=None
        if 'filename' in kwargs:
            filename=kwargs['filename']
            kwargs.pop('filename')
        if filename is None:
            self.task._POTCAR=POTCAR(dtype=dtype, path=path, **kwargs).output()
        else:
            self.task._POTCAR=POTCAR(dtype=dtype, path=path, filename=filename, **kwargs).output()
        return self
    
    def set_KPOINTS(self, **kwargs):
        """
        generate KPOINTS file
        
        Arguments:
            kwargs:
                path: path of KPOINTS file.
                filename: filename of KPOINTS file.
        Return:
            builder's ojbect.
        """
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.task.path
        filename=None
        if 'filename' in kwargs:
            filename=kwargs['filename']
            kwargs.pop('filename')
        if filename is None:
            self.task._POTCAR=KPOINTS(path=path, **kwargs).output()
        else:
            self.task._POTCAR=KPOINTS(path=path, filename=filename, **kwargs).output()
        return self
    
    def set_OPTCELL(self, **kwargs):
        """
        generate OPTCELL file
        
        Arguments:
            kwargs:
                path: path of OPTCELL file.
                filename: filename of OPTCELL file.
        Return:
            builder's ojbect.
        """
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.task.path
        filename=None
        if 'filename' in kwargs:
            filename=kwargs['filename']
            kwargs.pop('filename')
        if filename is None:
            self.task._OPTCELL=OPTCELL(path=path, **kwargs).output()
        else:
            self.task._OPTCELL=OPTCELL(path=path, filename=filename, **kwargs).output()
        return self
    
    def set_sumbitScript(self, **kwargs):
        """
        generate sumbit script file.
        
        Arguments:
            kwargs:
                path: path of OPTCELL file.
                filename: filename of OPTCELL file.
        Return:
            builder's ojbect.
        """
        path=None
        if 'path' in kwargs:
            path=kwargs['path']
            kwargs.pop('path')
        else:
            path=self.task.path
        filename=None
        if 'filename' in kwargs:
            filename=kwargs['filename']
            kwargs.pop('filename')
        if filename is None:
            self.task._submitScript=SubmitScript(path=path, **kwargs).output()
        else:
            self.task._submitScript=SubmitScript(path=path, filename=filename, **kwargs).output()
        return self
    
    # flag file
    def set_state_of_task(self, state_of_calculation, **kwargs):
        """
        set state of calculate task, and create a corresponding flag file with blank.
        
        Arguments:
            state_of_calculation: state of calculation (Enum-type).
            
            kwargs:
                path: path of OPTCELL file.
        """
        self.task.set_state(state_of_calculation=state_of_calculation, **kwargs)        
        return self
    
    def remove_state_of_task(self, **kwargs):
        """
        remove state of calculate task
        
        Arguments:
            kwargs:
                path: path of OPTCELL file.
        """
        self.task.remove_state(**kwargs)        
        return self

