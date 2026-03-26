# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import shutil
from compute.vasp.calculateTask import CalculateTask, CalculateTaskBuilder
from utils.variables import state_of_calculation


class PhononTask(CalculateTask):
    
    def __init__(self, name, path_of_parent, structure_of_parent):
        """
        collection of calculation's tasks
        
        Arguments:
            name: name of task.
            path_of_parent: primitive structure's path.
            structure_of_parent: primitive structure's object.
        """
        super().__init__(name=name, 
                         path=None, 
                         structure=None)
# =============================================================================
#         super(PhononTask, self).__init__(name=name, 
#                                          path=None, 
#                                          structure=None)
# =============================================================================
        
        self._path_of_parent=path_of_parent
        self._structure_of_parent=structure_of_parent
        
    @property
    def path_of_parent(self):
        return self._path_of_parent
    
    @property
    def structure_of_parent(self):
        return self._structure_of_parent
    
    def check(self):
        """
        check all childtask's states.
        
        Return:
            collection of states of childtask.
        """
        states=[]
        for path0 in self.path:
            states.append(super(PhononTask, self).check(path=path0))
        
        # update
        self._state=states
        return states
    
    def run(self):
        """
        run this task.
        """
        from time import sleep
        
# =============================================================================
#         # static method: onlu iterator the initialized list. cannot traverse the run-time task (dynamics).
#         for childTask in list(self.childTasks):
#             childTask.run()
# =============================================================================
            
        # dynamics method:
        while len(self._childTasks4run) > 0:
            childTask=self._childTasks4run.pop()
            if childTask.state == state_of_calculation.prepare:    
                childTask.run()
                print('\n{}-{}-running\n'.format(childTask.name, childTask.path))
            elif childTask.state == state_of_calculation.finished:
                print('\n{}-{}-finished\n'.format(childTask.name, childTask.path))
    
    # observer pattern
    def register(self, listener):
        """
        register this task to pool.
        
        Arguments:
            listener: listener's object.
        """
        
        for childtask in self._childTasks:
            listener.tasks.appendleft(childtask)
            childtask._listener=listener
#        self._listener=listener
        
    def deregister(self, childtask):
        """
        deregister a task from this listener.
        
        Arguments:
            childtask: childtask's object.
        """
        childtask._listener.tasks.remove(childtask)
        childtask._listener=None
    
    def notify_listener(self, childtask, event):
        """
        notify this listener.
        
        Arguments:
            childtask: childtask's objetct. 
            event: notification information.
        """
        childtask._listener.notify(childtask, event)

    # flag file
    def set_state(self, state_of_calculation, childtasks, **kwargs):
        """
        set state of calculate task, and create a corresponding flag file with blank.
        
        Arguments:
            state_of_calculation: state of calculation (Enum-type).
            childtasks: sub-tasks needed to change the state.
            paths: paths of corresponding sub-task.
        """
        # remove old state
        self.remove_state(childtasks=childtasks, **kwargs)
        
        paths=None
        if 'paths' in kwargs:
            paths=kwargs['paths']
            if len(paths) != len(childtasks):
                raise ValueError("missed some paths in 'paths'")
        else:
            paths=[s0.path for s0 in childtasks]

        for i in range(0, len(childtasks)):
            childtask=childtasks[i]
            path=paths[i]
            if childtask in self.childTasks:
                filename=None
                if state_of_calculation == state_of_calculation.prepare:
                    filename=state_of_calculation.prepare.value
                elif state_of_calculation == state_of_calculation.calculating:
                    filename=state_of_calculation.calculating.value
                elif state_of_calculation == state_of_calculation.finished:
                    filename=state_of_calculation.finished.value
                elif state_of_calculation == state_of_calculation.error:
                    filename=state_of_calculation.error.value
                elif state_of_calculation == state_of_calculation.unknown:
                    filename=state_of_calculation.unknown.value
                else:
                    raise ValueError('unknown state_of_calculation')
        
                # write new state
                with open('{}/{}'.format(path, filename), 'w') as f:
                    f.write('')
        
        # update
        self.check()
        
        return self
    
    def remove_state(self, childtasks, **kwargs):
        """
        set state of calculate task, and create a corresponding flag file with blank.
        
        Arguments:
            childtasks: sub-tasks needed to change the state.
            paths: paths of corresponding sub-task.
        """
        paths=None
        if 'paths' in kwargs:
            paths=kwargs['paths']
            if len(paths) != len(childtasks):
                raise ValueError("missed some paths in 'paths'")
        else:
            paths=[s0.path for s0 in childtasks]
            
        for i in range(0, len(childtasks)):
            childtask=childtasks[i]
            path=paths[i]
            if childtask in self.childTasks:
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

    
class PhononTaskBuilder(CalculateTaskBuilder):
    
    def __init__(self, name, path_of_parent, structure_of_parent, **kwargs):
        """
        build a phonon task.
        
        Arguments:
            name: name of phonon task.
            path_of_parent: parent path of phonon task.
            structure_of_parent: parent structure of phonon task.
            
            kwargs:
                isClear: if true, clear old data in phonon task's dictionary.
        """
#        super(PhononTaskBuilder, self).__init__(name, None, None, **kwargs)
        self.task=PhononTask(name=name, 
                                   path_of_parent=path_of_parent, 
                                   structure_of_parent=structure_of_parent)
        self.task._builder=self
        
#        self.calculateTask=self.phononTask
        
        self.isClear=False
        if 'isClear' in kwargs:
            self.isClear=kwargs['isClear']                
            
        if os.path.exists(path_of_parent):
            if self.isClear:
                shutil.rmtree(path_of_parent)
                os.makedirs(path_of_parent)
        else:
            os.makedirs(path_of_parent)
    
    
    # flag file
    def set_state_of_task(self, state_of_calculation, childtasks, **kwargs):
        """
        set state of calculate task, and create a corresponding flag file with blank.
        
        Arguments:
            state_of_calculation: state of calculation (Enum-type).
            childtasks: sub-tasks needed to change the state.
            paths: paths of corresponding sub-task.
        """
        self.task.set_state(state_of_calculation=state_of_calculation, childtasks=childtasks, **kwargs)        
        return self
    
    def remove_state_of_task(self, childtasks, **kwargs):
        """
        set state of calculate task, and create a corresponding flag file with blank.
        
        Arguments:
            childtasks: sub-tasks needed to change the state.
            paths: paths of corresponding sub-task.
        """
        self.task.remove_state(childtasks=childtasks, **kwargs)
        return self
        
    def generate_supercell(self, supercell, **kwargs):
        """
        call phonopy to generate POSCAR files needed for the phonon calculation.
        
        Arguments:
            supercell: size of super cell. i.e. [2,2,2]
            
            kwargs:
                tolerance (default=1e-6): tolerance for symmetry analysis.
                
        Return:
            builder's ojbect.
        """
        import os
        
        supercell="'"+ ' '.join(str(s0) for s0 in supercell)+"'"
        
        tolerance=1e-6
        if 'tolerance' in kwargs:
            tolerance=kwargs['tolerance']
        
        self.set_POSCAR(path=self.task.path_of_parent, structure=self.task.structure_of_parent)
#        os.system('cp {} {}'.format('/home/fu/workspace/eclipse/evolution4Fu/examples/SiAlLi/POSCAR', self.task.path_of_parent))
        
        path0=os.getcwd()
        os.chdir(self.task.path_of_parent)
        os.system('phonopy -d --dim={} --tolerance={}'.format(supercell, tolerance))
        os.chdir(path0)
        
        return self
    
# =============================================================================
#     def set_POSCAR_for_Phonon(self, infile, path):
#         """
#         generate POSCAR file for phonon calcualtion (phonopy).
#         
#         Arguments:
#             infile: infile generated by phonopy. i.e. xxx/POSCAR-001
#             path: saved path. i.e. xxx/001
#             
#         Return:
#             builder's ojbect.
#         """
#         os.system("sed '6i{}' {} > {}".format(os.popen('head -1 {}'.format(infile)).readline(), infile, path+'/POSCAR'))
#         return self
# =============================================================================
    
    def prepare(self, **kwargs):
        """
        generate childtasks according to the supercell structures from Phononpy.
        
        Arguments:
            
            kwargs:
                parameters_for_INCAR: customized parameters for INCAR (dictionary-type). i.e. {'EDIFF':1e-6, 'EDIFFG':-1e-4, ...}
            
            
        Return:
            builder's ojbect.
        """
        from iostream.read import Read
        from materials.structure import Structure
        
        import warnings
        warnings.warn('need to improve the code')
        
        path_of_parent=self.task.path_of_parent
        raw=Read(path='{}/POSCAR'.format(path_of_parent), dtype='poscar').run()
        elements=raw['elements']
        
        dirs=[s0.split('-')[1] for s0 in os.listdir(path_of_parent) if os.path.join(path_of_parent, s0) and s0.startswith('POSCAR-')]
        dirs=sorted(dirs, key=lambda x:int(x))
        
        for d in dirs:
            path0=os.path.join(path_of_parent, d)     
#            paths.append(path0)

            raw=Read(path='{}/POSCAR-{}'.format(path_of_parent, d), dtype='poscar', isContainedElement=False).run()
            raw['elements']=elements
            structure0=Structure().create(raw_structure=raw)
            
#            structures.append(structure0)
            
            if os.path.exists(path0):
                if self.isClear:
                    shutil.rmtree(path0)
                    os.makedirs(path0)
            else:
                os.makedirs(path0)
                
            # mehtod 1 cann't copy POSCAR file from parent dirctionary.
            builder0=CalculateTaskBuilder(name=self.task.name, 
                                          path=path0, 
                                          structure=structure0, 
                                          isClear=self.isClear)
            builder0.set_INCAR(type='scf') if not('parameters_for_INCAR' in kwargs) else builder0.set_INCAR(type='scf', parameters=kwargs['parameters_for_INCAR'])
            builder0.set_POSCAR()
            builder0.set_POTCAR()
            builder0.set_sumbitScript()
            builder0.set_state_of_task(state_of_calculation=state_of_calculation.prepare)
            builder0.set_parentTask(self.task)
#            self.task.childTasks.append(builder0.get_result())
            self.task.add_childTask(builder0.get_result())
            
# =============================================================================
#             # method 2
#             self.set_INCAR(path=path0)
#             self.set_KPOINTS(path=path0)
#             self.set_POSCAR(path=path0)
#             self.set_POTCAR(path=path0)
# =============================================================================

        # update path, structure, INCAR, POSCAR, POTCAR, submit, KPOINTS and OPTCELL of phonon task
        if self.task.childTasks != []:
            self.task._path=[s0.path for s0 in self.task.childTasks]
            self.task._structure=[s0.structure for s0 in self.task.childTasks]
        
            self.task._INCAR=[s0.INCAR for s0 in self.task.childTasks]
            self.task._POSCAR=[s0.POSCAR for s0 in self.task.childTasks]
            self.task._POTCAR=[s0.POTCAR for s0 in self.task.childTasks]
            self.task._submitScript=[s0.submitScript for s0 in self.task.childTasks]
            self.task._KPOINTS=[s0.KPOINTS for s0 in self.task.childTasks]
            self.task._OPTCELL=[s0.OPTCELL for s0 in self.task.childTasks]
            
            self.task.set_childTasks(self.task.childTasks)
            
        return self

