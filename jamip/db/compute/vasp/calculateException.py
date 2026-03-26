# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from utils.variables import state_of_calculation
from compute.task import Task, TaskBuilder
from compute.vasp.calculateTask import CalculateTaskBuilder

class CalculateException(Exception):
    
    def __init__(self):
        pass
        
    def sovle(self, task):
#        print('>>>>>>>>>> try to fix exception in calculation <<<<<<<<<<')
        builder1=CalculateTaskBuilder(name='{}-fix'.format(task.name), 
                                      path='{}-fix'.format(task.path),
                                      structure='{}-fix'.format(task.structure),
                                      isClear=True)
        builder1.set_INCAR(type='scf')
        builder1.set_POSCAR()
        builder1.set_POTCAR()
        builder1.set_KPOINTS()
        builder1.set_sumbitScript()
        builder1.set_state_of_task(state_of_calculation=state_of_calculation.finished)
# =============================================================================
#         builder1=TaskBuilder(name='{}-fix'.format(task.name),                                                                                                             
#                      path='{}-fix'.format(task.path),
#                      isClear=True)
# =============================================================================
        
        task1=builder1.get_result()
        if task.parentTask is None:
            task.job.append_task(task1, left_or_right='l')
            # update linke relationship
            builder1.set_doublyLink(prev_task=task)
        else:
            task.parentTask.add_childTask(task1)
        task1.register(listener=task.listener)
#        task1.run()
        
#        
#        print('!!!!', task.prev, task.next)
#        builder1.set_doublyLink(prev_task=task)
#        print('????', task1.prev, task1.next)
# =============================================================================
#         print('<< 2 <', task.name, task.prev, ' < ? > ',task.next, '>>>')
#         print('<< 3 <', task1.name, task1.prev, ' < ? > ',task1.next, '>>>')
# =============================================================================
