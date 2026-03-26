# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import deque
from src.utils.variables import state_of_calculation

class Pool:
    def __init__(self):
        """
        pool of jobs (listener@observer pattern).
        """
        self._jobs=deque()
        self._jobs4run=deque()
        
    def __str__(self):
        for task in self.jobs:
            if hasattr(self, 'state'):
                print('{0} | {1} | {2}'.format(self.path, self.name, self.state.value))
            else:
                print('{0} | {1}'.format(task.path, task.name))
                    
    @property
    def jobs(self):
        return self._jobs
        
    def append_job(self, job, left_or_right='right'):
        """
        append a job to the left side.
        
        Arguments:
            job:
            left_or_right (default='right'): if 'left', first-in-last-out (FILO); if 'right', first-in-first-out (FIFO).
            
        Return:
            job's object.
            """
        if left_or_right.lower().startswith('l'):
            self.jobs.appendleft(job)
            self._jobs4run.appendleft(job)
        elif left_or_right.lower().startswith('r'):
            self.jobs.append(job)
            self._jobs4run.append(job)
        else:
            raise ValueError('unknown left_or_right')
        job._pool=self
        
        return self
    
# =============================================================================
#     def notify(self, task, event):
#         """
#         handle the notification information.
#         
#         Arguments:
#             task: task's object.
#             event: notification information.
#         """
#         from src.compute.vasp.calculateException import CalculateException
#         from src.compute.vasp.check4vasp import Check4VASP
#         from src.compute.vasp.calculateTask import CalculateTaskBuilder
#         
#         try:
#             if 'finished' in event:
#                 Check4VASP(task=task).run()
#         except CalculateException as ce:
#             ce.sovle(pool=self, task=task)
# =============================================================================
            
            
            
    def state_of_all_jobs(self):
        """
        list the state of jobs in the pool.
        
        Return:
            list of states.
        """
        states=[job.check() for job in self.jobs]
        return states
    
    def get_jobs(self, state_of_calculation):
        """
        list filtered jobs by state of calculation.
        
        Arguemnts:
            state_of_calculation: state of calculation (Enum-type).
        """
        jobs=None
        if state_of_calculation == state_of_calculation.prepare:
            jobs=[job for job in self.jobs if state_of_calculation.prepare in job.check()]
        elif state_of_calculation == state_of_calculation.calculating:
            jobs=[job for job in self.jobs if state_of_calculation.calculating in job.check()]
        elif state_of_calculation == state_of_calculation.finished:
            jobs=[job for job in self.jobs if state_of_calculation.finished in job.check()]
        elif state_of_calculation == state_of_calculation.error:
            jobs=[job for job in self.jobs if state_of_calculation.error in job.check()]
        elif state_of_calculation == state_of_calculation.unknown:
            jobs=[job for job in self.jobs if state_of_calculation.unknown in job.check()]
        else:
            raise ValueError('unknown state_of_calculation')
        
        return jobs
    
    def get_tasks(self, state_of_calculation):
        """
        list filtered tasks by state of calculation.
        
        Arguemnts:
            state_of_calculation: state of calculation (Enum-type).
        """
        tasks=None
        
# =============================================================================
#         if state_of_calculation == state_of_calculation.prepare:
#             tasks=[task for task in self.tasks if task.check() == state_of_calculation.prepare]
#         elif state_of_calculation == state_of_calculation.calculating:
#             tasks=[task for task in self.tasks if task.check() == state_of_calculation.calculating]
#         elif state_of_calculation == state_of_calculation.finished:
#             tasks=[task for task in self.tasks if task.check() == state_of_calculation.finished]
#         elif state_of_calculation == state_of_calculation.error:
#             tasks=[task for task in self.tasks if task.check() == state_of_calculation.error]
#         elif state_of_calculation == state_of_calculation.unknown:
#             tasks=[task for task in self.tasks if task.check() == state_of_calculation.unknown]
#         else:
#             raise ValueError('unknown state_of_calculation')
# =============================================================================
        
        raise NotImplemented
        
        return tasks
    
# =============================================================================
#     def get_prepare_tasks(self):
#         """
#         list the prepare tasks in the pool.
#         """
#         prepare=[task for task in self.tasks if task.check() == state_of_calculation.prepare]
#         return prepare
#     
#     def get_calculating_tasks(self):
#         """
#         list the calculating tasks in the pool.
#         """
#         calculating=[task for task in self.tasks if task.check() == state_of_calculation.calculating]
#         return calculating
#     
#     def get_finished_tasks(self):
#         """
#         list the finished tasks in the pool.
#         """
#         finished=[task for task in self.tasks if task.check() == state_of_calculation.finished]
#         return finished
#     
#     def get_error_tasks(self):
#         """
#         list the error tasks in the pool.
#         """
#         error=[task for task in self.tasks if task.check() == state_of_calculation.error]
#         return error
#     
#     def get_unknown_tasks(self):
#         """
#         list the unknown tasks in the pool.
#         """
#         unknown=[task for task in self.tasks if task.check() == state_of_calculation.unknown]
#         return unknown
# =============================================================================
    

            
            