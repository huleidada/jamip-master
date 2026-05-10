# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import deque 
from utils.variables import state_of_calculation

class Job:
    id=0
    
    def __init__(self, name):
        """
        a job contain a set of tasks.
        
        Arguments:
            name: name of job.
        """
        self._name=name
        self._tasks=deque()
        self._tasks4run=deque()
        
        self._listener=None
        
        self._pool=None
        
        Job.id += 1
        self.id=Job.id
    
    @property
    def name(self):
        return self._name
    
    @property
    def tasks(self):
        return self._tasks
        
    def append_task(self, task, left_or_right='right'):
        """
        append a task to the left side.
        
        Arguments:
            task:
            left_or_right (default='right'): if 'left', first-in-last-out (FILO); if 'right', first-in-first-out (FIFO).
            
        Return:
            job's object.
        """
        if left_or_right.lower().startswith('l'):
            self.tasks.appendleft(task)
            self._tasks4run.appendleft(task)
        elif left_or_right.lower().startswith('r'):
            self.tasks.append(task)
            self._tasks4run.append(task)
        else:
            raise ValueError('unknown left_or_right')
        task._job=self
        
        return self
    
    @property
    def listener(self):
        return self._listener
        
    # observer pattern
    def register_all(self, listener):
        """
        register all tasks of the job to pool.
        
        Arguments:
            listener: listener's object.
        
        Return:
            job's object.
        """
        for task in self.tasks:
            if not task in listener.tasks:
#                listener.tasks.appendleft(task)
                task.register(listener=listener)
                task._listener=listener
            else:
                import warnings
                warnings.warn('exist task in self.tasks')
        return self
    
    def register(self, listener, task):
        """
        register all tasks of the job to pool.
        
        Arguments:
            listener: listener's object.
            task: a task's object in the job.
            
        Return:
            job's object.
        """
        import warnings
        # check
        if not task in self.tasks:
            warnings.warn('not exist in self.tasks')
        if not task in self._listener.tasks:
            warnings.warn('not exist in self._listener.tasks')
        
#        listener.tasks.appendleft(task)
        task.register(listener=listener)
        task._listener=listener
        
        return self
    
    def deregister_all(self):
        """
        deregister all tasks of the job from their listener.
        
        Return:
            job's object.
        """
        for task in self.tasks:    
#            self._listener.tasks.remove(task)
            task.deregister()
        self._listener=None
        return self
        
    def deregister(self, task):
        """
        deregister a task of the job from its listener.
        
        Arguments:
            task: a task's object in the job.
        
        Return:
            job's object.
        """
        import warnings
        
        # check
        if not task in self.tasks:
            warnings.warn('not exist in self.tasks')
        if not task in self._listener.tasks:
            warnings.warn('not exist in self._listener.tasks')

#        self._listener.tasks.remove(task)
        task.deregister()
            
        return self
    
    @property
    def pool(self):
        """
        pool
        |--job
        
        Return:
            pool's object.
        """
        return self._pool
    
    def check(self):
        """
        check the state of calculation
        
        Return:
            list of states.
        """
        states=[task.check() for task in self.tasks]
        return states
    
    def run(self):
        """
        run thsi jobs.
        """
# =============================================================================
#         # static method: onlu iterator the initialized list. cannot traverse the run-time task (dynamics).
#         for task in list(self.tasks):
#             print('\n\n+++++')
#             print('task', task.name, task.path)
#             print('\n\n+++++')
#             task.run()
# =============================================================================
            
        # dynamics method:
        while len(self._tasks4run) > 0:
            task=self._tasks4run.popleft()
            if task._childTasks4run is None:
                if task.state == state_of_calculation.prepare:    
                    task.run()
                    print('\n{}-{}-running\n'.format(task.name, task.path))
                elif task.state == state_of_calculation.finished:
                    print('\n{}-{}-finished\n'.format(task.name, task.path))
            else:
                task.run()   
