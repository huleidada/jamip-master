# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import shutil
import numpy as np
from abc import ABCMeta, abstractmethod


class Task:
    __metaclass__=ABCMeta
    
    id=0
    
    def __init__(self, name, path):
        """
        general task contain prepare task (i.e. structure operation, etc.), calculation task, posprocess task (data extraction).
        
        Arguments:
            name: name of task.
            path: store path of task data.
        """
        self._name=name
        self._path=path
        
        self._builder=None # taskBuilder
        
        self._job=None
        self._pool=None
        
        # doubly linked list
        self._prev=None
        self._next=None
#        self._peer=None
        
        self._listener=None
        
        self._parentTask=None # parent task's object
        self._childTasks=None # child tasks' object collection
        self._childTasks4run=None
        
        Task.id += 1
        self.id=Task.id
                
    def __str__(self):
        return '{0} | {1}'.format(self.path, self.name)
    
    @property
    def name(self):
        """
        Task name.
        """
        return self._name
    
    @property
    def path(self):
        """
        Task path.
        """
        return self._path
    
    @property
    def builder(self):
        """
        task builder.
        
        Return:
            taskBuilder's object.
        """
        return self._builder
    
    @property
    def job(self):
        """
        job
        |--task
        
        Return:
            job's object.
        """
        return self._job
    
# =============================================================================
#     @property
#     def pool(self):
#         """
#         task's pool.
#         
#         Return:
#             pool's object.
#         """
#         return self._pool
# =============================================================================
    
    @property
    def listener(self):
        """
        
        Return:
            listener's object.
        """
        return self._listener
    
    @property
    def isContainedChildtask(self):
        """
        whehter contain childtask.
        
        Return:
            boolean value.
        """
        result=None
        if isinstance(self.path, list):
            result=True
        elif isinstance(self.path, np.ndarray):
            result=True
        elif isinstance(self.path, str):
            result=False
        else:
            raise ValueError('unknown self.path')
            
        return result
    
    @property
    def parentTask(self):
        """
        parent task.
        
        Return
            parent task's object.
        """
        return self._parentTask
    
    def set_parentTask(self, task):
        """
        set parent task and update the link information.
        
        Arguments:
            task: parent task's object.
            
        Return:
            task's object.
        """
        self._parentTask=task
        if (task.childTasks is None) or (not self in task.childTasks):
            task.add_childTask(self)
        else:
            import warnings
            warnings.warn("this task has existed in childTasks")
            
        # update link information
        #check data
        self.set_prev(task.prev)
        self.set_next(task.next)
        
        return self
    
    @property
    def childTasks(self):
        """
        child tasks
        
        Return:
            child tasks' object collection.
        """
        return self._childTasks
    
    def set_childTasks(self, tasks):
        """
        set child tasks and update the link information.
        
        Arguments:
            tasks: child tasks' object collection (list-type).
            
        Return:
            task's object.
        """
        # check
        if isinstance(tasks, list):
            pass
        elif isinstance(tasks, np.ndarray):
            tasks=tasks.tolist()
        else:
            raise ValueError('unknown tasks')
            
#        self._childTasks=tasks
        for task in list(tasks):
            self.add_childTask(task)
        
        return self
    
    def add_childTask(self, task):
        """
        add a child task and update the link information.
        
        Arguments:
            task: child task's object.
            
        Return:
            task's object.
        """
        if self._childTasks is None:
            self._childTasks=[task]
            self._childTasks4run=[task]
        elif not task in self._childTasks:
            self._childTasks.append(task)
            self._childTasks4run.append(task)
        else:
            import warnings
            warnings.warn("this task has existed in childTasks")
        task.set_parentTask(self)
        
        # update link information
        task.set_prev(self.prev)
        task.set_next(self.next)
        
        
        return self
    
    def remove_childTask(self, task):
        """
        remove a child task.
        
        Arguments:
            task: child task's object.
            
        Return:
            task's object.
        """
        import warnings
        
        if self._childTasks is None:
            warnings.warn('childTasks is None')
        elif not task in self._childTasks:
            warnings.warn('not exist in childTasks')
        else:
            self._childTasks.remove(task)
            if task in self._childTasks4run:
                self._childTasks4run.remove(task)
        task.set_parentTask(None)
        
        # update link information
        task.set_prev(None)
        task.set_next(None)
        
        return self
    
    @property
    def prev(self):
        """
        previous task.
        
        Return:
            previous task's object.
        """
        # check consistency
        if not self._parentTask is None:
            if self._prev != self.parentTask.prev:
                self.set_prev(self.parentTask.prev)
        return self._prev
        
    def set_prev(self, task):
        """
        set previous task.
        
        Arguments:
            task: previous task
        
        Return:
            task's object.
        """
        self._prev=task
        return self
    
    @property
    def next(self):
        """
        next task.
        
        Return:
            next task's object.
        """
        # check consistency
        if not self._parentTask is None:
            if self._next != self.parentTask.next:
                self.set_next(self.parentTask.next)
        return self._next
    
    def set_next(self, task):
        """
        set next task.
        
        Arguments:
            task: next task.
            
        Return:
            task's object.
        """
        self._next=task
        return self
    
    def remove_doublyLink(self):
        """
        remove from the doubly link.
        
        Return:
            task's ojbect.
        """
        if not self.prev is None:
            self.prev.set_next(self.next)
        if not self.next is None:
            self.next.set_prev(self.prev)
        self.set_prev(None)
        self.set_next(None)
        return self
    
    @abstractmethod
    def run(self):
        """
        run this task.
        """
#        raise NotImplementedError
        print('\n{}-running'.format(self.name))
        if not self.listener is None:
            self.notify_listener(event='finished')
        
        
    # observer pattern
    def register(self, listener):
        """
        register this task to pool.
        
        Arguments:
            listener: listener's object.
            
        Return:
            task's object.
        """
        listener.tasks.appendleft(self)
        self._listener=listener
        return self
        
    def deregister(self):
        """
        deregister a task from this listener.
        
        Return:
            task's object.
        """
        self._listener.tasks.remove(self)
        self._listener=None
        return self
    
    def notify_listener(self, event):
        """
        notify this listener.
        
        Arguments:
            event: notification information.
            
        Return:
            task's object.
        """
        self._listener.notify(self, event)
        return self
        
        
class TaskBuilder:
    
    def __init__(self, name, path, **kwargs):
        """
        build a task.
        
        Arguments:
            name: name of task.
            path: store path of task data.
            
        kwargs:
                isClear: if true, clear old data in calculate task's dictionary.
        """
        self.task=Task(name=name,
                       path=path)
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
            
    def get_result(self):
        """
        Return:
            return task's object.
        """
        return self.task
    
    def set_doublyLink(self, **kwargs):
        """
        set doubly link relationship.
        
        Arguments:
            kwargs:
                prev_task: previous task's object.
                next_task: next task's object.
#                peer_tasks: peer tasks's object array.
                
        Return:
            builder's object
        """
        if 'prev_task' in kwargs:
            task=kwargs['prev_task']
            self.task.set_prev(task)
            self.task.set_next(task.next)
            if not task.next is None:
                task.next.set_prev(self.task)
            task.set_next(self.task)                
            
        if 'next_task' in kwargs:
            task=kwargs['next_task']
            self.task.set_prev(task.prev)
            self.task.set_next(task)
            if not task.prev is None:
                task.prev.set_next(self.task)
            task.set_prev(self.task)
        
        # update childTask
        tasks=[self.task, self.task.prev, self.task.next]
        for task in tasks:
            if hasattr(task, 'childTasks') and not task.childTasks is None:
                for childTask in task.childTasks:
                    childTask.set_prev(task.prev)
                    childTask.set_next(task.next)
            
        return self
    
    def remove_doublyLink(self):
        """
        remove doubly link relationship.
        
        Return:
            builder's object
        """
        prev_task=self.task.prev
        next_task=self.task.next
        
        # remove current node
        self.task.set_prev(None)
        self.task.set_next(None)
        
        # update
        prev_task.set_next(next_task)
        next_task.set_prev(prev_task)
        
        if not self.task.childTasks is None:
            for task in self.task.childTasks:
                task.set_prev(None)
                task.set_next(None)
        
        return self
    
    def set_parentTask(self, task):
        """
        set parent task.
        
        Arguments:
            task: parent task's object.
            
        Return:
            builder's object
        """
        self.task.set_parentTask(task=task)
        return self
    
    def set_childTasks(self, tasks):
        """
        set child tasks.
        
        Arguemnts:
            tasks: child tasks' object collection.
            
        Return:
            builder's object
        """
        self.task.set_childTasks(tasks=tasks)
        return self

