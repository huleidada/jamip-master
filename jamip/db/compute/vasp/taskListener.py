# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from collections import deque


class TaskListener:
    def __init__(self):
        """
        Arguments:
            task: task's ojbect.
        """
        self._tasks=deque()
    
    @property
    def tasks(self):
        """
        
        Return:
            tasks
        """
        return self._tasks
    
    def notify(self, task, event):
        """
        handle the notification information.
        
        Arguments:
            task: task's object.
            event: notification information.
        """
        from src.compute.vasp.calculateException import CalculateException
        from src.compute.vasp.check4vasp import Check4VASP
        from src.compute.vasp.calculateTask import CalculateTaskBuilder
        
        try:
            if 'finished' in event:
                Check4VASP(task=task).run()
        except CalculateException as ce:
            ce.sovle(task=task)
#            pass
#            print('Calculate Exception')
        