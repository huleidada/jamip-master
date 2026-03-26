# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod

class Check:
    """
    check the result od task. If finished with error, return error massage. 
    """
    __metaclass__=ABCMeta
    
    def __init__(self, task):
        """
        Arguments:
            task: task's object.
        """
        self.task=task
        
    @abstractmethod
    def run(self):
        raise NotImplementedError
