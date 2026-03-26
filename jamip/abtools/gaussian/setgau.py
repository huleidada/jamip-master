# 1. write qe.in % 

import os 
import logging
import numpy as np
from typing import Union, Tuple, Dict
from jamip.utils.logger import full_path

class SetGaussian:

    soft = 'gaussian'

    def __init__(self):
        super().__init__()
        self.__program = None 
        self.__potential = None
        # tasks
        self._tasks_ = None 
        self._overwrite_ = []
        # incar 
        self._energy_ = 1E-5
        self._force_ = None
        self._nbands_ = 1.2
        self._basis_ = "B3LYP/6-31G(d)" 
        self.label = "s1 tddft task"
        self.charge = 0
        self.mspin  = 1
        self.print_level = 'P'
   	
 
    @property 
    def program(self):
        return self.__program 

    @program.setter
    def program(self, value:str):
        """
        """
        if value in ['g16','g09']:
            self.__program = value
        else:
            raise IOError ('Invalid input gau.program')

    @property
    def energy(self):
        return self._energy_

    @energy.setter
    def energy(self, value:float):
        """
        """
        self._energy_ = value

    @property
    def force(self):
        return self._force_

    @force.setter
    def force(self, value:float):
        """
        """
        self._force_ = value

    @property
    def nbands(self):
        return self._nbands_

    @nbands.setter
    def nbands(self, value:float):
        """
        """
        self._nbands_ = value
   	
    @property
    def xc_func(self):
        return self._basis_
    
    @property
    def basis(self):
        return self._basis_

    @basis.setter
    def basis(self, value:str):
        """
        Required parameters, GGA & other group params
        gau.basis = '6-31G'
        """
        self._basis_ = value 

    @property
    def tasks(self):
        return self._tasks_
	
    @tasks.setter
    def tasks(self,value:str):
        """
        Required parameters, Task list 
        gau.tasks = 'opt sp'
        """
        from collections import OrderedDict
        
        for task in self.base_tasks:
            if task in value.lower():
                self._tasks_ = OrderedDict({task: {'task': value}})
                break
        else:
            self._tasks_ = OrderedDict({'sp': {'task': value}})

    def get_all_tasks(self):
        return self._tasks_

    @property
    def overwrite(self):
        return self._overwrite_

    @overwrite.setter
    def overwrite(self,value:str):
        tasks = []
        for task in value.split():
            if task in self.base_tasks:
                tasks.append(task)
        self._overwrite_ = list(set(tasks))
        
    @property
    def links(self):
        links_dict = {"frec": ['sp']}
        return links_dict

    @property
    def base_tasks(self):
        tasks_list = ["opt", 'sp', 'freq','irc','ircmax','scan','polar','admp','bomd',
                      'eet','force','stable','volume','scrf']
        return tasks_list
    
    def get_mspin(self):
        if self.mspin == 1:
            nelectrons = np.sum(self.structure.get_elements()) + self.charge
            if nelectrons % 2 == 1:  
                return 2
        return self.mspin
