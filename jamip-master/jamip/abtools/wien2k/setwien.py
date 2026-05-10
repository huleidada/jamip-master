# 1. write qe.in % 

import os 
import logging
import numpy as np
from typing import Union, Tuple, Dict
from jamip.utils.logger import full_path
from ..base.kpoints import Kpath, Kpoints 

class SetQE:

    soft = 'qe'

    def __init__(self):
        super().__init__()
        self.__program = None 
        self.__files = []
        # potential
        self.__potential = None
        self.potential_map = {}
        self.potxc = 'pbe'
        self.__magmom = True
        self.magmom_map = {}
        # tasks
        self._tasks_ = None 
        self._xc_ = []
        self._overwrite_ = []
        # incar 
        self._energy_ = 1E-5
        self._force_ = None
        self._cutwfc_ = 1.3
        self._cutrho_ = 1.3
        self._nbands_ = 1.2
        self.kpoints =  0.189
        self.kpath = Kpath()
        self._optcell = None
   	
    @property
    def potential(self):
        return self.__potential 

    @potential.setter
    def potential(self, value:Union[str,tuple]):
        """
        Required parameters
        > vasp.potential = "/public/apps/vasp/potential"
        > vasp.potential = "/public/apps/vasp/potential", {"pb": "Pb_d"}
        """
        if isinstance(value, str):
            self.__potential = full_path(value)
        elif len(value) == 2:
            self.__potential = full_path(value[0])
            self.potential_map = dict(value[1])
        else:
            logging.error("Invalid input: qe.potential")
 
    @property 
    def program(self):
        return self.__program 

    @program.setter
    def program(self, value:str):
        """

        """
        if os.path.isdir(value):
            self.__program = full_path(value)
        else:
            raise IOError ('Invalid input qe.program')

    @property
    def external_files(self):
        return self.__files

    @external_files.setter
    def external_files(self, value:Union[str,tuple]):
        """
        Optional parameters
        > vasp.external_files = "/public/apps/vasp/vdw.dat"
        > vasp.external_files = "/public/apps/vasp/vdw.dat", "/public/apps/vasp/OPTCELL"
        """
        if isinstance(value, str):
            self.__file = [full_path(value)]
        elif isinstance(value, tuple):
            self.__file = [full_path(path) for path in value]
        else:
            logging.error("Invalid input: qe.external_files")

    @property
    def energy(self):
        return self._energy_

    @energy.setter
    def energy(self, value:float):
        """
        Optional parameters, etot_conv_thr 
        qe.energy = 1e-6  ->   etot_conv_thr = 1E-6
        """
        self._energy_ = value

    @property
    def force(self):
        return self._force_

    @force.setter
    def force(self, value:float):
        """
        Optional parameters, forc_conv_thr
        qe.force = 1e-2  ->   forc_conv_thr = -0.01
        """
        self._force_ = value

    @property
    def ecutwfc(self):
        return self._cutwfc_

    @ecutwfc.setter
    def ecutwfc(self, value:float):
        """
        Optional parameters, 
        qe.ecutwfc = 1.3  ->   ecutwfc = ENMAX * 1.3
        qe.ecutwfc = 45   ->    ecutwfc = 45
        """
        self._cutwfc_ = value

    @property
    def ecutrho(self):
        return self._cutrho_

    @ecutrho.setter
    def ecutrho(self, value:float):
        """
        Optional parameters, 
        qe.ecutrho = 1.3  ->   ecutrho = ENMAX * 1.3
        qe.ecutrho = 360  ->   ecutrho = 360
        """
        self._cutrho_ = value

    @property
    def nbands(self):
        return self._nbands_

    @nbands.setter
    def nbands(self, value:float):
        """
        Optional parameters, NBANDS
        vasp.nbands = 1.3  ->   NBANDS = NBANDS* 1.3
        vasp.nbands = 300  ->   NBANDS = 300
        """
        self._nbands_ = value
   	
    @property
    def xc_func(self):
        return self._xc_

    @xc_func.setter
    def xc_func(self,value:str):
        """
        Required parameters, GGA & other group params 
        vasp.xc_func = 'soc'
        """
        import re

        xc_list = re.findall('[a-z]+',value.lower())
        self._xc_ = set()

        for xc in xc_list:
            if xc in ['pbe', 'pbesol','91','pe','rp','ps','am','pz','soc','gw','hse']:
                if xc == 'pbe': xc = 'pe'
                elif xc == 'pbesol': xc = 'ps'
                self._xc_.add(xc)

    @property
    def kpoints(self):
        return self._kpoints_

    @kpoints.setter
    def kpoints(self, *args):
        '''
        '''
        if isinstance(args[0], tuple): 
            args = args[0]
        self._kpoints_ = Kpoints(*args)

    @property
    def tasks(self):
        return self._tasks_
	
    @tasks.setter
    def tasks(self,value:str):
        """
        Required parameters, Task list 
        vasp.tasks = 'relax scf'
        """
        from jamip.abtools.diyflow import get_diy_modules
        from collections import OrderedDict
      
        incar_dict = OrderedDict()
        diy_tasks = get_diy_modules()
        relax = set()

        for task in value.split():

            # relax %
            if task in ['ions', 'shape', 'volume']:
                relax.add(task)
            elif task in ['relax','vc-relax']:
                relax.update({'ions', 'shape', 'volume'})

            # base tasks %
            elif task in self.base_tasks or task in diy_tasks:
                incar_dict[task] = {}

        if len(relax):
            if relax == {'ions'}: 
                incar_dict['relax'] = {'calculation':'relax'} 
            elif relax == {'ions','shape','volume'}: 
                incar_dict['relax'] = {'calculation':'vc-relax'}
            incar_dict.move_to_end('relax', last=False)

        self._tasks_ = incar_dict

    @property
    def overwrite(self):
        return self._overwrite_

    @overwrite.setter
    def overwrite(self,value:str):
        from jamip.abtools.diyflow import get_diy_modules
        diy_tasks = get_diy_modules()
        tasks = []
        for task in value.split():
            if task in self.base_tasks or task in diy_tasks:
                tasks.append(task)
        self._overwrite_ = list(set(tasks))
        
    @property
    def links(self):
        links_dict = {"dos": ['scf'], "band": ['scf'], "projwfc": ['scf'],
                      "phdos": ['scf'], "phband": ['scf'],
                      "elastic": ['scf'], "poisson": ['scf'], "bader": ['scf'],
                      "cohp": ['scf'], "boltztrap": ['scf']}
        return links_dict

    @property
    def base_tasks(self):
        tasks_list = ["relax","md","scf","band","dos","projwfc","optics"
                      "phonon","phband","phdos","softmode","gruneisen",
                      "elastic","poisson"]
        return tasks_list

    def get_all_tasks(self):
        from jamip.abtools.diyflow import get_requires

        all_tasks = []
        for task in self.tasks:
            if task not in all_tasks:
                all_tasks.append(task)
        for task in get_requires(self.tasks):
            if task not in all_tasks:
                all_tasks.append(task)
        for xc in self.xc_func:
            if xc == 'pe': continue
            if xc not in all_tasks:
                all_tasks.append(xc)
        return all_tasks

    def get_potential(self, incar):
        """
        Set the POTCAR according to the species of elements in structure.
        
        return dict{'ecutwfc':wfc, 'ecutrho':rho}
        """	
         
        from os.path import join, exists, abspath 
        import re

        elements = list(incar.structure.species_of_elements)
        potential_lib = self.get_pseudo_licrary(self.potential)
        files = [None]*len(elements)

        for i,elm in enumerate(elements):
            if elm in self.potential_map and self.potential_map[elm] in potential_lib[elm]:
                files[i] = self.potential_map[elm]

            # set potential by auto %
            elif elm in potential_lib:
                pots = potential_lib[elm]
                files[i] = pots[np.argmin([len(pot) for pot in pots])]

        if None in files:
            raise KeyError("No useable potential in the pseudopotential library !")

        # read potentials %
        wfcs = []
        rhos = []
        for elm,filename in zip(elements,files):

            # search wfc&rho
            wfc = rho = None
            with open(join(self.potential, filename),'r') as f:
                for line in f:
                    if 'wavefunctions' in line:
                        wfc = re.findall(r':\s*(\d+\.?\d*)\s+Ry',line)[0]
                    elif 'charge density' in line:
                        rho = re.findall(r':\s+(\d+\.?\d*)\s+Ry',line)[0]
                    if wfc and rho:
                        break    
            wfcs.append(wfc)
            rhos.append(rho)
                    
        # update incar %
        incar['pseudo_dir'] =  self.potential
        wfc = np.array(wfc,dtype=float)
        rho = np.array(rho,dtype=float)
        if 'ecutwfc' not in incar:
            if self.ecutwfc > wfc.min():
                incar['ecutwfc'] = self.ecutwfc
            else:
                incar['ecutwfc'] = min(self.ecutwfc * wfc.max(), 150)
        if 'ecutrho' not in incar:
            if self.ecutrho == None:
                incar['ecutrho'] = incar['ecutwfc'] * 8
            elif self.ecutrho > rho.min():
                incar['ecutrho'] = self.ecutrho
            else:
                incar['ecutrho'] = min(self.ecutrho * rho.max(), 1200)

        return files

    def get_pseudo_licrary(self, path:str):

        from os.path import join, exists
        from collections import defaultdict
        import re

        # self.pseudo = 'rrkjus'
        # self.potxc = 'pbe'
        description = '(rel)?-?(starNl|starhNl)?-?(pz|vwn|pbe|blyp|pw91|tpss|column)-?([spdfnl]*)-(ae|mt|bhs|vbc|van|rrkj|rrkjus|kjpaw|bpaw)[^a-zA-Z0-9]'
        pseudo_lib = defaultdict(list)

        for filename in os.listdir(path):
            tmp = filename.split('.')
            if tmp[-1] != 'UPF': continue
            element = tmp[0]
            _,_,xc,state,pseudo = re.findall(description,filename)[0]
            if xc == self.potxc and pseudo == self.pseudo:
                pseudo_lib[element].append(filename)
 
        return pseudo_lib

