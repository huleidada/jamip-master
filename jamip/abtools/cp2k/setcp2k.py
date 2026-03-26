from __future__ import annotations

import os 
import logging
import numpy as np
from jamip.utils.logger import full_path
from ..base.kpoints import Kpath, Kpoints 
import pathlib

class SetCP2K:

    soft = 'cp2k'

    def __init__(self):
        super().__init__()
        self.__program = None 
        self._data_dir = None
        self.__files = []
        # potential
        self.potential = 'GTH-PBE'
        self.basis_set = 'DZVP-MOLOPT-SR-GTH' 
        self.potential_map = {}
        self.basis_set_map = {}
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
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value:str):
        """
        Required parameters
        > cp2k.data_dir = "/public/apps/vasp/potential"
        """
        self._data_dir = full_path(value)
 
    @property 
    def program(self):
        return self.__program 

    @program.setter
    def program(self, value:str):
        """
        cp2k.program='/public/apps/cp2k-2024.1/exe/Linux-intel-x86_64-minimal'
        """
        if os.path.isdir(value):
            self.__program = full_path(value)
        else:
            raise IOError ('Invalid input cp2k.program')

    @property
    def external_files(self):
        return self.__files

    @external_files.setter
    def external_files(self, value: str | tuple):
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
            logging.error("Invalid input: cp2k.external_files")

    @property
    def energy(self):
        return self._energy_

    @energy.setter
    def energy(self, value:float):
        """
        Optional parameters, etot_conv_thr 
        cp2k.energy = 1e-6  ->   etot_conv_thr = 1E-6
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
        #self._xc_ = value

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
        import re

        elements = list(incar.structure.species_of_elements)
        basis_set = {key:None for key in elements}
        potential = {key:None for key in elements}

        # set potential by auto %
        for elm in elements:
            if elm in self.basis_set_map:
                basis_set[elm] = self.basis_set_map[elm][0]
            if elm in self.potential_map:
                potential[elm] = self.potential_map[elm][0]

        if None in basis_set.values():
            raise KeyError("No useable potential in the pseudopotential library !")
        if None in potential.values():
            raise KeyError("No useable potential in the pseudopotential library !")

        return potential

    def set_basis_set(self, 
                      basis_set_file_name:str, 
                      potential_file_name:str,
                      **kwargs):

        from collections import defaultdict
        import re

        path = self.data_dir
        if path is None:
            path = os.environ['CP2K_DATA_DIR']
        path = pathlib.Path(path)

        maps = defaultdict(list)
        with open(path/basis_set_file_name, 'r') as f:
            for line in f:
                if line[0] == '#': continue
                if self.basis_set in line:
                    if len(line.split()) != 3: continue
                    specie, bs, potential = line.split()
                    maps[specie].append(potential)
        self.basis_set_map = maps
 
        maps = defaultdict(list)
        with open(path/potential_file_name, 'r') as f:
            for line in f:
                if line[0] == '#': continue
                if self.potential in line:
                    if len(line.split()) != 3: continue
                    specie, ps, potential = line.split()
                    maps[specie].append(potential)
        self.potential_map = maps
