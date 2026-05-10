# 1. write poscar % 
# 2. write potcar % 
# 3. write kpoints % 
# 4. write incar % 
# 5. write vdwkernel % 

import os 
import numpy as np
from collections import namedtuple
from typing import Union, Dict
from jamip.utils.logger import full_path
from ..base.kpoints import Kpath, Kpoints
from .vdw import VDW
import logging

Potential = namedtuple('Potential', ['formula','files', 'enmax', 'enmin', 'nelect'])

class SetVasp(VDW):
  
    soft = "vasp"

    def __init__(self):
        super().__init__()
        self.__program = None
        self.__files = []
        # potential
        self.__potential = None
        self.potential_map = {}
        self.__magmom = False
        self.magmom_map = {}
        self.__ldau = False
        self.ldau_map = {}
        self.soc_map = None
        self._optcell = None
        self.dimension = 3
        # incar 
        self._energy_ = 1E-5
        self._force_ = None
        self._cutoff_ = 1.3
        self._nbands_ = 1.2
        self._optics_nbands_ = None 
        self.kpoints =  0.189
        self.kpath = Kpath()
        self.kpoints_opt = False
        self.force_create_kpoints = False
        self.force_copy_poscar = False
        # tasks
        self._tasks_ = None
        self._xc_ = []
        self._accelerate_ = False
        self._overwrite_ = []
   	
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
            logging.error("Invalid input: vasp.potential")

    @property
    def magmom(self):
        return self.__magmom

    @magmom.setter
    def magmom(self, value:Union[Dict[str, int], bool]):
        """
        Optional parameters
        > vasp.magmom = {'Cs':0, 'Pb':4, 'I':0}
        > vasp.magmom = False   # set all magmom = 0
        """
        if isinstance(value, bool):
            self.__magmom = value
        else:
            self.__magmom = True
            self.magmom_map = value
    
    @property
    def ldau(self):
        return self.__ldau

    @ldau.setter
    def ldau(self, value:Union[Dict[str, dict], bool]):
        if isinstance(value, bool):
            self.__ldau = value
        else:
            self.__ldau = True
            self.ldau_map = value

    @property 
    def program(self):
        return self.__program 

    @program.setter
    def program(self, value:Union[str,dict]):
        """
        Required parameters
        > vasp.program = "/public/apps/vasp/bin/vasp_std"
        > vasp.program = {"std": "/public/apps/vasp/bin/vasp_std", 
                          "ncl": "/public/apps/vasp/bin/vasp_ncl"}
        """
        if isinstance(value, str):
            self.__program = full_path(value) 
        elif isinstance(value, dict):
            self.__program = {key:full_path(path) for key,path in value.items()}
        else:
            logging.error("Invalid input: vasp.program")

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
            logging.error("Invalid input: vasp.external_files")
       	
    @property
    def optcell(self):
        return self._optcell

    @optcell.setter
    def optcell(self,value):
        """
        Optional parameters, create file "OPTCELL"
        vasp.optcell = 110 or '110' or 1,1,0
        """
        import re

        result = re.findall('[01]',str(value))
        if len(result) == 3:
            self._optcell = '{0[0]:s}{0[1]:s}{0[2]:s}'.format(result) 
        else:
            logging.error("Invalid input: vasp.optcell")

    def get_atuo_groups(self, xc:str):
        # soc %
        if xc == 'soc':
            soc_map = self.soc_map
            if soc_map is None:
                soc_map = ['In','Sn','Sb','Te','I',
                           'W','Re','Os','Ir','Pt',
                           'Au','Hg','Tl','Pb','Bi']
            union = set(soc_map) & set(self.structure.species_of_elements)
            if len(union) > 0:
                return True
        elif xc == 'u':
            u_map = ['Sc', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 
                     'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
            union = set(u_map) & set(self.structure.species_of_elements)
            if len(union) > 0:
                return True

        return False

    @property
    def energy(self):
        return self._energy_

    @energy.setter
    def energy(self, value:float):
        """
        Optional parameters, EDIFF 
        vasp.energy = 1e-6  ->   EDIFF = 1E-6
        """
        self._energy_ = value

    @property
    def force(self):
        return self._force_

    @force.setter
    def force(self, value:float):
        """
        Optional parameters, EDIFFG
        vasp.force = 1e-2  ->   EDIFFG = -0.01
        """
        self._force_ = value

    @property
    def cutoff(self):
        return self._cutoff_

    @cutoff.setter
    def cutoff(self, value:float):
        """
        Optional parameters, ENCUT
        vasp.cutoff = 1.3  ->   ENCUT = ENMAX * 1.3
        vasp.cutoff = 520  ->   ENCUT = 520
        """
        self._cutoff_ = value

    @property
    def nbands(self):
        # electric or nbandsgw
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
    def optics_nbands(self):
        # optics
        return self._optics_nbands_ if self._optics_nbands_ else self._nbands_

    @optics_nbands.setter
    def optics_nbands(self, value:float):
        """
        Optional parameters, NBANDS
        vasp.nbands = 1.3  ->   NBANDS = NBANDS* 1.3
        vasp.nbands = 300  ->   NBANDS = 300
        """
        self._optics_nbands_ = value

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
    def xc_func(self):
        return self._xc_

    @xc_func.setter
    def xc_func(self,value:str):
        """
        Required parameters, GGA & other group params 
        vasp.xc_func = 'soc'
        """
        import re

        xc_list = re.findall('[a-z0-9]+',value.lower())
        self._xc_ = set()

        for xc in xc_list:
            if xc in ['pbe', 'pbesol','91','pe','rp','ps','am','pz','soc','hse','ldau']:
                if xc == 'pbe': xc = 'pe'
                elif xc == 'pbesol': xc = 'ps'
                self._xc_.add(xc)

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

    @property
    def tasks(self):
        return self._tasks_
	
    @tasks.setter
    def tasks(self,value:str):
        """
        Required parameters, Task list 
        vasp.tasks = 'relax scf'
        """
        from jamip.abtools.diyflow import get_diy_modules#,import_diy_module
        from collections import OrderedDict
      
        incar_dict = OrderedDict()
        diy_tasks = get_diy_modules()
        relax = set()

        for task in value.split():

            # relax %
            if task in ['ions', 'shape', 'volume']:
                relax.add(task)
            elif task == 'relax':
                relax.update({'ions', 'shape', 'volume'})
        
            # # md %
            # if task in ['nve', 'nvt', 'npt', 'mlmd']:
            #     incar_dict['md'] = {'mdtype': task}
            # elif task == 'nve':
            #     incar_dict['md'] = {'smass':-3}
            # elif task == 'nvt':
            #     incar_dict['md'] = {'smass':0, 'iwavpr':11}
            # elif task == 'npt':
            #     incar_dict['md'] = {'pmass':5, 'mdalgo':3, 'pstress':1, 
            #                         'langevin_gamma':"10 10", "langevin_gamma_l":1}

            # base tasks %
            elif task in self.base_tasks or task in diy_tasks:
                incar_dict[task] = {}

        if len(relax):
            if relax == {'ions'}: 
                incar_dict['relax'] = {'isif': 2}
            elif relax == {'ions','shape','volume'}: 
                incar_dict['relax'] = {'isif': 3}
            elif relax == {'ions','shape'}: 
                incar_dict['relax'] = {'isif': 4}
            elif relax == {'shape'}: 
                incar_dict['relax'] = {'isif': 5}
            elif relax == {'shape','volume'}: 
                incar_dict['relax'] = {'isif': 6}
            elif relax == {'volume'}: 
                incar_dict['relax'] = {'isif': 7}
            elif relax == {'ions','volume'}: 
                incar_dict['relax'] = {'isif': -1}
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
    def accelerate(self):
        """
        Required parameters, Whether to accelerate during optimization calculations 
        vasp.accelerate = {ediff: 1E-3}, {ediff: 1E-4}
        """
        return self._accelerate_

    @accelerate.setter
    def accelerate(self,value):
        if isinstance(value,bool):
            if value is True:
                step1 = {'kspacing':0.5, 'ediff':1E-3, 'nsw':30, 'istart': 0, 'icharg': 2}
                step2 = {'kspacing':0.4, 'ediff':1E-4, 'ediffg':-0.05, 'istart': 0, 'icharg': 2}
                self._accelerate_ = [step1, step2]

        elif isinstance(value,dict):
            self._accelerate_ = [value]
 
        else:
            acc = []
            for i in value:
                if isinstance(i,dict):
                    acc.append(i)
            if len(acc):
                self._accelerate_ = acc

    @property
    def links(self):
        links_dict = {"dos": ['scf'], "band": ['scf'], "born": ['scf'], "stm": ['scf'],
                      "emass": ['scf', 'band'], "partchg": ['scf', 'band'],
                      "emc": ['scf', 'band'], "chgdiff": ['scf'], "zpe": ['scf'], 
                      "hse_gap": ['scf','band'], "hse_band": ['scf'], "hse_emass": ['hse_band'], 
                      "meta_gap": ['scf','band'], "meta_band": ['scf'],
                      "unfolding": ['scf'], "deformation":['scf','band'],
                      "optics": ['scf'], "dielectric": ['scf'], "jdos": ['optics'],
                      "diag": ['scf'], "gw": ['diag'], "bse": ['gw','diag'],
                      "singlet": ['scf'], "triplet": ['scf'], "shg": ['optics'], 
                      "elastic": ['scf'], "poisson": ['scf'], "amset": ['scf'],#'dielectric','elastic'],
                      "fc2": ['scf'], "fc3": ['scf'], "dfpt": ['scf'], "tdep": ["md"], "shengbte": ["tdep"],
                      "softmode":['fc2'], "gruneisen": ['fc2'], "raman": ['fc2'], 
                      "bader": ['scf'], "cohp": ['scf'], "boltztrap": ['scf'], "elf": ['scf']}
        return links_dict

    @property
    def base_tasks(self):
        from .vaspflow import get_base_modules
        return get_base_modules()
  
    @property
    def suggest_potcar(self):
        return None

    def get_potential_files(self, structure):
        """
        workflow function
        """
        from .utils import paw_potcar

        elements = list(structure.species_of_elements)
        potcar_lib = self.get_potcar_library(self.potential)
        files = [None]*len(elements)
        priority = ['_3','_2','','_sv','_pv','_d','_s','_h']

        for i,elm in enumerate(elements):
            # custom rule %
            if elm in self.potential_map and self.potential_map[elm] in potcar_lib[elm]:
                files[i] = potcar_lib[elm][self.potential_map[elm]]
            # default rule %
            elif elm in potcar_lib:
                if paw_potcar[elm] in potcar_lib[elm]:
                    files[i] = potcar_lib[elm][paw_potcar[elm]]
                else:
                    for tag in priority:
                        if elm+tag in potcar_lib[elm]:
                            files[i] = potcar_lib[elm][elm+tag]
                            break
                for tag in priority:
                    if elm+tag in potcar_lib[elm]:
                        files[i] = potcar_lib[elm][elm+tag]
                        break
        return files

    def get_potential(self, structure, files=None):
        import re

        elements = list(structure.species_of_elements)
        if files is None:
            files = self.get_potential_files(structure)
        elif len(files) != len(elements):
            raise ValueError('The number of files and elements are inconsistent.')

        # set encut & zval %
        enmax = []                                 
        enmin = []       
        zval = []
        for i,potcar in enumerate(files):
            if potcar is None:
                raise OSError('Failed find potential of %s' %elements[i])
            # read potcar %                
            with open(potcar,'r') as f:
                for line in f:
                    if 'ENMAX' in line:                  
                        enmax.extend(re.findall(r'ENMAX\s*=\s*(\d+\.\d+)',line))
                        enmin.extend(re.findall(r'ENMIN\s*=\s*(\d+\.\d+)',line))
                        break

            # read potcar %
            with open(potcar,'r') as f:
                for line in f:
                    if 'ZVAL' in line:  
                        zval.extend(re.findall(r'ZVAL\s*=\s*(\d*\.\d+)',line))
                        break
        assert len(enmax) == len(enmin) == len(zval) == len(files)

        # sum nelm %
        nelect = 0
        for i, num in enumerate(structure.number_of_atoms):
            nelect += num * float(zval[i])
        enmax = np.array(enmax,dtype=float).max()
        enmin = np.array(enmin,dtype=float).min()

        return Potential(formula=structure.get_formula(), files=files, enmax=enmax, enmin=enmin, nelect=nelect)

    @classmethod
    def get_potcar_library(self, path:str):

        from collections import defaultdict 
 
        potcar_lib = defaultdict(dict) 
        for file in os.listdir(path):
            potcar = os.path.join(path,file,"POTCAR")
            if os.path.exists(potcar): 
                element = file.split('_')[0]
                potcar_lib[element][file] = potcar

        return potcar_lib

    def get_vdW(self, incar):

        files = []
        if self.vdw != None and incar.name not in self.vdWdeny:

            incar.vdw = self.vdw
            params = self.get_vdW_params(incar.structure.species_of_elements)
            incar.group_update('vdw', params)

            if incar.vdw in ['B86', 'B88', 'DF2', 'rDF2', 'rPBE','oPBE', 'rVV10'] and self.vdWdat != None:
                files.append(self.vdWdat)

        return files

    def get_dipole(self, incar, **kwargs):

        if incar.get('ldipol'):
            params = {'ldipol': True}
            atoms = incar.structure
            if 'idipol' not in incar:
                params['idipol'] = 3
            if 'dipol' not in incar:
                weights = [atom.elementinfo.mass for atom in atoms.atomic_position]
                center_of_mass = np.average(atoms.get_positions(type='direct'), weights=weights, axis=0)
                params['DIPOL'] = center_of_mass 

            incar.group_update('ldipol', params)

    def get_ldau(self, incar, **kwargs):
        from jamip.utils.logger import load_yaml

        orb_d = ['Sc','V','Cr','Mn','Fe','Co','Ni','Cu','Zr','Nb','Mo',
                 'Ru','Rh','Pd','Ag','Hf','Ta','W','Re','Os','Ir','Pt','Au']
        orb_f = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho',
                 'Er','Tm','Yb','Lu']
        ldaul = []
        ldauu = []
        ldauj = []

        if incar.get('ldau') or self.ldau or 'u' in incar.xc_func:
            template = full_path('~/.jamip/env/ldau.yaml') 
            lda = load_yaml(template)
            for elm in incar.structure.species_of_elements:
                # ldaul
                if elm in orb_d:
                    ldaul.append(2)
                elif elm in orb_f:
                    ldaul.append(3)
                else:
                    ldaul.append(-1)

                # u & j
                if elm in self.ldau_map:
                    ldauu.append(self.ldau_map[elm])
                    ldauj.append(0)
                else:
                    ldauu.append(lda['LDAUU'].get(elm,0))
                    ldauj.append(lda['LDAUJ'].get(elm,0))

            params = {'ldau': True}
            if 'ldautype' not in incar:
                params['ldautype'] = 2
            if 'ldaul' not in incar:
                params['ldaul'] = ' '.join([str(i) for i in ldaul])
            if 'ldauu' not in incar:             
                params['ldauu'] = ' '.join([str(i) for i in ldauu])
            if 'ldauj' not in incar:    
                params['ldauj'] = ' '.join([str(i) for i in ldauj])
            if 'lmaxmix' not in incar:    
                params['lmaxmix'] = max(max(ldaul)*2, 2)

            incar.group_update('ldau', params)

    def get_magmom(self, incar, **kwargs):

        multi = 3 if incar.spin == 3 else 1
        natom = sum(incar.structure.number_of_atoms)
        # auto %
        if self.magmom is False or len(self.magmom_map) == 0:
            #return '%d*0' %natom*multi
            return None
        else:
            magmom = ''
            for elm, num in zip(incar.structure.species_of_elements, incar.structure.number_of_atoms):
                mag = self.magmom_map[elm] if elm in self.magmom_map else 1
                magmom += '%d*%d ' %(num*multi, mag)
            return magmom
            

