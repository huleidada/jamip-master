import os
import numpy as np
from pathlib import Path


class __Vasptools:

    tasklist = ['potcar', 'bond', 'standard', 'kpath', 'kpath2d', 'clean']

    def __init__(self,*args,**kwargs):
        """
        与output不同，vasptools模块需要读取POSCAR等结构文件(potcar, bond kpath)
        这里统一采用通配符来识别, 不支持任务池文件
        例如: jp -v kpath -f *.cif
              jp -v bond -f Output/*/scf/POSCAR
        """
        self.tasks = []
        self.paths = []

    def run(self,params,**kwargs):
        import glob

        # set paths %
        if 'pool' in params:
            # TODO read task Pool
            for path in params['pool']:
                paths = glob.glob(str(path))
                if len(paths) == 0:
                    raise OSError('Path not exists!')
                self.paths.extend(paths)

        if len(self.paths) == 0:
            self.paths.append(os.getcwd())

        # run tasks %
        task = params['vasp_tools']
        if task == 'standard':
            self.standard()
        elif task == 'bond':
            self.bonding()
        elif task == 'phonon':
            self.write_force()
        elif task == 'potcar':
            self.write_potcar()
        elif task == 'backup':
            self.backup()
        elif task == 'kpath':
            self.kpath()
        elif task == 'kpath2d':
            self.kpath2d()
        elif task == 'clean':
            self.clean()
        elif task == 'dim':
            self.dim()

    def clean(self):
        """
        glob.glob(path + "/**/WAVECAR",recursive=True)
        remove: REPORT / PCDAT / XDATCAR
        """
        from os.path import isfile,join
        import glob

        recursive = ['REPORT','PCDAT', 'XDATCAR', 'WAVECAR', 'CHGCAR', 'CHG']
        include = [] 
        exclude = []

        while(True):
            in_content = input("Notice that you are deleting files. Type Y/N to continue: ")
            if in_content.lower() == "y":
                break
            if in_content.lower() == "n":
                return 0
            else:
                print("Invalid input. Please try again.")

        #os.remove(path)
        num = 0
        mem = 0
        for stdin in self.paths:
            for filename in recursive: 
                for path in glob.glob(stdin + "/**/" + filename ,recursive=True):
                    mem += os.path.getsize(path) / 1024 / 1024
                    num += 1
                    print(path, os.path.getsize(path))
            
        print("clean finished. %s files were deleted, with a total memory of %.1f MB" %(num, mem))

    def standard(self):
        from jamip.abtools.vasp.vaspio import VaspIO
        from jamip.analysis.base.test import get_ftype
        from jamip.abtools.vasp.check import CheckStatus
        from jamip.structure import read
        import shutil

        if os.path.exists('cifs'):
            result = input("The directory already exists. Do you want to run the command again?").lstrip()
            if not (len(result) and result[0].lower() == 'y'):
                print('exit')
                exit()
        else:
            os.makedirs('cifs')

        # 有效输入路径为jamip计算目录和结构文件，其中jamip目录根据.status确定稳定结构
        num = 0
        # try read all possible structure models %
        for path in self.paths:
            path = Path(path) 
            try:
                if path.is_file() and get_ftype(path.name):
                    s = read(path)
                    if get_ftype(path.name) == 'poscar':
                        VaspIO.write_poscar(s, 'cifs', name=path.name)
                    else:
                        VaspIO.write_poscar(s, 'cifs', name=path.stem+'.vasp')
                        #path.unlink()
                    num += 1
                elif (path / '.status').exists():
                    status = CheckStatus.load_status(path)
                    if 'scf' in status and status['scf']['status']:
                        shutil.copy(path / status['scf']['path'] / 'CONTCAR', 'cifs/' + path.name)
                        num += 1
                    elif 'relax' in status and status['relax']['status']:
                        shutil.copy(path / status['relax']['path'] / 'CONTCAR', 'cifs/' + path.name)
                        num += 1
            except:
                print('Skip path %s' %path)
                
        print("standard successed : %s" %num)

    def kpath(self):    
        from jamip.analysis.base import Finder

        num = 0
        # 有效输入路径结构文件和含有POSCAR的目录
        for path in self.paths:
            path = Path(path) 
            # try:
            if path.is_file() and Finder.seek_structure_type(path.name):
                self.kprint(path) 
                num += 1
            elif (path/'POSCAR').exists():
                self.kprint(path/'POSCAR') 
                num += 1
            # except:
            #     print('Skip path %s' %path)

        print("kpath successed : %s" %num)

    def kpath2d(self):    
        from jamip.analysis.base import Finder

        num = 0
        # 有效输入路径结构文件和含有POSCAR的目录
        for path in self.paths:
            path = Path(path) 
            # try:
            if path.is_file() and Finder.seek_structure_type(path.name):
                self.kprint(path, dim=2) 
                num += 1
            elif (path/'POSCAR').exists():
                self.kprint(path/'POSCAR', dim=2) 
                num += 1
            # except:
            #     print('Skip path %s' %path)

        print("kpath successed : %s" %num)

    def bonding(self):
        from jamip.analysis.base.test import get_ftype
        from jamip.structure.bonding import Bonding
        from jamip.structure import read

        # 有效输入路径结构文件和含有POSCAR的目录
        for path in self.paths:
            path = Path(path) 
            try:
                if path.is_file() and get_ftype(path.name):
                    structure = read(path)
                    bd = Bonding(structure, mtehod='min')
                    print(path)
                    print(bd.data.full_repr())
                elif (path/'POSCAR').exists():
                    path = path/'POSCAR'
                    structure = read(path)
                    bd = Bonding(structure, mtehod='min')
                    print(path)
                    print(bd.data.full_repr())
            except:
                pass

    def dim(self):
        from jamip.structure import read, Structure
        from jamip.analysis.base.test import get_ftype
        from jamip.structure.dimension import DimensionAnalysis
        import spglib

        for path in self.paths:
            path = Path(path) 
            #try:
            if True:
                if path.is_file() and get_ftype(path.name):
                    structure = read(path)
                    primcell=spglib.find_primitive(structure.to_cell(), symprec=1e-3)
                    s = Structure.from_cell(primcell)
                    us = DimensionAnalysis(s)
                    us.debug = True
                    us.set_valence()
                    us.set_bonding()
                    units = us.search_cutoff()
                    print(path)
                    print(units)
                elif (path/'POSCAR').exists():
                    path = path/'POSCAR'
                    structure = read(path)
                    primcell=spglib.find_primitive(structure.to_cell(), symprec=1e-3)
                    s = Structure.from_cell(primcell)
                    us = DimensionAnalysis(s)
                    us.set_bonding()
                    units = us.search_cutoff()
                    print(path)
                    print(units)
            #except:
                pass
            
    def write_potcar(self):

        from jamip.utils.logger import load_yaml, dump_yaml
        from jamip.abtools.vasp.setvasp import SetVasp
        from jamip.analysis.base import FinderSet
        from jamip.structure import read

        potenv = Path.home() / '.jamip' / 'env' / 'pot.yaml'
        potcar_map = load_yaml(potenv)
        if potcar_map is None: 
            potcar_map = {}

        if potcar_map.get('vasp') and Path(potcar_map['vasp']).exists():
            print('Potentials read from %s' %potcar_map['vasp'])
            potdir = potcar_map['vasp']

        else:
            potdir = input('Please input VASP Pseudopotential library path: ')
            if Path(potdir).exists():
                potcar_map['vasp'] = potdir
                dump_yaml(potcar_map, potenv)
            else:
                raise OSError('Pseudopotential library %s not exists!' %potdir)

        potcar_lib = SetVasp.get_potcar_library(potdir)
        finder = FinderSet(self.paths, task='structure')

        for path in finder.stdin:
            try:
                structure = read(path)
                elements = list(structure.species_of_elements)
                files = [None]*len(elements)
                names = [None]*len(elements)
         
                # set potential by auto %
                for i,elm in enumerate(elements):
                    # custom rule %
                    if elm in potcar_map and potcar_map[elm] in potcar_lib[elm]:
                        files[i] = potcar_lib[elm][potcar_map[elm]]
                        names[i] = potcar_map[elm]
                    # default rule %
                    elif elm in potcar_lib:
                        for tag in ['_3','_2','','_sv','_pv','_d','_s','_h']:
                            if elm+tag in potcar_lib[elm]:
                                files[i] = potcar_lib[elm][elm+tag]
                                names[i] = elm+tag
                                break
                    else:
                        raise KeyError("No useable pseudopotential for %s !" %elm)
          
                output = path.parent / 'POTCAR'
                if output.exists(): 
                    os.rename(output, path.parent / "POTCAR_bak")
                 
                print('Potentials {0} > {1}'.format(' + '.join(names), output))
                with open(output,'w') as f:
                    for potcar in files:
                        with open(potcar, 'r') as p:
                            f.write(p.read())
            except: 
                print('Generate failed for %s' %path)

    def kprint(self, path, dim=3):
        from jamip.utils.brillouin_zone import HighSymmetryKpath, HighSymmetryKpath2d
        from jamip.abtools.base.kpoints import BandPath
        from jamip.structure import read

        structure = read(path)
        if dim == 2:
            bz = HighSymmetryKpath2d()
        else:
            bz = HighSymmetryKpath()
        kpoint = bz.get_HSKP(structure.to_cell(), symprec=1e-3)
        kpath = []
        for points in kpoint['Path']:
            kpath.append(points)

        bandpath = BandPath.from_symbols(kpath, kpoint['Kpoints']) 

        if dim == 2:
            print(path,' LG: {0}({1})  PG: {2}'.format(bz.layergroup,bz.lgnum,bz.pointgroup))
        else:
            print(path,' SG: {0}({1})  PG: {2}'.format(bz.spacegroup,bz.sgnum,bz.pointgroup))
        for i in range(len(bandpath)):
            print(' "{0} {1}",'.format(bandpath.sites[i],bandpath.numbers[i]))
        

