import os
import numpy as np
from pathlib import Path


class __CP2ktools:

    tasklist = ['poscar', 'bond', 'standard', 'kpath', 'kpath2d', 'clean']

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
        task = params['cp2k_tools']
        if task == 'standard':
            self.standard()
        elif task == 'bond':
            self.bonding()
        elif task == 'phonon':
            self.write_force()
        elif task == 'poscar':
            self.write_poscar()
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
            
    def write_poscar(self):

        from jamip.abtools.cp2k.cp2kio import CP2KIO
        from jamip.structure import write

        for path in Path.cwd().glob('*.inp'):
            try:
                structure = CP2KIO.load_structure(path.parent)
                write(structure, f'{path.stem}.vasp')
            except: 
                print(f'load failed for {path}')
