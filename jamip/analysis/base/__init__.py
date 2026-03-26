from pathlib import Path

VASPFILES = {\
    'band': ['EIGENVAL','PROCAR','WAVECAR'],
    'dos': ['DOSCAR','PROCAR'],
    'md': ['XDATCAR'],
    'boltztrap': ['boltztrap.dat','SCF.trace'],
    'phonon':['phonopy_disp.yaml','phonopy.yaml']
    }


class Finder:

    _stdin = None
    _file = None

    def __init__(self,path,task=None,soft='vasp'):
        self.task = task
        self.soft = soft     
        self.stdin = path
        self.multi = False

    def get_task_files(self,task,soft:str):
        '''
        get task files
        '''
        files = []
        if soft == 'vasp':
            files = ['OUTCAR','vasprun.xml']
            if task in VASPFILES:
                files += VASPFILES[task]
        elif soft == 'qe':
            if task is None: task = 'scf'
            files.append(task+'.xml')
        elif soft == 'gaussian':
            if task is None: task = 'opt'
            files.append(task+'.out')
        elif soft == 'cp2k':
            files = ['cp2k.inp','cp2k.log']
        else:
            raise ValueError('Soft %s not supported!' % soft)

        return files

    def seek_entry_type(self,path:str,task:str):
        '''
        seek entry type
        '''
        path = Path(path)
        # check if path is a jamip directory
        if path.is_dir() and (path/'.status').exists():
            return 'jamip'
          
        files = self.get_task_files(task, self.soft)
        for file in files:
            if (path/file).exists():
                return file

    def seek_multi_entries(self, path:str, task:str):
        """
        seek multi entry dirs
        """
        root = Path(path)
        files = self.get_task_files(task, self.soft)
        entries = []
        for path in root.iterdir():
            if path.is_dir():
                for file in files:
                    if (path/file).exists():
                        entries.append(path)
                        break
        return entries

    @classmethod
    def seek_structure_type(cls, filename:str):
        '''
        seek structure type
        '''
        ftype = None
        if filename.endswith('.cif'):
            ftype='cif'
        elif filename.endswith('.xyz'):
            ftype='xyz'
        elif filename.endswith('.mol'):
            ftype='mol'
        elif filename.endswith('.vasp'):
            ftype = 'poscar'
        elif filename.endswith('.struct'):
            ftype = 'poscar'
        elif 'CONTCAR' in filename or 'POSCAR' in filename:
            ftype='poscar'

        return ftype

    @classmethod
    def get_func(cls, soft:str):
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.qe.qexml import Xml
        from jamip.analysis.cp2k.output import LogFinder

        if soft == 'vasp':
            return GrepOutcar()
        elif soft == 'qe':
            return Xml()
        elif soft == 'cp2k':
            return LogFinder()
        else:
            raise ValueError(f'Soft {soft} not supported!')

    @property
    def file(self):
        return self._file
    
    @file.setter
    def file(self,value:str):
        self._file = value

    @property
    def stdin(self):
        return self._stdin

    @stdin.setter
    def stdin(self,path:str):
        '''
        get calculation type, return absolute path if exists
        '''
        self._stdin = None
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("File not found!")

        # get entry type
        file = self.seek_entry_type(path, self.task)
        if file != None:
            self.file = file
            self._stdin = path
        elif getattr(self, 'multi', False):
            self._stdin = path
        else:
            raise ValueError("File not found!")

    @property
    def scfdir(self):

        if self.file == 'jamip':
            if self.soft == 'vasp':
                path = self._stdin/'scf'
            elif self.soft == 'qe':
                # return join(self._stdin, 'qerun', 'scf.xml')
                path = self._stdin/'qerun'
            else:
                raise ValueError('Soft %s not supported!' % self.soft)
        else:
            path =self._stdin
        
        if path.exists():
            return path
        
    @property
    def stdcell(self):

        if self.soft == 'vasp':
            if self.file == 'jamip':
                contcar = self._stdin/'scf'/'CONTCAR'
                poscar = self._stdin/'scf'/'POSCAR'
            else:                
                contcar = self._stdin/'CONTCAR'
                poscar = self._stdin/'POSCAR'
        else:
            raise FileNotFoundError('No stdcell found!')

        if contcar.exists():
            return contcar
        elif poscar.exists():
            return poscar
        else:
            raise OSError('Seek std_structure failed!')

    def grep(self,value,path=None):
        '''
        grep base information from scf calculation
        '''

        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.qe.qexml import Xml
        from jamip.analysis.cp2k.output import LogFinder
        if path is None: path = self.scfdir
        func = getattr(self.get_func(self.soft), value)
        return func(path)

class FinderSet(Finder):

    _stdin = None
    _files = None
    task = None
    recursion = True
 
    def __init__(self,path,task=None,soft='vasp',depth=1, recursion=True):
        self.recursion = recursion
        self.task = task
        self.soft = soft
        self.depth = depth
        self.stdin = path

    @property
    def stdin(self):
        return self._stdin

    @property
    def files(self):
        return self._files

    @stdin.setter
    def stdin(self,value):
        import warnings
        import os
        paths = []
        files = []

        assert (value is None) == False, 'value must be Pathlike or List(Pathlike), Not None'

        try:
            os.stat(value)
            value = [value]
        except:
            pass

        def traverse_file(dir, depth, max_depth):
            for path in dir.iterdir():
                if path.is_file():
                    yield path
                elif path.is_dir() and depth < max_depth:
                    traverse_file(path, depth+1, max_depth)

        def traverse_dir(dir, depth, max_depth):
            for path in dir.iterdir():
                if path.is_dir():
                    yield depth, path
                    if depth < max_depth:
                        traverse_dir(path, depth+1, max_depth)


        if self.task == 'structure':
            for path in value:
                path = Path(path)
                if not path.exists():
                    warnings.warn(f'Path {path} not found!')
                else:
                    for file in traverse_file(path, 0, self.depth):
                        ftype = self.seek_structure_type(file.name)
                        if ftype != None:
                            paths.append(file)
                            files.append(ftype)

        else:
            for path in value:
                path = Path(path)
                if not path.exists():
                    warnings.warn(f'Path {path} not found!')
                else:
                    ftype = self.seek_entry_type(path, self.task)
                    if ftype != None:
                        paths.append(path)
                        files.append(ftype)
                        if not self.recursion:
                            continue

                    max_depth = self.depth
                    if self.depth > 0:
                        gen = traverse_dir(path, 1, self.depth)
                        for depth, file in gen:
                            if not self.recursion and depth > max_depth:
                                continue

                            ftype = self.seek_entry_type(file, self.task) 
                            if ftype != None:
                                paths.append(file)
                                files.append(ftype)
                                max_depth = depth

        self._stdin = paths
        self._files = files

    @classmethod
    def from_pool(cls, path, task=None, soft='vasp'):
        from jamip.compute.pool import Pool
        paths = []
        try:
            root = Path(path).absolute().parent
            with Pool.open(path, 'r') as pool:
                for dir in pool.keys():
                    if (root/dir).exists():
                        paths.append(root/dir)
        except:
            print('PathError: Invalid poolfile')
        finder = FinderSet(paths, task, soft, depth=0)
        return finder 
