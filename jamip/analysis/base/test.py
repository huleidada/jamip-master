from pathlib import Path

class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())

def get_ftype(filename):

    ftype = None
    if filename.endswith('.cif'):
        ftype='cif'
    elif filename.endswith('.xyz'):
        ftype='xyz'
    elif filename.endswith('.mol'):
        ftype='mol'
    elif filename.endswith('.vasp'):
        ftype = 'poscar'
    elif 'CONTCAR' in filename:
        ftype='poscar'
    elif 'POSCAR' in filename:
        ftype='poscar'

    return ftype

def get_soft(filename):
    if filename == 'OUTCAR':
        return 'vasp'
    elif filename == '.status':
        return 'jamip'

def seekpath(path):

    def workpath(path):
        """yield calculation path or all paths in input paths"""
        if path.is_dir():
            for f in path.iterdir():
                yield from workpath(path / f)
        elif path.is_file():
            filetype = get_soft( path.name )
            if filetype != None:
                yield ( path.parent, filetype )

    return IterableAdapter(lambda: workpath(path))

def seekpath_with_depth(path, depth=1):

    def workpath(path,depth):
        """yield calculation path or all paths in input paths"""
        if path.is_dir() and depth > -1:
            for f in path.iterdir():
                yield from workpath( path / f, depth-1)
        elif path.is_file:
            filetype = get_soft( path.name )
            if filetype != None:
                yield ( path.parent, filetype)

    return IterableAdapter(lambda: workpath(path, depth))
  
def seekpool(path):
    from jamip.compute.pool import Pool
    root = dirname(path)

    def workpath():
        """yield calculation path or all paths in pool"""
        with Pool.open(path) as pool:
            for outdir in pool:
                if isinstance(join(root, outdir, '.status')):
                    yield join(root, outdir)

    return IterableAdapter(lambda: workpath())



class Finder:
    """Basic method for querying properties and paths"""

    @staticmethod
    def grep(iterable, value:str):
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.qe.qexml import GrepXml

        def distribute():
            for path,ftype in iterable:
                if ftype == 'vasp':
                    func = getattr(GrepOutcar(),value)
                elif ftype == 'qe':
                    func = getattr(GrepXml(),value)
                elif ftype == 'jamip':
                    func = getattr(GrepOutcar(),value)
                else:
                    raise ValueError('Unknown grep module.')
                yield path, func(path)

        return IterableAdapter(distribute)

    @staticmethod
    def stdscf(iterable):

        def distribute():
            for path,ftype in iterable:
                if ftype == 'jamip':
                    if exists(join(path, 'scf', 'OUTCAR')):
                        path = join(path, 'scf', 'OUTCAR')
                        ftype = 'vasp'
                    #elif exists(join(self._stdin, 'qerun', 'scf.xml')):
                    #    return join(self._stdin, 'qerun', 'scf.xml')
                    else:
                        raise OSError('Failed to find a scf directory.')

                yield path, ftype

        return IterableAdapter(distribute)

    @staticmethod
    def seek_structure(iterable):

        def cellpath(iterable):
            """yield calculation path or all paths in pool"""
            for path in iterable:
                if path.is_dir():
                    yield from cellpath(path.iterdir())
                elif path.is_file():
                    if get_ftype( path.name ) != None:
                        yield path.resolve()

        return IterableAdapter(lambda: cellpath(iterable))

    @staticmethod
    def seek_calc_with_depth(iterable, depth=1):

        def workpath(iterable,depth):
            """yield calculation path or all paths in input paths"""
            for path in iterable:
                if path.is_dir() and depth > -1:
                    yield from workpath( path.iterdir(), depth-1)
                elif path.is_file():
                    filetype = get_soft( path.name )
                    if filetype != None:
                        yield ( path.resolve().parent, filetype)
 
        return IterableAdapter(lambda: workpath(iterable, depth))
