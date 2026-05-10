"=========================================================="
"introduction of functions:" \
"runtime: show the running time of the script" \
"FindKeys: find the keyword in the given line"
""
def runtime():
    import time as tm
    return tm.strftime("%Y-%m-%d %H:%M:%S",tm.localtime(tm.time()))

def find_lcsubstr(s1:str, s2:str):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]
    mmax=0
    p=0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p]

def get_stdout(paths:list):
    import os.path
    
    if len(paths) == 1:
        return os.path.dirname(paths[0])
    elif len(paths) > 1:
        return os.path.commonpath(paths)
        #return find_lcsubstr(paths[0], paths[-1])

def relink(stdin, stdout, filename):
    import os.path
    import os

    infile = os.path.join(stdin, filename)
    outfile = os.path.join(stdout, filename)
    if not os.path.exists(infile):
        raise OSError('File %s not exists in %s' %(filename,stdin))
    # rm stdout / filename
    if os.path.exists(outfile) or os.path.islink(outfile):
        os.unlink(outfile)
    relpath = os.path.relpath(infile, start=stdout)
    os.symlink(relpath, outfile)

def is_periodic(lst):
    from itertools import cycle

    def is_k_periodic(lst, k):
        # we want the returned part to repeat at least twice... 
        # otherwise every list is periodic (1 period of its full self)
        if len(lst) < k // 2: 
            return False

        return all(x == y for x, y in zip(lst, cycle(lst[:k])))

    for k in range(1, (len(lst) // 2) + 1):
        if is_k_periodic(lst, k):
            return tuple(lst[:k])

    return None

def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(instance):
        if not hasattr(instance, attr_name):
            setattr(instance, attr_name, func(instance))
        return getattr(instance, attr_name)

    return _lazy_property

class CopyFile(object):
    """class to copy files
    static method: loop_files;
    method: load_files:
    """

    def load_files(self, stdin, stdout, loop=False, include=None, exclude=None):
        """
        load_files: copy file from 'stdin' directory to 'stodut' directory;
        :param stdin: the original directory includes the files wait for copy;
        :param stdout: the object directory;
        :param loop: bool, directory or documents, False: documents;
        :param include: given files to copies;
        :param exclude: given files to ignore;
        :return: None
        """
        import os

        if not os.path.exists(stdout): os.makedirs(stdout)
        if include is not None:
            for f in include:
               src = os.path.join(stdin, f)
               os.system('cp -r {0} {1}'.format(src,stdout))
        return
        n=2
        if loop is True: n = 1
        copyfile = os.walk(stdin).next()[n]

        if exclude is not None:
            for exf in exclude:
                if exf in copyfile:
                    copyfile.remove(exf)
        
        for f in copyfile:
            src = os.path.join(stdin, f)
            os.system('cp -r {0} {1}'.format(src,stdout))
 
