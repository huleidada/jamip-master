import os
import numpy as np
from jamip.analysis.vasp import GrepOutcar
from jamip.analysis.base import FinderSet
from pathlib import Path


__task_dict__ = {'dielectric':('dielectric','dielectric_ionic'),
                 'cbvb':('bandgap','cbm-kpoint','vbm-kpoint'),
                 'bandgap':('indirect','direct','ismetal'),
                 'metal_bandgap':('indirect','direct','full_band','empty_band'),
                 'boltztrap':('H-mass','e-mass'),
                 'emass':('H-mass','e-mass'),
                 'emhm':('x-vbm','y-vbm','z-vbm','x-cbm','y-cbm','z-cbm'),
                 'deformation':('vbm-dp','cbm-dp'),
                 'work_function':('work_function',),
                 'relax_free_energy':('free_energy',),
                 'free_energy_per_atom':('free_energy',),
                 'converge':(),
                 'dp':(),
                 'status':('relax','scf'),
                 'formula':('formula',)}


class __OutputData:
#(GrepOutcar):

    tasklist = ['plot', 'csv', 'sort', 'form', 'hdf5']

    def __init__(self,*args,**kwargs):
        self.paths = []
        self.files = []
        self.tasks = []
        self.depth = 2
        self.soft = 'vasp'

    def __getattr__(self, meth_name):
        from jamip.analysis.base import Finder 
        return getattr(Finder.get_func(self.soft), meth_name)

    @property
    def path(self):
        return self.paths
    @property
    def stdin(self):
        return self.paths

    @stdin.setter
    def stdin(self,value:str):
        from jamip.compute.pool import Pool

        value = Path(value)

        if value.is_dir():
            fs = FinderSet(path=value, depth=self.depth, soft=self.soft, recursion=False)
        elif value.is_file():
            fs = FinderSet.from_pool(value, soft=self.soft)
        else:
            raise OSError('PathError: Invalid directory')

        if fs.stdin != None:
            self.paths.extend(fs.stdin)
            self.files.extend(fs.files)

    def run(self,params,path=None,**kwargs):

        # set soft %
        if 'soft' in params:
            self.soft = params['soft']

        # set paths %
        if 'pool' in params:
            for path in params['pool']:
                if path.exists():
                    self.stdin = path
        elif path is None:
            self.stdin = os.getcwd()
        else:
            for dir in path:
                if path.exists():
                    self.stdin = dir

        if len(self.paths) == 0:
            print('Warning! Invalid input path')
            exit()
        print('Total = %s' %len(self.paths))
      
        # set tasks %
        tasks = []
        for t in params['output']:
            if t in self.tasklist:
                self.tasks.append(t)
            else:
                tasks.append(t)

        # load datas %
        if len(tasks):
            dataset = self.getdatas(tasks)
        elif 'hdf5' in self.tasks:
            return self.hdf5()
        else:
            print('Warning! No vaild task was found')

        # run main_project %
        if 'plot' in self.tasks:
            self.plot(tasks)
        if 'csv' in self.tasks:
            self.csv(dataset)
        if 'sort' in self.tasks:
            self.sort(dataset)
        if 'form' in self.tasks or len(self.tasks) == 0:
            self.form(dataset)

    def getdatas(self, tasks:list):
        import pandas as pd 
        from jamip.utils.utils import get_stdout

        # basename %
        dataset = []
        root = get_stdout(self.stdin)
        dat = [os.path.relpath(path,root) for path in self.stdin]
        dataset.append(pd.DataFrame(dat, columns=['path'], index=self.stdin))

        # property
        for task in set(tasks):

            #func = getattr(self,task)(self.stdin[0])
            #print(func)
            # check whether vaild task %
            state = self.find_task_belong(task)
            if state == 0:     # Invaild task %
                print('Invaild_task: %s' %task)
                continue
        
            # Vaild task %
            dat = []
            success = 0
            func = getattr(self,task)
            # run all path %
            for path in self.stdin:
                try:
                    scfdir = path/'scf'
                    #print(scfdir)
                    # GrepOutcar moudle %
                    if state == 2 and scfdir.exists():
                        dat.append(func(scfdir))
                    # local moudle %
                    else:
                        dat.append(func(path))
                    success+=1
                except:
                   print('IOError: read %s task-%s failed' %(path,task))
                   dat.append(None) 
                
            # conclusion %
            print('%s success = %d' %(task,success))

            if state == 1:
                label = __task_dict__[task]
                null = [None]*len(label)
                for i,v in enumerate(dat):
                    if v == None:
                        dat[i] = null
            else:
                label = [task]
                        
            dataset.append(pd.DataFrame(dat, columns=label, index=self.stdin))
        dataset = pd.concat(dataset,axis=1)
        return dataset

    def find_task_belong(self, meth_name):
        '''check whether task belong to main_class or sub_class
           return 0 : Invalid task
           return 1 : belong to main_class, path use main_director
           return 2 : belong to sub_class, path use stdin/scf'''

        state = 2 if hasattr(self, meth_name) else 0

        for ty in type(self).mro():
            if meth_name in ty.__dict__:
                if ty == type(self):
                    state = 1
                    break

        return state 

    def formula(self,path):
        '''Structure formula'''
        from jamip.analysis.vasp import Finder
        formula = ''
        sep = ' ' if 'csv' in self.tasks else ''
        for elm,num in Finder(path).grep('atominfo'):
            if num != 1:
                formula += '%s%d%s' %(elm,num,sep)
            else:
                formula += '%s%s' %(elm,sep)
        return formula.rstrip()

    def relax_free_energy(self,path):
        '''free energy from relax calculations '''
        from jamip.analysis.vasp import Finder
        path = Path(path) 
        task = 'relax'
        result = None
        if self.soft == 'vasp':
            from jamip.abtools.vasp.check import CheckStatus
            if (path/'.status').exists():
                status = CheckStatus.load_status(path)
                if task in status and status[task]['status']:
                    stdin = path / status[task]['path']
                    result = np.round(Finder(stdin).grep('free_energy'), 6)
        elif self.soft == 'qe':
            pass
        else:
            raise KeyError('unknown soft')

        return result

    def free_energy_per_atom(self, path):
        '''free energy from relax calculations '''
        from jamip.analysis.vasp import Finder
        finder = Finder(path, soft=self.soft)
        task = 'scf'
        result = [None]
        if self.soft == 'vasp':
            result = np.round(finder.grep('free_energy'), 6)
            natom = finder.grep('nions')
            result = [result / natom]
        elif self.soft == 'qe':
            pass
        elif self.soft == 'cp2k':
            result = np.round(finder.grep('free_energy'), 6)
            natom = finder.grep('natoms')
            result = [result / natom]
        else:
            raise KeyError('unknown soft')
        return result

    def status(self,path):
        '''job status from .status '''
        from jamip.analysis.vasp import Finder
        from jamip.abtools.vasp.check import CheckStatus
        result = [None,None]
        if os.path.exists(os.path.join(path,'.status')):
            status = CheckStatus.load_status(path)
            for i,task in enumerate(('relax','scf')):
                if task in status:
                    result[i] = status[task]['status']
        return result

    def converge(self,path):
        '''converge params test'''
        from jamip.analysis.vasp import Finder
        from os.path import join, exists, relpath
        stdin = join(path,'converge')
        columns = list(__task_dict__['converge'])
        datas = [None]*len(columns)
        if exists(stdin):
            for root,dirs,files in os.walk(stdin):
                if 'OUTCAR' in files:
                    key = relpath(root, stdin).replace('/','-')
                    value = np.round(Finder(root).grep('free_energy'), 6)
                    if key in columns:
                        datas[columns.index(key)] = value
                    else:
                        columns.append(key)
                        datas.append(value)
            __task_dict__['converge'] = tuple(columns)
            return datas
        else:
            return None
            
    def dielectric(self,path):
        '''dielectric dielectric_ionic'''
        from jamip.analysis.vasp import OpticsFinder
        of = OpticsFinder(path)
        diel_e = np.round(of.get_dielectric_const(),6)
        diel_ion = np.round(of.get_dielectric_const_of_ionic(),6)
        return diel_e,diel_ion

    def cbvb(self,path):
        '''cbvb data'''
        from jamip.analysis.vasp import BandFinder
        cbmvbm = BandFinder(path).get_data().get_cbmvbm()
        bandgap = np.around(cbmvbm['cbm'].energy - cbmvbm['vbm'].energy,4)
        cbm_kpoint = '{0[0]}, {0[1]}, {0[2]}'.format(np.round(cbmvbm['cbm'].kpoints,6))
        vbm_kpoint = '{0[0]}, {0[1]}, {0[2]}'.format(np.round(cbmvbm['vbm'].kpoints,6))
        return bandgap,cbm_kpoint,vbm_kpoint

    def metal_bandgap(self,path):
        '''indirect_bandgap direct_bandgap ismetal'''
        from jamip.analysis.vasp import BandFinder
        # bf = BandFinder(path).get_data()
        bf = BandFinder(path).get_data(source='EIGENVAL')
        cbvbs = bf.get_metal_cbvb()
        fb, eb = cbvbs[0]
        fb += 1
        vbm = np.max(bf.bands[0,:,fb,0])
        cbm = np.min(bf.bands[0,:,eb,0])
        direct = np.max(bf.bands[0,:,eb,0]-bf.bands[0,:,fb,0])
        indirect = cbm - vbm
        return indirect, direct, fb, eb

    def bandgap(self,path):
        '''indirect_bandgap direct_bandgap ismetal'''
        if self.soft == 'vasp':
            from jamip.analysis.vasp import BandFinder
            if (path/"PROCAR_OPT").exists():
                bf = BandFinder(path).get_data(source='PROCAR_OPT')
            else:
                bf = BandFinder(path).get_data(source='EIGENVAL')
        elif self.soft == 'qe':
            from jamip.analysis.qe import BandFinder
            bf = BandFinder(path).get_data()
        else:
            raise KeyError('unknown soft')

        gap = bf.get_bandgap()
        isdirect = False if gap['direct'] > gap['indirect'] else True
        return gap['indirect'],gap['direct'], bf.metal

    def deformation(self,path):
        '''cbm-dp vbm-dp'''
        from jamip.analysis.vasp import BandFinder
        dp = BandFinder(path).get_deformation_potential()
        print(dp)
        cbms, vbms = [], []
        for key,value in dp.items():
            if 'cbm' in key: cbms.append(value)
            elif 'vbm' in key: vbms.append(value)
        vbmdp = np.around(np.mean(vbms), 4)
        cbmdp = np.around(np.mean(cbms), 4)
        return vbmdp, cbmdp

    def boltztrap(self,path):
        '''H-mass e-mass'''
        from jamip.analysis.vasp.boltztrap import BoltztrapFinder
        bf = BoltztrapFinder(path)
        return bf.get_effective_mass()

    def work_function(self,path):
        '''work_function'''
        from jamip.analysis.vasp import BandFinder
        from jamip.analysis.vasp import Finder
        stdin = Finder(path).scfdir
        vaccum_level = GrepOutcar().locpot(stdin,axis='z')[0]
        fermi_energy = BandFinder(stdin).get_fermi()
        return vaccum_level - fermi_energy

    def emass(self,path):
        '''H-mass e-mass'''
        from jamip.analysis.vasp import BandFinder
        bf = BandFinder(path)
        emass = bf.get_emass()
        if len(emass) == 6:
            vbm_emass = np.around(1/np.mean(1/np.array([emass['vbm-x'].mass,emass['vbm-y'].mass,emass['vbm-z'].mass])),3)
            cbm_emass = np.around(1/np.mean(1/np.array([emass['cbm-x'].mass,emass['cbm-y'].mass,emass['cbm-z'].mass])),3)
        elif len(emass) == 4:
            vbm_emass = np.around(1/np.mean(1/np.array([emass['vbm-x'].mass,emass['vbm-y'].mass])),3)
            cbm_emass = np.around(1/np.mean(1/np.array([emass['cbm-x'].mass,emass['cbm-y'].mass])),3)
        else:
            vbm_emass = np.around(emass['vbm'].mass,3)
            cbm_emass = np.around(emass['cbm'].mass,3)
        return vbm_emass,cbm_emass

    def emhm(self,path):
        '''H-mass e-mass'''
        from jamip.analysis.vasp import BandFinder
        bf = BandFinder(path)
        emass = bf.get_emass()
        if len(emass) == 6:
            return emass['vbm-x'].mass,emass['vbm-y'].mass,emass['vbm-z'].mass,emass['cbm-x'].mass,emass['cbm-y'].mass,emass['cbm-z'].mass
        else:
            return emass['vbm-x'].mass,emass['vbm-y'].mass,None,emass['cbm-x'].mass,emass['cbm-y'].mass,None

    def csv(self,dataset):
        import pandas as pd
        if len(self.stdin) > 1:
            filename = self.paths[0].absolute().parent.name + '.csv'
        else:
            filename = self.paths[0].name + '.csv'
        print(filename)
        dataset.to_csv(filename,index=False)

    def plot(self,tasks):
        '''
        simple plot model 
        plot up to 2 properties
        x-axis is sorted-index, only plot datas not None 
        if index <= 10 , x_params is path
        else, plot hist
        '''
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from jamip.analysis.vasp import BandFinder, GrepOutcar
        from matplotlib.ticker import MultipleLocator
        from collections import defaultdict

        parents = defaultdict(list)

        for path in self.stdin:
            for task in set(tasks):
                if task == 'cbvb':
                    bf = BandFinder(path)
                    bfd = bf.get_data()
                    bands = bfd.bands
                    cbvb = bfd.get_metal_cbvb()
                    cbmvbm = bfd.get_metal_cbmvbm()
                    for spin in range(len(bands)):
                        for ib in range(cbvb[spin][0],cbvb[spin][1]+1):
                            occ = np.mean(bands[spin,:,ib,1])
                            c = 'r' if occ > 0.5 else 'b'
                            plt.plot(bfd.get_xkpt(), bands[spin,:,ib,0], c=c)
                            plt.plot(bfd.get_xkpt(), bands[spin,:,ib,0],'o')
                    plt.axhline(np.max(bands[spin,:,cbvb[spin][0],0]), linestyle='--', c='b')
                    plt.axhline(np.min(bands[spin,:,cbvb[spin][1],0]), linestyle='--', c='r')
                    plt.axhline(bfd.fermi, linestyle='--', c='k')
                    plt.savefig(os.path.join(bf.stdin,'edge.png'),dpi=144)

                elif task == 'emass':
                    bf = BandFinder(path)
                    if bf.file != 'jamip':
                        if path.parent not in parents[task]:
                            bf = BandFinder(path.parent)
                            parents[task].append(path.parent)
                        else:
                            continue
                    #bf = BandFinder(path)
                    for dir in bf.emassdirs:
                        if 'vbm' in dir.name:
                            bfd = BandFinder(dir).get_data()
                            bands = bfd.bands
                            cbvb = bfd.get_cbvb()
                            cbmvbm = bfd.get_cbmvbm()
                            for spin in range(len(bands)):
                                plt.plot(bfd.get_xkpt(), bands[spin,:,cbvb[spin][0],0])
                                plt.plot(bfd.get_xkpt(), bands[spin,:,cbvb[spin][0],0],'o',label=dir.name)
                    plt.legend()
                    plt.savefig('hmass.png',dpi=144)
                    plt.clf()
                    for dir in bf.emassdirs:
                        if 'cbm' in dir.name:
                            bfd = BandFinder(dir).get_data()
                            bands = bfd.bands
                            cbvb = bfd.get_cbvb()
                            cbmvbm = bfd.get_cbmvbm()
                            for spin in range(len(bands)):
                                plt.plot(bfd.get_xkpt(), bands[spin,:,cbvb[spin][1],0])
                                plt.plot(bfd.get_xkpt(), bands[spin,:,cbvb[spin][1],0],'o',label=dir.name)
                    plt.legend()
                    plt.savefig('emass.png',dpi=144)

                elif task == 'dp':
                    bf = BandFinder(path)
                    if bf.file != 'jamip':
                        if path.parent not in parents[task]:
                            bf = BandFinder(path.parent)
                            parents[task].append(path.parent)
                        else:
                            continue
                    df = bf.get_deformation_potential_data(vacuum=None, core=None)
                    for axis in 'xyz':
                        if df[axis].sum() <= 1: continue
                        group = df[df[axis]]
                        print(group)
                        scale = group['scale'].values
                        Ecbm = group['Ecbm'].values - group['vacuum'].values
                        #Evbm = group['Evbm'] - group['vacuum']
                        indices = np.argsort(scale)
                        x = scale[indices]
                        y = Ecbm[indices]
                        plt.plot(x,y)
                        plt.plot(x,y,'o',label='cbm-'+axis)
                       
                    plt.legend()
                    plt.savefig('dpcbm.png',dpi=144)
                    plt.clf()
                    for axis in 'xyz':
                        if df[axis].sum() <= 1: continue
                        group = df[df[axis]]
                        print(group)
                        scale = group['scale'].values
                        #Ecbm = group['Ecbm'] - group['vacuum']
                        Evbm = group['Evbm'].values - group['vacuum'].values
                        indices = np.argsort(scale)
                        x = scale[indices]
                        y = Evbm[indices]
                        print(axis,'v',group['vacuum'].values[indices])
                        print(axis,'e',group['Evbm'].values[indices])
                        print(axis,'t',Evbm[indices])
                        plt.plot(x,y)
                        plt.plot(x,y,'o',label='vbm-'+axis)
                    plt.legend()
                    plt.savefig('dpvbm.png',dpi=144)


                elif task == 'energy':
                    oszicar = GrepOutcar().oszicar(path)
                    if oszicar.shape[0] > 1:
                        plt.plot(np.arange(oszicar.shape[0]), oszicar[:,0])
                    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
                    plt.savefig(os.path.join(path,'energy.png'))
                else:
                    print("Error!")

    def sort(self,dataset):
        from os.path import split
        def title(task):
            print('\n+'+'-'*(len(task)+12)+'+')
            print('| Property: %s |' %task)
            print('+'+'-'*(len(task)+12)+'+')

        for task in dataset.keys():
            if dataset[task].dtypes != object: 
                title(task)
                for i,v in zip(np.argsort(dataset[task]),np.sort(dataset[task])):
                    print(' %s, %s' %(split(self.stdin[i])[-1],v))

    def form(self,dataset):
        from jamip.utils.views import shellform
        # init lists %
        #dataset = dataset.to_dict(orient='records')
        shellform(dataset)

    def hdf5(self):
        import h5py

        if os.path.exists('info.hdf5'):
            with h5py.File("info.hdf5", 'r') as h5:
                #print(list(h5.keys()))
                h5.visititems(print)
