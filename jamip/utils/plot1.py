import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from jamip.analysis.vasp import *
from jamip.analysis.vasp import ProFinder, BandFinder, DosFinder, OpticsFinder, PhononFinder
from jamip.analysis.base import FinderSet
from dataclasses import dataclass
import numpy as np
import threading
import pathlib

@dataclass
class pltvars:
    """
    global variables
    emin, emax: energy range
    limit: energy range for dos plot
    scissor: energy shift for band/dos above fermi level
    title: title of the plot
    xlabel: xlabel of the plot
    ylabel: ylabel of the plot
    """
    emin: float=0.0
    emax: float=0.0
    limit: float=0.0
    scissor: float=0.0
    title: str=''
    xlabel: str=''
    ylabel: str=''

class globalvar:
    """
    global variables for different plot types
    """
    band   = pltvars(emin=-2, emax=4, ylabel='E (eV)')
    dos    = pltvars(emin=-1, emax=3, limit=0.04, xlabel='Energy (eV)', ylabel=r'$PDOS\ (states/eV/\AA^{3})$')
    absorb = pltvars(emin=0,  emax=5, xlabel='Energy (eV)', ylabel=r'$Absorb (cm^{-1}$)')
    tdm    = pltvars(ylabel=r'$TDM\ P2^2(Debye^2)$')
    diel   = pltvars(emin=0,  emax=5, xlabel='Energy (eV)', ylabel='$Arb.Unit$')
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = object.__new__(cls)  
        return cls._instance

class globalcmap:
    """
    global cmaps
    """
    cmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', "#31d3e6"]
    _instance = None
    _instance_lock = threading.Lock()
    #CMAP = ['g','b','r','y','m','orange','c','cyan','yellow','violet','brown',\
    #        'lime','deepskyblue','gold','darkorchid','greenyellow','r']

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = object.__new__(cls)  
        return cls._instance
    
    @classmethod
    def cycle(cls):
        from itertools import cycle
        return cycle(cls.cmap)


JOBLIST = {'band': 'band', 'fatband':'band', 'hseband':'band',
           'unfolding':'band', 'phband': 'band', 'gruneisen': 'band',
           'cutoff_conv':'base', 'kpoints_conv':'base', 'softmode':'base',
           'absorb':'absorb', 'refrace':'absorb', 'dielectric':'absorb', 'shg': 'absorb',
           'dos': 'dos', 'cohp':'dos', 'phdos': 'dos', 'spdos': 'dos',
           'tdm': 'tdm'}
__all__ = ['globalvar', 'globalcmap', 'Plot', 'VaspPlot', 'QEPlot', 'JOBLIST']

class Plot(FinderSet):

    def __init__(self, path=None, soft:str='vasp', depth=1, recursion=False, **kwargs):
        self.plotter = None
        self.savedir = None
        self.figtype = '.png'
        self.depth = depth
        self.soft = soft
        self.task = None
        self.recursion = recursion
        if path != None:
            self.stdin = path
        self.fig_kw = {} 

    @property
    def soft(self):
        return self.plotter.soft if self.plotter != None else None

    @soft.setter
    def soft(self, value:str):
        if value.lower() == 'vasp':
            self.plotter = VaspPlot(self)
        elif value.lower() == 'qe':
            self.plotter = QEPlot(self)
        elif value.lower() == 'cp2k':
            self.plotter = CP2KPlot(self)
        else:
            raise ValueError('Unknown soft.')

    def set_figure(self, jobs, width_ratios, height_ratios, **kwargs):
        """
        jobs: [[job1, job2], [job3, ...], ...]
        width_ratios/height_ratios: [w1, w2, ...], [h1, h2, ...]
        """
        jobs = np.array(jobs)
        assert len(jobs.shape) == 2, jobs.shape
        assert jobs.shape[0] == len(height_ratios)
        assert jobs.shape[1] == len(width_ratios)
        joblist = []
        for job in jobs.ravel():
            if job in JOBLIST.keys():
                joblist.append(job)
        key = tuple(np.unique(joblist))
        kwargs['height_ratios'] = height_ratios
        kwargs['width_ratios'] = width_ratios
        self.fig_kw[key] = kwargs
        self.plotter.set_axes(jobs, **kwargs)

    def plots(self,*args,**kwargs):
        '''
        main plot function
        '''
        jobs = []
        for job in args:
            if job in JOBLIST.keys():
                jobs.append(job)

        # job number %
        if len(jobs) == 0:
            raise KeyError("Invalid input job name.") 
        # path number %
        if len(self.stdin) == 0:
            raise IOError('No valid path for %s-plot' %job)
        # create plot directory %
        jobname = ''.join(jobs)
        if len(self.stdin) > 1:
            self.savedir = pathlib.Path(jobname+'plot')
            if not self.savedir.exists():
                self.savedir.mkdir()

        # load fig kwargs
        key = tuple(np.unique(jobs))
        fig_kw = {}
        if key in self.fig_kw:
            fig_kw.update(self.fig_kw[key])

        # group style %
        basestyle = pathlib.Path.home()/'.jamip'/'viewer'/'base.mplstyle'
        success = 0
        for path in self.stdin:

            plt.style.use(basestyle)
            self.plotter.set_axes(jobs, **fig_kw)

            kwargs['fname'] = None
            for job in jobs:
                self.plot(job, path, **kwargs)

            if len(self.stdin) > 1:
                self.save(fname=path.absolute().stem+self.figtype)
            else:
                self.save(fname=jobname+self.figtype)
            success += 1
        print('Success plot : %d' %success)
        del self.plotter.axes

    def multiplot(self, job:str, **kwargs):
        '''
        continue plot job in one subplot
        '''
        # job number %
        if job not in JOBLIST.keys():
            raise KeyError("Invalid input job name.") 
        # path number %
        if len(self.stdin) == 0:
            raise IOError('No valid path for %s-plot' %job)
        
        # load fig kwargs
        key = job,
        fig_kw = {}
        if key in self.fig_kw:
            fig_kw.update(self.fig_kw[key])

        # group style %
        basestyle = pathlib.Path.home()/'.jamip'/'viewer'/'base.mplstyle'
        plt.style.use(basestyle)
        self.plotter.set_axes([job], **fig_kw)
        success = 0
        for path in self.stdin:
            kwargs['fname'] = None
            self.plot(job, path, **kwargs)
            success += 1

        self.save(fname='%ss.png')
        print('Success plot : %d' %success)
        del self.plotter.axes


    def plot(self, job, path, **kwargs):

        #print(job)
        #try:
        if True: 
            if job == 'band':
                self.plotter.plot_band(path,**kwargs)
            elif job == 'fatband':
                self.plotter.plot_fat_band(path,**kwargs) 
            elif job == 'dos':
                self.plotter.plot_dos(path,**kwargs)
            elif job == 'spdos':
                self.plotter.plot_single_point_dos(path,**kwargs)
            elif job == 'tdm':
                self.plotter.plot_tdm(path,**kwargs)
            elif job == 'absorb':
                self.plotter.plot_absorb(path,**kwargs)
            elif job == 'refract':
                self.plotter.plot_absorb(path,ptype='refract',**kwargs)
            elif job == 'dielectric':
                self.plotter.plot_dielfunc(path,**kwargs)
            elif job == 'unfolding':
                self.plotter.plot_unfolding(path,**kwargs)
            elif job == 'cohp':
                self.plotter.plot_cohp(path,**kwargs)
            elif job == 'shg':
                self.plotter.plot_shg(path,**kwargs)
            elif job == 'hseband':
                self.plotter.plot_hse_band(path,**kwargs)
            elif job == 'cutoff_conv':
                self.plotter.plot_converge(path,'cutoff',**kwargs)
            elif job == 'kpoints_conv':
                self.plotter.plot_converge(path,'kpoints',**kwargs)
            elif job == 'phband':
                self.plotter.phband(path,**kwargs)
            elif job == 'phdos':
                self.plotter.phdos(path,**kwargs)
            elif job == 'gruneisen':
                self.plotter.gruneisen(path,**kwargs)
            elif job == 'softmode':
                self.plotter.softmode(path,**kwargs)
            else:
                print("Invalid jobname: %s" %job)
        #except:
            # warnings.warn()
            #print('job %s failed in %s' %(job, path))
        
    def save(self, fname=None, **kwargs):
        '''
        get figure name and save figure
        1. if input fname, named by fname
        2. if plot single figure, named by jobs
        3. if plot mutliple figure, named by entry dir.name        
        '''

        if fname != None:        
            if self.savedir != None:
                fname = self.savedir/fname
                
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

class BasePlot:
    
    @property
    def axes(self):
        return self._axes
    
    @axes.setter
    def axes(self,axes):
        self._axes = axes

    @axes.deleter
    def axes(self):
        del self._axes

    def set_axes(self, jobs:list, figsize=None, sharey=True, **kwargs):

        self.axes = None
        if 'width_ratios' in kwargs and 'height_ratios' in kwargs:
            if isinstance(jobs[0], str): jobs = [jobs]
            fig, axes = plt.subplot_mosaic(jobs, figsize=figsize, sharey=sharey, **kwargs)
            self.axes = axes
        elif len(jobs) > 1:
            rows, height, width = self.fast_mosaic(jobs)
            figsize = kwargs.get('figsize', (10*sum(width),10*sum(height)))
            fig, axes = plt.subplot_mosaic(rows, width_ratios=width, height_ratios=height, 
                                           figsize=figsize, sharey=sharey, **kwargs)
                                           #,wspace=wspace,hspace=hspace)
            self.axes = axes
        elif figsize != None:
            fig, axes = plt.subplots(figsize=figsize, **kwargs)
            self.axes = {jobs[0]: axes}

    @classmethod
    def fast_mosaic(self, jobs):
        '''plot nx2 subplot '''
        rows, row1, row2 = [], [], []
        height, width = [1], []
        for job in jobs:
            if JOBLIST[job] in ['band','base','absorb']:
                width.append(1)
                row1.append(job) 
            elif JOBLIST[job] == 'dos':
                width.append(0.4)
                row1.append(job) 
        rows.append(row1)

        for job in jobs:
            if JOBLIST[job] == 'tdm':
                if len(row2) == 0:
                    row2 = ['.'] * len(row1)
                for i,r in enumerate(row1):
                    if r == 1:
                        row2[i] = job
                        break
                else:
                    row2[0] = job
                height.append(0.3)
        if len(row2) > 0:
            rows.append(row2)

        return rows, height, width

    def set_style(self, job:str):
        """
        set matplotlib style file and activate subplot   
        """
        mplstyle = lambda job: pathlib.Path.home()/'.jamip'/'viewer'/(job+'.mplstyle')

        # stylefile %
        if mplstyle(job).exists():
            plt.style.use(mplstyle(job))
        elif mplstyle(JOBLIST[job]).exists():
            plt.style.use(mplstyle(JOBLIST[job]))
        else:
            plt.style.use(mplstyle('base'))

        if self.axes != None:
            return plt.subplot(self.axes[job].get_subplotspec())
        else:
            return plt.gcf()

    def _plot_swap(self, plt, swap_axes=False, swap_figure=False, **kwargs):
  
        if swap_axes and swap_figure:
            fig = plt.gcf()
            height,width = fig.get_figheight(),fig.get_figwidth()
            fig.set_figwidth(height)
            fig.set_figheight(width)

        if 'xlabel' in kwargs:
            if swap_axes:
                plt.ylabel(kwargs['xlabel'])
            else:
                plt.xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            if swap_axes:
                plt.xlabel(kwargs['ylabel'])
            else:
                plt.ylabel(kwargs['ylabel'])
        if 'xlim' in kwargs:
            if swap_axes:
                plt.ylim(*kwargs['xlim'])
            else:
                plt.xlim(*kwargs['xlim'])
        if 'ylim' in kwargs:
            if swap_axes:
                plt.xlim(*kwargs['ylim'])
            else:
                plt.ylim(*kwargs['ylim'])
            

class VaspPlot(BasePlot):

    soft = 'vasp'

    def __init__(self, builder=None):
        self.builder = builder
        self.axes = None

    def _plot_bandgap(self,plt,xkpt:np.ndarray,cv:dict,scissor=0):
        '''
        plot bandgap for band plot
        '''
        xvbm = xkpt[cv['vbm'].ikpt]
        xcbm = xkpt[cv['cbm'].ikpt]
        xtol = xkpt.max()/100
        Ecbm = cv['cbm'].energy-cv['vbm'].energy+scissor
        # vbm point
        plt.scatter([xvbm],[0],color='r',s=40)
        # cbm point
        # plt.plot([max(0,xcbm-xtol),min(xkpt[-1],xcbm+xtol)],[Ecbm,Ecbm],color='r') 
        plt.plot([xcbm],[Ecbm],'_',color='r')
        # Connection between cbm and vbm
        plt.plot([xcbm,xcbm],[0,Ecbm],c='r',lw=2)
        # Label the band gap value
        xtext = xcbm+xtol if xcbm<xkpt[-1]*2/3 else xcbm-xkpt[-1]/6
        plt.text(xtext,Ecbm/2,'{:.3f} eV'.format(Ecbm))

    def _plot_legend(self, labels, **kwargs):
        from itertools import cycle
        
        if 'color' in kwargs:
            color = cycle(kwargs['color'])
        else:
            color = globalcmap.cycle()
        # create colormap and legend %
        colormap = []
        linemap = []
        for i, label in enumerate(labels):
            c = next(color)
            colormap.append(c)
            pl,=plt.plot([0,1],[0,1],c=c,label=label)
            linemap.append(pl)
        colormap = np.array(colormap)
        if kwargs.get('legend',True):
            plt.legend(loc=1, fontsize='large', framealpha=0.9)
        for line in linemap:
            line.remove()
        
        return colormap
        
    def plot_band(self,path,kpath=None,source='eigenval',fname='band.png',**kwargs):
        '''
        plot band-figure base on OUTCAR
        kwargs:
            bandgap: whether to label band-gap value
        '''
        # Initializes the data retrieval module & pyplot & colormap %
        fig = self.set_style('band')
        g = globalvar.band
        color = globalcmap.cycle()

        # grep main datas %
        bf = BandFinder(path).get_data(source=source)
        if kpath != None:
            bf.regroup(kpath).remove_duplicates()
        else:
            bf.remove_duplicates()
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # add bandgap %
        cv = bf.get_cbmvbm()
        if bf.metal:
            shift = bf.fermi
        else:
            shift = cv['vbm'].energy
            # plot bandgap %
            if kwargs.get('bandgap',True):
                self._plot_bandgap(plt,xkpt,cv,g.scissor)
            print(shift, g.scissor)

        for bands_ispin in bf.bands:
            ybands = bands_ispin[...,0] - shift
            if g.scissor != 0:
                ybands[:,cv['cbm'].iband:] += g.scissor
            plt.plot(xkpt,ybands,c=next(color))

        plt.axhline(c='r', linestyle='--')
        plt.xlim(0,xkpt[-1])
        plt.ylim(g.emin,g.emax)
        plt.ylabel(g.ylabel)        
        if fname != None: self.builder.save(fname)
        return plt

    def plot_fat_band(self,
                      path,
                      ptype='base',
                      proj='emax',
                      kpath=None, 
                      linestype='-', 
                      alpha=0.75,
                      max_z:int=1,
                      interpolation:int=2,
                      source='procar',
                      fname='fatband.png',
                      **kwargs):
        '''
        plot band-figure base on PROCAR
        kwargs:
            path: Calculation directory.
            ptype: Default ploting type. 'emax'/'lmax'/'mmax'
            proj: Custom ploting type. [[element/atom-index,],[orbit,],label]
              e.g. [[['Al','Ga'],['px','py','pz']],
                    [['N'],['s']],
                    [[129,],['dz2']]]
            or:
              e.g. [[['Al','Ga'],['px','py','pz'],'Al-Ga-p'],
                    [['Al','Ga'],['s'],'Al-Ga-s'],
                    [[129,],['dz2'],'Al-Ga-dz2'],'Ga-129-dz2']]

            kpath: regroup hym-kpoints path. likes [[Gamma,M,K,Gamma,A]]
            legend: whether to plot legend.
            bandgap: whether to plot bandgap.
            interpolation: Interpolate the calculated K point data.
            fname: figure name. when the fname is None, the figure will not be saved.            
        '''
        from matplotlib.collections import LineCollection
        from scipy.interpolate import interp1d
        fig = self.set_style('fatband')
        g = globalvar.band 

        # grep main datas %        
        pf = ProFinder(path).get_data(source=source)
        if kpath != None:
            pf = pf.regroup(kpath).remove_duplicates()
        else:
            pf = pf.remove_duplicates()
        bands = pf.bands
        procar, labels = pf.projection(proj)
        xkpt = pf.get_xkpt()
        xticks, xlabels = pf.get_xticks()

        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)  

        # add bandgap %
        cv = pf.get_cbmvbm()
        if pf.metal:
            shift = pf.fermi     
        else:
            shift = cv['vbm'].energy
            # plot bandgap %
            if kwargs.get('bandgap', True):
                self._plot_bandgap(plt,xkpt,cv,g.scissor)
        
        # data interpolation initialize %
        if interpolation:
            newkpt = np.linspace(xkpt.min(),xkpt.max(),int(len(xkpt)*interpolation))
            f1 = interp1d(xkpt, bands, kind='linear', axis=1)
            f2 = interp1d(xkpt, procar, kind='linear', axis=1)
            xkpt = newkpt
            bands = f1(xkpt)
            procar = f2(xkpt) * plt.rcParams['lines.markersize'] 
        else:
            bands = bands
            procar = procar * plt.rcParams['lines.markersize'] 

        if ptype == 'base':
            procar_index = np.flip(np.argsort(procar, axis=-1), axis=-1)
            sorted_procar = np.flip(np.sort(procar, axis=-1), axis=-1)

            # plot legend and get colormap %
            if isinstance(proj, str):
                unids = np.unique(procar_index[...,:max_z])
                labels = [labels[i] for i in unids]
                cmap = self._plot_legend(labels, **kwargs)

                colormap = ['k'] * (max(unids)+1)
                for i,j in enumerate(unids):
                    i = i % len(cmap)
                    colormap[j] = cmap[i]
                colormap = np.array(colormap)
            else:
                colormap = self._plot_legend(labels, **kwargs)
            
            # plot band.png %
            for i,bands_ispin in enumerate(bands):
                ybands = bands_ispin[...,0] - shift
                ybands[:,cv['cbm'].iband:] += g.scissor

                # plot %
                for nband in range(len(ybands[0])):
                    if ybands[:,nband].min() > g.emax or ybands[:,nband].max() < g.emin: continue
                    for j in range(max_z):
                        if max_z == 1:
                            plt.plot(xkpt,ybands[:,nband],c='grey',lw=1, alpha=0.5)
                            try:
                                plt.scatter(xkpt,ybands[:,nband],s=sorted_procar[i,:,nband,j],color=colormap[procar_index[i,:,nband,j]])
                            except:
                                print(i,j,nband)
                                raise ValueError("max_z must be less than procar_index.")
                        else:
                            # plt.scatter(xkpt,ybands[:,nband],s=sorted_procar[i,:,nband,j],marker='o',color='none',
                            #        edgecolors=colormap[procar_index[i,:,nband,j]])
                            # plt.scatter(xkpt,ybands[:,nband],s=np.sum(sorted_procar[i,:,nband,j:], axis=-1),marker='o',
                            #             color=colormap[procar_index[i,:,nband,j]], alpha=0.8)
                            plt.scatter(xkpt,ybands[:,nband],s=np.sum(sorted_procar[i,:,nband,j:], axis=-1)**2,marker='o',
                                        c=colormap[procar_index[i,:,nband,j]], alpha=1)                                                        
                            # points = np.array([xkpt,ybands[:,nband]]).T.reshape(-1, 1, 2)
                            # weights = np.sum(sorted_procar[i,:,nband,j:], axis=-1)
                            # colors = colormap[procar_index[i,:,nband,j]]
                            # segments = np.concatenate([points[:-1], points[1:]], axis=1)
                            # lc = LineCollection(segments, linewidths=weights, color=colors)
                            # ax.add_collection(lc)

        elif ptype in ['rg','rb','gb','rgb']:
            # filled 
            if len(labels) == 2:
                if ptype == 'rb':
                    procar = np.concatenate((procar[...,:1],np.zeros_like(procar[...,:1]),procar[...,1:]),axis=-1)
                elif ptype == 'gb':
                    procar = np.concatenate((np.zeros_like(procar[...,:1]),procar),axis=-1)
                elif ptype == 'rg':
                    procar = np.concatenate((procar,np.zeros_like(procar[...,:1])),axis=-1)
                else:
                    raise ValueError("ptype must be rb/gb/rg for two labels.")
            elif len(labels) != 3:
                raise ValueError("proj must be two or three labels.")
        
            # legend
            if kwargs.get('legend',True):                
                self._plot_legend(labels, color=ptype, **kwargs)

            # plot rgb type figure
            for i,bands_ispin in enumerate(bands):
                ybands = bands_ispin[...,0] - shift
                ybands[:,cv['cbm'].iband:] += g.scissor

                for iband in range(ybands.shape[1]):
                    segments = [np.c_[xkpt,ybands[:,iband]]]
                    rgbs = procar[i,:,iband] 
                    rgbsum = np.sum(rgbs, axis=-1)[:,None]
                    rgbs = np.divide(rgbs, rgbsum, out=np.zeros_like(rgbs), where=rgbsum!=0) * np.array([1, 0.5, 1])
                    rgbs = [tuple(list(v)+[alpha]) for v in rgbs][1:]
                    lc = LineCollection(segments, colors=rgbs)
                    ax.add_collection(lc)

        plt.axhline(color='r',linestyle='--')
        plt.xlim(0,xkpt[-1])
        plt.ylim(g.emin,g.emax)
        plt.ylabel(g.ylabel)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_band_transitions(self,path,search_energies,dE=0.1,plot_tdm=False,
                              ptype='band',source='procar',fname='transition.png',**kwargs):
        '''
        plot band transitions base on PROCAR
        search_energies: list of search energies
        dE: transition energy range
        ptype: 'proj' or 'band'
        plot_legend: True or False
        fname: output file name
        kwargs:
            label: label
            legend: True or False
        '''                  
        if ptype == 'proj':
            plt=self.plot_fat_band(path,fname=None,ptype='rgb', legend=False,**kwargs)
        else:
            plt=self.plot_band(path,plot_bandgap=False,fname=None, legend=False,**kwargs)

        # if plot_tdm:
        #     from jamip.analysis.vasp.wavecar import GrepWavecar
        #     w = GrepWavecar(path)
        #     w.wavecar()

        color = globalcmap.cycle()            
        bf = BandFinder(path).get_data(source=source).remove_duplicates()
        all_transitions = bf.get_all_transitions()
        xkpt = bf.get_xkpt()

        # find band edge and get shift        
        cv = bf.get_cbmvbm()
        shift = bf.get_fermi() if bf.metal else cv['vbm'].energy
        
        labels = ['%s ev' %energy for energy in search_energies]
        colormap = self._plot_legend(labels, **kwargs)

        linemap=[]
        for i,search_energy in enumerate(search_energies):            
            for ispin,transitions in enumerate(all_transitions):
                for ikpt,vb,cb in np.argwhere(np.abs(transitions-search_energy)<dE):
                    kpt = xkpt[ikpt]                       # x coords
                    vbE = bf.bands[ispin,ikpt,vb,0]-shift  # y coords
                    deltaE = transitions[ikpt,vb,cb]       # height 
                    plt.arrow(kpt,vbE,0,deltaE,color=colormap[i])
                    xtext = kpt+xkpt[-1]/100 if kpt<xkpt[-1]*2/3 else kpt-xkpt[-1]/6
                    plt.annotate('{:.3f}'.format(deltaE),(xtext,vbE+deltaE*2/3))
                    # if plot_tdm:
                    #     tdm = w.dipole([ispin,ikpt,vb],[ispin,ikpt,trans['cb']])
                    #     plt.annotate('{:.3f}'.format(tdm),(xtext,vbE+tE/3))

        if kwargs.get('legend',True): 
            plt.legend(loc=1, fontsize='large', framealpha=0.9)
        if fname != None: self.builder.save(fname)
        return plt
        
    def plot_single_point_dos(self, path, kpoint=None, proj='lmax', nedos:int=601, sigma:float=0.02,
                              swap_axes:bool=False, norm:bool=False, source='procar', fname='spdos.png', **kwargs):
        '''
        plot single points dos base on PROCAR
        '''
        fig = self.set_style('dos')
        g = globalvar.dos
        pf = ProFinder(path).get_data()
        color = globalcmap.cycle()

        # search kpoint %
        if kpoint == None: 
            kpoint = np.zeros((1,3))
        else:
            kpoint = np.array(kpoint).reshape(1,3)

        # kpoint position -> ikpt %
        d = np.sum(np.abs(pf.kpoints - kpoint), axis=1)
        if np.min(d) > 1e-4 :
            raise ValueError("Search Kpoint %s Failed." %kpoint)
        ikpt = np.argmin(d)
        energy,procar,labels=pf.single_point(proj, ikpt, nedos=nedos, sigma=sigma)

        # extra: shift & cutoff
        cv = pf.get_cbmvbm()
        shift = pf.fermi if pf.metal else cv['cbm'].energy
        energy -= shift
        imin,imax = sum(energy<g.emin),sum(energy<g.emax)
        energy = energy[imin:imax]
        width = (energy[-1] - energy[0]) / len(energy)
        procar = procar[:,imin:imax,...] 
            
        for i,bands_ispin in enumerate(pf.bands):
            # yp = np.cumsum(procar[i,:,0], axis=-1)
            bottom = np.zeros_like(energy)
            ysum = np.sum(procar[i,:,0],axis=1) if norm else 1 
            cutoff = np.where(ysum < ysum.max()*0.01)
            procar[:,cutoff] = 0
            # plot %
            for j,label in enumerate(labels):
                # percentage
                yp = procar[i,:,0,j] / ysum
                # swap_axes %
                if swap_axes:
                    plt.barh(energy,yp,height=width,left=bottom,label=label)
                else:
                    plt.bar(energy,yp,width=width,bottom=bottom,label=label)
                bottom += yp

        plt.legend(loc=2)
        # swap_axes %
        self._plot_swap(plt, xlabel=g.ylabel, xlim=(energy[0],energy[-1]),swap_axes=swap_axes,swap_figure=True)
        if fname != None: self.builder.save(fname)
        return plt
    
    def plot_dos(self, path, proj='edos', cutoff:float=0.003, swap_axes:bool=False,
                 tdos:bool=False, pdos:bool=True, source='doscar', fname='dos.png',**kwargs):
        '''
        plot dos-figure base on DOSCAR

        params:
            path: data source, a density of states calculation directory for VASP or JAMIP  
            proj: Plot type, {'edos', 'ldos', 'mdos'}, DOS summation of element, L and M orbitals respectively
            cutoff: the cutoff value of DOS. curves below this value will not be shown in the figure.
            tdos: Whether to plot the total DOS.
            pdos: Whether to plot the projected DOS.
            fname: figure name. when the fname is None, the figure will not be saved.            
        '''
        fig = self.set_style('dos')
        g = globalvar.dos

        # initialize dos data %         
        df = DosFinder(path).get_data(source=source).per_volume()
        dos_energy = df.energy
        # get shift %
        bf = BandFinder(DosFinder(path).banddir).get_data()
        cv = bf.get_cbmvbm()
        print(bf.metal)
        if bf.metal:
            dos_energy -= bf.fermi
        else:
            print(cv['vbm'].energy, g.scissor)
            dos_energy -= cv['vbm'].energy 
            if g.scissor:
                # get first inflection point above fermi
                zero_point = np.argmax((dos_energy[:-1]>0) & (np.diff(df.tdos[0])>0))
                dos_energy[zero_point:] += g.scissor
                
        imin = max(sum(dos_energy<g.emin)-1,0)
        imax = min(sum(dos_energy<g.emax)+1,len(dos_energy))
        dos_energy = dos_energy[imin:imax]
        if isinstance(df.pdos, np.ndarray):
            proj_dos, labels = df.projection(proj)
            proj_dos = proj_dos[...,imin:imax]
            total_dos = df.tdos[...,imin:imax]
        else:
            total_dos = df.tdos[...,imin:imax]
            pdos = False
            tdos = True

        # plot %
        if tdos:
            if len(total_dos) == 1:
                tdos_labels = {'total': 1}
            elif len(total_dos) == 2:
                tdos_labels = {'spin_up': 1, 'spin_down': -1}

            for i,label in enumerate(labels):
                dos_spin = total_dos[i] * tdos_labels[label]
                x,y = (dos_energy, dos_spin) if not swap_axes else (dos_spin, dos_energy)
                plt.plot(x,y,label=label,c='k')

        if pdos:
            color = globalcmap.cycle()
            for dos,label in zip(proj_dos,labels):
                if isinstance(proj,str) and max(np.abs(dos)) < cutoff: continue
                x,y = (dos_energy, dos) if not swap_axes else (dos, dos_energy)
                co = next(color)
                plt.plot(x,y,label=label,c=co)

        # spin & limit
        limit = g.limit
        _limit = -limit if df.spin == 2 else 0
        print(swap_axes)
        self._plot_swap(plt, xlim=(g.emin,g.emax),ylim=(_limit,limit),swap_axes=swap_axes)

        # swap_axes %
        if not swap_axes:
            plt.axvline(c='r',linestyle='--')
            if df.spin == 2:
                plt.axhline(linewidth=1,color='black',linestyle='--')
            plt.xlabel(g.xlabel)
            plt.ylabel(g.ylabel)
        else:
            plt.axhline(c='r',linestyle='--')
            if df.spin == 2:
                plt.axvline(linewidth=1,color='black',linestyle='--')
            ax = plt.gca()
            xticks = ax.get_xticks()[1:]
            plt.xticks(xticks)
            ax.yaxis.tick_right()

        # labels -> ncol
        ncol = kwargs.get('ncol',1)
        plt.legend(loc=1, ncol=ncol)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_absorb(self,path, ptype='absorb',source='xml',directions=None,fname='absorb.png'):
        '''
        plot optics-figure base on vasprun.xml
        '''
        fig = self.set_style('absorb')
        color = globalcmap.cycle()
        g = globalvar.absorb

        of = OpticsFinder(path).get_data(source=source)
        if ptype == 'absorb':
            # cm-1
            data = of.absorb()
        elif ptype == 'refract':
            data = of.refract()
        elif ptype == 'reflect':
            data = of.reflect()
        else:
            raise ValueError('Optional job include absorb and refract.')

        # plot directions 
        if directions == None:
            if ptype == 'absorb':
                plt.axes(yscale='log')
                plt.ylim(1e-0,1e7)
            plt.plot(of.energy,np.mean(data,axis=1),c='b')

        else:                
            for direction in directions:
                tmp = []
                for i,d in enumerate('xyz'):
                    if d in direction:
                        tmp.append(data[:,i])
                plt.plot(of.energy,np.mean(tmp,axis=0),c=next(color),label=direction)    
            plt.ylim(bottom=0)
            plt.legend()

        plt.xlim(g.emin,g.emax)
        plt.xlabel(g.xlabel)
        plt.ylabel(g.ylabel)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_polarization(self,path, source='xml',directions=('x','y'),fname='polarization.png'):
        '''
        plot optics-figure base on vasprun.xml
        '''
        fig = self.set_style('absorb')
        color = globalcmap.cycle()
        g = globalvar.absorb

        of = OpticsFinder(path).get_data(source=source)
        # cm-1
        data = of.absorb()
        maps = {'x': data[:,0], 'y': data[:,1], 'z': data[:,2],}
        dx = maps[directions['x']]
        dy = maps[directions['y']]
        polar = (dx - dy) / (dx + dy)
        plt.axhline(0,ls='--',color='k')
        plt.plot(of.energy, polar, c='r')
        
        plt.xlim(g.emin,g.emax)
        plt.ylim(-1,1)
        plt.xlabel('Energy (eV)')
        plt.ylabel('Degree of polarization')
        plt.figtext(0.9,0.9,'x',ha='center',va='center')
        plt.annotate('', xy=(0.9, 0.88), xycoords='figure fraction',
                     xytext=(0.9, 0.75), textcoords='figure fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=7))
        plt.figtext(0.9,0.2,'y',ha='center',va='center')
        plt.annotate('', xy=(0.9, 0.22), xycoords='figure fraction',
                     xytext=(0.9, 0.35), textcoords='figure fraction',
                     arrowprops=dict(facecolor='black', shrink=0.06, width=0.5, headwidth=5, headlength=7))
        plt.tight_layout()
        if fname != None: self.builder.save(fname)
        return plt

    def plot_dielfunc(self,path,ptype=['imag','real'],directions=['x','z'],source='xml',fname='dielfunc.png',**kwargs):
        '''
        plot optics-figure base on vasprun.xml
        '''
        fig = self.set_style('dielectric')
        color = globalcmap.cycle()
        g = globalvar.diel

        # data reshape %
        of = OpticsFinder(path).get_data(source=source)
        energy = of.energy
        imag = of.imag()
        real = of.real()

        if isinstance(ptype,str):  ptype = [ptype]

        # plot %
        for inc in ptype:
            value = imag if inc == 'imag' else real
            for direction in directions:
                data = []
                for i,d in enumerate(('x','y','z')):
                    if d in direction:
                        data.append(value[:,i])
                plt.plot(energy,np.mean(data,axis=0),c=next(color),label='%s-%s'%(inc,direction))

        plt.xlim(0, g.emax)
        plt.ylabel(g.ylabel)
        plt.xlabel(u'Energy (eV)')
        plt.legend(frameon=False)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_tdm_from_waveder(self, path, fname='tdm.png',**kwargs):
        '''
        plot tdm-figure base on WAVEDER
        '''
        from jamip.analysis.vasp.waveder import Waveder
        fig = self.set_style('tdm')
        g = globalvar.tdm

        # get kpath %
        bf = BandFinder(path).get_data()
        xkpt = np.arange(len(bf.kpoints))
        cbvbs = bf.get_cbvb()
        
        # set x axis % 
        #xticks = [0] + np.cumsum(bf.kpath.insert).tolist()
        #xticks[-1] -= 1
        #xticks = [xkpt[i] for i in xticks]
        #for i in xticks:
        #    plt.axvline(i,c='black')
        #ax = plt.gca()
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(kpath)

        # get dipole-mement from wavecar %
        cder, nodesn_i_dielectric_function, wplasmon = Waveder.from_file(path)
        for ispin,(cb,vb) in enumerate(cbvbs):
            #tdm = np.abs(cder[ispin,:,cb,vb])**2
            print(cder.shape, vb,cb)
            tdm = np.abs(cder[vb,cb,:,ispin,:])**2
            # nbands, nelect, nk, ispin, 3
            #cder_data = cder.reshape((3, ispin, nk, nelect, nbands)).T
            print(cb, vb, tdm.shape)
            print(xkpt.shape, np.sum(tdm, axis=-1).shape)
            plt.plot(xkpt,np.sum(tdm, axis=-1))
            for i,j in enumerate('xyz'):
                plt.plot(xkpt, tdm[:,i], label=j)
        plt.ylim(bottom=0)
        plt.xlim(0,xkpt[-1])
        plt.ylabel(g.ylabel)
        plt.legend()
        if fname != None: self.builder.save(fname)
        return plt

    def plot_tdm(self, path, kpath=None, source='eigenval',fname='tdm.png',**kwargs):
        '''
        plot tdm-figure base on WAVECAR
        '''
        fig = self.set_style('tdm')
        g = globalvar.tdm

        # get kpath %
        bf = BandFinder(path).get_data(source=source).remove_duplicates()
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # get dipole-mement from wavecar %
        bf = BandFinder(path).get_data_from_wavecar().get_tdm().remove_duplicates()
        assert len(xkpt) == bf.bands.shape[1], "K-point inconsistent, did you save wavecar in band calculation?"
        for ispin,tdm in enumerate(bf.tdms):
            plt.plot(xkpt,np.sum(tdm,axis=-1))
            for i,j in enumerate('xyz'):
                plt.plot(xkpt, tdm[:,i], label=j)
        plt.ylim(bottom=0)
        plt.xlim(0,xkpt[-1])
        plt.ylabel(g.ylabel)
        plt.legend()
        if fname != None: self.builder.save(fname)
        return plt

    def plot_cpd(self, path, kpath=None, source='eigenval',fname='cpd.png',**kwargs):
        '''
        plot cpdm-figure base on WAVECAR
        '''
        fig = self.set_style('tdm')
        g = globalvar.tdm

        # get kpath %
        bf = BandFinder(path).get_data(source=source).remove_duplicates()
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # get dipole-mement from wavecar %
        bf = BandFinder(path).get_data_from_wavecar().get_cpd().remove_duplicates()
        assert len(xkpt) == bf.bands.shape[1], "K-point inconsistent, did you save wavecar in band calculation?"
        for ispin,cpd in enumerate(bf.cpds):
            plt.plot(xkpt, cpd) #, label=j)
        #plt.ylim(bottom=0)
        plt.xlim(0,xkpt[-1])
        plt.ylabel(g.ylabel)
        plt.legend()
        if fname != None: self.builder.save(fname)
        return plt

    def plot_converge(self, path, ptype='cutoff', fname='energy.png',**kwargs):
        """
        """
        from jamip.analysis.vasp.outcar import GrepOutcar
        import re
  
        fig = self.set_style('base')
        relax = pathlib.Path(path) / 'relax'
        energys = []
        times = []
 
        for dir in relax.iterdir():
            value = re.match(r'%s-(\d+\.?\d*)' %ptype, dir.name)
            if value and (dir/'OUTCAR').exists():
                value = float(value.group())
                try:
                    energy = GrepOutcar().free_energy(dir)
                    energys.append([value, energy])
                except:
                    pass
                try:
                    cputime = GrepOutcar().cputime(dir)
                    times.append([value, cputime])
                except:
                    pass
                
        if len(times) == 0:
            raise Exception('Warning! %s converge test for %s are all unfinished. ')
        
        energys = np.array(energys, dtype=float)
        energys = energys[np.argsort(energys[:,0])]
        times = np.array(times, dtype=float)
        times = times[np.argsort(times[:,0])]
        
        # plot energys % 
        plt.xticks(energys[:,0])
        if ptype == 'cutoff':
            plt.xlabel('E$_{cutoff}$ (eV)')
        elif ptype == 'kpoints':
            plt.xlabel('$kspacing$')
                    
        ax1 = plt.gca()
        ax1.plot(energys[:,0], energys[:,1], c='b')
        ax1.set_ylabel('Energy (eV)')
        if fname != None: self.builder.save(fname)
        return plt    

    def plot_hse_band(self,path,proj=None,interpolation=None,fname='hse.png',**kwargs):
        """
        """
        from scipy.interpolate import interp1d

        path = pathlib.Path(path)
        if (path/'.status').exists():
            path = path/'electric'/'hse_band'
        if not (path/'KPATH.in').exists():
            raise OSError("HSE band calculation failed!")

        fig = self.set_style('band')
        g = globalvar.band
        # get main datas %
        if proj == None:
            bf = BandFinder(path).get_data()      
        else:
            bf = ProFinder(path).get_data()

        kpath,insert = BandFinder.read_kpath(path)
        bf.remove_grid(len(bf.kpoints) - np.sum(insert))
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # set x axis with insert % 
        # xticks = [0]
        # tmp = 0
        # for i in insert:
        #     tmp += i
        #     if i > 1:
        #         xticks.append(xkpt[tmp])
        #         plt.axvline(xkpt[tmp],c='black')

        # add bandgap %
        cv = bf.get_cbmvbm()
        if bf.metal:
            shift = bf.fermi
        else:
            shift = cv['vbm'].energy
            if kwargs.get('bandgap', True):
                self._plot_bandgap(plt,xkpt,cv,g.scissor)

        if interpolation:
            newkpt = np.linspace(xkpt.min(),xkpt.max(),int(len(xkpt)*interpolation))
            f1 = interp1d(xkpt, bf.bands, kind='linear', axis=1)
            xkpt = newkpt
            bands = f1(xkpt)
        else:
            bands = bf.bands

        if proj != None:            
            procar, labels = bf.projection(proj)
            procar = procar * plt.rcParams['lines.markersize']**2 * 0.8
            if interpolation:
                f2 = interp1d(xkpt, procar, kind='linear', axis=1)
                procar = f2(xkpt)
            # get sorted procar %
            procar_max_index = np.argmax(procar, axis=-1)
            procar_max_value = np.max(procar, axis=-1)

            # plot legend and get colormap %
            labels = [labels[i] for i in np.unique(procar_max_index)]
            colormap = self._plot_legend(labels, **kwargs)

            # plot band.png %
            for i,bands_ispin in enumerate(bands):
                ybands = bands_ispin[...,0] - shift
                ybands[:,cv['cbm'].iband:] += g.scissor

                for nband in range(len(ybands[0])):
                    if ybands[:,nband].min() > g.emax or ybands[:,nband].max() < g.emin: continue
                    # plt.plot(xkpt,ybands[:,nband],c='k')
                    plt.scatter(xkpt,ybands[:,nband],s=procar_max_value[i,:,nband],
                                color=colormap[procar_max_index[i,:,nband]])

        else:
            # plot band.png %
            for bands_ispin in bands:
                ybands = bands_ispin[...,0] - shift
                ybands[:,cv['cbm'].iband:] += g.scissor
                plt.plot(xkpt,ybands,c='b')

        plt.axhline(c='r', linestyle='--')
        plt.ylabel(g.ylabel)
        plt.ylim(g.emin,g.emax)
        plt.xlim(0,xkpt[-1])
        if fname != None: self.builder.save(fname)
        return plt

    def plot_unfolding(self,path,smear=False,proj=False,interpolation:int=3,fname='unfold.png',**kwargs):
        """
        plot band-unfolding base on POSCAR, WAVECAR
        kwargs:
            path: calculation directory
            primcell: primcell file
            dim: trans matrix from primcell to supercell
            ---
            smear: whether to plot with smear
            fname: save figure name
            cache: whether to cache plot data
        """
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.vasp.band import GrepKpath
        from jamip.analysis.vasp.wavecar import Wavecar
        from jamip.structure import read
        from jamip.analysis.vasp.band import kpath2list

        path = pathlib.Path(path)
        if (path/'.status').exists():
            path = path/'electric'/'unfolding'
        if not (path/'KPATH.in').exists() or not (path/'GPOINTS').exists():
            raise OSError('Lack of essential k-points files')

        fig = self.set_style('band')
        g = globalvar.band
        fermi = GrepOutcar().fermi_energy(path)

        # get primcell transformation matrix %
        matrix = None
        if 'primcell' in kwargs and 'dim' in kwargs:
            primcell = kwargs['primcell']
            dim = kwargs['dim']
            matrix = np.array(dim).astype(int)

        # procar
        #p           = procar()
        #atomic_whts = [p.get_pw(0)[:,kmap,:], p.get_pw("1:18")[:,kmap,:], p.get_pw("18:54")[:,kmap,:]]

        # wavecar
        w = Wavecar.from_file(path)
        if matrix != None: 
            w.M = matrix
        else:
            try:
                primcell = w.read_primcell()
                matrix = w.M
                cell = read(path/'POSCAR')
                delta = cell.lattice - np.dot(matrix, primcell.lattice)
                assert np.sqrt(np.sum(delta**2)) <= 0.5, "delta %.4f out of range!" %delta
            except:
                print("""Warning! No primcell input, following code may work ...
                    import spglib
                    primcell = spglib.find_primitive(cell)
                    matrix = np.rint(np.dot(cell[0],np.linalg.inv(primcell[0])), dtype=int)
                    plot_unfolding(path, primcell=primcell, dim=matrix)
                    """)
                
        K = w.read_unfolding()
        sw = w.spectral_weight(K)

        # kpath-in %
        kpath,insert = GrepKpath.read_kpath(path)
        kpath = kpath2list(kpath)
        # get xkpt %
        rec_cell = np.linalg.inv(primcell.lattice)
        delta = np.linalg.norm((K[1:]-K[:-1]) @ rec_cell, axis=1)
        delta = np.insert(delta,0,0)
        xkpt = np.cumsum(delta)
        disc = np.where(np.abs(np.diff(delta))>1e-4)[0]
        if len(disc) > 1:
            for i in range(1,len(disc)):
                if disc[i]-disc[i-1] == 1:
                    k0 = disc[i-1]
                    k1 = disc[i]
                    xkpt[k1:] -= (xkpt[k1]-xkpt[k0]+1e-8)

        # plot kpoint symbols %
        xticks = [0]
        tmp = 0
        for i in insert:
            tmp += i
            if i > 1:
                xticks.append(xkpt[tmp])
                plt.axvline(xkpt[tmp],c='black')

        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(kpath)
        plt.ylabel('Energy (eV)')
        plt.xlim(0,xkpt[-1])
        plt.ylim(g.emin,g.emax)

        # plot band.png %
        if smear:
            if interpolation:
                xkpt, e0, sf = w.spectral_function_with_interpolation(xkpt, interpolation, nedos=4000, sigma=0.02)
            else:
                e0, sf = w.spectral_function(nedos=4000,sigma=0.02)

            e0 -= fermi
            imin,imax = np.sum(e0 < g.emin) , np.sum(e0 < g.emax)
            e0 = e0[imin:imax]
            sf = sf[:,imin:imax]

            # datamap %
            # sfm = np.mean(sf) 
            # sf[np.where(sf > sfm*10)] = sfm*10
            # sf[np.where(sf > sfm/10)] = sfm/10 + np.log(sf[np.where(sf > sfm/10)]/sfm*10)
            X, Y = np.meshgrid(xkpt, e0)
            for i in range(w.wfc._nspin):
                ax.contourf(X, Y, sf[i], cmap='jet')
                #ax.pcolormesh(X, Y, sf[i], cmap='jet', shading='auto')

        else:
            scale = plt.rcParams['lines.markersize']**2 / 2 / np.max(sw[...,1])
            for i in range(w.wfc._nspin):
                for nb in range(w.wfc._nbands):
                    plt.scatter(xkpt,sw[i,:,nb,0]-fermi, s=sw[i,:,nb,1]*scale, c='b')    
        if fname != None: self.builder.save(fname)
        return plt

    def plot_cohp(self,path,ptype='coop',dtype='t',swap_axes=False,fname='cohp.png',**kwargs):
        """
        Plot cohp calculation result.

        Args:
            path (str): Path of cohp calculation.
            ptype (str): Type of plot.
            dtype (str): Type of orbit. 't,p,m'
            swap_axes (bool): Swap axes x & y.
            fname (str): Name of figure.
            kwargs: Other arguments.

        Returns:
            Figure.
        """
        from jamip.analysis.vasp.cohp import COHPFinder

        path = pathlib.Path(path)
        if (path/'.status').exists():
            path = path/'electric'/'cohp'
        if not (path/'lobsterin').exists():
            raise OSError('Cannot find cohp calculation!')
       
        g = globalvar.dos
        fig = self.set_style('cohp')
        
        if ptype.lower() == 'cohp' or ptype.lower() == 'coop':
            if ptype.lower() == 'coop':
                df = COHPFinder(path).read_coop(path, dtype=dtype)
            else:
                df = COHPFinder(path).read_cohp(path, dtype=dtype)
            energy = df['energy']
            for label in df.columns:
                if label == 'energy': continue
                if swap_axes:
                    plt.plot(-df[label],energy, label=label)
                    plt.fill_betweenx(energy, 0, -df[label], facecolor = 'cyan')
                else:
                    plt.plot(energy,-df[label], label=label)
                    plt.fill_between(energy, 0, -df[label], facecolor = 'cyan')
            plt.axvline(0,c='black')
            plt.axhline(0,c='black',linestyle='--')
            self._plot_swap(plt,swap_axes=swap_axes,swap_figure=True,xlabel="Energy (eV)",ylabel="-COHP (E)",xlim=(g.emin,g.emax))

        elif ptype.lower() == 'icohp':
            if ptype.lower() == 'icoop':
                df = COHPFinder(path).read_icoop(path, dtype=dtype)
            else:
                df = COHPFinder(path).read_icohp(path, dtype=dtype)
            energy = df['energy']
            for label in df.columns:
                if label == 'energy': continue
                x,y = (-df[label],energy) if swap_axes else (energy,-df[label])
                plt.plot(x,y,label=label)
            plt.axvline(0,c='black')
            plt.axhline(0,c='black',linestyle='--')
            self._plot_swap(plt,swap_axes=swap_axes,swap_figure=True,xlabel="Energy (eV)",ylabel="-ICOHP (E)",xlim=(g.emin,g.emax))

        plt.legend()
        if fname != None: self.builder.save(fname)
        return plt

    def plot_boltztrap(self, path, ptype='seebeck', cbvb:str='vb', fname='boltztrap.png',
                       cmin=1e18, cmax=1e21, **kwargs):
        '''
        Plot boltztrap calculation result.

        Args:
            path (str): Path of boltztrap calculation.
            ptype (str): Type of plot.
            cbvb (str): cb or vb.
            fname (str): Name of figure.
            kwargs: Other arguments.

        Returns:
            Figure.
        '''
        from jamip.analysis.vasp.boltztrap import BoltztrapFinder
        fig = self.set_style('base')

        # get data (pd.DataFrame) %
        bf = BoltztrapFinder(path).get_trace()
        imin, imax = bf.get_carrier(cbvb)        
        xdata = bf.data['N'][imin:imax]
        xdata = xdata if cbvb == 'vb' else -xdata
        # x: N, y: ptype
        for key in bf.data.columns:
            if ptype.lower() in key.lower():
                ydata = bf.data[key][imin:imax]
                break
        else:
            raise KeyError('ptype not found!')
            
        # axis
        plt.axes(xscale='log')
        plt.xlim(cmin,cmax)
        plt.plot(xdata,ydata,label=ptype)
        plt.ylabel(ptype)

        if fname != None: self.builder.save(fname)
        return plt


    def plot_shg(self,path,ptype='real',fname='shg.png',**kwargs):
        '''
        plot shg base on SHG_xxx
        path: calculation direction
        directions: filenames
        '''
        import re

        path = pathlib.Path(path)
        if (path/'.status').exists():
            path = path/'optic'/'shg'
        # search datafiles % 
        files = []
        if 'directions' in kwargs:
            for d in kwargs['directions']:
                subpath = path/'SHG_%s' %d
                if subpath.is_file():
                    files.append(subpath)
        else:
            if path.is_file() and re.match('SHG_[0-9]{3}', path.name):
                files.append(path)
            elif path.is_dir():
                for file in path.iterdir():
                    if re.match('SHG_[0-9]{3}',file.name):
                        files.append(file)
        if len(files) == 0:
            raise OSError('SHG datafile not exists!')

        fig = self.set_style('absorb')
        g = globalvar.absorb

        for file in files:
            with open(file,'r') as f:
                data = []
                for line in f:
                   data.append(line.split())
                data = np.array(data,dtype=float)
               
                plt.figure(figsize=(10,6))
                plt.plot(data[:,0],data[:,1],label='real')
                plt.plot(data[:,0],data[:,2],label='imag')
                plt.plot(data[:,0],data[:,3],label='abs')
                plt.xlim(0,5)
                pngname = file.name.split('.')[0] + '.png'
                self.save(pngname)


    def softmode(self,path,fname='softmode.png',**kwargs):
        '''
        plot softmode-figure base on OUTCARs
        '''
        fig = self.set_style('base')
        pf = PhononFinder(path)
        dat = []
        for key,value in pf.get_softmode_result().items():
            dat.append([key[4:],value])
       
        dat = np.array(dat,dtype=float)
        dat = dat[np.argsort(dat[:,0],axis=0)]
        dat[:,1] -= dat[0,1]
        yrange = np.max(dat[:,1]) - np.min(dat[:,1])
        magnitudes = np.floor(np.log10(yrange))
        mult = np.power(10,magnitudes)
        if yrange / mult < 5:
            mult /= 2
       
        plt.plot(dat[:,0]*0.5,dat[:,1])
        plt.xlim(0,3)
        plt.xlabel('amplitude')
        plt.ylabel('Energy (eV)')
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(mult))
        plt.grid(linestyle='-.')
        plt.title('softmode')
        if fname != None: plt.savefig(fname)
        return plt

    def phband(self, path, npoints=51, kpoints=None, fname='phband.png', **kwargs):
        '''
        plot phonon-spectrum-figure base on FORCE_SETS

        Args:
            kpoints: {'Kpoints': {'X': [0.5, 0, 0], 'M': [0.5, 0.5, 0]},
                      'Path': [['X','M']]}
        '''
        from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
        from jamip.utils.brillouin_zone import HighSymmetryKpath
        from jamip.structure.atomic_number import number
        from jamip.utils.convert import kpath2list

        fig = self.set_style('phband')

        # structure %
        pf = PhononFinder(path)
        try:
            phonon = pf.get_phonon()
            phonon = pf.set_force_constants(phonon)#, mode='FORCE_CONSTANTS')
            phonon.symmetrize_force_constants()
            unitcell = phonon.unitcell
        except:
            raise OSError("Function not support")

        # get kpath %
        if kpoints is None: 
            symbols = [number[i] for i in unitcell.symbols]
            cell = (unitcell.cell, unitcell.scaled_positions, symbols)
            bz = HighSymmetryKpath()
            kpoints = bz.get_HSKP(cell)
 
        paths = [[kpoints['Kpoints'][i] for i in p] for p in kpoints['Path']]
        kpath = kpoints['Path']

        qpoints, connections = get_band_qpoints_and_path_connections(paths, npoints=npoints)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=kpath)
       
        max_freq = max([np.max(fq) for fq in phonon._band_structure.frequencies])
        max_dist = phonon._band_structure.distances[-1][-1]
        xscale = max_freq / max_dist * 1.5
        distances_scaled = [d * xscale for d in phonon._band_structure.distances]
        spts = [p[0] for p in distances_scaled]
        spts.append(distances_scaled[-1][-1])
       
        #plt.figure(figsize=(6,6),dpi=144)
        plt.title('Phonon Spectrum', pad=0.1, fontsize=14)
        plt.ylabel('Frequency (THz)')
        plt.axhline(linestyle='--',linewidth=1.5,color='green')
        plt.xlim(0,spts[-1])
        ax = plt.gca()
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
       
        ax.set_xticks(spts)
        ax.set_xticklabels(kpath2list(kpath))
       
        for i, (d, f, c) in enumerate(zip(distances_scaled, phonon._band_structure.frequencies, phonon._band_structure.path_connections)):
            curves = plt.plot(d, f, c='r')
            plt.axvline(d[0], linewidth=1, c='black', alpha=0.5)
        if fname != None: plt.savefig(fname)
        return plt

    def phdos(self, path, project=False, mesh=[10,10,10], swap_axes=False, fname='phdos.png', **kwargs):
        '''
        plot phonon-dos-figure base on FORCE_SETS
        '''
        from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

        fig = self.set_style('phdos')
        g = globalvar.dos

        # structure %
        pf = PhononFinder(path)
        try:
            phonon = pf.get_phonon()
            phonon = pf.set_force_constants(phonon)
            unitcell = phonon.unitcell
        except:
            raise OSError("Function not support")

        # set mesh %
        phonon.run_mesh(mesh=mesh,is_mesh_symmetry=False,with_eigenvectors=True)
        phonon.run_projected_dos(xyz_projection=project)
       
        # get range %
        energy = phonon._pdos.frequency_points 
        edos = {}
        if len(unitcell.symbols) == len(phonon._pdos.partial_dos):
            for elm,dos in zip(unitcell.symbols,phonon._pdos.partial_dos):
                if elm not in edos:
                    edos[elm] = []
                edos[elm].append(dos)
       
        elif len(unitcell.symbols)*3 == len(phonon._pdos.partial_dos):
            for i, elm in enumerate(unitcell.symbols):
                if elm not in edos:
                    edos['%s-x' %elm] = []
                    edos['%s-y' %elm] = []
                    edos['%s-z' %elm] = []
                edos['%s-x' %elm].append(phonon._pdos.partial_dos[i*3])
                edos['%s-y' %elm].append(phonon._pdos.partial_dos[i*3+1])
                edos['%s-z' %elm].append(phonon._pdos.partial_dos[i*3+2])
       
        for elm,dos in edos.items():
            dos = np.sum(dos,axis=0)#[imin:imax]
            x,y = (energy, dos) if not swap_axes else (dos, energy)
            plt.plot(x, y, label=elm)
            #plt.plot(phonon._pdos.frequency_points, dos, label=elm)
       
        ax = plt.gca()
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        if not swap_axes:
            plt.xlim(g.emin, g.emax)
            plt.ylim(bottom=0)
            plt.axvline(c='r',linestyle='--')
            plt.xlabel('Frequency (THz)')
        else:
            plt.xlim(left=0)
            xticks = ax.get_xticks()[1:]
            plt.xticks(xticks)
            plt.axhline(c='r',linestyle='--')
            #if 'phband' in self.group:
            #    ax.sharey(self.group['phband'])

        #plt.title('Density of Phonon States', pad=1)
        plt.legend()
        if fname != None: plt.savefig(fname)
        return plt

    def gruneisen(self, path, mesh=[10,10,10], fname='gruneisen',**kwargs):
        '''
        plot gruneisen-figure base on FORCE_SETS
        orig - minus - plus
        '''
        fig = self.set_style('base')

        # structure %
        pf = PhononFinder(path)
        gruneisen = pf.get_gruneisen_from_info()
        gruneisen.set_mesh(mesh)
        mesh = gruneisen._mesh
        # plt = gruneisen.plot_mesh()
        ax = plt.gca()
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')

        n = len(mesh._gamma.T) - 1
        for i, (g, freqs) in enumerate(zip(mesh._gamma.T, mesh._frequencies.T)):
            color = (1. / n * i, 0, 1./ n * (n - i))
            plt.plot(freqs, g, 'o', color=color)

        plt.xlabel('Frequency (THz)')
        plt.ylabel('gruneisen')
        if fname != None: plt.savefig(fname)
        return plt

class QEPlot(BasePlot):

    def __init__(self, builder=None):
        self.builder = builder
        self.axes = None
        self.soft = 'qe'

    def plot_band(self, path, fname='band.png', **kwargs):
        '''
        plot band-figure base on band.xml
        '''
        from jamip.analysis.qe import BandFinder
        fig = self.set_style('band')
        color = globalcmap.cycle()            
        g = globalvar.band

        # grep main datas %
        bf = BandFinder(path).get_data()
        bands=bf.bands
        kpoints = bf.kpoints
        rec_vector = bf.rec_vector
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # add bandgap %
        cv = bf.get_cbmvbm()
        if cv['gap'] > 0:
            shift = cv['vbm'].energy
            # plot bandgap %
            xvbm = xkpt[cv['vbm'].ikpt]
            xcbm = xkpt[cv['cbm'].ikpt]
            Ecbm = cv['cbm'].energy-cv['vbm'].energy+g.scissor
            plt.scatter([xvbm],[0],color='r',s=40)
            plt.plot([max(0,xcbm-xkpt[-1]/100),min(xkpt[-1],xcbm+xkpt[-1]/100)],[Ecbm,Ecbm],color='r')
            plt.plot([xcbm,xcbm],[0,Ecbm],c='r',lw=2)
            if xcbm>xkpt[-1]*2/3:
                plt.text(xcbm-xkpt[-1]/6,Ecbm/2,'{:.3f} eV'.format(Ecbm))
            else:
                plt.text(xcbm+xkpt[-1]/100,Ecbm/2,'{:.3f} eV'.format(Ecbm))
        else:
            shift = bf.get_fermi()

        ybands = bands[...,0] - shift
        if g.scissor != 0:
            ybands[:,cv['cbm'].iband:] += g.scissor
        plt.plot(xkpt,ybands,c=next(color))
        plt.ylim(g.emin,g.emax)
        plt.xlim(0,xkpt[-1])
        if fname != None: plt.savefig(fname)
        return plt

    def plot_dos(self, path, ptype='tdos', swap_axes=False, cutoff=0.003, fname='dos.png', **kwargs):
        '''
        plot band-figure base on dos.xml
        '''
        from jamip.analysis.qe import DosFinder
        fig = self.set_style('dos')
        g = globalvar.dos

        # get main data %
        df = DosFinder(path).get_data()
        if ptype == 'tdos':
            dos = df.tdos
        if ptype == 'pdos':
            dos = df.pdos

        dos_energy = df.energy-df.get_vbm()-g.scissor
        imin = sum(dos_energy<g.emin)
        imax = sum(dos_energy<g.emax)
        dos_energy = dos_energy[imin:imax]
        dos = dos[...,imin:imax] / df.volume

#            print(Ecum[-1] * estep, nelect, Ecum[-1])
        print(df.get_vbm())

        limit = g.limit
        if df.spin == 2:
            _limit = -limit if limit != None else None
        else:
            _limit = 0

        if not swap_axes == None:
            plt.ylim(_limit,limit)
            plt.xlim(g.emin,g.emax)
            plt.axvline(c='r',linestyle='--')
            plt.xlabel(g.xlabel)
            plt.ylabel(g.ylabel)
        else:
            plt.xlim(_limit,limit)
            plt.ylim(g.emin,g.emax)
            plt.axhline(c='r',linestyle='--')
            plt.xlabel("${}$".format(ptype.upper()))

        if ptype == 'tdos':
            if df.spin == 1:
                x,y = (dos, dos_energy) if swap_axes else (dos_energy, dos)
                plt.plot(x,y)
        elif ptype == 'pdos':
            for i,element in enumerate(df.elements):
                dos_atom = dos[i]
                if df.spin == 1:
                    for k in range(3):
                        if cutoff and max(np.abs(dos_atom[k])) < cutoff: continue
                        x,y = (dos_energy, dos_atom[k]) if swap_axes else (dos_atom[k], dos_energy)
                        plt.plot(x,y,label=element+'-'+df.orbits[k])

        if ptype != 'tdos' or df.spin ==2:
            plt.legend(loc=1)
        if fname != None: plt.savefig(fname)
        return plt

class CP2KPlot(BasePlot):

    def __init__(self, builder=None):
        self.builder = builder
        self.axes = None
        self.soft = 'cp2k'

    def plot_band(self, path, fname='band.png', **kwargs):
        '''
        plot band-figure base on band.xml
        '''
        from jamip.analysis.cp2k import BandFinder
        fig = self.set_style('band')
        color = globalcmap.cycle()            
        g = globalvar.band

        # grep main datas %
        bf = BandFinder(path).get_data()
        bands=bf.bands
        kpoints = bf.kpoints
        rec_vector = bf.rec_vector
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # add bandgap %
        cv = bf.get_cbmvbm()
        if cv['gap'] > 0:
            shift = cv['vbm'].energy
            # plot bandgap %
            xvbm = xkpt[cv['vbm'].ikpt]
            xcbm = xkpt[cv['cbm'].ikpt]
            Ecbm = cv['cbm'].energy-cv['vbm'].energy+g.scissor
            plt.scatter([xvbm],[0],color='r',s=40)
            plt.plot([max(0,xcbm-xkpt[-1]/100),min(xkpt[-1],xcbm+xkpt[-1]/100)],[Ecbm,Ecbm],color='r')
            plt.plot([xcbm,xcbm],[0,Ecbm],c='r',lw=2)
            if xcbm>xkpt[-1]*2/3:
                plt.text(xcbm-xkpt[-1]/6,Ecbm/2,'{:.3f} eV'.format(Ecbm))
            else:
                plt.text(xcbm+xkpt[-1]/100,Ecbm/2,'{:.3f} eV'.format(Ecbm))
        else:
            shift = bf.get_fermi()

        ybands = bands[...,0] - shift
        if g.scissor != 0:
            ybands[:,cv['cbm'].iband:] += g.scissor
        plt.plot(xkpt,ybands,c=next(color))
        plt.ylim(g.emin,g.emax)
        plt.xlim(0,xkpt[-1])
        if fname != None: plt.savefig(fname)
        return plt

    def plot_dos(self, path, ptype='tdos', swap_axes=False, cutoff=0.003, fname='dos.png', **kwargs):
        '''
        plot band-figure base on dos.xml
        '''
        from jamip.analysis.cp2k import DosFinder
        fig = self.set_style('dos')
        g = globalvar.dos

        # get main data %
        df = DosFinder(path).get_data()
        if ptype == 'tdos':
            dos = df.tdos
        if ptype == 'pdos':
            dos = df.pdos

        dos_energy = df.energy-df.get_vbm()-g.scissor
        imin = sum(dos_energy<g.emin)
        imax = sum(dos_energy<g.emax)
        dos_energy = dos_energy[imin:imax]
        dos = dos[...,imin:imax] / df.volume

#            print(Ecum[-1] * estep, nelect, Ecum[-1])
        print(df.get_vbm())

        limit = g.limit
        if df.spin == 2:
            _limit = -limit if limit != None else None
        else:
            _limit = 0

        if not swap_axes == None:
            plt.ylim(_limit,limit)
            plt.xlim(g.emin,g.emax)
            plt.axvline(c='r',linestyle='--')
            plt.xlabel(g.xlabel)
            plt.ylabel(g.ylabel)
        else:
            plt.xlim(_limit,limit)
            plt.ylim(g.emin,g.emax)
            plt.axhline(c='r',linestyle='--')
            plt.xlabel("${}$".format(ptype.upper()))

        if ptype == 'tdos':
            if df.spin == 1:
                x,y = (dos, dos_energy) if swap_axes else (dos_energy, dos)
                plt.plot(x,y)
        elif ptype == 'pdos':
            for i,element in enumerate(df.elements):
                dos_atom = dos[i]
                if df.spin == 1:
                    for k in range(3):
                        if cutoff and max(np.abs(dos_atom[k])) < cutoff: continue
                        x,y = (dos_energy, dos_atom[k]) if swap_axes else (dos_atom[k], dos_energy)
                        plt.plot(x,y,label=element+'-'+df.orbits[k])

        if ptype != 'tdos' or df.spin ==2:
            plt.legend(loc=1)
        if fname != None: plt.savefig(fname)
        return plt

