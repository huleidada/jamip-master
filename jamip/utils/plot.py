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
from itertools import cycle

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
    Global colormap provider with shared color cycler state.
    
    Usage:
    # Get next color (shared state)
    color = GlobalCmap().next_color()  # returns next color in sequence
    
    # Get color by index
    color = GlobalCmap()[2]  # returns '#2ca02c'
    """
    _DEFAULT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', "#31d3e6"
    ]    
    #CMAP = ['g','b','r','y','m','orange','c','cyan','yellow','violet','brown',\
    #        'lime','deepskyblue','gold','darkorchid','greenyellow','r']
    
    _instance = None
    _instance_lock = threading.Lock()
    _shared_cycler = None
    _cycler_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cmap = cls._DEFAULT_COLORS.copy()
                    cls._shared_cycler = cycle(cls._instance._cmap)
        return cls._instance
    
    def __init__(self):
        self._cmap = self._DEFAULT_COLORS.copy()
    
    def __getitem__(self, index):
        return self._cmap[index % len(self._cmap)]
    
    def __len__(self):
        return len(self._cmap)
    
    @classmethod
    def next_color(cls):
        """Get the next color in the shared sequence (thread-safe)"""        
        if cls._shared_cycler is None:
            with cls._cycler_lock:
                if cls._shared_cycler is None:
                    cls._shared_cycler = cycle(cls._DEFAULT_COLORS)
                    
        with cls._cycler_lock:
            return next(cls._shared_cycler)
    
    @property
    def colors(self):
        """Return a copy of the current color list"""
        return self._cmap.copy()
    
    @classmethod
    def set_colors(cls, new_colors):
        """Update the color list with new colors and reset the cycler"""
        with cls._instance_lock:
            cls._cmap = list(new_colors)
            cls._shared_cycler = cycle(cls._cmap)
    
    @classmethod
    def reset(cls):
        """Reset to default colors and cycler"""
        with cls._instance_lock:
            cls._cmap = cls._DEFAULT_COLORS.copy()
            cls._shared_cycler = cycle(cls._cmap)
        return cls

def get_path_label(path):
    path = pathlib.Path(path).resolve()
    if path.suffix != '':
        return path.stem
    elif path.parent == path:
        return None
    else:
        return get_path_label(path.parent)

JOBLIST = {'band': 'band', 'fatband':'band', 'hseband':'band',
           'unfolding':'band', 'phband': 'band', 'gruneisen': 'band',
           'cutoff_conv':'base', 'kpoints_conv':'base', 'softmode':'base',
           'absorb':'absorb', 'refrace':'absorb', 'dielectric':'absorb', 'shg': 'absorb',
           'dos': 'dos', 'cohp':'dos', 'phdos': 'dos', 'spdos': 'dos',
           'tdm': 'tdm', 'ldos': 'dos'}
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

    def plots(self, *args, **kwargs):
        '''
        Plot multiple jobs in one figure with subplots.
        args: job names, e.g. 'band', 'dos', 'tdm'
        kwargs: plot parameters, e.g. fname, figsize, sharey, etc.
        '''
        jobs = [job for job in args if job in JOBLIST]        
        if len(jobs) == 0:
            raise KeyError("Invalid input job name(s).")
        jobname = '_'.join(jobs)

        # Create output directory if multiple paths
        if len(self.stdin) == 0:
            raise IOError('No valid path for plots.')  
        elif len(self.stdin) > 1:
            self.savedir = pathlib.Path(f'{jobname}_plots')
            self.savedir.mkdir(exist_ok=True)

        # Figure config
        basestyle = pathlib.Path.home()/'.jamip'/'viewer'/'base.mplstyle'
       
        # load fig kwargs
        key = tuple(np.unique(jobs))
        fig_kw = self.fig_kw.get(key, {})

        # update kwargs
        kwargs['fname'] = None
        if len(set([JOBLIST[i] for i in jobs]) & set(['band','dos'])) == 2:
            kwargs['swap_axes'] = True

        success = 0
        for path in self.stdin:
            
            plt.style.use(basestyle)
            self.plotter.set_axes(jobs, **fig_kw)
            fname = path.absolute().stem+self.figtype if len(self.stdin) > 1 else jobname+self.figtype

            for job in jobs:
                self.plot(job, path, **kwargs)
            self.save(fname=fname)
            success += 1

        print(f'Successfully plotted {success} figure(s).')
        del self.plotter.axes

    def plot_multi_data(self, job:str, paths:list=None, ncol=1, labels=None, **kwargs):
        '''
        Plot the same job for multiple paths on one subplot for comparison.
        '''
        if job not in JOBLIST:
            raise KeyError(f"Invalid input job name: {job}")
        
        if paths is None:
            paths = self.stdin
        if len(paths) == 0:
            raise IOError(f'No valid path for {job}-plot')

        # load fig kwargs
        key = (job,)
        fig_kw = self.fig_kw.get(key, {})
        self.plotter.set_axes([job,], **fig_kw)
        basestyle = pathlib.Path.home()/'.jamip'/'viewer'/'base.mplstyle'
        plt.style.use(basestyle)

        # update kwargs
        kwargs['fname'] = None
        kwargs['legend'] = False
        kwargs['reset_color'] = False
        # if 'return_handles' not in kwargs:
        #     kwargs['return_handles']=True

        for path in paths:
            self.plot(job, path, **kwargs)

        labels = []
        lines = []
        for key, value in self.plotter.legends.items():
            labels.extend(f'{key.split("-")[0]}-{i}' for i in value[1])
            lines.extend(value[0])
        # print(labels, lines)
        # print(self.plotter.legends)
        # print(len(labels), len(lines))  
        self.plotter.set_legend(lines, labels, ncol=ncol)
        self.save(fname=f'{job}_compare.png')
        print(f'Successfully compared {len(self.stdin)} {job} plots.')
        del self.plotter.axes


    def plot_multi_params(self, jobs:list, paths:list=None, params:list=None, **kwargs):
        '''
        Plot the same job for multiple paths with different parameters on one subplot for comparison.
        '''
        try:
            maps = []
            joblist = []
            for row in jobs:
                for job in row:
                    joblist.append(job)
                    if job not in JOBLIST:
                        raise KeyError(f"Invalid input job name: {job}")
                xrange = range(len(joblist)-len(row), len(joblist))
                maps.append(list(xrange))
        except:
            raise KeyError(f"Invalid input jobs format: {jobs}")
        
        if paths is None:
            paths = self.stdin
        elif not isinstance(paths, list):
            paths = [paths]

        # check paths & params
        if len(paths) == 0 or len(params) == 0:
            raise IOError(f'No valid path or params for {job}-plot')

        # load fig kwargs
        key = (job,)
        fig_kw = self.fig_kw.get(key, {})
        self.plotter.set_axes([job,], **fig_kw)
        basestyle = pathlib.Path.home()/'.jamip'/'viewer'/'base.mplstyle'
        plt.style.use(basestyle)

        sharex = fig_kw.get('sharex', True)
        sharey = fig_kw.get('sharey', True)
        gridspec_kw = fig_kw.get('gridspec_kw', {'wspace': 0, 'hspace': 0})
        fig, axdict = plt.subplot_mosaic(maps, sharex=sharex, sharey=sharey, 
                                         gridspec_kw=gridspec_kw, **fig_kw)

        # update kwargs
        kwargs['fname'] = None
        kwargs['reset_color'] = False
        for i,job in enumerate(joblist):
            plt.sca(axdict[i])
            i1 = int(i / len(joblist) * len(paths))
            i2 = i % len(params)
            kwargs.update(params[i2])
            self.plot(job, paths[i1], **kwargs)

        if len(maps) > 1:
            ax = plt.gca()
            ax.set_yticks(ax.get_yticks()[:-1])
            plt.ylabel('')
        elif len(maps[0]) > 1:
            ax = plt.gca()
            ax.set_xticks(ax.get_xticks()[:-1])
            plt.xlabel('')

        self.save(fname=f'{job}_compare.png')
        print(f'Successfully compared {len(self.stdin)} {job} plots.')
        del self.plotter.axes

    def plot(self, job, path, **kwargs):

        #try:
        if True: 
            if job == 'band':
                self.plotter.plot_band(path,**kwargs)
            elif job == 'fatband':
                self.plotter.plot_fat_band(path,**kwargs) 
            elif job == 'dos':
                self.plotter.plot_dos(path,**kwargs)
            elif job == 'ldos':
                self.plotter.plot_local_dos(path,**kwargs)
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
                print(f"Invalid jobname: {job}")
        #except:
            # warnings.warn()
            #print(f'job {job} failed in {path}')

    def save(self, fname=None, **kwargs):
        '''
        get figure name and save figure
        1. if input fname, named by fname
        2. if plot single figure, named by jobs
        3. if plot mutliple figure, named by entry dir.name        
        '''

        if fname != None:        
            if self.savedir != None:
                fname = self.savedir / fname

            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            # clear globalcmap
            globalcmap.reset()

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

    def set_axes(self, jobs:list, figsize=None, sharey=True, wspace=None, hspace=None, **kwargs):

        self.axes = None
        if 'width_ratios' in kwargs and 'height_ratios' in kwargs:
            if isinstance(jobs[0], str): jobs = [jobs]
            fig, axes = plt.subplot_mosaic(jobs, figsize=figsize, sharey=sharey, **kwargs)
            self.axes = axes

        elif len(jobs) > 1:
            rows, height, width = self.fast_mosaic(jobs)
            figsize = kwargs.get('figsize', (10*sum(width),10*sum(height)))
            fig, axes = plt.subplot_mosaic(rows, width_ratios=width, height_ratios=height,
                                           figsize=figsize, sharey=sharey,
                                           gridspec_kw={'wspace': wspace, 'hspace': hspace},
                                           **kwargs)

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
        mplstyle = lambda job: pathlib.Path.home()/'.jamip'/'viewer'/f'{job}.mplstyle'

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

    def set_legend(self, lines, labels, **kwargs):
        
        if len(lines):
            plt.legend(lines, labels, **kwargs)

def swap_axes_support(func):
    def wrapper(self, *args, swap_axes=False, swap_figure=False, **kwargs):
        result = func(self, *args, swap_axes=swap_axes, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            plt, swap_params = result
        else:
            plt, swap_params = result, {}

        if swap_axes and swap_figure:
            fig = plt.gcf()
            height,width = fig.get_figheight(), fig.get_figwidth()
            fig.set_figwidth(height)
            fig.set_figheight(width)

        # Set axis labels
        xlabel = swap_params.get('xlabel')
        ylabel = swap_params.get('ylabel')
        if xlabel:
            (plt.ylabel if swap_axes else plt.xlabel)(xlabel)
        if ylabel:
            (plt.xlabel if swap_axes else plt.ylabel)(ylabel)

        # Set axis limits
        xlim = swap_params.get('xlim')
        ylim = swap_params.get('ylim')
        if xlim:
            (plt.ylim if swap_axes else plt.xlim)(*xlim)
        if ylim:
            (plt.xlim if swap_axes else plt.ylim)(*ylim)
        
        # Set axvline/axhline
        axvline = swap_params.get('axvline')
        axhline = swap_params.get('axhline')        
        if axvline:
            (plt.axhline if swap_axes else plt.axvline)(**axvline)
        if axhline:
            (plt.axvline if swap_axes else plt.axhline)(**axhline)

        if swap_axes and swap_params.get('merge', False):
            ax = plt.gca()
            ax.set_xticks(ax.get_xticks()[1:])
            plt.ylabel('')
        # else:
        #     ax = plt.gca()
        #     ax.set_yticks(ax.get_yticks()[:-1])
        #     plt.ylabel('')

        if swap_params.get('fname', None):
            self.builder.save(swap_params['fname'])

        return plt
    return wrapper            

class VaspPlot(BasePlot):

    soft = 'vasp'

    def __init__(self, builder=None):
        self.builder = builder
        self.axes = None
        self.legends = {}

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

    def set_linestyle(self, labels, color=None, reset_color=True, **kwargs):
        """ """                       
        from itertools import cycle        

        if color is not None:
            color = cycle(color)
        elif reset_color:
            globalcmap.reset()

        # create colormap and legend %
        colormap = []
        lines = []
        for label in labels:
            if color is not None:
                c = next(color)
            else:
                c = globalcmap.next_color()
            colormap.append(c)
            pl,=plt.plot([],[],c=c,label=label)
            lines.append(pl)
        colormap = np.array(colormap)
        # if kwargs.get('legend',True):
        #     plt.legend(loc=1, fontsize='large', framealpha=0.9)
        for line in lines:
            line.remove()
        
        return colormap, lines
        
    def plot_band(self,path,kpath=None,source='eigenval',legend=False,fname='band.png',reset_color=True,**kwargs):
        '''
        plot band-figure base on OUTCAR
        kwargs:
            bandgap: whether to label band-gap value
        '''
        # Initializes the data retrieval module & pyplot & colormap %
        fig = self.set_style('band')
        g = globalvar.band
        color = globalcmap.reset() if reset_color else globalcmap

        # grep main datas %
        bf = BandFinder(path).get_data(source=source)
        if kpath != None: bf.regroup(kpath)
        bf.remove_duplicates()
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks[:-1]:
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

        labels = []
        lines = []
        for ispin, bands_ispin in enumerate(bf.bands):
            ybands = bands_ispin[...,0] - shift
            if g.scissor != 0:
                ybands[:,cv['cbm'].iband:] += g.scissor
            pl = plt.plot(xkpt,ybands,c=color.next_color())
            lines.append(pl)
            labels.append(f'spin {ispin+1}')
        
        if legend:
            self.set_legend(lines, labels, loc=1)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-band'] = lines, labels

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
                      legend=True, 
                      alpha=0.75,
                      ispin=False,
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
        fig = self.set_style('fatband')
        g = globalvar.band 

        # grep main datas %        
        pf = ProFinder(path).get_data(source=source)
        if kpath != None: pf = pf.regroup(kpath)
        pf.remove_duplicates()
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
        '''
        if interpolation:
            xkpt, bands, procar = pf.get_interpolation(procar,interpolation)
            procar = procar * plt.rcParams['lines.markersize'] 
        else:
            bands = bands
            procar = procar * plt.rcParams['lines.markersize'] 
        '''
        bands = bands
        procar = procar * plt.rcParams['lines.markersize'] 

        if ptype == 'base':
            procar_index = np.flip(np.argsort(procar, axis=-1), axis=-1)
            sorted_procar = np.flip(np.sort(procar, axis=-1), axis=-1)

            if pf.spin == 2:
                if ispin is False:
                    labels = np.repeat(labels, 2)
                else:
                    labels = [f'{key}-{suffix}' for key in labels for suffix in ('up', 'down')]

            # plot legend and get colormap %
            if isinstance(proj, str):
                if pf.spin == 2:
                    unids1 = np.unique(procar_index[0,...,:max_z])
                    unids2 = np.unique(procar_index[1,...,:max_z]) + procar_index.shape[-1]
                    unids = np.r_[unids1, unids2]
                else:
                    unids = np.unique(procar_index[...,:max_z])

                labels = [labels[i] for i in unids]
                cmap, lines = self.set_linestyle(labels, **kwargs)

                colormap = ['k'] * (max(unids)+1)
                for i,j in enumerate(unids):
                    i = i % len(cmap)
                    colormap[j] = cmap[i]
                colormap = np.array(colormap)
            else:
                colormap, lines = self.set_linestyle(labels, **kwargs)

            if legend:
                self.set_legend(lines, labels, loc=1)
            else:
                key = get_path_label(path)
                self.legends[f'{key}-fatband'] = lines, labels
            
            # plot band.png %
            for i,bands_ispin in enumerate(bands):
                ybands = bands_ispin[...,0] - shift
                ybands[:,cv['cbm'].iband:] += g.scissor
                cmap = colormap[i::pf.spin]

                # plot %
                for nband in range(len(ybands[0])):
                    if ybands[:,nband].min() > g.emax or ybands[:,nband].max() < g.emin: continue
                    for j in range(max_z):
                        if max_z == 1:
                            plt.plot(xkpt,ybands[:,nband],c='grey',lw=1, alpha=0.5)
                            try:
                                plt.scatter(xkpt,ybands[:,nband],s=sorted_procar[i,:,nband,j],color=cmap[procar_index[i,:,nband,j]])
                            except:
                                print(i,j,nband)
                                raise ValueError("max_z must be less than procar_index.")
                        else:
                            # plt.scatter(xkpt,ybands[:,nband],s=sorted_procar[i,:,nband,j],marker='o',color='none',
                            #        edgecolors=colormap[procar_index[i,:,nband,j]])
                            # plt.scatter(xkpt,ybands[:,nband],s=np.sum(sorted_procar[i,:,nband,j:], axis=-1),marker='o',
                            #             color=colormap[procar_index[i,:,nband,j]], alpha=0.8)
                            plt.scatter(xkpt,ybands[:,nband],s=np.sum(sorted_procar[i,:,nband,j:], axis=-1)**2,marker='o',
                                        c=cmap[procar_index[i,:,nband,j]], alpha=1)                                                        
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
            colormap, lines = self.set_linestyle(labels, color=ptype, **kwargs)
            if legend:
                self.set_legend(lines, labels, loc=1)
            else:
                key = get_path_label(path)
                self.legends[f'{key}-fatband'] = lines, labels

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

    def plot_band_transitions(self,path,search_energies,dE=0.1,legend=True,plot_tdm=False,
                              ptype='band',source='procar',fname='transition.png',reset_color=True,**kwargs):
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
      
        bf = BandFinder(path).get_data(source=source).remove_duplicates()
        all_transitions = bf.get_all_transitions()
        xkpt = bf.get_xkpt()

        # find band edge and get shift        
        cv = bf.get_cbmvbm()
        shift = bf.get_fermi() if bf.metal else cv['vbm'].energy
        
        labels = [f'{energy} ev' for energy in search_energies]
        colormap, lines = self.set_linestyle(labels, reset_color=reset_color, **kwargs)        
        if legend: 
            self.set_legend(lines, labels, loc=1)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-transition'] = lines, labels

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

        if fname != None: self.builder.save(fname)
        return plt

    @swap_axes_support
    def plot_local_dos(self, path, nedos:int=301, sigma:float=0.05,emin=None,emax=None, vmin=-7,vmax=1, 
                       swap_axes:bool=False, source='procar', fname='ldos.png', reset_color=True, **kwargs):
        '''
        plot single points dos base on PROCAR
        '''
        from jamip.analysis.vasp.chgcar import Chgcar
        fig = self.set_style('dos')
        g = globalvar.dos
        bf = BandFinder(path)
        if emin != None and emax != None:
            energy = np.linspace(emin, emax, nedos)
            energy,ldos=bf.get_ldos(energy=energy, sigma=sigma, rec_vector='CHGCAR')
        else:
            energy,ldos=bf.get_ldos(nedos=nedos, sigma=sigma, rec_vector='CHGCAR')
        ldos = np.array(ldos).sum(axis=0)
        ldos = np.log(np.clip(ldos, 1e-9, 1e9))
        lattice_c = Chgcar.from_file(bf.stdin/"CHGCAR").structure._cell.c

        # extra: shift & cutoff
        bf = bf.get_data(rec_vector='CHGCAR')
        cv = bf.get_cbmvbm()
        #shift = bf.fermi #if bf.metal else cv['cbm'].energy
        #shift = cv['cbm'].energy
        #print(energy, shift)
        #energy -= shift

        cm = "hot"                              # Coloring type
        fig, ax = plt.subplots(figsize = (12,6))# Figure size plotted
        plt.xlabel(r'z ($\AA$)', fontsize=21)    # X axis name
        plt.ylabel(r'$E-E_{\mathrm{Vac}}$ (eV)', fontsize=21)       # Y axis name
        vmin = -7.0                             # Minimum of color scale
        vmax = 1.0
        
        ng = np.arange(ldos.shape[1])
        ng = ng / ng.max() * lattice_c 
        X, Y = np.meshgrid(ng, energy)
        #z[:,2] = np.clip(z[:,2], 1e-9, 1e9)
        #to_plot = np.log(z[:,2])
        bc=ax.pcolormesh(X, Y, ldos,cmap=cm,vmin=vmin, vmax=vmax)
        cbar=plt.colorbar(bc,ticks=[vmin,vmax], shrink=0.7, pad = 0.1)
        cbar.ax.set_yticklabels([vmin,vmax],size=21)
        cbar.ax.set_ylabel(r"$\log(LDOS)$", rotation=270,size=21)
        #ax.set_ylim(ymin,ymax)
        #ax.set_xlim(xmin,xmax)

        return plt, {'ylim': (energy[0],energy[-1]), 'xlim': (0,lattice_c), 'fname': fname}
            
    @swap_axes_support
    def plot_single_point_dos(self, path, kpoint=None, proj='lmax', nedos:int=601, sigma:float=0.02, legend:bool=True,
                              swap_axes:bool=False, norm:bool=False, source='procar', fname='spdos.png', reset_color=True, **kwargs):
        '''
        plot single points dos base on PROCAR
        '''
        fig = self.set_style('dos')
        g = globalvar.dos
        pf = ProFinder(path).get_data()

        # search kpoint %
        if kpoint == None: 
            kpoint = np.zeros((1,3))        
        
        if isinstance(kpoint, int):
            ikpt = kpoint
            assert ikpt < len(pf.kpoints), f"Kpoint index {ikpt} is out of range."
        else:
            kpoint = np.array(kpoint).reshape(1,3)        
            # kpoint position -> ikpt %
            d = np.sum(np.abs(pf.kpoints - kpoint), axis=1)
            if np.min(d) > 1e-4 :
                raise ValueError(f"Search Kpoint {kpoint} Failed.")
            ikpt = np.argmin(d)
            print(f"Search Kpoint Successfully. ikpt: {ikpt}")

        energy,procar,labels=pf.single_point(proj, ikpt, nedos=nedos, sigma=sigma)

        # extra: shift & cutoff
        cv = pf.get_cbmvbm()
        shift = pf.fermi if pf.metal else cv['cbm'].energy
        energy -= shift
        imin,imax = sum(energy<g.emin),sum(energy<g.emax)
        energy = energy[imin:imax]
        width = (energy[-1] - energy[0]) / len(energy)
        procar = procar[:,imin:imax,...] 

        lines = []            
        for i,bands_ispin in enumerate(pf.bands):
            # yp = np.cumsum(procar[i,:,0], axis=-1)
            bottom = np.zeros_like(energy)
            ysum = 1
            if norm:
                ysum = np.sum(procar[i,:,0],axis=1) 
                cutoff = np.where(ysum < ysum.max()*0.01)
                procar[:,cutoff] = 0
            # plot %
            for j,label in enumerate(labels):
                # percentage
                yp = procar[i,:,0,j] / ysum
                # swap_axes %
                if swap_axes:
                    pl,=plt.barh(energy,yp,height=width,left=bottom,label=label)
                else:
                    pl,=plt.bar(energy,yp,width=width,bottom=bottom,label=label)
                bottom += yp
                lines.append(pl)

        if legend:
            self.set_legend(lines, labels, loc=2)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-spdos'] = lines, labels

        # swap_axes %
        # self._plot_swap(plt, xlabel=g.ylabel, xlim=(energy[0],energy[-1]),swap_axes=swap_axes,swap_figure=True)
        # if fname != None: self.builder.save(fname)
        return plt, {'xlim': (energy[0],energy[-1]), 'xlabel': g.xlabel, 'ylabel': g.ylabel, 'fname': fname}
    
    @swap_axes_support
    def plot_dos(self, path, proj='edos', cutoff:float=0.003, swap_axes:bool=False, fill:bool=False, spin:int=None,
                 tdos:bool=False, pdos:bool=True, source='doscar', legend=True, fname='dos.png', reset_color=True, **kwargs):
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
        df = DosFinder(path).get_data(source=source, spin=spin).per_volume()
        dos_energy = df.energy
        # get shift %
        bf = BandFinder(DosFinder(path).banddir).get_data()
        cv = bf.get_cbmvbm()
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
            proj_dos, proj_labels = df.projection(proj)
            proj_dos = proj_dos[...,imin:imax]
            total_dos = df.tdos[...,imin:imax]
        else:
            total_dos = df.tdos[...,imin:imax]
            pdos = False
            tdos = True

        lines = []
        labels = []
        # plot %
        if tdos:
            tdos_labels = {'spin_up': 1, 'spin_down': -1} if df.spin == 2 else {'total': 1}
            for i,label in enumerate(tdos_labels):
                dos_spin = total_dos[i] * tdos_labels[label]
                x,y = (dos_energy, dos_spin) if not swap_axes else (dos_spin, dos_energy)
                pl,=plt.plot(x, y, label=label, c='k')
                lines.append(pl)
                labels.append(label)

        if pdos:
            color = globalcmap.reset() if reset_color else globalcmap
            for dos,label in zip(proj_dos,proj_labels):
                if isinstance(proj,str) and max(np.abs(dos)) < cutoff: continue
                x,y = (dos_energy, dos) if not swap_axes else (dos, dos_energy)
                co = color.next_color()
                pl,=plt.plot(x,y,label=label,c=co)
                if fill:
                    if swap_axes:
                        plt.fill_betweenx(y, 0, x,
                                        where=(x > x.min()),
                                        color=co, alpha=0.3,
                                        interpolate=True)
                    else:
                        plt.fill_between(x, 0, y,
                                        where=(y > y.min()),
                                        color=co, alpha=0.3,
                                        interpolate=True)
                lines.append(pl)
                labels.append(label)

        # spin & limit
        limit = g.limit
        _limit = -limit if df.spin == 2 else 0
        axvline = {'c':'r','linestyle':'--'}
        axhline = {'c':'black','linestyle':'--'} if df.spin == 2 else None

        # swap_axes %
        if swap_axes:
            # need set xlim first %
            # ax = plt.gca()
            # ax.set_xticks(ax.get_xticks()[1:])
            plt.gca().yaxis.tick_right()

        # labels -> ncol
        ncol = kwargs.get('ncol',1)
        if legend:
            self.set_legend(lines, labels, loc=1, ncol=ncol)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-dos'] = lines, labels

        return plt, {'xlim': (g.emin,g.emax), 'ylim': (_limit,limit), 'xlabel': g.xlabel, 'ylabel': g.ylabel, 'fname': fname,        
                     'axvline': axvline, 'axhline': axhline, 'merge': True}

    def plot_absorb(self, path, ptype='absorb', source='xml', directions=None, legend=True, fname='absorb.png',reset_color=True):
        '''
        plot optics-figure base on vasprun.xml
        '''
        fig = self.set_style('absorb')
        color = globalcmap.reset() if reset_color else globalcmap
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

        lines = []
        # plot directions 
        if directions == None:
            if ptype == 'absorb':
                plt.axes(yscale='log')
                plt.ylim(1e-0,1e7)
            pl,=plt.plot(of.energy,np.mean(data,axis=1),c='b')
            lines.append(pl)
            directions = ['absorb']

        else:                
            for direction in directions:
                tmp = []
                for i,d in enumerate('xyz'):
                    if d in direction:
                        tmp.append(data[:,i])
                pl,=plt.plot(of.energy,np.mean(tmp,axis=0),c=color.next_color(),label=direction)    
                lines.append(pl)
            plt.ylim(bottom=0)

        if legend:
            self.set_legend(lines, directions, loc=1)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-absorb'] = lines, directions

        plt.xlim(g.emin,g.emax)
        plt.xlabel(g.xlabel)
        plt.ylabel(g.ylabel)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_polarization(self, path, source='xml', directions=('x','y'), legend=True, fname='polarization.png',reset_color=True,**kwargs):
        '''
        plot optics-figure base on vasprun.xml
        '''
        fig = self.set_style('absorb')
        color = globalcmap.reset() if reset_color else globalcmap
        g = globalvar.absorb

        of = OpticsFinder(path).get_data(source=source)
        # cm-1
        data = of.absorb()
        maps = {'x': data[:,0], 'y': data[:,1], 'z': data[:,2],}
        dx = maps[directions['x']]
        dy = maps[directions['y']]
        polar = (dx - dy) / (dx + dy)
        plt.axhline(0,ls='--',color='k')
        pl, = plt.plot(of.energy, polar, c='r')
        
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
        
        if legend is False:
            key = get_path_label(path)
            self.legends[f'{key}-polarization'] = [pl], ['polarization']
        else:
            self.set_legend([pl], ['polarization'], loc=1)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_dielfunc(self, path,ptype=['imag','real'], directions=['x','z'], source='xml', fname='dielfunc.png', legend=True, reset_color=True, **kwargs):
        '''
        plot optics-figure base on vasprun.xml
        '''
        fig = self.set_style('dielectric')
        color = globalcmap.reset() if reset_color else globalcmap
        g = globalvar.diel

        # data reshape %
        of = OpticsFinder(path).get_data(source=source)
        energy = of.energy
        imag = of.imag()
        real = of.real()

        if isinstance(ptype, str):  ptype = [ptype]
        labels = []
        lines = []

        # plot %
        for inc in ptype:
            value = imag if inc == 'imag' else real
            for direction in directions:
                data = []
                for i,d in enumerate(('x','y','z')):
                    if d in direction:
                        data.append(value[:,i])
                label = f'{inc}-{direction}'
                pl, = plt.plot(energy,np.mean(data,axis=0),c=color.next_color(),label=label)
                labels.append(label)
                lines.append(pl)

        # legend %
        if legend:
            self.set_legend(lines, labels, frameon=False)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-dielfunc'] = lines, labels

        plt.xlim(0, g.emax)
        plt.ylabel(g.ylabel)
        plt.xlabel(u'Energy (eV)')

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

    def plot_tdm(self, path, kpath=None, source='eigenval', legend=True, fname='tdm.png',**kwargs):
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
        for i in xticks[:-1]:
            plt.axvline(i,c='black')
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)

        # get dipole-mement from wavecar %
        bf = BandFinder(path).get_data_from_wavecar().get_tdm().remove_duplicates()
        assert len(xkpt) == bf.bands.shape[1], "K-point inconsistent, did you save wavecar in band calculation?"

        labels = []
        lines = []
        for ispin,tdm in enumerate(bf.tdms):
            plt.plot(xkpt,np.sum(tdm,axis=-1))
            for i,j in enumerate('xyz'):
                pl, = plt.plot(xkpt, tdm[:,i], label=j)
                lines.append(pl)
                labels.append(j)
        if legend:
            self.set_legend(lines, labels)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-tdm'] = lines, labels

        plt.ylim(bottom=0)
        plt.xlim(0,xkpt[-1])
        plt.ylabel(g.ylabel)
        if fname != None: self.builder.save(fname)
        return plt

    def plot_cpd(self, path, kpath=None, source='eigenval',legend=True,fname='cpd.png',**kwargs):
        '''
        plot Circular Polarization dipole-moment figure base on WAVECAR
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

        # get dipole-moment from wavecar %
        bf = BandFinder(path).get_data_from_wavecar().get_cpd().remove_duplicates()
        assert len(xkpt) == bf.bands.shape[1], "K-point inconsistent, did you save wavecar in band calculation?"

        lines = []
        labels = []
        for ispin,cpd in enumerate(bf.cpds):
            pl,=plt.plot(xkpt, cpd) #, label=j)
            lines.append(pl)
            labels.append(f'spin{ispin}')
        if legend:
            self.set_legend(lines, labels)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-cpd'] = lines, labels

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

    def plot_hse_band(self,path,proj=None,interpolation=None,legend=True,fname='hse.png',**kwargs):
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
        for i in xticks[:-1]:
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
            colormap, lines = self.set_linestyle(labels, **kwargs)

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
            lines = []
            labels = []
            for bands_ispin in bands:
                ybands = bands_ispin[...,0] - shift
                ybands[:,cv['cbm'].iband:] += g.scissor
                pl,=plt.plot(xkpt,ybands,c='b')
                lines.append(pl)
                labels.append(f'spin{i}')

        if legend:
            self.set_legend(lines, labels)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-band'] = lines, labels

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

    @swap_axes_support
    def plot_cohp(self, path, ptype='coop', dtype='t', swap_axes=False, fill=True, legend=True, fname='cohp.png', **kwargs):
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
        lines = []
        labels = []

        if ptype.lower() == 'cohp' or ptype.lower() == 'coop':
            if ptype.lower() == 'coop':
                df = COHPFinder(path).read_coop(path, dtype=dtype)
            else:
                df = COHPFinder(path).read_cohp(path, dtype=dtype)
            energy = df['energy']
            for label in df.columns:
                if label == 'energy': continue
                x,y = (-df[label],energy) if swap_axes else (energy,-df[label])
                pl,=plt.plot(x,y, label=label)
                lines.append(pl)
                labels.append(label)
                if fill:
                    if swap_axes:
                        plt.fill_betweenx(y, 0, x, facecolor = 'cyan')
                    else:
                        plt.fill_between(x, 0, y, facecolor = 'cyan')
            plt.axvline(0,c='black')
            plt.axhline(0,c='black',linestyle='--')
            ylabel="-COHP (E)"

        elif ptype.lower() == 'icohp' or ptype.lower() == 'icoop':
            if ptype.lower() == 'icoop':
                df = COHPFinder(path).read_icoop(path, dtype=dtype)
            else:
                df = COHPFinder(path).read_icohp(path, dtype=dtype)
            energy = df['energy']
            for label in df.columns:
                if label == 'energy': continue
                x,y = (-df[label],energy) if swap_axes else (energy,-df[label])
                pl,=plt.plot(x,y,label=label)
                lines.append(pl)
                labels.append(label)
            plt.axvline(0,c='black')
            plt.axhline(0,c='black',linestyle='--')
            ylabel="-COHP (E)"

        if legend:
            self.set_legend(lines, labels)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-cohp'] = lines, labels
        if fname != None: self.builder.save(fname)
        return plt, {'xlabel':"Energy (eV)", 'ylabel':ylabel, 'xlim':(g.emin,g.emax), 'fname': fname}

    def plot_boltztrap(self, path, ptype='seebeck', cbvb:str='vb', legend=True, fname='boltztrap.png',
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
        pl,=plt.plot(xdata,ydata,label=ptype)
        plt.ylabel(ptype)
        if legend:
            self.set_legend([pl], [ptype])
        else:
            key = get_path_label(path)
            self.legends[f'{key}-boltztrap'] = [pl], [ptype]

        if fname != None: self.builder.save(fname)
        return plt

    def plot_md(self, path, ptype='temperature', legend=True, fname='md.png', **kwargs):
        '''
        Plot molecular dynamics calculation result.

        Args:
            path (str): Path of molecular dynamics calculation.
            fname (str): Name of figure.
            kwargs: Other arguments.

        Returns:
            Figure.
        '''
        from jamip.analysis.vasp.md import MDFinder
        fig = self.set_style('base')

        # get data (pd.DataFrame) %
        md = MDFinder(path).get_data()
        if md is None:
            raise OSError('No MD data found!')

        # plot %
        pl,=plt.plot(md['time'], md[ptype], label=ptype)
        plt.xlabel('Time (fs)')
        if ptype == 'temperature':
            plt.ylabel('Temperature (K)')
        elif ptype == 'pressure':
            plt.ylabel('Pressure (GPa)')
        elif ptype == 'volume':
            plt.ylabel('Volume (A^3)')
        else:
            plt.ylabel(ptype)
        if legend:
            self.set_legend([pl], [ptype])
        else:
            key = get_path_label(path)
            self.legends[f'{key}-md'] = [pl], [ptype]

        if fname != None: self.builder.save(fname)
        return plt

    def plot_shg(self,path,ptype='real',legend=True,fname='shg.png',**kwargs):
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
                subpath = path/f'SHG_{d}'
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

    def softmode(self,path,legend=True,fname='softmode.png',**kwargs):
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

    def phband(self, path, npoints=51, kpoints=None, legend=True, fname='phband.png', **kwargs):
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
        if legend:
            self.set_legend(curves, ['phband'])
        else:
            key = get_path_label(path)
            self.legends[f'{key}-phband'] = curves, ['phband']
        if fname != None: plt.savefig(fname)
        return plt

    @swap_axes_support
    def phdos(self, path, project=False, mesh=[10,10,10], swap_axes=False, legend=True, fname='phdos.png', **kwargs):
        '''
        plot phonon-dos-figure base on FORCE_SETS
        '''
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
                    edos[f'{elm}-x'] = []
                    edos[f'{elm}-y'] = []
                    edos[f'{elm}-z'] = []
                edos[f'{elm}-x'].append(phonon._pdos.partial_dos[i*3])
                edos[f'{elm}-y'].append(phonon._pdos.partial_dos[i*3+1])
                edos[f'{elm}-z'].append(phonon._pdos.partial_dos[i*3+2])
       
        lines = []
        labels = []
        for elm,dos in edos.items():
            dos = np.sum(dos,axis=0)#[imin:imax]
            x,y = (energy, dos) if not swap_axes else (dos, energy)
            pl,=plt.plot(x, y, label=elm)
            lines.append(pl)
            labels.append(elm)
            #plt.plot(phonon._pdos.frequency_points, dos, label=elm)
       
        ax = plt.gca()
        ax.xaxis.set_tick_params(which='both', direction='in')
        ax.yaxis.set_tick_params(which='both', direction='in')
        if swap_axes:
            xticks = ax.get_xticks()[1:]
            plt.xticks(xticks)
            #if 'phband' in self.group:
            #    ax.sharey(self.group['phband'])
        #plt.title('Density of Phonon States', pad=1)
        # xrange = energy.max() - energy.min()
        # xlim = (energy.min()-0.1*xrange, energy.max()+0.1*xrange)

        if legend:
            self.set_legend(lines, labels)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-phdos'] = lines, labels
        return plt, {'ylim':(0,), 'xlabel':'Frequency (THz)', 'axvline':{'c':'r','linestyle':'--'}, 'merge':True, 'fname': fname}

    def gruneisen(self, path, mesh=[10,10,10], legend=True, fname='gruneisen',**kwargs):
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

        lines = []
        labels = []
        n = len(mesh._gamma.T) - 1
        for i, (g, freqs) in enumerate(zip(mesh._gamma.T, mesh._frequencies.T)):
            color = (1. / n * i, 0, 1./ n * (n - i))
            pl,= plt.plot(freqs, g, 'o', color=color)
            lines.append(pl)
            labels.append(f'{i}')

        plt.xlabel('Frequency (THz)')
        plt.ylabel('gruneisen')
        if legend:
            self.set_legend(lines, labels)
        else:
            key = get_path_label(path)
            self.legends[f'{key}-gruneisen'] = lines, labels
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
        color = globalcmap.reset() if reset_color else globalcmap    
        g = globalvar.band

        # grep main datas %
        bf = BandFinder(path).get_data()
        bands=bf.bands
        kpoints = bf.kpoints
        rec_vector = bf.rec_vector
        xkpt = bf.get_xkpt()
        xticks, xlabels = bf.get_xticks()
        
        # set x axis % 
        for i in xticks[:-1]:
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
        plt.plot(xkpt,ybands,c=color.next_color())
        plt.ylim(g.emin,g.emax)
        plt.xlim(0,xkpt[-1])
        if fname != None: plt.savefig(fname)
        return plt

    # @swap_axes_support
    def plot_dos(self, path, ptype='tdos', swap_axes=False, cutoff=0.003, legend=True, fname='dos.png', **kwargs):
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

    def plot_band(self, path, legend=True, fname='band.png', reset_color=True, **kwargs):
        '''
        plot band-figure base on band.xml
        '''
        from jamip.analysis.cp2k import BandFinder
        fig = self.set_style('band')
        color = globalcmap.reset() if reset_color else globalcmap
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
        plt.plot(xkpt,ybands,c=color.next_color())
        plt.ylim(g.emin,g.emax)
        plt.xlim(0,xkpt[-1])
        if fname != None: plt.savefig(fname)
        return plt

    def plot_dos(self, path, ptype='tdos', swap_axes=False, cutoff=0.003, legend=True, fname='dos.png', **kwargs):
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

