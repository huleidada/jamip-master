from jamip.structure import read
from .outcar import GrepOutcar
from vaspwfc import vaspwfc
from unfold import unfold
import numpy as np
import pathlib
import sys

"""https://github.com/QijingZheng/VaspBandUnfolding"""
# install: 
# pip install git+https://github.com/QijingZheng/VaspBandUnfolding

# constant %
lsoc = False  # if calculation with vasp_ncl
lgam = False  # if calculation with vasp_gam
gam_half = 'x'
RYTOEV   = 13.605826
TPI    = 2 * np.pi
AUTOA    = 0.529177249
HSQDTM = RYTOEV * AUTOA * AUTOA
AUTDEBYE = 2.541746
# HSQDTM    =  hbar**2/(2*ELECTRON MASS)

class Wavecar(unfold):

    def __init__(self, path:str, lsorbit:bool=False, lgamma:bool=False):
        self._lsoc = lsorbit
        self._lgam = lgamma
        self.path = pathlib.Path(path)
        self.wfc = vaspwfc(path, lsorbit=lsorbit, lgamma=lgamma)
        self.M = None
        self.KPTS = None

    @classmethod
    def from_file(cls, path:str, lsorbit=None, lgamma=False):
        p = pathlib.Path(path)
        if p.is_dir():
            p = p / 'WAVECAR'
        if not p.exists():
            raise OSError("WAVECAR not exists!")

        if lsorbit is None:
            lsorbit = GrepOutcar().lsorbit(p.parent)
        return cls(p, lsorbit, lgamma)

    def read_primcell(self, path=None):
        import re

        if path is None:
            path = self.path.parent/'PRIMCELL'
        if not path.exists():
            raise IOError("File:%s not exists!" %path.name)

        primcell = read(path, ftype='vasp')
        dim = re.findall(r'-?\d+', primcell.comment_line)
        assert len(dim) == 9, 'Fail to get dim from PRIMCELL'
        self.M = np.array(dim, dtype=int).reshape(3,3)

        return primcell

    def read_kpoints(self, path=None):
        if path is None:
            path = self.path.parent/'KPOINTS'
        if not path.exists():
            raise IOError("File: %s not exists!" %path)
        with open(path, 'r') as f:
            for line in f:
                if line.lstrip()[0].lower() == 'r':
                    break
            KPTS = []
            for line in f:
                if len(line.split()) == 4:
                    KPTS.append(line.split())
            KPTS = np.array(KPTS,dtype=np.float64)
        if path.name == 'KPOINTS': self.KPTS = KPTS
        return KPTS

    def read_unfolding(self):

        if self.M is None:
           raise ValueError("Matrix must be set first.")

        KPOINTS = self.path.parent/'KPOINTS'
        GPOINTS = self.path.parent/'GPOINTS'

        # read kpts
        KPTS = self.read_kpoints(KPOINTS)
        GPTS = self.read_kpoints(GPOINTS)
        PKPTS = (KPTS[:,:3] + GPTS[:,:3]) @ np.linalg.inv(self.M)

        return PKPTS

    def get_bands(self):
        bands = np.stack([self.wfc._bands, self.wfc._occs], axis=-1)
        return bands

    @property
    def kgrid(self):
        def get_indices(n):
            x = np.arange(n, dtype=int)
            x[n // 2 + 1:] -= n 
            return x

        dx, dy, dz = self.wfc._ngrid
        fx = get_indices(dx) 
        fy = get_indices(dy) 
        fz = get_indices(dz) 

        gz, gy, gx = np.array(
            np.meshgrid(fz, fy, fx, indexing='ij')
        ).reshape((3, -1))
        return np.array([gx, gy, gz], dtype=float).T

    def gvectors(self, ikpt=0):
        '''
        Generate the G-vectors that satisfies the following relation
            (G + k)**2 / 2 < ENCUT
        '''
        import scipy.constants as sc
        from vasp_constant import HSQDTM 

        kvec = self.wfc._kvecs[ikpt]
        #HSQDTM = sc.hbar**2 / (2 * sc.electron_mass)
        #print(HSQDTM)
        KENERGY = HSQDTM * np.linalg.norm(
            np.dot(self.kgrid + kvec[None, :], 2*np.pi*self.wfc._Bcell), axis=1
        )**2
        # find Gvectors where (G + k)**2 / 2 < ENCUT
        #print(self.wfc._encut)
        #print(np.max(KENERGY))
        Gvec = self.kgrid[np.where(KENERGY < self.wfc._encut)[0]]
        return Gvec

    def slow_grid(self, ispin=0, ikpt=0, iband=0, axis='z'):

        coeff = self.wfc.readBandCoeff(ispin+1, ikpt+1, iband+1)
        gvectors = self.gvectors(ikpt=ikpt)
        kvec = self.wfc._kvecs[ikpt]
        assert len(coeff) == len(gvectors), f"coeffs: {len(coeff)} != gvectors: {len(gvectors)}..."
        
        def get_indices(n):
            return np.arange(n, dtype=int) / n

        dx, dy, dz = self.wfc._ngrid
        #dz = int((dz-1)/2*10+1)
        fx = get_indices(dx) 
        fy = get_indices(dy) 
        fz = get_indices(dz) 

        gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing='ij')).reshape((3, -1))
        grid = np.array([gx, gy, gz], dtype=float).T
        matrix = np.zeros((len(grid), len(gvectors)), dtype=np.complex64)
        print('gs',grid.shape)
        print('ks',gvectors.shape)

        #for i,rvec in enumerate(grid):
        #    for j,ivec in enumerate(gvectors+kvec):
        #        csum = coeff[j] * np.exp(2j * np.pi * np.dot(ivec, rvec))
        #        matrix[i,j] = csum

        # 计算 ivec = gvectors + kvec
        ivec = gvectors + kvec  # 形状为 (M, 3)
        # 计算 np.dot(ivec, rvec) 的矩阵形式
        #dot_product = np.dot(grid, ivec.T)  # 形状为 (N, M)
        dot_product = np.linalg.multi_dot([grid, ivec.T])
        # 计算 exp(2j * np.pi * dot_product)
        exp_term = np.exp(2j * np.pi * dot_product)  # 形状为 (N, M)
        # 将 coeff 广播到与 exp_term 相同的形状，并进行逐元素乘法
        matrix = coeff[np.newaxis, :] * exp_term  # 形状为 (N, M)
        print(matrix.shape)

        # calculate wsum
        csum = np.sum(matrix, axis=1) / np.sqrt(self.wfc._Omega)
        csum2 = np.abs(csum)**2 
        wsum = np.sum(csum2)
        print(wsum)
        print(self.wfc._Omega)

        if axis == 'z':
            zsum = np.sum(csum2.reshape(dz, dy, dx), axis=(1,2)) / wsum
            latticec = np.linalg.norm(self.wfc._Acell[2])
            return latticec*fz, zsum 

    def fast_grid(self, ispin=0, ikpt=0, iband=0):
        from mpi4py import MPI
        import numpy as np
        import time

        coeff = self.wfc.readBandCoeff(ispin+1, ikpt+1, iband+1)
        gvectors = self.gvectors(ikpt=ikpt)
        kvec = self.wfc._kvecs[ikpt]
        assert len(coeff) == len(gvectors), f"coeffs: {len(coeff)} != gvectors: {len(gvectors)}..."
        
        def get_indices(n):
            return np.arange(n, dtype=int) / n

        dx, dy, dz = self.wfc._ngrid
        #dz = int((dz-1)/2*4+1)
        #dx = int(dx/10)
        #dy = int(dy/10)

        #dz = int((dz-1)/2*10+1)
        fx = get_indices(dx) 
        fy = get_indices(dy) 
        fz = get_indices(dz) 

        gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing='ij')).reshape((3, -1))
        #grid = np.array([gx, gy, gz], dtype=float).T
        #ivec = gvectors + kvec  # 形状为 (M, 3)
        
        # 初始化 MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #print('rank',rank)
        #print('size',size)
        
        A = np.array([gx, gy, gz], dtype=float).T
        B = gvectors + kvec  # 形状为 (M, 3)
        C = coeff
        
        # 广播 B 到所有进程
        #A = comm.bcast(A, root=0)
        B = comm.bcast(B, root=0)
        C = comm.bcast(C, root=0)
        
        # 分块 A 并分发到各个进程
        chunk_size = min(dx,dy,dz)#11 #len(A) // size
        num_chunks = len(A) // chunk_size
        #A_local = np.zeros((chunk_size, 3))
        #comm.Scatter(A, A_local, root=0)
        log_step = max(dx,dy,dz)

        if rank == 0:
            print(f"Total chunks: {num_chunks}")
            next_chunk = 0  # 下一个要分配的任务
            completed_chunks = 0  # 已完成的任务数
            start_time = time.time() 
            #results = np.zeros((len(A),), dtype=np.float64)  
            results = np.zeros((len(A),), dtype=np.complex128)  
        
            # 初始任务分配
            for worker in range(1, size):
                if next_chunk < num_chunks:
                    comm.send(next_chunk, dest=worker, tag=1)
                    next_chunk += 1
                else:
                    comm.send(None, dest=worker, tag=1)  # 发送结束信号
        
            # 接收结果并分配新任务
            while completed_chunks < num_chunks:
                status = MPI.Status()
                result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)
                worker_rank = status.source
                chunk_idx = result[0]
                chunk_result = result[1]
                # 存储结果
                start = chunk_idx * chunk_size
                end = start + chunk_size
                results[start:end] = chunk_result
                completed_chunks += 1

                # 输出进度
                if completed_chunks % log_step == 0: 
                    progress = completed_chunks / num_chunks * 100
                    elapsed_time = time.time() - start_time
                    print(f"Progress: {progress:.2f}% | Completed chunks: {completed_chunks}/{num_chunks} | "
                          f"Elapsed time: {elapsed_time:.2f}s")
        
                # 分配新任务
                if next_chunk < num_chunks:
                    comm.send(next_chunk, dest=worker_rank, tag=1)
                    next_chunk += 1
                else:
                    comm.send(None, dest=worker_rank, tag=1)  # 发送结束信号
        
            #MPI.Finalize()
            print("Final result shape:", results.shape)
            csum2 = np.abs(results / np.sqrt(self.wfc._Omega))**2
            zsum = np.sum(csum2.reshape(dz, dy, dx), axis=(1,2)) / np.sum(csum2) #wsum
            #zsum = comm.bcast(zsum, root=0)
            latticec = np.linalg.norm(self.wfc._Acell[2])
            return latticec*fz, zsum 
        
        else:
            # 工作进程等待任务并计算
            while True:
                chunk_idx = comm.recv(source=0, tag=1)
                if chunk_idx is None:  # 收到结束信号
                    break
        
                # 计算当前块
                start = chunk_idx * chunk_size
                end = start + chunk_size
                local_A = A[start:end, :]  # 获取当前块的数据
                #local_result = np.sum(np.dot(local_A, B.T), axis=1)
                dot_product= np.dot(local_A, B.T)            # 形状为 (size, M)
                local_exp_term = np.exp(2j * np.pi * dot_product)  # 形状为 (size, M)
                local_result = np.sum(C[None,:] * local_exp_term, axis=1) # 形状为 (size, )
        
                # 发送结果回主进程
                comm.send((chunk_idx, local_result), dest=0, tag=2)

        

    @property
    def bands(self):
        return self.wfc._bands
        
    @property
    def kvecs(self):
        return self.wfc._kvecs
        
    @property
    def kpoints(self):
        return self.kvecs

    @property
    def rec_cell(self):
        return self.wfc._Bcell

    def spectral_weight(self, kpoints, path=None, overwrite=False):

        if path is None: path = self.path.parent

        if overwrite and (path/'sw.npy').exists():
            self.SW = np.load(path/'sw.npy')
        else:
            fileObj = open('unfold.log', 'a+', 1)
            savedStdout, sys.stdout = sys.stdout, fileObj
            self.SW = unfold.spectral_weight(self, kpoints)
            np.save(path/'sw.npy', self.SW)
            sys.stdout = savedStdout
           
        return self.SW

    def spectral_function_with_interpolation(self, xkpts, interpolation=3, kind='linear', **kwargs):
        from scipy.interpolate import interp1d

        sw = self.SW
        assert len(xkpts) == sw.shape[1]
        nkpt = int(len(xkpts)*interpolation)
        kpts = np.linspace(xkpts.min(),xkpts.max(),nkpt)
        f1 = interp1d(xkpts, sw[...,0], kind=kind, axis=1)
        f2 = interp1d(xkpts, sw[...,1], kind=kind, axis=1)
        self.SW = np.stack([f1(kpts), f2(kpts)], axis=-1)
        e0, sf = self.spectral_function(**kwargs)
        self.SW = sw

        return kpts, e0, sf

    def tdm(self, ki, kj):
        '''
        calculate Transition Dipole Moment (TDM) between two KS states.
        K = [ispin,ikpt,iband]
        shift value for vaspwfc

        Return:
            tuple: e0, e1, dE, tdm
            e0: energy_i (float, unit eV)
            e1: energy_j (float, unit eV)
            dE: energy_j - energy_i (float, unit eV)
            tdm: TDM in x,y,z direction. (complex array, (3,))
        '''
        ispin, ikpt, iband = ki
        ki = ispin+1, ikpt+1, iband+1

        ispin, ikpt, iband = kj
        kj = ispin+1, ikpt+1, iband+1

        return self.wfc.get_dipole_mat(ki, kj)

    def cpd(self, ki, kj):

        e0,e1,dE,tdm = self.tdm(ki, kj)

        x_real = tdm[0].real
        x_imag = tdm[0].imag
        y_real = tdm[1].real
        y_imag = tdm[1].imag
        z_real = tdm[2].real
        z_imag = tdm[2].imag

        value_p = ((x_real-y_imag)**2+(x_imag+y_real)**2)/2
        value_n = ((x_real+y_imag)**2+(x_imag-y_real)**2)/2
        value = (value_p-value_n)/(value_p+value_n)
        return value

    def write_elf(self, info, ngrid=None, output='ELFCAR'): 
        from .chgcar import Chgcar
        from .band import Outcar

        # read kpts from outcar %
        if self.KPTS is None or self.KPTS.shape[1] != 4:
            self.KPTS = Outcar.from_file(self.path.parent)._get_kpoint(weight=True)
        kptw = self.KPTS[:,3]
       
        elf = self.wfc.elf(kptw=kptw, ngrid=ngrid, warn=False)
        elf = np.array(elf[0])
        Chgcar.write(elf, info, output)

    @classmethod
    def merge_wavecar(cls, dirs, lsorbit:bool=False, lgamma:bool=False):

        # constants
        AUTOA    = 0.529177249
        RYTOEV   = 13.605826
        PI     = 3.141592653589793238
        TPI    = 2 * PI
       
        h1 = None
        h2 = None
        rtag = None
        ngrid = None
        recls = []
        nkpts = []

        for path in dirs:
  
            p = pathlib.Path(path)
            if p.is_dir():
                p = p / 'WAVECAR'
            if not p.exists():
                raise OSError("WAVECAR not exists!")
  
            with open(p, 'rb') as f:
                head1 = np.fromfile(f, dtype=np.float64, count=3) # recl, nspin, rtag
                f.seek(int(head1[0]))
                head2 = np.fromfile(f, dtype=np.float64, count=12) 
  
            # check data size
            if h1 is None:
                h1 = head1
                h2 = head2
                if int(h1[2]) == 45200:
                    rtag = np.complex64
                elif int(h1[2]) == 45210:
                    rtag = np.complex128
                else:
                    raise OSError("unsupport vasp format")
  
                # get FFT grid
                cell = np.array(h2[3:]).reshape(3,3)
                anorm = np.linalg.norm(cell, axis=1)
                cutoff = np.ceil(np.sqrt(h2[2] / RYTOEV) / (TPI / (anorm / AUTOA)))
                ngrid = np.array(2 * cutoff + 1, dtype=int)
  
            else:
                assert h1[1] == head1[1], "nspin different"
                assert h2[1] == head2[1], "nbands different"
                assert h2[2] == head2[2], "encut different"
  
            recls.append(int(head1[0]))
            nkpts.append(int(head2[0]))

        with open('WAVECAR_sum', 'wb') as f:
  
            nrecl = max(recls)
            h1[0] = nrecl
            h1.tofile(f, format='%f')
            f.seek(nrecl)
            h2[0] = sum(nkpts)
            h2.tofile(f, format='%f')
  
            for spin in range(int(h1[1])):
                ikpt=0
                for i,path in enumerate(dirs):
   
                    recl = recls[i]
                    nkpt = nkpts[i]
                    nband = int(h2[1])
   
                    p = pathlib.Path(path)
                    if p.is_dir():
                        p = p / 'WAVECAR'
                    if not p.exists():
                        raise OSError("WAVECAR not exists!")
   
                    with open(p, 'rb') as g:
                        for ii in [spin,]:  # ispin
                            for jj in range(nkpt):  # nkpts
   
                                # read bands & kpoints
                                rec = 2 + ii*nkpt*(nband+1) + jj*(nband+1) 
                                g.seek(rec * recl)
                                dump = np.fromfile(g, dtype=np.float64, count=4+3*nband)
                                rec2 = 2 + ii*(nkpt+ikpt)*(nband+1) + (ikpt+jj)*(nband+1)
                                f.seek(rec2 * nrecl)
                                dump.tofile(f, format='%f')
                                nplws = int(dump[0])
   
                                # read wavecar coeff
                                for kk in range(nband):  # nbands
                                    rec = 2 + ii*nkpt*(nband+1) + jj*(nband+1) + (kk + 1)
                                    g.seek(rec * recl)
                                    dump = np.fromfile(g, dtype=rtag, count=nplws)
                                    rec2 = 2 + ii*(nkpt+ikpt)*(nband+1) + (ikpt+jj)*(nband+1) + (kk + 1)
                                    f.seek(rec2 * nrecl)
                                    dump.tofile(f, format='%f')
   
                    ikpt += nkpt
