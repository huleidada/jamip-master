import os
import numpy as np
from os.path import exists,join
from jamip.abtools.vasp.vaspio import VaspIO

class Unfolding:
 
    def __init__(self,builder):
        self.obj = builder

    def __getattr__(self,attr):
        return getattr(self.obj, attr)

    def diy_calculator(self):

        task_id = 'unfolding'
        # set stdin & stdout %
        stdin = None
        if len(self.links[task_id]):
            stdin = self.tasks[self.links[task_id][0]].path
        stdout = join(self.rootdir,'electric','unfolding')
        if not exists(stdout): os.makedirs(stdout)

        # add parameters %
        incar = self.tasks[task_id]
        incar.structure = self.load_structure(stdin)
        self.clear_status(task_id)
        # get params from stdin %
        if 'nbands' not in incar:
            self.get_nbands(incar, stdin)
        #if "encut" not in incar:
        #    incar["encut"] = 520

        # get primcell and kpath %
        incar.kpoints = self.get_unfolding_kpath(incar, stdout)

        # update task status %
        self.calculator(task_id, stdout, stdin)
        status = incar.get_status(stdout)
        self.write_status(status, stdout)
            
    def encut_enough(self,stdin,energy):
        from jamip.analysis.vasp.outcar import GrepOutcar

        # get encut from scf calculation %
        if stdin != None and os.path.exists(stdin):
            encut = GrepOutcar().encut(stdin)
            if encut >= energy:
                return True
        else:
            raise OSError('encut grep filed.')

        return False

    def get_unfolding_kpath(self, incar, stdout):
        from jamip.utils.logger import full_path
        from jamip.abtools.base.kpoints import Kpoints
        from jamip.structure import read, Structure
        import spglib

        # supercell %
        structure = incar.structure
        mesh = incar.pop('mesh', 0.01)
        if 'dim' in incar and 'primcell' in incar:
            dim = incar.pop('dim').split()
            if len(dim) == 3:
                trans = np.diag(np.array(dim,dtype=int))
            elif len(dim) == 9:
                trans = np.array(dim,dtype=int).reshape(3,3)
            else:
                raise
            path = full_path(incar.pop('primcell'))
            primcell = read(path)
        else:
            cell = incar.structure.to_cell()
            primcell = spglib.find_primitive(cell)
            primcell = Structure.from_cell(primcell)
            trans = np.dot(cell[0],np.linalg.inv(primcell.lattice))
            trans = np.rint(trans).astype(int)

        # distance check %
        delta = structure.lattice - np.dot(trans, primcell.lattice)
        norm = np.sqrt(np.sum(delta**2))
        if norm > 0.5: 
            raise ValueError('Primcell and supercell do not match, delta = %.4f' %norm)
        dim = np.array(trans, dtype=str).reshape(-1).tolist()
        primcell._comment = 'DIM = {}'.format(' '.join(dim))
        VaspIO.write_poscar(primcell, stdout, 'PRIMCELL')

        # kpath -> meshkpath -> bandkpt %
        prec = "suggest"
        kpath = self.get_kpath('band', prec=prec, structure=primcell)
        kpath.value.set_mesh(primcell.lattice, mesh=mesh)
        kpoint = kpath.get_reciprocal_kpoints()

        K = np.dot(kpoint.value[:,:3],trans.T)
        G = np.rint(K).astype(int)
        kpts = Kpoints("Reciprocal",np.round(K-G, 6))
        gpts = Kpoints("Reciprocal",G)
        # save
        VaspIO.write_kpoints(gpts, stdout, 'GPOINTS')
        VaspIO.write_kpoints(kpath, stdout, 'KPATH.in')

        return kpts

    @classmethod
    def plot(self,path,dim=None,primcell=None,smear=False):
        from jamip.analysis.vasp.outcar import GrepOutcar
        from jamip.analysis.vasp.wavecar import GrepWavecar
        from jamip.analysis.vasp.band import Kpath
        from jamip.utils.plot import globalvar
        from jamip.structure import read
        import matplotlib.pyplot as plt
        import spglib

        if exists(join(path,'.status')):
            path = join(path,'electric','unfolding')
        if not self.check(path):
            raise OSError('Band unfolding calculation Failed!')

        # fermi %
        fermi = GrepOutcar().fermi_energy(path)
        cell = read(join(path,'CONTCAR')).bandStructure()
        if dim is not None:
            trans = np.asarray(dim)
            primcell = read(primcell).bandStructure()
        else:
            primcell = spglib.find_primitive(cell)
            trans = np.dot(np.linalg.inv(cell[0]),primcell[0])

        # wavecar
        w = GrepWavecar(path)
        w.wavecar()
        w.trans = np.dot(np.linalg.inv(primcell[0]),cell[0])
        K,G = w.read_unfolding()
        # spectral-weight %
        if exists('sw.npy'):
            sw = np.load('sw.npy')
        else:
            sw = w.spectral_weight(G)
            np.save('sw.npy',sw)
        # kpath-in %
        kpath,index = Kpath.read_kpath(path)
        delta = np.linalg.norm(np.mat(np.diff(K, axis=0))* np.mat(w.rec_cell)* np.mat(w.trans), axis=1)
        xkpt = np.concatenate(([0,], np.cumsum(delta)))

        plt.rcParams['figure.figsize'] = 12,12
        plt.gcf()
        plt.ylabel('Energy (eV)')
        plt.xlim(0,xkpt[-1])
        plt.ylim(globalvar.band.emin,globalvar.band.emax)
        xticks = []
        # plot kpoint symbols %
        for i in index[:-1]:
            xticks.append(xkpt[i])
            if i == 0: continue
            plt.axvline(xkpt[i],c='black')
        xticks.append(xkpt[-1])
        ax = plt.gca()
        ax.set_xticks(xticks)
        ax.set_xticklabels(kpath)

        # plot band.png %
        if smear:
            e0, sf = w.spectral_function(sw,nedos=4000,sigma=0.01)
            e0 -= fermi
            r0,r1 = np.sum(e0 < -3) , np.sum(e0 < 3)
            e0 = e0[r0:r1]
            sf = sf[:,r0:r1]
            # datamap %
            sfm = np.mean(sf) 
            sf[np.where(sf > sfm*10)] = sfm*10
            sf[np.where(sf > sfm/10)] = sfm/10 + np.log(sf[np.where(sf > sfm/10)]/sfm*10)
            X, Y = np.meshgrid(xkpt, e0)
            for i in range(w.nspin):
                ax.contourf(X, Y, sf[i], cmap='jet')
        else:
            for i in range(w.nspin):
                for nb in range(w.nbands):
                    plt.scatter(xkpt,w.bands[i,:,nb]-fermi,
                                s=sw[i,:,nb]*30 ,c='b')
