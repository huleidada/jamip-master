import numpy as np
import os

class BondAtom(object):

    def __init__(self,Bond,kid,bonds):
        self.origin = (kid,bonds)
        self.kernel = Bond.elements[kid]
        self.position = Bond.direct_positions[kid]
        # create datas %
        self._bond_e = [Bond.elements[bid] for bid in bonds]
        self._bond_d = [Bond.distances[kid][bid] for bid in bonds]
        self._bond_c = [Bond.direct_positions[bid] for bid in bonds]
        del Bond

    def get_bondlen(self,type='mean'):
        if type == 'mean':
            return self.meanbond
        elif type == 'min':
            return self.minbond

    @property
    def minbond(self):
        return np.min(self._bond_d)

    @property
    def meanbond(self):
        return np.mean(self._bond_d)

    @property
    def environment(self):
        env = {}
        for atom in self._bond_e:
            if atom not in env:
                env[atom] = 1
            else :
                env[atom] += 1
        envstr = ''
        envsort = np.sort(list(env.keys()))
        for atom in envsort:
            if env[atom] >1:
                envstr += '%s%d' %(atom,env[atom])
            else:
                envstr += atom
        return envstr

    def __repr__(self):
        return self._print

    @property
    def _print(self):
        prints = []
        prints.append('\nAtom:{0:2s}  tot:{1} '.format(self.kernel,len(self._bond_d)))
        for atom,distance in zip(self._bond_e,self._bond_d):
            prints.append('{0:>2s} : {1:.4f}'.format(atom,distance))
        return '  '.join(prints)

class Bonding(object):

    def __init__(self,structure,bondrange=0.3,**kwargs):
        self.bondrange = bondrange
        # load structure %
        self.elements = [atom.element.symbol for atom in structure.atoms]
        self.direct_positions = [atom.position for atom in structure.atoms]
        self.lattice = structure.lattice
        # load structure 2 %
        self.positions = np.dot(self.direct_positions, self.lattice)
        self.distances = distances = self.get_all_distances(structure,min=True)
        # create 3*3 supercell %
        self.vector = self.get_vector(structure.lattice)

        bondset = []
        for i in range(self.distances.shape[0]):
            # get nearest atoms %
            bmin = None
            num = 1
            for bondlen in np.sort(distances[i])[1:]:
                if bmin == None:
                    bmin = bondlen
                elif bondlen-bmin > self.bondrange:
                    break
                num += 1

            # get nearest atoms ids %
            bonds = np.argsort(distances[i])[1:num]
            allbonds = self.get_equal(i,bonds)
            #print('bond :',allbonds)
            bondset.append(BondAtom(self,i,allbonds))
        # save property %
        self.data = bondset

    def get_vector(self,lattice):
        vector=np.zeros(3)
        for i in lattice:
            vector=np.vstack((vector,vector+i,vector-i))
        return vector

    def get_equal(self,kid,bids):
        bondlist = []
        kernel = self.positions[kid]
        for bid in bids:
            bond = self.positions[bid]
            bondlen = self.distances[kid][bid]
            if np.linalg.norm(kernel-bond) - bondlen > 1e-8:
                equal = 0
            else: equal = 1
            for v in self.vector[1:]:
                if np.linalg.norm(kernel-bond+v) - bondlen < 1e-8:
                    equal += 1
            if equal > 0:
                bondlist.extend([bid]*equal)
        bondlist = np.array(bondlist)
        return bondlist

    def get_all_distances(self, structure, min=True):
        natoms = structure.natoms
        distances = np.zeros((natoms,natoms))
        for i,kernel in enumerate(structure.atoms):
            for j,atom in enumerate(structure.atoms):
                if i==j : continue
                rec = np.subtract(kernel.position,atom.position)
                if min:
                    rec = rec-np.round(rec)
                distance = np.dot(rec,structure.lattice)
                distances[i][j] = np.linalg.norm(distance)
        return distances


