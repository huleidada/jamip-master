from scipy.special import erf, erfc
import numpy as np

# import scipy.constants as sc
# CONV_FACT = 1e10 * sc.e / (4 * np.pi * sc.epsilon_0)
CONV_FACT=14.399787146105348
sqrt_pi = np.math.sqrt(np.pi)

class EwaldSolver(object):
    
    def __init__(self, structure, rmax=None, kmax=None):
        '''            
        rmax (float): Real space cutoff radius dictating how
            many terms are used in the real space sum. Defaults to None,
            which means determine automatically using the formula given
            in gulp 3.1 documentation.
        kmax (float): Reciprocal space cutoff radius.
            Defaults to None, which means determine automatically using
            the formula given in gulp 3.1 documentation.
        eta (float): The screening parameter. Defaults to None, which means
            determine automatically.
        acc_factor (float): No. of significant figures each sum is
            converged to.                    
        compute_forces (bool): Whether to compute forces. False by
            default since it is usually not needed.
        '''
        self.structure = structure
        accf = np.math.sqrt(np.log(10**12))
        self.alpha = ( len(structure) / np.sqrt(2) / structure.volume**2 ) ** (1/6) * sqrt_pi

        self.rmax = rmax or accf / self.alpha
        self.kmax = kmax or 2*accf*self.alpha

        self._initialized = False
        self._dimension = None
        self._calc_force = False

    def add_charge(self, value=dict):
        charges = []
        for atom in self.structure.atomic_positions:
            atom.charge = value[atom.specie]
            charges.append(atom.charge)
        self.charges = np.array(charges, dtype=float)
        return self

    def force(self):

        if not self._initialized:
            self._calc_recip()
            self._calc_real()
            self._calc_epoint()
            self._initialized = True

        total_force = self.f_recip + self.f_real

        return total_force 

    def total_energy(self):

        if not self._initialized:
            self._calc_recip()
            self._calc_real()
            self._calc_epoint()
            self._initialized = True

        total_energy=np.sum(self.e_recip)+np.sum(self.epoint)+np.sum(self.e_real)
        if sum(self.charges) != 0 and self._dimension == 3:
            total_energy += self._calc_background()

        return total_energy

    def site_energy(self, i):

        if not self._initialized:
            self._calc_recip()
            self._calc_real()
            self._calc_epoint()
            self._initialized = True

        return np.sum(self.e_recip[i]) + np.sum(self.e_real[i]) + self.epoint[i] 

    def _calc_epoint(self):
        self.epoint = -CONV_FACT*self.alpha/sqrt_pi*self.charges**2

    def _calc_recip(self):
        pass

    def _celc_real(self):
        pass

class Ewald2D(EwaldSolver):
    '''reference
    https://github.com/WagnerGroup/pyqmc/pull/399
    https://doi.org/10.1063/1.479595
    '''

    def __init__(self, structure, rmax=None, kmax=None):
        super().__init__(structure, rmax=rmax, kmax=kmax)
        self._dimension = 2
        self.set_grid()
        
    def set_grid(self):

        lattice = self.structure.lattice
        a = int(self.rmax / self.structure._cell.a ) + 1
        b = int(self.rmax / self.structure._cell.b ) + 1
        grid = np.mgrid[-a:a+1, -b:b+1].reshape(2,-1).T
        length = np.linalg.norm(grid @ lattice[:2], axis=1)
        self.real_grid = grid[np.where(length < self.rmax)]

        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi
        a,b,c = np.linalg.norm(rec_lattice, axis=1)
        a = int(self.kmax / a) + 1
        b = int(self.kmax / b ) + 1
        grid = np.mgrid[-a:a+1, -b:b+1].reshape(2,-1).T
        #g = int(self.kmax)
        #grid = np.mgrid[-g:g+1, -g:g+1].reshape(2,-1).T
        grid = grid[grid.any(axis=1)]
        length = np.linalg.norm(grid @ rec_lattice[:2], axis=1)
        self.recip_grid = grid[np.where(length < self.kmax)]

        self.area = np.linalg.norm(np.cross(lattice[0], lattice[1]))

    def _calc_recip(self):
        '''
        long range energy
        '''
        alpha = self.alpha 
        natom = len(self.structure)
        energy = np.zeros((natom, natom))
        force = np.zeros((natom, 3))

        coords = self.structure.get_positions(type='cartesian')
        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi

        # g = int(self._kmax) 
        # grid = np.mgrid[-g:g+1, -g:g+1].reshape(2,-1).T
        # grid = grid[grid.any(axis=1)]
        # gs = grid @ rec_lattice[:2]
        gs = self.recip_grid @ rec_lattice[:2]
        g1s = np.linalg.norm(gs, axis=1) 

        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[i] * self.charges[j]
                zij = disp[2]                
                exp1_xy = np.exp(g1s*zij)*erfc(g1s/(2*alpha)+alpha*zij)
                exp2_xy = np.exp(-g1s*zij)*erfc(g1s/(2*alpha)-alpha*zij)
                factor = exp1_xy + exp2_xy
                energy[i,j] += charge*np.sum(np.cos(np.dot(gs, disp))*factor/g1s)/2
                energy[i,j] += -charge*(zij*erf(alpha*zij)+np.exp(-(alpha*zij)**2)/(alpha*sqrt_pi))

                if self._calc_force:
                    
                    # F_xy
                    factor = exp1_xy + exp2_xy
                    force[i,:] += 2*charge*np.sum((np.sin(np.dot(gs,disp))*factor)[:,None]*gs)

                    # F_z_1
                    exp_z1 = np.exp(g1s*zij-(g1s/(2*alpha)+alpha*zij)**2)
                    exp_z2 = -np.exp(-g1s*zij-(g1s/(2*alpha)-alpha*zij)**2)
                    factor = -exp1_xy+exp2_xy+(exp_z1+exp_z2)*2*alpha/sqrt_pi
                    force[i,2] += charge*np.sum(np.cos(np.dot(gs,disp))*factor/g1s) 

                    # F_Z_0
                    force[i,2] += 2*charge*np.sum(erf(alpha*zij))

        self.e_recip = CONV_FACT*np.pi*energy/self.area
        self.f_recip = CONV_FACT*np.pi*force/self.area

    def _calc_real(self):
        '''
        short range energy
        '''
        alpha = self.alpha 
        natom = len(self.structure)
        energy = np.zeros((natom, natom))
        force = np.zeros((natom, 3))

        coords = self.structure.get_positions(type='cartesian')
        lattice = self.structure.lattice

        # r = int(self.rmax) 
        # grid = np.mgrid[-r:r+1, -r:r+1].reshape(2,-1).T
        # gs = grid @ lattice[:2]
        gs = self.real_grid @ lattice[:2]

        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[i] * self.charges[j]                
                disps = gs + disp
                g2s = np.sum(disps**2, 1)
                inds = g2s > 1e-8
                ds = np.sqrt(g2s[inds])
                energy[i,j] += np.sum(charge*erfc(alpha*ds)/(ds))

                if self._calc_force:
                    part1=(2*alpha/sqrt_pi)*np.exp(-(alpha*alpha)*ds**2)/ds**2
                    part2 = erfc(alpha*ds)/(ds**3)
                    force[i,:] += charge*np.sum((part1+part2)[:,None]*disps[inds], axis=0)

        self.e_real = CONV_FACT / 2 * energy
        self.f_real = CONV_FACT * force

    def get_madelung_potential(self, position, scaled=True):

        alpha = self.alpha 
        natom = len(self.structure)
        lattice = self.structure.lattice
        coords = self.structure.get_positions(type='cartesian')
        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi

        position = np.ravel(position)
        assert position.shape == (3,)
        if scaled: position = np.dot(position, lattice)

        gs = self.recip_grid @ rec_lattice[:2]
        g1s = np.linalg.norm(gs, axis=1) 

        potential = np.zeros(natom)
        for i in range(natom):
            disp = coords[i] - position
            charge = self.charges[i] 
            zij = disp[2]                
            exp1_xy = np.exp(g1s*zij)*erfc((g1s/(2*alpha))+alpha*zij)
            exp2_xy = np.exp(-g1s*zij)*erfc((g1s/(2*alpha))-alpha*zij)
            factor = exp1_xy + exp2_xy
            potential[i] += charge*np.sum(np.cos(np.dot(gs, disp))*factor/g1s)
            potential[i] += -charge*2*(zij*erf(alpha*zij)+np.exp(-(alpha*zij)**2)/(alpha*sqrt_pi))

        potential_recip = CONV_FACT*np.pi*potential/self.area

        gs = self.real_grid @ lattice[:2]
        potential_self = 0 

        potential = np.zeros(natom)
        for i in range(natom):
            disp = coords[i] - position 
            charge = self.charges[i]       
            disps = gs + disp
            g2s = np.sum(disps**2, 1)
            inds = g2s > 1e-8
            ds = np.sqrt(g2s[inds])
            potential[i] += np.sum(charge*erfc(alpha*ds)/(ds))
            if np.min(g2s) < 1e-8:
                potential_self = CONV_FACT*2*self.alpha*self.charges[i]/sqrt_pi

        potential_real = CONV_FACT*potential
        potential = np.sum(potential_recip) + np.sum(potential_real) - potential_self

        return potential

    def get_madelung_constant(self):
        alpha = self.alpha 
        natom = len(self.structure)
        
        elements = self.structure.get_elements()
        coords = self.structure.get_positions(type='cartesian')
        lattice = self.structure.lattice
        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi

        gs = self.recip_grid @ rec_lattice[:2]
        g1s = np.linalg.norm(gs, axis=1) 

        potential = np.zeros((natom, natom))
        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[j] 
                #zij = disp[2] - np.round(disp[2])
                zij = disp[2]
                exp1_xy = np.exp(g1s*zij)*erfc((g1s/(2*alpha))+alpha*zij)
                exp2_xy = np.exp(-g1s*zij)*erfc((g1s/(2*alpha))-alpha*zij)
                factor = exp1_xy + exp2_xy
                potential[i,j] += charge*np.sum(np.cos(np.dot(gs, disp))*factor/g1s)
                potential[i,j] += -charge*2*(zij*erf(alpha*zij)+np.exp(-(alpha*zij)**2)/(alpha*sqrt_pi))

            potential_recip = CONV_FACT*np.pi*potential/self.area

        gs = self.real_grid @ lattice[:2]

        potential = np.zeros((natom, natom))
        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j] 
                charge = self.charges[j]       
                disps = gs + disp
                g2s = np.sum(disps**2, 1)
                inds = g2s > 1e-8
                ds = np.sqrt(g2s[inds])
                potential[i] += np.sum(charge*erfc(alpha*ds)/(ds))

        potential_real = CONV_FACT*potential

        potential_self = CONV_FACT*2*self.alpha*self.charges/sqrt_pi

        potential = np.sum(potential_recip,axis=1) + np.sum(potential_real,axis=1) - potential_self
                
        # 2d distance
        '''
        vector = coords[:,None,:] - coords[None,:,:]
        vector = vector - np.floor(vector)
        # shape = (4, natom, natom, 2)
        grid = np.mgrid[-1:1, -1:1, -1:1].reshape(3,-1).T
        allvector = vector[None,:,:,:] + grid[:,None,None,:]
        allvector = allvector @ lattice
        ds = np.min(np.linalg.norm(allvector[:,:,:,:2], axis=-1), axis=0)
        row, col = np.diag_indices_from(ds)
        ds[row,col] = np.sum(np.linalg.norm(lattice, axis=1))
        distances = ds #self.structure.get_all_distances()
        '''

        distances = self.structure.get_all_distances()
        a0 = (self.structure.volume)**(1/3)
        madelung_nn = 0
        madelung_a0 = 0
        for i in range(natom):
            nn_idx = np.argmin(distances[i])
            nn = np.min(distances[i])
            madelung_nn += abs(potential[i]*nn/CONV_FACT) * abs(self.charges[i]) 
            madelung_a0 += abs(potential[i]*a0/CONV_FACT) * abs(self.charges[i]) 

        Z = self.structure.get_formula_units_Z()
        madelung_nn = madelung_nn/Z
        madelung_a0 = madelung_a0/Z

        return madelung_nn, madelung_a0

class Ewald3D(EwaldSolver):

    def __init__(self, structure, rmax=None, kmax=None):
        super().__init__(structure, rmax=rmax, kmax=kmax)
        self._dimension = 3
        self.set_grid()

    def set_grid(self):

        lattice = self.structure.lattice
        a = int(self.rmax / self.structure._cell.a ) + 1
        b = int(self.rmax / self.structure._cell.b ) + 1
        c = int(self.rmax / self.structure._cell.c ) + 1
        grid = np.mgrid[-a:a+1, -b:b+1, -c:c+1].reshape(3,-1).T
        #self.real_grid = grid
        length = np.linalg.norm(grid @ lattice, axis=1)
        self.real_grid = grid[np.where(length < self.rmax)]

        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi
        a,b,c = np.linalg.norm(rec_lattice, axis=1)
        a = int(self.kmax / a) + 1
        b = int(self.kmax / b ) + 1
        c = int(self.kmax / c ) + 1
        # g = int(self.kmax) + 1
        # grid = np.mgrid[-g:g+1, -g:g+1, -g:g+1].reshape(3,-1).T
        grid = np.mgrid[-a:a+1, -b:b+1, -c:c+1].reshape(3,-1).T
        grid = grid[grid.any(axis=1)]
        #self.recip_grid = grid
        length = np.linalg.norm(grid @ rec_lattice, axis=1)
        self.recip_grid = grid[np.where(length < self.kmax)]

    def _calc_recip(self):

        alpha = self.alpha 
        natom = len(self.structure)
        energy = np.zeros((natom, natom))
        force = np.zeros((natom, 3))
        
        coords = self.structure.get_positions(type='cartesian')
        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi

        # g = int(self._kmax) 
        # grid = np.mgrid[-g:g+1, -g:g+1, -g:g+1].reshape(3,-1).T
        # grid = grid[grid.any(axis=1)]
        # gs = grid @ rec_lattice
        gs = self.recip_grid @ rec_lattice
        g2s = np.sum(gs**2, 1)     

        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[i] * self.charges[j]
                for k,k2 in zip(gs,g2s): 
                    structure_factor = charge*(np.cos(np.dot(k,disp)))
                    exponential = np.exp(-k2/(4.0*alpha*alpha))/k2
                    energy[i,j] += exponential*structure_factor   

                    if self._calc_force:
                        structure_factor = charge*(np.sin(np.dot(k,disp)))
                        force[i,:] += exponential*structure_factor*k   

 
        self.e_recip = CONV_FACT*2*np.pi*energy/(self.structure.volume)
        if self._calc_force:
            self.f_recip = CONV_FACT*4*np.pi*force/(self.structure.volume)
    
    def _calc_real(self):

        alpha = self.alpha 
        natom = len(self.structure)
        energy = np.zeros((natom, natom))
        force = np.zeros((natom, 3))
        
        coords = self.structure.get_positions(type='cartesian')
        lattice = self.structure.lattice
        gs = self.real_grid @ lattice

        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[i] * self.charges[j]
                disps = disp + gs
                g2s = np.sum(disps**2, 1)
                inds = g2s > 1e-8
                ds = np.sqrt(g2s[inds])
                energy[i,j] += np.sum(charge*erfc(alpha*ds)/(ds))

                if self._calc_force:
                    part1 = 2*alpha/sqrt_pi*np.exp(-alpha**2*ds**2)/(ds**2)
                    part2 = erfc(alpha*ds)/(ds**3)
                    force[i,:] += charge*np.sum((part1+part2)[:,None]*disps[inds], axis=0)


        self.e_real = CONV_FACT*1/2*energy      
        if self._calc_force:
            self.f_real = CONV_FACT*force

    def _calc_background(self):
        return -CONV_FACT*(np.pi*(np.sum(self.charges)**2))/ (2*self.alpha**2 * self.structure.volume)

    def get_madelung_potential(self, position, scaled=True, type=None):

        alpha = self.alpha 
        natom = len(self.structure)
        coords = self.structure.get_positions(type='cartesian')
        lattice = self.structure.lattice
        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi
        volume = self.structure.volume

        position = np.ravel(position)
        assert position.shape == (3,)
        if scaled: position = np.dot(position, lattice)
        
        gs = self.recip_grid @ rec_lattice
        g2s = np.sum(gs**2, 1)     

        potential = np.zeros(natom)
        for i in range(natom):
            disp = position - coords[i]
            charge = self.charges[i]
            for k,k2 in zip(gs,g2s): 
                structure_factor = charge*(np.cos(np.dot(k,disp)))
                exponential = np.exp(-k2/(4.0*alpha*alpha))/k2
                potential[i] += exponential*structure_factor   

        potential_recip = CONV_FACT*4*np.pi*potential/(volume)
        potential_self = 0 

        gs = self.real_grid @ lattice

        potential = np.zeros(natom)
        for i in range(natom):
            disp = position - coords[i] 
            charge = self.charges[i] 
            disps = disp + gs
            g2s = np.sum(disps**2, 1)
            inds = g2s > 1e-8
            ds = np.sqrt(g2s[inds])
            potential[i] += np.sum(charge*erfc(alpha*ds)/(ds))
            if np.min(g2s) < 1e-8:
                potential_self = CONV_FACT*2*self.alpha*self.charges[i]/sqrt_pi

        potential_real = CONV_FACT*potential
        

        potential = np.zeros(natom)
        for i in range(natom):
            disp = position - coords[i] 
            dsq = np.sum(disp**2)
            charge = self.charges[i] 
            potential[i] = charge*dsq

        if type == 'sp':
            potential_ex = -potential*2*CONV_FACT*np.pi/volume**3
        elif type == 'pl':
            potential_ex = -potential*2*CONV_FACT*np.pi/volume
        else:
            potential_ex = 0
        
        potential = np.sum(potential_recip) + np.sum(potential_real) - potential_self + np.sum(potential_ex)
        #print('@', potential_recip, potential_real, potential_self)

        return potential

    def get_madelung_constant(self):
        alpha = self.alpha 
        natom = len(self.structure)
        
        elements = self.structure.get_elements()
        coords = self.structure.get_positions(type='cartesian')
        lattice = self.structure.lattice
        rec_lattice = self.structure._cell.reciprocal * 2 * np.pi
        distances = self.structure.get_all_distances()
        a0 = (self.structure.volume)**(1/3)

        gs = self.recip_grid @ rec_lattice
        g2s = np.sum(gs**2, 1)     

        potential = np.zeros((natom, natom))
        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[j]
                for k,k2 in zip(gs,g2s): 
                    structure_factor = charge*(np.cos(np.dot(k,disp)))
                    exponential = np.exp(-k2/(4.0*alpha*alpha))/k2
                    potential[i,j] += exponential*structure_factor   

        potential_recip = CONV_FACT*4*np.pi*potential/(self.structure.volume)

        gs = self.real_grid @ lattice
        g2s = np.sum(gs**2, 1)     

        potential = np.zeros((natom, natom))
        for i in range(natom):
            for j in range(natom):
                disp = coords[i] - coords[j]
                charge = self.charges[j] 
                disps = disp + gs
                g2s = np.sum(disps**2, 1)
                inds = g2s > 1e-8
                ds = np.sqrt(g2s[inds])
                potential[i,j] += np.sum(charge*erfc(alpha*ds)/(ds))

        potential_real = CONV_FACT*potential   
       
        potential_self = CONV_FACT*2*self.alpha*self.charges/sqrt_pi

        potential = np.sum(potential_recip,axis=1) + np.sum(potential_real,axis=1) - potential_self
                
        madelung_nn = 0
        madelung_a0 = 0
        for i in range(natom):
            nn_idx = np.argmin(distances[i])
            nn = np.min(distances[i])
            madelung_nn += abs(potential[i]*nn/CONV_FACT) * abs(self.charges[i])
            madelung_a0 += abs(potential[i]*a0/CONV_FACT) * abs(self.charges[i])

        Z = self.structure.get_formula_units_Z()
        madelung_nn = madelung_nn/2/Z
        madelung_a0 = madelung_a0/2/Z

        return madelung_nn, madelung_a0

if __name__ == '__main__':

    vaspfile = './ewald/Cu2Te2.vasp'

    '''
    # calc with jamip
    s1 = read(vaspfile)
    ewald = Ewald2D(s1)
    ewald.add_charge({'Cu':1, 'Te':-2})
    E = ewald.total_energy()

    ewald = Ewald3D(s1)
    ewald.add_charge({'Cu':1, 'Te':-2})
    E = ewald.total_energy()

    # compare with pymatgen
    s2 = mps.from_file(vaspfile)
    s2.add_oxidation_state_by_element({'Cu':1, 'Te':-2})
    ewald = EwaldSummation(s2, compute_forces=False)
    for i in range(len(s1)):
        site_energy = ewald.get_site_energy(i)
        print(i, site_energy)
    print(ewald.total_energy)
    print(ew._recip)
    print(ew._real)
    print(ew._point)
    print('recip:',sum(sum(ew._recip)))
    print('real',sum(sum(ew._real)))
    print('point',sum(ew._point))
    print('cell',ew._charged_cell_energy)
    '''

