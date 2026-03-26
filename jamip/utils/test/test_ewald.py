from jamip.analysis.vasp.ewald import Ewald2D, Ewald3D
from jamip.structure import read
import numpy as np
import pytest


class Test_ewald:

    def test_2d_force(self,request):
        # 按二维结构计算材料的ewald能
        vaspfile = request.config.rootdir / 'testfile/structure/Cu2Te2.vasp'
        # move atoms z-center
        s1 = read(vaspfile)
        positions = s1.get_positions() + np.array([0,0,0.5])
        positions = positions - np.floor(positions)
        s1.atomic_positions = positions
 
        # calc with jamip
        ewald = Ewald2D(s1)
        ewald.add_charge({'Cu':1, 'Te':-2})
        ewald._calc_force = True
        E = ewald.total_energy()
        F = ewald.force()
        recip1 = ewald.f_recip
        real1 = ewald.f_real

        print(E)
        print(F)
        print('recip')
        print(recip1)
        print('real')
        print(real1)
        #TODO: assert value function
        assert abs(E - (-66.527)) < 1e-3
        assert abs(F[-1,-1] - (-1.668)) < 1e-3
        

    def test_3d_force(self,request):
        from pymatgen.core.structure import Structure       
        from pymatgen.analysis.ewald import EwaldSummation
         
        # 按三维结构计算材料的ewald能
        vaspfile = request.config.rootdir / 'testfile/structure/Cu2Te2.vasp'

        # calc with jamip
        s1 = read(vaspfile)
        ewald = Ewald3D(s1)
        ewald.add_charge({'Cu':1, 'Te':-2})
        ewald._calc_force = True
        F1 = ewald.force()
        recip1 = ewald.f_recip
        real1 = ewald.f_real
 
        # compare with pymatgen
        s2 = Structure.from_file(vaspfile)
        s2.add_oxidation_state_by_element({'Cu':1, 'Te':-2})
        ewald = EwaldSummation(s2, compute_forces=True)
        F2 = ewald.forces
        force = np.sum(ewald._recip)
        real2 = np.sum(ewald._real)
        epoint2 = np.sum(ewald._point)
        #assert abs(recip1 - recip2) < 1e-4
        #assert abs(real1 - real2) < 1e-4
        assert np.sum(abs(F2 - F1)) < 1e-4, "Total_energy_jamip = %f\tTotal_energy_pymatgen = %f" %(F1,F2)
 
    def test_3d_madelung_constant(self,request):
        from pymatgen.core.structure import Structure       
        from pymatgen.analysis.ewald import EwaldSummation
        '''https://zhuanlan.zhihu.com/p/615288766'''
         
        # 按三维结构计算材料的ewald能
        vaspfile = request.config.rootdir / 'testfile/structure/NaCl.vasp'
        vaspfile2 = request.config.rootdir / 'testfile/structure/NaCl-cubic.vasp'

        # calc with primcell
        s1 = read(vaspfile)
        ewald = Ewald3D(s1)
        ewald.add_charge({'Na':1, 'Cl':-1, })
        potential1 = ewald.get_madelung_potential([0,0,0])
        constants1 = ewald.get_madelung_constant()[0]

        s1 = read(vaspfile2)
        ewald = Ewald3D(s1)
        ewald.add_charge({'Na':1, 'Cl':-1, })
        potential2 = ewald.get_madelung_potential([0,0,0])
        constants2 = ewald.get_madelung_constant()[0]

        assert np.sum(abs(constants1 - 1.747565)) < 1e-4, "madelung_constant_jamip_prim = %f" %constants1
        assert np.sum(abs(constants2 - 1.747565)) < 1e-4, "madelung_constant_jamip_super = %f" %constants2
        assert np.sum(abs(potential1 - potential2)) < 1e-4, "mp_prim = %f\tmp_super = %f" %(potential1, potential2)
 
    def test_3d(self,request):
        from pymatgen.core.structure import Structure       
        from pymatgen.analysis.ewald import EwaldSummation
         
        # 按三维结构计算材料的ewald能
        vaspfile = request.config.rootdir / 'testfile/structure/Cu2Te2.vasp'

        # calc with jamip
        s1 = read(vaspfile)
        ewald = Ewald3D(s1)
        ewald.add_charge({'Cu':1, 'Te':-2})
        E1 = ewald.total_energy()
        recip1 = np.sum(ewald.e_recip)
        real1 = np.sum(ewald.e_real)
        epoint1 = np.sum(ewald.epoint)
 
        # compare with pymatgen
        s2 = Structure.from_file(vaspfile)
        s2.add_oxidation_state_by_element({'Cu':1, 'Te':-2})
        ewald = EwaldSummation(s2, compute_forces=False)
        #for i in range(len(s1)):
        #    site_energy = ewald.get_site_energy(i)
        #    print(i, site_energy)
        E2 = ewald.total_energy
        recip2 = np.sum(ewald._recip)
        real2 = np.sum(ewald._real)
        epoint2 = np.sum(ewald._point)
        assert abs(recip1 - recip2) < 1e-4
        assert abs(real1 - real2) < 1e-4
        assert abs(epoint1 - epoint2) < 1e-3
        assert abs(E2 - E1) < 1e-3, "Total_energy_jamip = %f\tTotal_energy_pymatgen = %f" %(E1,E2)

if __name__ == '__main__':
    pytest.main(['-vs'])
