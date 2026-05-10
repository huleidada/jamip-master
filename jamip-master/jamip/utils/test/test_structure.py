import pytest
from jamip.structure import read,write,Structure
import numpy as np

# 1. 读取cif文件，访问常用属性
# 2. 正确的读写vasp/cif/xyz文件
# 3. 分析

class Test_structure:

    structure = None

    def test_01(self,tmp_path_factory,request):
        # 测试读写功能
        # 读取软件包根目录下的测试文件
        cifdir = request.config.rootdir / 'testfile/structure'
        s0 = read(cifdir/'MoS2.vasp')
        d = tmp_path_factory.mktemp("structure-io")
        write(s0, d/'POSCAR', ftype='poscar')
        write(s0, d/'MoS2.cif')
        write(s0, d/'MoS2.xyz')
        s1 = read(d/'MoS2.cif')
        s2 = read(d/'POSCAR')
        np.testing.assert_allclose(s0.lattice, s1.lattice, rtol=0, atol=1e-6)
        np.testing.assert_allclose(s0.lattice, s2.lattice, rtol=0, atol=1e-8)
 
    def test_02(self,request):
        cifdir = request.config.rootdir / 'testfile/structure'
        s0 = read(cifdir/'MoS2.vasp')
        # 测试访问属性
        np.testing.assert_equal(s0.species_of_elements, ['Mo','S'])
        np.testing.assert_equal(s0.number_of_atoms, [2,4])
        np.testing.assert_equal(s0.get_formula(), 'Mo2S4')
        np.testing.assert_equal(s0.get_formula(reduced=True), 'MoS2')
        np.testing.assert_allclose(s0.volume, s0._cell.volume, rtol=1e-8, atol=0)

    def test_03(self,tmp_path_factory,request):
        # 面向计算的属性设置
        cifdir = request.config.rootdir / 'testfile/structure'
        d = tmp_path_factory.mktemp("structure-io")
        s0 = read(cifdir/'MoS2.vasp')
        lattice0 = s0.lattice
        # 添加附加属性
        s0.initial_velocity = np.ones((6,3))
        s0.select_dynamic = [True,True,False,False,False,False]
        np.testing.assert_equal(s0.atomic_positions[0].velocity, np.ones(3))
        assert s0.atomic_positions[0].freeze == (True,True,True)
        s0.scale_factor = 1.01

        # 读写附加属性
        write(s0, d/'POSCAR', ftype='poscar')
        s1 = read(d/'POSCAR')
        np.testing.assert_equal(s1.atomic_positions[0].velocity, np.ones(3))
        np.testing.assert_allclose(s0.lattice*1.01, s1.lattice, rtol=0, atol=1e-8)
        assert s1.atomic_positions[0].freeze == (True,True,True)

    def _04(self):
        # 分析成键环境
        from jamip.structure.bonding import Bonding
        from jamip.structure.dimension import DimensionAnalysis, connectivity_index
        import numpy as np
 
        bond = Bonding(s)
        print(bond.data.full_repr())
        bd = bond.get_bond_by_atom('Mo')
        bd = bond.get_angle_by_atom('Mo')
        bd = bond.get_bond_by_pair(['Mo','S'])
        print(np.mean(bd[:,-1]))
       
        print(list(bond.data.classify()))
        print(list(bond.data.get_coordination_matrix()))
       
        dim = DimensionAnalysis(s)  # 初始化结构类
        dim.set_valence()
        dim.set_bonding()
        print(dim.valences)
        units = dim.search_cutoff()
        print(units)
        print(units[0].rank)
        print(units[0].valences)

if __name__ == '__main__':
    pytest.main(['-vs'])
