from jamip.structure import Structure,read,write
from jamip.modeling.structureFactory import StructureFactory
from jamip.modeling.interfaceFactory import InterfaceFactory
import numpy as np
import pytest
import pathlib



class Test_slme:

    def test_twist(self,request):

        cif = request.config.rootdir / 'testfile/structure/MoS2.vasp'

        # interface match
        s0 = read(cif)
        interface = InterfaceFactory((s0,s0))
        angles = np.arange(12,20,0.1)
        #for row in interface.twister_match(angles, nmiller=10, tolerate_vector_mismatch=0.04):
        #    print(row)

        for row in interface.twister_match(angles, nmiller=10, fix_angle=60, tolerate_vector_mismatch=0.04, tolerate_angle_mismatch=0.01):
            angle, matrix1, matrix2, tol = row
            filename = '%.1f-%.4f.vasp' %(angle, tol[0]) 
            interface._layers[0].supercell_matrix = matrix1
            interface._layers[1].supercell_matrix = matrix2
            s1 = interface.attach(spacing=3)
            write(s1, filename)
            break

    def test_zsl(self,request):

        cif1 = request.config.rootdir / 'testfile/structure/MoS2.vasp'
        cif2 = request.config.rootdir / 'testfile/structure/WS2.vasp'

        # interface match
        s0 = read(cif1)
        s1 = read(cif2)
        interface = InterfaceFactory((s0,s1))

        for row in interface.match_zsl(max_area=400, area_tol=0.1, length_tol=0.05, angle_tol=0.08):
            print(row)
            miller1, miller2, vector1, vector2 = row
            # filename = '%.1f-%.4f.vasp' %(angle, tol[0]) 
            # print(filename)
            interface._layers[0].supercell_matrix = miller1
            interface._layers[1].supercell_matrix = miller2
            s2 = interface.attach(spacing=3)
            write(s2, 'zwl.vasp')
            break    


if __name__ == '__main__':
    pytest.main(['-vs'])

    # rootdir = pathlib.Path('/public/home/slluo/test/jamip-test/')
    # cif = rootdir / 'testfile/structure/MoS2.vasp'
    # s0 = read(cif)
    # interface = InterfaceFactory((s0,s0))
    # angles = np.arange(12,20,0.1)
    # #for row in interface.twister_match(angles, nmiller=10, tolerate_vector_mismatch=0.04):
    # #    print(row)

    # for row in interface.twister_match(angles, nmiller=10, fix_angle=60, tolerate_vector_mismatch=0.04, tolerate_angle_mismatch=0.01):
    #     print(row)
    #     angle, matrix1, matrix2, tol = row
    #     filename = '%.1f-%.4f.vasp' %(angle, tol[0]) 
    #     print(filename)
    #     interface._layers[0].supercell_matrix = matrix1
    #     interface._layers[1].supercell_matrix = matrix2
    #     s1 = interface.attach(spacing=3)
    #     write(s1, filename)
    #     # break

    # rootdir = pathlib.Path('/public/home/slluo/test/jamip-test/')
    # cif = rootdir / 'testfile/structure/MoS2.vasp'
    # cif2 = rootdir / 'testfile/structure/WS2.vasp'

    # # interface match
    # s0 = read(cif)
    # s1 = read(cif2)
    # interface = InterfaceFactory((s0,s1))

    # for row in interface.match_zsl(max_area=400, area_tol=0.1, length_tol=0.05, angle_tol=0.08):
    #     print(row)
    #     miller1, miller2, vector1, vector2 = row
    #     # filename = '%.1f-%.4f.vasp' %(angle, tol[0]) 
    #     # print(filename)
    #     interface._layers[0].supercell_matrix = miller1
    #     interface._layers[1].supercell_matrix = miller2
    #     s2 = interface.attach(spacing=3)
    #     write(s2, 'zwl.vasp')
    #     break    